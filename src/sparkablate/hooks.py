"""Hook-based ablation of transformer components.

Works against Hugging Face causal LMs without modifying model code. Three
intervention sites per decoder layer, plus whole-layer skip:

- head:  zero/mean-patch the per-head slice of the attention output
         projection's *input* (pre-o_proj), which removes that head's
         contribution to the residual stream.
- attn:  replace the whole attention block's output.
- mlp:   replace the MLP block's output.
- layer: skip the decoder layer entirely (output = input hidden states).

Modes: "zero" writes zeros; "mean" writes per-dimension mean activations
collected from a calibration pass (mean ablation avoids pushing the model
off-distribution as hard as zeroing does).
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import torch
from torch import nn

ATTN_NAMES = ("self_attn", "attn", "attention", "self_attention")
MLP_NAMES = ("mlp", "feed_forward", "ffn", "block_sparse_moe")
OPROJ_NAMES = ("o_proj", "out_proj", "c_proj", "dense", "wo")
LAYER_CONTAINER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "transformer.blocks",
)

KINDS = ("head", "attn", "mlp", "layer")
MODES = ("zero", "mean")
DIRECTION_OPS = ("project_out", "add")


@dataclass(frozen=True)
class AblationSpec:
    """One ablation intervention. ``heads`` is only used when kind == 'head'."""

    kind: str
    layer: int
    heads: tuple[int, ...] = ()
    mode: str = "zero"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.kind == "head" and not self.heads:
            raise ValueError("kind='head' requires at least one head index")
        if self.kind == "layer" and self.mode == "mean":
            raise ValueError("layer ablation is a skip; mode='mean' does not apply")

    def label(self) -> str:
        if self.kind == "head":
            return f"L{self.layer}.heads[{','.join(map(str, self.heads))}].{self.mode}"
        if self.kind == "layer":
            return f"L{self.layer}.skip"
        return f"L{self.layer}.{self.kind}.{self.mode}"


@dataclass(frozen=True)
class DirectionSpec:
    """A residual-stream direction intervention.

    - ``project_out``: x ← x − coefficient·(x·v̂)v̂ at the embedding output and
      every decoder layer output (coefficient 1.0 = full directional ablation).
    - ``add``: x ← x + coefficient·v̂, typically at a single layer (steering).

    ``layers='all'`` hooks the embedding output plus every decoder layer;
    a tuple of layer indices hooks exactly those decoder layers (no embedding).
    The vector comes from ``vector_path`` (a DirectionArtifact .pt file) or is
    passed directly to ``AblationManager.apply_direction``.
    """

    op: str = "project_out"
    coefficient: float = 1.0
    layers: tuple[int, ...] | str = "all"
    vector_path: str | None = None

    def __post_init__(self):
        if self.op not in DIRECTION_OPS:
            raise ValueError(f"op must be one of {DIRECTION_OPS}, got {self.op!r}")
        if self.layers != "all":
            object.__setattr__(self, "layers", tuple(self.layers))

    def label(self) -> str:
        where = "all" if self.layers == "all" else ",".join(map(str, self.layers))
        return f"dir.{self.op}[L{where}]x{self.coefficient:g}"


@dataclass
class LayerHandles:
    layer: nn.Module
    attn: nn.Module
    mlp: nn.Module
    o_proj: nn.Module


@dataclass
class ArchInfo:
    layers: list[LayerHandles]
    num_heads: int
    head_dim: int
    num_kv_heads: int = 0
    # which name table entry matched each component ('container' may be 'fallback')
    resolution: dict[str, str] = field(default_factory=dict)
    # set to a reason string when per-head slicing assumptions don't hold
    head_slicing_error: str | None = None

    @property
    def num_layers(self) -> int:
        return len(self.layers)


def _resolve_path(root: nn.Module, path: str) -> nn.Module | None:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _find_child(module: nn.Module, names: tuple[str, ...]) -> tuple[nn.Module | None, str | None]:
    for name in names:
        child = getattr(module, name, None)
        if isinstance(child, nn.Module):
            return child, name
    return None, None


def _child_names(module: nn.Module) -> list[str]:
    return [n for n, _ in module.named_children()]


def _find_layer_container(model: nn.Module) -> tuple[nn.ModuleList, str]:
    for path in LAYER_CONTAINER_PATHS:
        obj = _resolve_path(model, path)
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj, path
    # Fallback: largest ModuleList whose elements look like decoder layers.
    candidates = [
        m
        for m in model.modules()
        if isinstance(m, nn.ModuleList)
        and len(m) > 0
        and _find_child(m[0], ATTN_NAMES)[0] is not None
        and _find_child(m[0], MLP_NAMES)[0] is not None
    ]
    if not candidates:
        raise RuntimeError(
            "Could not locate the decoder layer stack; this architecture needs "
            "an entry in LAYER_CONTAINER_PATHS (sparkablate/hooks.py). "
            f"Top-level children: {_child_names(model)}"
        )
    return max(candidates, key=len), "fallback"


def _oproj_in_width(module: nn.Module) -> int | None:
    """Input width of the attention output projection, if statically knowable."""
    if isinstance(module, nn.Linear):
        return module.in_features
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        # transformers Conv1D (GPT-2 style) stores weight as [in, out]
        if type(module).__name__ == "Conv1D":
            return weight.shape[0]
        return weight.shape[1]
    return None


def introspect(model: nn.Module, strict: bool = True) -> ArchInfo:
    """Map out the decoder layers and per-layer attention/MLP submodules.

    With ``strict=True`` (default), an architecture that only resolves via the
    heuristic fallback raises instead of silently guessing — better to fail at
    load time than to run a long sweep against the wrong modules. Pass
    ``strict=False`` to accept the fallback with a warning.

    Note on layernorm placement (e.g. Gemma-2/3 post-block layernorms): block
    interventions act on the attn/MLP module *output*, i.e. pre-`post_*_layernorm`
    where such norms exist. Mean activations are captured at the same site they
    are replayed, so mean-mode semantics stay consistent per architecture.
    """
    container, source = _find_layer_container(model)
    if source == "fallback":
        msg = (
            "Decoder layer stack found only via heuristic fallback (largest "
            "ModuleList with attn+mlp children) — add this architecture's path "
            "to LAYER_CONTAINER_PATHS in sparkablate/hooks.py to make detection "
            "explicit. Pass strict=False to proceed anyway."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    layers = []
    attn_name = mlp_name = oproj_name = None
    for idx, layer in enumerate(container):
        attn, attn_name = _find_child(layer, ATTN_NAMES)
        mlp, mlp_name = _find_child(layer, MLP_NAMES)
        if attn is None or mlp is None:
            missing = "attention" if attn is None else "MLP"
            raise RuntimeError(
                f"Layer {idx} ({type(layer).__name__}) has no recognizable {missing} "
                f"submodule; its children are {_child_names(layer)}. Add the right "
                "name to ATTN_NAMES/MLP_NAMES in sparkablate/hooks.py."
            )
        o_proj, oproj_name = _find_child(attn, OPROJ_NAMES)
        if o_proj is None:
            raise RuntimeError(
                f"Attention {type(attn).__name__} has no recognizable output "
                f"projection; its children are {_child_names(attn)}. Add the right "
                "name to OPROJ_NAMES in sparkablate/hooks.py."
            )
        layers.append(LayerHandles(layer=layer, attn=attn, mlp=mlp, o_proj=o_proj))

    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads

    head_slicing_error = None
    width = _oproj_in_width(layers[0].o_proj)
    if width is not None and width != num_heads * head_dim:
        head_slicing_error = (
            f"o_proj input width {width} != num_heads*head_dim "
            f"{num_heads * head_dim}; per-head slicing is unavailable for this "
            "architecture (attn/mlp/layer kinds still work)"
        )

    resolution = {
        "container": source,
        "attn": attn_name or "?",
        "mlp": mlp_name or "?",
        "o_proj": oproj_name or "?",
    }
    return ArchInfo(
        layers=layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        resolution=resolution,
        head_slicing_error=head_slicing_error,
    )


def _first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_first(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + tuple(output[1:])
    return tensor


class _MeanAccumulator:
    """Running per-dimension mean over all batch/sequence positions."""

    def __init__(self):
        self.total: torch.Tensor | None = None
        self.count = 0

    def update(self, x: torch.Tensor):
        flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        s = flat.sum(dim=0)
        self.total = s if self.total is None else self.total + s
        self.count += flat.shape[0]

    def mean(self) -> torch.Tensor:
        if self.total is None or self.count == 0:
            raise RuntimeError("No activations captured during calibration")
        return self.total / self.count


class AblationManager:
    """Applies and removes ablation hooks on a loaded model."""

    def __init__(self, model: nn.Module, strict: bool = True):
        self.model = model
        self.arch = introspect(model, strict=strict)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # means[(layer_idx, site)] -> float32 vector; sites: oproj_in, attn_out, mlp_out
        self.means: dict[tuple[int, str], torch.Tensor] = {}

    # ---------------------------------------------------------------- means

    def calibrate_means(self, batches, forward_fn=None) -> None:
        """Run calibration batches and record per-site mean activations.

        ``batches`` yields input_ids tensors; ``forward_fn(input_ids)`` may be
        supplied to customize the forward pass (defaults to a plain call with
        use_cache=False).
        """
        accs: dict[tuple[int, str], _MeanAccumulator] = {}
        handles = []

        def capture_pre(key):
            def hook(module, args):
                accs.setdefault(key, _MeanAccumulator()).update(args[0])
            return hook

        def capture_fwd(key):
            def hook(module, args, output):
                accs.setdefault(key, _MeanAccumulator()).update(_first_tensor(output))
            return hook

        for i, lh in enumerate(self.arch.layers):
            handles.append(lh.o_proj.register_forward_pre_hook(capture_pre((i, "oproj_in"))))
            handles.append(lh.attn.register_forward_hook(capture_fwd((i, "attn_out"))))
            handles.append(lh.mlp.register_forward_hook(capture_fwd((i, "mlp_out"))))

        try:
            with torch.no_grad():
                for input_ids in batches:
                    if forward_fn is not None:
                        forward_fn(input_ids)
                    else:
                        device = next(self.model.parameters()).device
                        self.model(input_ids=input_ids.to(device), use_cache=False)
        finally:
            for h in handles:
                h.remove()

        self.means = {key: acc.mean() for key, acc in accs.items()}

    def _mean_for(self, layer: int, site: str) -> torch.Tensor:
        key = (layer, site)
        if key not in self.means:
            raise RuntimeError(
                f"Mean activations for layer {layer}/{site} not available; "
                "run calibrate_means() before applying mode='mean' ablations."
            )
        return self.means[key]

    # ---------------------------------------------------------------- hooks

    def apply(self, spec: AblationSpec | DirectionSpec) -> None:
        if isinstance(spec, DirectionSpec):
            self.apply_direction(spec)
            return
        lh = self.arch.layers[spec.layer]
        if spec.kind == "head":
            if self.arch.head_slicing_error:
                raise RuntimeError(self.arch.head_slicing_error)
            self._handles.append(
                lh.o_proj.register_forward_pre_hook(self._head_prehook(spec))
            )
        elif spec.kind == "attn":
            self._handles.append(
                lh.attn.register_forward_hook(self._block_hook(spec, "attn_out"))
            )
        elif spec.kind == "mlp":
            self._handles.append(
                lh.mlp.register_forward_hook(self._block_hook(spec, "mlp_out"))
            )
        elif spec.kind == "layer":
            self._handles.append(
                lh.layer.register_forward_hook(self._layer_skip_hook(), with_kwargs=True)
            )

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def apply_direction(self, spec: DirectionSpec, vector: torch.Tensor | None = None) -> None:
        if vector is None:
            if spec.vector_path is None:
                raise ValueError("DirectionSpec needs vector_path or an explicit vector")
            from sparkablate.directions import DirectionArtifact

            vector = DirectionArtifact.load(spec.vector_path).vector
        v = vector.detach().to(torch.float32)
        v = v / v.norm()
        hook = self._direction_hook(spec, v)

        if spec.layers == "all":
            self._handles.append(
                self.model.get_input_embeddings().register_forward_hook(hook)
            )
            targets = range(self.arch.num_layers)
        else:
            targets = spec.layers
        for i in targets:
            self._handles.append(self.arch.layers[i].layer.register_forward_hook(hook))

    @contextlib.contextmanager
    def applied(self, specs: AblationSpec | DirectionSpec | list):
        if isinstance(specs, (AblationSpec, DirectionSpec)):
            specs = [specs]
        try:
            for spec in specs:
                self.apply(spec)
            yield self
        finally:
            self.clear()

    # ------------------------------------------------------------ hook impls

    def _head_prehook(self, spec: AblationSpec):
        head_dim = self.arch.head_dim
        num_heads = self.arch.num_heads
        bad = [h for h in spec.heads if not 0 <= h < num_heads]
        if bad:
            raise ValueError(f"Head indices {bad} out of range (num_heads={num_heads})")
        mean_vec = self._mean_for(spec.layer, "oproj_in") if spec.mode == "mean" else None

        def hook(module, args):
            x = args[0]
            expected = num_heads * head_dim
            if x.shape[-1] != expected:
                raise RuntimeError(
                    f"o_proj input dim {x.shape[-1]} != num_heads*head_dim {expected}; "
                    "head slicing assumption does not hold for this architecture"
                )
            x = x.clone()
            for h in spec.heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                if mean_vec is None:
                    x[..., sl] = 0
                else:
                    x[..., sl] = mean_vec[sl].to(dtype=x.dtype, device=x.device)
            return (x,) + tuple(args[1:])

        return hook

    def _block_hook(self, spec: AblationSpec, site: str):
        mean_vec = self._mean_for(spec.layer, site) if spec.mode == "mean" else None

        def hook(module, args, output):
            out = _first_tensor(output)
            if mean_vec is None:
                repl = torch.zeros_like(out)
            else:
                repl = mean_vec.to(dtype=out.dtype, device=out.device).expand_as(out).contiguous()
            return _replace_first(output, repl)

        return hook

    @staticmethod
    def _direction_hook(spec: DirectionSpec, unit_vector: torch.Tensor):
        cache: dict[torch.device, torch.Tensor] = {}

        def hook(module, args, output):
            x = _first_tensor(output)
            v = cache.get(x.device)
            if v is None:
                v = unit_vector.to(x.device)
                cache[x.device] = v
            x32 = x.to(torch.float32)
            if spec.op == "project_out":
                new = x32 - spec.coefficient * (x32 @ v).unsqueeze(-1) * v
            else:  # add
                new = x32 + spec.coefficient * v
            return _replace_first(output, new.to(x.dtype))

        return hook

    @staticmethod
    def _layer_skip_hook():
        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise RuntimeError("Could not recover input hidden_states for layer skip")
            return _replace_first(output, hidden)

        return hook
