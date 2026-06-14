"""Bake ablation interventions permanently into model weights.

Where :mod:`sparkablate.hooks` applies interventions as *runtime* forward hooks
that are removed afterwards, this module rewrites the model's own parameters so
the intervention survives ``save_pretrained``. The result is a standard Hugging
Face checkpoint that loads in stock transformers / vLLM / TGI (and, after a
documented GGUF conversion, LM Studio) with **no custom modeling code and no
hooks**.

Every supported intervention acts on a linear projection that writes to the
residual stream, so each has an exact equivalent static weight edit:

- ``head`` zero   → zero the head's input columns of ``o_proj.weight``.
- ``attn`` zero   → zero ``o_proj.weight`` (and its bias if any).
- ``mlp`` zero    → zero ``down_proj.weight`` (and its bias if any).
- ``layer`` skip  → ``attn`` zero + ``mlp`` zero (the layer becomes an identity).
- ``direction`` ``project_out`` → orthogonalize every residual-stream *writer*
  (the input embedding plus every ``o_proj`` / ``down_proj``) against the unit
  direction: ``W ← W − coef·v̂(v̂ᵀW)``. This is the standard "abliteration"
  weight form of directional ablation: each write's output is forced orthogonal
  to v̂, which equals projecting v̂ out at every residual write.

Deferred (see the project README): ``mean`` mode (would require injecting a bias
where LLaMA-family projections have none, which stock vLLM/GGUF loaders drop),
and ``direction`` over a *subset* of layers (path-dependent, no static weight
equivalent).
"""

from __future__ import annotations

import json
import os
import time

import torch
from torch import nn

from sparkablate import __version__
from sparkablate.hooks import (
    AblationSpec,
    ArchInfo,
    DirectionSpec,
    _find_layer_container,
)


class BakeError(ValueError):
    """Raised when a spec cannot be baked into weights faithfully."""


# --------------------------------------------------------------------- helpers

def _is_conv1d(module: nn.Module) -> bool:
    # transformers' GPT-2-style Conv1D stores weight as [in, out] (vs nn.Linear
    # [out, in]); the residual/output dimension is therefore axis 1, not 0.
    return type(module).__name__ == "Conv1D"


def _hidden_axis(module: nn.Module) -> int:
    """Axis of ``module.weight`` indexing the output (residual) dimension."""
    return 1 if _is_conv1d(module) else 0


def _project_param(param: nn.Parameter, v: torch.Tensor, coef: float, hidden_axis: int) -> None:
    """In place: remove the v̂ component from every hidden-vector in ``param``.

    Math is done in float32 then cast back to the parameter's dtype, mirroring
    the runtime direction hook (hooks.py) and ``_MeanAccumulator``.
    """
    orig_dtype = param.dtype
    w = param.data.to(torch.float32)
    if w.ndim == 1:  # a bias added to the residual stream
        w = w - coef * torch.dot(w, v) * v
    elif hidden_axis == 0:  # [hidden, in]; output dim is the rows
        w = w - coef * torch.outer(v, v @ w)
    else:  # [n, hidden]; each row is a hidden-vector (embedding / Conv1D)
        w = w - coef * torch.outer(w @ v, v)
    param.data.copy_(w.to(orig_dtype))


def _zero_module_output(module: nn.Module) -> None:
    module.weight.data.zero_()
    if getattr(module, "bias", None) is not None:
        module.bias.data.zero_()


def _spec_summary(spec: AblationSpec | DirectionSpec) -> dict:
    if isinstance(spec, AblationSpec):
        return {"kind": spec.kind, "layer": spec.layer,
                "heads": list(spec.heads), "mode": spec.mode}
    return {"kind": "direction", "op": spec.op, "coefficient": spec.coefficient,
            "layers": spec.layers if spec.layers == "all" else list(spec.layers),
            "vector_path": spec.vector_path}


# ----------------------------------------------------------------- per-kind bake

def bake_head_zero(arch: ArchInfo, layer: int, heads) -> None:
    if arch.head_slicing_error:
        raise BakeError(arch.head_slicing_error)
    o_proj = arch.layers[layer].o_proj
    head_dim, num_heads = arch.head_dim, arch.num_heads
    bad = [h for h in heads if not 0 <= h < num_heads]
    if bad:
        raise BakeError(f"head indices {bad} out of range (num_heads={num_heads})")
    conv1d = _is_conv1d(o_proj)
    for h in heads:
        sl = slice(h * head_dim, (h + 1) * head_dim)
        if conv1d:  # weight [in, out]: input dims are rows
            o_proj.weight.data[sl, :] = 0
        else:       # weight [out, in]: input dims are columns
            o_proj.weight.data[:, sl] = 0


def bake_attn_zero(arch: ArchInfo, layer: int) -> None:
    _zero_module_output(arch.layers[layer].o_proj)


def bake_mlp_zero(arch: ArchInfo, layer: int) -> None:
    down_proj = arch.layers[layer].down_proj
    if down_proj is None:
        raise BakeError(
            f"layer {layer} MLP has no single output projection (MoE block?); "
            "baking 'mlp'/'layer' for this architecture is not supported in v1"
        )
    _zero_module_output(down_proj)


def bake_layer_skip(arch: ArchInfo, layer: int) -> None:
    bake_attn_zero(arch, layer)
    bake_mlp_zero(arch, layer)


def bake_direction_project_out(model: nn.Module, arch: ArchInfo,
                               vector: torch.Tensor, coefficient: float = 1.0) -> None:
    """Orthogonalize every residual-stream writer against ``vector``."""
    v = vector.detach().to(torch.float32).flatten()
    norm = v.norm()
    if norm == 0:
        raise BakeError("direction vector is zero")
    v = v / norm

    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise BakeError("could not locate the input embedding to orthogonalize")
    _project_param(emb.weight, v, coefficient, hidden_axis=1)  # [vocab, hidden]

    for i, lh in enumerate(arch.layers):
        if lh.down_proj is None:
            raise BakeError(
                f"layer {i} MLP has no single output projection (MoE block?); "
                "baking 'direction' for this architecture is not supported in v1"
            )
        for module in (lh.o_proj, lh.down_proj):
            ha = _hidden_axis(module)
            _project_param(module.weight, v, coefficient, ha)
            if getattr(module, "bias", None) is not None:
                _project_param(module.bias, v, coefficient, ha)


def prune_skipped_layers(model: nn.Module, layer_indices) -> list[int]:
    """Physically delete decoder layers and update ``config.num_hidden_layers``.

    Applied last (after all index-based edits) so earlier specs still reference
    original layer indices. Returns the sorted list of pruned indices.
    """
    container, _ = _find_layer_container(model)
    pruned = sorted(set(layer_indices), reverse=True)
    for i in pruned:
        del container[i]
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(container)
    return sorted(set(layer_indices))


# -------------------------------------------------------------------- orchestrate

def bake_specs(model: nn.Module, arch: ArchInfo, specs, *,
               prune_layers: bool = False) -> dict:
    """Apply every spec to the model's weights in a compatibility-safe order.

    Phase 1: structural zero/skip edits (using original layer indices).
    Phase 2: direction ``project_out`` (whole-stack only).
    Phase 3: optional physical pruning of skipped layers.

    Returns a dict describing what was baked (for the model card / metadata).
    Raises :class:`BakeError` for any spec that cannot be baked faithfully.
    """
    if isinstance(specs, (AblationSpec, DirectionSpec)):
        specs = [specs]

    ablations = [s for s in specs if isinstance(s, AblationSpec)]
    directions = [s for s in specs if isinstance(s, DirectionSpec)]

    # ----- validation up front (fail before touching any weights) -----
    for s in ablations:
        if not 0 <= s.layer < arch.num_layers:
            raise BakeError(f"layer {s.layer} out of range (num_layers={arch.num_layers})")
        if getattr(s, "mode", "zero") == "mean":
            raise BakeError(
                "mean-mode ablation cannot be baked: it needs a constant output, "
                "i.e. a bias on o_proj/down_proj that LLaMA-family models lack and "
                "stock vLLM/GGUF loaders drop. Use mean mode as a runtime hook "
                "(ablate run / compare), or bake zero mode."
            )
    for d in directions:
        if d.op != "project_out":
            raise BakeError(f"only direction op 'project_out' can be baked, got {d.op!r}")
        if d.layers != "all":
            raise BakeError(
                "baking a direction over a layer subset is unsupported: per-layer "
                "projection is path-dependent and has no static weight equivalent. "
                "Use layers='all' (the find-direction production path) or a runtime hook."
            )

    interventions: list[dict] = []
    skip_layers: list[int] = []

    # ----- phase 1: structural edits -----
    for s in ablations:
        if s.kind == "head":
            bake_head_zero(arch, s.layer, s.heads)
        elif s.kind == "attn":
            bake_attn_zero(arch, s.layer)
        elif s.kind == "mlp":
            bake_mlp_zero(arch, s.layer)
        elif s.kind == "layer":
            if prune_layers:
                skip_layers.append(s.layer)
            else:
                bake_layer_skip(arch, s.layer)
        else:  # pragma: no cover - AblationSpec validates kind
            raise BakeError(f"cannot bake kind {s.kind!r}")
        interventions.append(_spec_summary(s))

    # ----- phase 2: direction -----
    for d in directions:
        vector = _resolve_direction_vector(d)
        bake_direction_project_out(model, arch, vector, d.coefficient)
        summary = _spec_summary(d)
        summary["provenance"] = _direction_provenance(d)
        interventions.append(summary)

    # ----- phase 3: prune (last) -----
    pruned = prune_skipped_layers(model, skip_layers) if skip_layers else []

    return {"interventions": interventions, "pruned_layers": pruned}


def _resolve_direction_vector(spec: DirectionSpec) -> torch.Tensor:
    if spec.vector_path is None:
        raise BakeError("direction spec needs a vector_path to bake")
    from sparkablate.directions import DirectionArtifact

    return DirectionArtifact.load(spec.vector_path).vector


def _direction_provenance(spec: DirectionSpec) -> dict:
    if spec.vector_path is None:
        return {}
    from sparkablate.directions import DirectionArtifact

    art = DirectionArtifact.load(spec.vector_path)
    return {"vector_path": spec.vector_path, "layer_key": art.layer_key,
            "position": art.position, "meta": art.meta}


# ------------------------------------------------------------------------- save

def gguf_command(out_dir: str) -> str:
    """The llama.cpp invocation that turns this checkpoint into a GGUF for LM Studio."""
    name = os.path.basename(os.path.normpath(out_dir)) or "model"
    return f"python convert_hf_to_gguf.py {out_dir} --outfile {name}.gguf"


def save_baked(model: nn.Module, tokenizer, out_dir: str, *, base_model: str,
               interventions: list[dict], dtype: str, pruned_layers=None,
               extra: dict | None = None, log=print) -> dict:
    """Write the baked model + tokenizer + provenance metadata and model card."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "base_model": base_model,
        "sparkablate_version": __version__,
        "dtype": dtype,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "interventions": interventions,
        "pruned_layers": list(pruned_layers or []),
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, "sparkablate_bake.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    gguf = gguf_command(out_dir)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_model_card(meta, gguf))

    log(f"wrote baked checkpoint to {out_dir}")
    log("usable directly with transformers / vLLM / TGI (HF safetensors).")
    log(f"for LM Studio (GGUF), convert with:\n  {gguf}")
    meta["gguf_command"] = gguf
    return meta


def _model_card(meta: dict, gguf: str) -> str:
    lines = [
        f"# {meta['base_model']} (component-ablated)",
        "",
        "Produced by [sparkablate](https://github.com/) "
        f"v{meta['sparkablate_version']} by baking component ablations directly "
        "into the weights. This is a standard Hugging Face checkpoint — it loads "
        "in stock `transformers`, vLLM, and TGI with no custom modeling code.",
        "",
        f"- **Base model:** `{meta['base_model']}`",
        f"- **Saved dtype:** {meta['dtype']}",
        f"- **Baked at:** {meta['created_at']}",
        "",
        "## Interventions baked in",
        "",
    ]
    for iv in meta["interventions"]:
        if iv["kind"] == "direction":
            prov = iv.get("provenance", {}) or {}
            extra = ""
            if prov.get("layer_key") is not None:
                extra = f" (from {prov['layer_key']} @ pos {prov.get('position')})"
            lines.append(
                f"- **direction / {iv['op']}** × {iv['coefficient']:g} over all "
                f"residual writes{extra} — refusal-style directional ablation "
                "(weights orthogonalized against the direction)."
            )
        else:
            heads = f" heads {iv['heads']}" if iv["kind"] == "head" else ""
            lines.append(f"- **{iv['kind']} {iv['mode']}** at layer {iv['layer']}{heads}.")
    if meta["pruned_layers"]:
        lines.append(f"- **pruned layers:** {meta['pruned_layers']} "
                     "(physically removed; `num_hidden_layers` updated).")
    lines += [
        "",
        "## Use with LM Studio (GGUF)",
        "",
        "LM Studio runs GGUF, not raw safetensors. Convert with llama.cpp:",
        "",
        "```bash",
        gguf,
        "```",
        "",
        "## Caveats",
        "",
        "- Baked from the chosen precision; for directional ablation, baking from "
        "a float32-loaded model is recommended to avoid a faint residual along the "
        "direction after casting back to a half precision.",
        "- This is a deliberately modified model intended for research and "
        "evaluation (e.g. safety red-teaming). Test before relying on it.",
    ]
    return "\n".join(lines) + "\n"
