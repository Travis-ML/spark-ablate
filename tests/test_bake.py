"""Weight-baking equivalence: a baked + reloaded checkpoint must reproduce the
runtime hook exactly, with a clean state dict (no custom code, no stray biases).

All CPU, tiny randomly-initialized LLaMA (GQA), no downloads. Embeddings are
untied so the directional case is exactly separable from the unembedding.
"""

import tempfile

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sparkablate.bake import (
    BakeError,
    bake_direction_project_out,
    bake_specs,
)
from sparkablate.hooks import AblationManager, AblationSpec, DirectionSpec, introspect

VOCAB = 128
NUM_LAYERS = 3
NUM_HEADS = 4
NUM_KV_HEADS = 2
HIDDEN = 64


def make_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    m = LlamaForCausalLM(cfg)
    m.eval()
    m.requires_grad_(False)
    return m


def make_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (2, 16))


def logits(model, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def reload_from(tmp: str):
    model, info = LlamaForCausalLM.from_pretrained(tmp, output_loading_info=True)
    model.eval()
    model.requires_grad_(False)
    return model, info


# ----------------------------------------------------------- zero / skip kinds

@pytest.mark.parametrize(
    "spec",
    [
        AblationSpec(kind="head", layer=1, heads=(2,)),
        AblationSpec(kind="head", layer=0, heads=(0, 3)),
        AblationSpec(kind="attn", layer=0),
        AblationSpec(kind="mlp", layer=2),
        AblationSpec(kind="layer", layer=1),
    ],
    ids=lambda s: s.label(),
)
def test_baked_reload_matches_hook(spec):
    model = make_model()
    ids = make_ids()

    mgr = AblationManager(model)
    with mgr.applied(spec):
        hooked = logits(model, ids)

    arch = introspect(model, strict=True)
    bake_specs(model, arch, [spec])
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded, info = reload_from(tmp)
        assert not info["unexpected_keys"], "baked checkpoint has unexpected keys"
        baked = logits(reloaded, ids)

    assert torch.allclose(hooked, baked, atol=1e-5), \
        "baked+reloaded logits must match the runtime hook"


def test_stacked_specs_bake():
    model = make_model()
    ids = make_ids()
    specs = [
        AblationSpec(kind="head", layer=0, heads=(1,)),
        AblationSpec(kind="mlp", layer=2),
    ]
    mgr = AblationManager(model)
    with mgr.applied(specs):
        hooked = logits(model, ids)

    arch = introspect(model, strict=True)
    bake_specs(model, arch, specs)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded, info = reload_from(tmp)
        assert not info["unexpected_keys"]
        baked = logits(reloaded, ids)
    assert torch.allclose(hooked, baked, atol=1e-5)


# ------------------------------------------------------------------- direction

def _project_out_hook(unit_v: torch.Tensor, coef: float):
    def hook(module, args, output):
        x = output[0] if isinstance(output, tuple) else output
        x32 = x.to(torch.float32)
        new = (x32 - coef * (x32 @ unit_v).unsqueeze(-1) * unit_v).to(x.dtype)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new
    return hook


def test_direction_bake_matches_per_write_projection():
    """Orthogonalizing every residual writer == projecting v out of each write."""
    model = make_model()
    ids = make_ids()
    arch = introspect(model, strict=True)

    torch.manual_seed(7)
    v = torch.randn(HIDDEN)
    v = v / v.norm()
    coef = 1.0

    # Reference: hook every residual-stream write (embedding + o_proj + down_proj).
    handles = [model.get_input_embeddings().register_forward_hook(_project_out_hook(v, coef))]
    for lh in arch.layers:
        handles.append(lh.o_proj.register_forward_hook(_project_out_hook(v, coef)))
        handles.append(lh.down_proj.register_forward_hook(_project_out_hook(v, coef)))
    try:
        hooked = logits(model, ids)
    finally:
        for h in handles:
            h.remove()

    bake_direction_project_out(model, arch, v, coef)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded, info = reload_from(tmp)
        assert not info["unexpected_keys"], "no bias / extra keys should appear"
        baked = logits(reloaded, ids)

    assert torch.allclose(hooked, baked, atol=1e-4)


def test_direction_bake_via_specs_and_artifact():
    """End-to-end through bake_specs using a saved DirectionArtifact."""
    from sparkablate.directions import make_artifact

    model = make_model()
    arch = introspect(model, strict=True)
    torch.manual_seed(3)
    v = torch.randn(HIDDEN)

    with tempfile.TemporaryDirectory() as tmp:
        art_path = f"{tmp}/direction.pt"
        make_artifact(v, "tiny-llama", ("L1", -1), ["harm"], ["safe"]).save(art_path)
        spec = DirectionSpec(op="project_out", layers="all", vector_path=art_path)
        out = bake_specs(model, arch, [spec])

    assert out["interventions"][0]["kind"] == "direction"
    assert out["interventions"][0]["provenance"]["layer_key"] == "L1"


# ----------------------------------------------------------------------- pruning

def test_prune_matches_hooked_skip():
    model = make_model()
    ids = make_ids()
    spec = AblationSpec(kind="layer", layer=1)

    mgr = AblationManager(model)
    with mgr.applied(spec):
        hooked = logits(model, ids)

    arch = introspect(model, strict=True)
    out = bake_specs(model, arch, [spec], prune_layers=True)
    assert out["pruned_layers"] == [1]
    assert model.config.num_hidden_layers == NUM_LAYERS - 1

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded, info = reload_from(tmp)
        assert not info["unexpected_keys"] and not info["missing_keys"]
        baked = logits(reloaded, ids)
    assert torch.allclose(hooked, baked, atol=1e-5)


# ------------------------------------------------------------------------ guards

def test_mean_mode_is_rejected():
    model = make_model()
    arch = introspect(model, strict=True)
    with pytest.raises(BakeError, match="mean-mode"):
        bake_specs(model, arch, [AblationSpec(kind="mlp", layer=0, mode="mean")])


def test_direction_subset_is_rejected():
    model = make_model()
    arch = introspect(model, strict=True)
    spec = DirectionSpec(op="project_out", layers=(0,), vector_path="unused.pt")
    with pytest.raises(BakeError, match="subset"):
        bake_specs(model, arch, [spec])


def test_layer_out_of_range_is_rejected():
    model = make_model()
    arch = introspect(model, strict=True)
    with pytest.raises(BakeError, match="out of range"):
        bake_specs(model, arch, [AblationSpec(kind="attn", layer=99)])


def test_save_baked_writes_card_and_metadata():
    from sparkablate.bake import save_baked

    model = make_model()
    arch = introspect(model, strict=True)
    out = bake_specs(model, arch, [AblationSpec(kind="attn", layer=0)])
    # a minimal tokenizer-free save would fail; use the model's own config dir round-trip
    import os

    from transformers import AutoTokenizer  # noqa: F401  (ensure transformers present)

    class _StubTokenizer:
        def save_pretrained(self, d):
            with open(os.path.join(d, "tokenizer_stub.txt"), "w") as f:
                f.write("stub")

    with tempfile.TemporaryDirectory() as tmp:
        meta = save_baked(model, _StubTokenizer(), tmp, base_model="tiny-llama",
                          interventions=out["interventions"], dtype="float32",
                          pruned_layers=out["pruned_layers"], log=lambda *a: None)
        assert os.path.exists(os.path.join(tmp, "README.md"))
        assert os.path.exists(os.path.join(tmp, "sparkablate_bake.json"))
        assert os.path.exists(os.path.join(tmp, "config.json"))
        assert "convert_hf_to_gguf.py" in meta["gguf_command"]
