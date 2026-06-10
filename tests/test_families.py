"""Per-family integration smoke tests on tiny randomly-initialized models.

Each family gets a tiny config (CPU-only, no downloads). Asserts that
introspection resolves explicitly (no heuristic fallback) and that every
intervention kind changes logits and restores them bit-for-bit on removal.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from sparkablate.hooks import AblationManager, AblationSpec

VOCAB = 128
HIDDEN = 64
LAYERS = 3
HEADS = 4

TINY = dict(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    intermediate_size=128,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
)


def _cfg(name, **kw):
    import transformers

    cls = getattr(transformers, name, None)
    if cls is None:
        pytest.skip(f"{name} not available in installed transformers")
    return cls(**kw)


FAMILY_BUILDERS = {
    "gpt2": lambda: _cfg(
        "GPT2Config", vocab_size=VOCAB, n_positions=64, n_embd=HIDDEN, n_layer=LAYERS, n_head=HEADS
    ),
    "gpt_neox": lambda: _cfg("GPTNeoXConfig", max_position_embeddings=64, **TINY),
    "qwen2": lambda: _cfg("Qwen2Config", num_key_value_heads=2, max_position_embeddings=64, **TINY),
    "gemma2": lambda: _cfg(
        "Gemma2Config",
        num_key_value_heads=2,
        head_dim=HIDDEN // HEADS,
        sliding_window=16,
        max_position_embeddings=64,
        **TINY,
    ),
    "phi3": lambda: _cfg(
        "Phi3Config", num_key_value_heads=2, max_position_embeddings=64, pad_token_id=0, **TINY
    ),
    "olmo": lambda: _cfg("OlmoConfig", num_key_value_heads=2, max_position_embeddings=64, **TINY),
    "mixtral": lambda: _cfg(
        "MixtralConfig",
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        **TINY,
    ),
}

# Mixtral's MoE block was `block_sparse_moe` before transformers ~4.58, `mlp` after.
EXPECTED_MLP_NAMES = {"mixtral": ("mlp", "block_sparse_moe")}


@pytest.fixture(scope="module", params=sorted(FAMILY_BUILDERS))
def family_model(request):
    torch.manual_seed(0)
    cfg = FAMILY_BUILDERS[request.param]()
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    model.requires_grad_(False)
    return request.param, model


@pytest.fixture()
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (2, 16))


def logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def test_introspection_resolves_explicitly(family_model):
    family, model = family_model
    mgr = AblationManager(model)  # strict: raises on heuristic fallback
    assert mgr.arch.num_layers == LAYERS
    assert mgr.arch.num_heads == HEADS
    assert mgr.arch.resolution["container"] != "fallback"
    assert mgr.arch.resolution["mlp"] in EXPECTED_MLP_NAMES.get(family, ("mlp",))
    assert mgr.arch.head_slicing_error is None


@pytest.mark.parametrize("kind", ["head", "attn", "mlp", "layer"])
def test_ablation_changes_and_restores(family_model, input_ids, kind):
    _, model = family_model
    mgr = AblationManager(model)
    spec = AblationSpec(kind=kind, layer=1, heads=(2,) if kind == "head" else ())
    base = logits(model, input_ids)
    with mgr.applied(spec):
        ablated = logits(model, input_ids)
    restored = logits(model, input_ids)

    assert not torch.allclose(base, ablated), f"{kind} ablation should change outputs"
    assert torch.equal(base, restored), "removing hooks must restore outputs exactly"


def test_mean_calibration(family_model, input_ids):
    _, model = family_model
    mgr = AblationManager(model)
    mgr.calibrate_means([input_ids])
    for site in ("oproj_in", "attn_out", "mlp_out"):
        vec = mgr.means[(1, site)]
        assert vec.shape == (HIDDEN,)
        assert torch.isfinite(vec).all()


def test_strict_mode_rejects_unmappable_model():
    class Weird(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stuff = torch.nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="LAYER_CONTAINER_PATHS"):
        AblationManager(Weird())
