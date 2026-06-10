"""CPU tests against a tiny randomly-initialized LLaMA-style model.

These verify the intervention semantics exactly — no downloads, no GPU:
- hooks change logits and removal restores them bit-for-bit
- layer skip equals deleting the layer's residual contribution
- head ablation touches only the targeted head's slice
- mean calibration produces correctly shaped, finite vectors
- perplexity eval is deterministic and responds to ablation
"""

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sparkablate.eval import evaluate_perplexity, iter_eval_batches
from sparkablate.hooks import AblationManager, AblationSpec

VOCAB = 128
NUM_LAYERS = 3
NUM_HEADS = 4
NUM_KV_HEADS = 2
HIDDEN = 64


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=64,
    )
    m = LlamaForCausalLM(cfg)
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture()
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (2, 16))


def logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def test_introspection(model):
    mgr = AblationManager(model)
    assert mgr.arch.num_layers == NUM_LAYERS
    assert mgr.arch.num_heads == NUM_HEADS
    assert mgr.arch.head_dim == HIDDEN // NUM_HEADS


@pytest.mark.parametrize(
    "spec",
    [
        AblationSpec(kind="head", layer=1, heads=(2,)),
        AblationSpec(kind="attn", layer=0),
        AblationSpec(kind="mlp", layer=2),
        AblationSpec(kind="layer", layer=1),
    ],
    ids=lambda s: s.label(),
)
def test_ablation_changes_and_restores(model, input_ids, spec):
    mgr = AblationManager(model)
    base = logits(model, input_ids)
    with mgr.applied(spec):
        ablated = logits(model, input_ids)
    restored = logits(model, input_ids)

    assert not torch.allclose(base, ablated), "ablation should change outputs"
    assert torch.equal(base, restored), "removing hooks must restore outputs exactly"


def test_layer_skip_matches_manual_deletion(model, input_ids):
    """Skipping layer i via hook == running the model without that layer."""
    mgr = AblationManager(model)
    with mgr.applied(AblationSpec(kind="layer", layer=1)):
        hooked = logits(model, input_ids)

    container = model.model.layers
    kept = torch.nn.ModuleList([container[0], container[2]])
    saved = model.model.layers
    model.model.layers = kept
    try:
        manual = logits(model, input_ids)
    finally:
        model.model.layers = saved

    assert torch.allclose(hooked, manual, atol=1e-5)


def test_head_ablation_is_slice_local(model, input_ids):
    """Zeroing all heads == zeroing the whole attention output (minus o_proj bias-free path)."""
    mgr = AblationManager(model)
    all_heads = AblationSpec(kind="head", layer=0, heads=tuple(range(NUM_HEADS)))
    whole_attn = AblationSpec(kind="attn", layer=0)
    with mgr.applied(all_heads):
        a = logits(model, input_ids)
    with mgr.applied(whole_attn):
        b = logits(model, input_ids)
    # o_proj(0) == 0 since LLaMA o_proj has no bias, so these must match.
    assert torch.allclose(a, b, atol=1e-5)


def test_single_head_differs_from_other_head(model, input_ids):
    mgr = AblationManager(model)
    with mgr.applied(AblationSpec(kind="head", layer=0, heads=(0,))):
        a = logits(model, input_ids)
    with mgr.applied(AblationSpec(kind="head", layer=0, heads=(1,))):
        b = logits(model, input_ids)
    assert not torch.allclose(a, b)


def test_mean_calibration_and_ablation(model, input_ids):
    mgr = AblationManager(model)
    mgr.calibrate_means([input_ids])
    vec = mgr.means[(1, "oproj_in")]
    assert vec.shape == (HIDDEN,)
    assert torch.isfinite(vec).all()

    spec = AblationSpec(kind="head", layer=1, heads=(2,), mode="mean")
    base = logits(model, input_ids)
    with mgr.applied(spec):
        ablated = logits(model, input_ids)
    assert not torch.allclose(base, ablated)


def test_mean_without_calibration_raises(model):
    mgr = AblationManager(model)
    with pytest.raises(RuntimeError, match="calibrate_means"):
        with mgr.applied(AblationSpec(kind="mlp", layer=0, mode="mean")):
            pass


def test_stacked_specs(model, input_ids):
    mgr = AblationManager(model)
    specs = [
        AblationSpec(kind="head", layer=0, heads=(1,)),
        AblationSpec(kind="mlp", layer=2),
    ]
    base = logits(model, input_ids)
    with mgr.applied(specs):
        ablated = logits(model, input_ids)
    assert not torch.allclose(base, ablated)
    assert torch.equal(base, logits(model, input_ids))


def test_perplexity_eval(model):
    torch.manual_seed(2)
    corpus = torch.randint(0, VOCAB, (1024,))
    batches = list(iter_eval_batches(corpus, seq_len=32, batch_size=2, max_batches=4))
    assert len(batches) == 4 and batches[0].shape == (2, 32)

    r1 = evaluate_perplexity(model, batches)
    r2 = evaluate_perplexity(model, batches)
    assert r1["perplexity"] == r2["perplexity"]
    assert math.isfinite(r1["perplexity"]) and r1["perplexity"] > 0

    mgr = AblationManager(model)
    with mgr.applied(AblationSpec(kind="layer", layer=0)):
        r3 = evaluate_perplexity(model, batches)
    assert r3["nll"] != r1["nll"]


def test_spec_validation():
    with pytest.raises(ValueError):
        AblationSpec(kind="head", layer=0)  # heads required
    with pytest.raises(ValueError):
        AblationSpec(kind="layer", layer=0, mode="mean")  # skip has no mean mode
    with pytest.raises(ValueError):
        AblationSpec(kind="nope", layer=0)


def test_out_of_range_head_raises(model):
    mgr = AblationManager(model)
    with pytest.raises(ValueError, match="out of range"):
        mgr.apply(AblationSpec(kind="head", layer=0, heads=(NUM_HEADS,)))
