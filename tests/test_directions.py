"""Activation capture, diff-of-means directions, and direction interventions."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sparkablate.capture import ActivationRecorder, CaptureResult, _last_token_indices
from sparkablate.directions import (
    DirectionArtifact,
    candidate_keys,
    diff_of_means,
    make_artifact,
)
from sparkablate.hooks import AblationManager, AblationSpec, DirectionSpec

VOCAB = 128
HIDDEN = 64
NUM_LAYERS = 3


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
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


def test_last_token_indices_padding():
    ids = torch.zeros(2, 5, dtype=torch.long)
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert _last_token_indices(ids, right).tolist() == [2, 4]
    assert _last_token_indices(ids, left).tolist() == [4, 4]
    assert _last_token_indices(ids, None).tolist() == [4, 4]


def test_recorder_shapes_and_restore(model, input_ids):
    rec = ActivationRecorder(model)
    base = logits(model, input_ids)
    result = rec.record([input_ids, input_ids], positions=(-1, -2))

    expected_keys = {("emb", -1), ("emb", -2)} | {
        (f"L{i}", off) for i in range(NUM_LAYERS) for off in (-1, -2)
    }
    assert set(result.activations) == expected_keys
    for t in result.activations.values():
        assert t.shape == (4, HIDDEN)
        assert t.dtype == torch.float32
    assert torch.equal(base, logits(model, input_ids)), "capture hooks must be removed"


def test_recorder_respects_attention_mask(model):
    ids = torch.randint(0, VOCAB, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, 5:] = 0  # row 0: last real token at index 4
    rec = ActivationRecorder(model)
    masked = rec.record([{"input_ids": ids, "attention_mask": mask}])
    full = rec.record([ids[:1, :5]])  # row 0 truncated to its real tokens
    assert torch.allclose(masked[("L1", -1)][0], full[("L1", -1)][0], atol=1e-5)


def test_diff_of_means_unit_norm():
    a = CaptureResult({("L0", -1): torch.randn(8, HIDDEN)})
    b = CaptureResult({("L0", -1): torch.randn(8, HIDDEN)})
    d = diff_of_means(a, b)
    assert torch.allclose(d[("L0", -1)].norm(), torch.tensor(1.0), atol=1e-6)
    with pytest.raises(ValueError, match="different"):
        diff_of_means(a, CaptureResult({("L1", -1): torch.randn(8, HIDDEN)}))


def test_candidate_keys_excludes_late_layers():
    dirs = {(f"L{i}", -1): torch.ones(4) for i in range(10)}
    dirs[("emb", -1)] = torch.ones(4)
    keep = candidate_keys(dirs, num_layers=10, layer_fraction=0.8)
    assert ("emb", -1) in keep
    assert ("L7", -1) in keep and ("L8", -1) not in keep


def test_project_out_zeroes_component(model, input_ids):
    torch.manual_seed(3)
    v = torch.randn(HIDDEN)
    mgr = AblationManager(model)
    rec = ActivationRecorder(model)

    # manager hooks registered first, so the recorder sees projected outputs
    mgr.apply_direction(DirectionSpec(op="project_out"), vector=v)
    try:
        result = rec.record([input_ids])
    finally:
        mgr.clear()

    v_hat = v / v.norm()
    for key, acts in result.activations.items():
        comp = acts @ v_hat
        assert comp.abs().max() < 1e-4, f"residual component along v at {key}"


def test_apply_direction_with_explicit_vector(model, input_ids):
    torch.manual_seed(3)
    v = torch.randn(HIDDEN)
    mgr = AblationManager(model)
    base = logits(model, input_ids)
    mgr.apply_direction(DirectionSpec(op="project_out"), vector=v)
    ablated = logits(model, input_ids)
    mgr.clear()
    assert not torch.allclose(base, ablated)
    assert torch.equal(base, logits(model, input_ids))


def test_add_zero_coefficient_is_identity(model, input_ids):
    v = torch.randn(HIDDEN)
    mgr = AblationManager(model)
    base = logits(model, input_ids)
    mgr.apply_direction(DirectionSpec(op="add", coefficient=0.0, layers=(1,)), vector=v)
    out = logits(model, input_ids)
    mgr.clear()
    assert torch.allclose(base, out, atol=1e-6)


def test_add_single_layer_changes_output(model, input_ids):
    torch.manual_seed(4)
    v = torch.randn(HIDDEN)
    mgr = AblationManager(model)
    base = logits(model, input_ids)
    mgr.apply_direction(DirectionSpec(op="add", coefficient=4.0, layers=(1,)), vector=v)
    out = logits(model, input_ids)
    mgr.clear()
    assert not torch.allclose(base, out)


def test_artifact_roundtrip(tmp_path):
    torch.manual_seed(5)
    v = torch.randn(HIDDEN)
    art = make_artifact(v / v.norm(), "test/model", ("L1", -1),
                        ["bad prompt"], ["good prompt"], extra_meta={"note": "t"})
    path = str(tmp_path / "dir.pt")
    art.save(path)
    loaded = DirectionArtifact.load(path)
    assert torch.allclose(loaded.vector, art.vector)
    assert loaded.layer_key == "L1" and loaded.position == -1
    assert loaded.meta["n_harmful"] == 1 and loaded.meta["note"] == "t"
    assert loaded.meta["harmful_sha256"] != loaded.meta["harmless_sha256"]


def test_spec_from_artifact_path(model, input_ids, tmp_path):
    torch.manual_seed(6)
    v = torch.randn(HIDDEN)
    path = str(tmp_path / "dir.pt")
    make_artifact(v, "test/model", ("L1", -1), ["a"], ["b"]).save(path)

    mgr = AblationManager(model)
    base = logits(model, input_ids)
    with mgr.applied(DirectionSpec(vector_path=path)):
        ablated = logits(model, input_ids)
    assert not torch.allclose(base, ablated)
    assert torch.equal(base, logits(model, input_ids))


def test_mixed_spec_stack(model, input_ids, tmp_path):
    v = torch.randn(HIDDEN)
    path = str(tmp_path / "dir.pt")
    make_artifact(v, "test/model", ("L0", -1), ["a"], ["b"]).save(path)

    mgr = AblationManager(model)
    base = logits(model, input_ids)
    specs = [AblationSpec(kind="mlp", layer=2), DirectionSpec(vector_path=path)]
    with mgr.applied(specs):
        ablated = logits(model, input_ids)
    assert not torch.allclose(base, ablated)
    assert torch.equal(base, logits(model, input_ids))


def test_direction_spec_validation():
    with pytest.raises(ValueError, match="op"):
        DirectionSpec(op="rotate")
    assert DirectionSpec(layers=[0, 2]).layers == (0, 2)
    assert DirectionSpec().label() == "dir.project_out[Lall]x1"
