"""Sweep generation and config parsing tests (no model needed)."""

import pytest

from sparkablate.config import AblationConfig, EvalConfig, ExperimentConfig, ModelConfig
from sparkablate.runner import generate_specs, resolve_layers


def make_cfg(**ablation) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(name="dummy"),
        eval=EvalConfig(),
        ablation=AblationConfig(**ablation),
    )


def test_resolve_layers():
    assert resolve_layers("all", 4) == [0, 1, 2, 3]
    assert resolve_layers([1, 3], 4) == [1, 3]
    with pytest.raises(ValueError):
        resolve_layers([4], 4)


def test_layer_sweep():
    conds = generate_specs(make_cfg(sweep="layers"), num_layers=3, num_heads=4)
    assert len(conds) == 3
    assert all(c[0].kind == "layer" for c in conds)


def test_head_sweep_counts():
    conds = generate_specs(make_cfg(sweep="heads", layers=[0, 2]), num_layers=3, num_heads=4)
    assert len(conds) == 8  # 2 layers x 4 heads
    assert {c[0].heads for c in conds} == {(0,), (1,), (2,), (3,)}


def test_custom_sweep():
    conds = generate_specs(
        make_cfg(
            sweep="custom",
            mode="mean",
            targets=[
                {"kind": "head", "layer": 1, "heads": [0, 2]},
                {"kind": "layer", "layer": 2, "mode": "zero"},
            ],
        ),
        num_layers=3,
        num_heads=4,
    )
    assert conds[0][0].heads == (0, 2)
    assert conds[0][0].mode == "mean"  # inherits sweep default
    assert conds[1][0].kind == "layer"


def test_introspect_model_output(monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    import sparkablate.runner as runner

    torch.manual_seed(0)
    tiny = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    ))
    monkeypatch.setattr(runner, "load_model", lambda name, **kw: (tiny, None))

    lines = []
    rc = runner.introspect_model("tiny", log=lines.append)
    out = "\n".join(lines)
    assert rc == 0
    assert "layers: 3" in out
    assert "kv_heads: 2" in out
    assert out.count("PASS") == 4


def test_chat_template_prompts():
    from sparkablate.eval import apply_chat_template_prompts

    class FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    out = apply_chat_template_prompts(FakeTok(), ["hi", "yo"], system="sys")
    assert out == ["system:sys|user:hi|assistant:", "system:sys|user:yo|assistant:"]


def test_yaml_roundtrip(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(
        """
model: {name: test/model, dtype: float32}
eval: {dataset: data.txt, seq_len: 64}
ablation: {sweep: mlp, layers: [0, 1]}
output: {dir: out}
""",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(str(cfg_file))
    assert cfg.model.name == "test/model"
    assert cfg.eval.seq_len == 64
    assert cfg.ablation.sweep == "mlp"
    assert cfg.output_dir == "out"
