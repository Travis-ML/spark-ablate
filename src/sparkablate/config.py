"""YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = None
    trust_remote_code: bool = False
    strict_introspect: bool = True


@dataclass
class EvalConfig:
    dataset: str = "wikitext"
    seq_len: int = 1024
    batch_size: int = 2
    max_batches: int | None = 16
    calibration_batches: int = 8
    # generation-based metrics (metric: refusal | both); perplexity path ignores these
    metric: str = "perplexity"          # perplexity | refusal | both
    prompts: str | None = None          # prompt file for generation metrics
    refusal_patterns: list[str] = field(default_factory=list)  # [] = built-in defaults
    max_new_tokens: int = 64
    chat_template: bool = False         # wrap prompts via tokenizer.apply_chat_template


@dataclass
class AblationConfig:
    mode: str = "zero"            # zero | mean
    sweep: str = "layers"         # layers | heads | mlp | attn | custom
    layers: list[int] | str = "all"
    # custom sweep entries: [{kind, layer, heads?, mode?}, ...]
    targets: list[dict] = field(default_factory=list)
    # direction conditions appended to any sweep: [{path, op?, coefficient?, layers?}, ...]
    directions: list[dict] = field(default_factory=list)


@dataclass
class DirectionConfig:
    """Inputs for `ablate find-direction` (difference-of-means extraction)."""

    harmful_prompts: str
    harmless_prompts: str
    positions: list[int] = field(default_factory=lambda: [-1])
    val_fraction: float = 0.2     # held out for candidate selection
    layer_fraction: float = 0.8   # candidates only from layers below this depth
    nll_guard: float = 0.5        # max ΔNLL on harmless val for a candidate to qualify
    artifact: str | None = None   # output path; default {output_dir}/direction.pt


@dataclass
class ExperimentConfig:
    model: ModelConfig
    eval: EvalConfig
    ablation: AblationConfig
    output_dir: str = "results/experiment"
    direction: DirectionConfig | None = None

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        try:
            model = ModelConfig(**raw["model"])
        except KeyError as e:
            raise ValueError("config requires a 'model' section with at least 'name'") from e
        ev = EvalConfig(**raw.get("eval", {}))
        ab = AblationConfig(**raw.get("ablation", {}))
        direction = DirectionConfig(**raw["direction"]) if "direction" in raw else None
        out = raw.get("output", {}).get("dir", "results/experiment")
        return ExperimentConfig(model=model, eval=ev, ablation=ab, output_dir=out,
                                direction=direction)
