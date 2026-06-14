# sparkablate

Component-ablation toolkit for open-weight causal LMs, built for single-node
research on an NVIDIA DGX Spark (and developed/tested on CPU anywhere).

It answers questions of the form *"what happens to this model if I knock out
this piece?"* with causal interventions applied via forward hooks — no model
surgery, no forked modeling code:

| kind    | intervention                                                        |
|---------|---------------------------------------------------------------------|
| `head`  | zero/mean-patch one or more attention heads (pre-`o_proj` slice)    |
| `attn`  | zero/mean-patch an entire attention block's output                  |
| `mlp`   | zero/mean-patch an entire MLP block's output                        |
| `layer` | skip a decoder layer entirely (output = input hidden states)        |

Modes: **zero** ablation (output → 0) and **mean** ablation (output → average
activation over a calibration corpus). Mean ablation is generally the more
defensible intervention for importance claims, since zeroing pushes layernorm
statistics off-distribution and inflates measured importance.

Architecture support is auto-detected (LLaMA / Mistral / Qwen / GPT-NeoX /
OPT-style layouts); GQA models are handled correctly because head slicing
operates on the `o_proj` input, which is always `num_attention_heads × head_dim`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[plots,datasets,dev]"
```

### On the DGX Spark

The Spark is aarch64 + Blackwell, so use either:

- **NVIDIA's PyTorch NGC container** (recommended; ships a CUDA-enabled
  aarch64 torch): mount this repo and `pip install -e .` inside it, or
- a native venv with an aarch64 CUDA torch wheel per NVIDIA's current DGX
  Spark instructions, then `pip install -e ".[plots,datasets]"`.

The 128 GB unified memory comfortably holds ~70B models in bf16 alongside
calibration state — set `dtype: bfloat16`, `device: auto` and skip
`device_map` entirely (single logical device).

The GB10 is compute capability **sm_121**: `load_model` checks the torch
build's arch list at load time and warns if sm_121 kernels are missing
(symptom otherwise: slow PTX-JIT or "no kernel image" mid-sweep).

### On a Mac

`device: auto` falls back to Apple MPS (then CPU), and `bfloat16` is
silently swapped for `float16` on MPS where bf16 support is incomplete.
Develop and smoke-test anywhere; run real sweeps on the Spark.

## Quickstart

```bash
# 1. Validate the install end-to-end with a 135M model (CPU-friendly):
ablate run -c configs/smoke_test.yaml
ablate report results/smoke

# 2. Layer-skip sweep over a real model:
ablate run -c configs/layer_sweep.yaml
ablate report results/llama31-8b-layer-sweep      # ranked table + plots

# 3. Per-head mean-ablation sweep (writes a layer×head Δ-NLL heatmap):
ablate run -c configs/head_sweep.yaml

# 4. Qualitative check — greedy generation with vs. without an ablation:
ablate compare -m meta-llama/Llama-3.1-8B --kind head --layer 14 --heads 3 7 \
    --prompt "The capital of France is"
```

Every run writes `results.jsonl` (one row per condition, written
incrementally so long sweeps are interrupt-safe) plus `summary.json`.
`ablate report` prints conditions ranked by Δ NLL and renders plots
(bar chart per layer for layer/mlp/attn sweeps, heatmap for head sweeps).

## Baking a usable model

The sweeps above *measure* importance with temporary hooks. `ablate bake` instead
writes the chosen intervention **permanently into the weights** and saves a
standalone Hugging Face checkpoint — no hooks, no custom modeling code — that
loads directly in `transformers`, **vLLM**, and TGI:

```bash
# Abliteration: bake a refusal direction out of every residual write.
ablate bake -m meta-llama/Llama-3.1-8B \
    --direction results/refusal/direction.pt --op project_out \
    --out models/llama31-8b-abliterated

# Knock out components permanently (zero ablation):
ablate bake -m meta-llama/Llama-3.1-8B --kind head --layer 14 --heads 3 7 \
    --out models/llama31-8b-noL14h3h7

# From a config (applies ablation.targets + ablation.directions jointly):
ablate bake -c configs/custom_targets.yaml --out models/baked
```

Each run writes the checkpoint plus `sparkablate_bake.json` (provenance: base
model, interventions, direction sha256s) and a `README.md` model card.

**What bakes (all exact, no new parameters, stock-loadable):**

| kind                       | weight edit                                                       |
|----------------------------|-------------------------------------------------------------------|
| `head` / `attn` / `mlp` zero | zero the writing projection's relevant weights (+ bias if any)   |
| `layer` skip               | zero `o_proj`+`down_proj` → identity layer (`--prune-layers` deletes it) |
| `direction` `project_out`  | orthogonalize embedding + every `o_proj`/`down_proj` against v̂   |

Notes and current limits:

- **Precision:** bake in `float32` (the default for `bake`); baking directional
  ablation from a half precision can leave a faint residual along the direction.
- **Mean mode is not bakeable.** A constant output needs a bias on
  `o_proj`/`down_proj`, which LLaMA/Mistral/Qwen lack and vLLM/GGUF loaders drop.
  Use mean mode as a runtime hook (`ablate run` / `compare`).
- **`direction` bakes only with `layers: all`** (subset projection is
  path-dependent — no static weight equivalent).
- **LM Studio (GGUF):** the checkpoint is safetensors. Convert with llama.cpp —
  `ablate bake` prints and records the exact
  `python convert_hf_to_gguf.py <out_dir> --outfile model.gguf` command.

## Config reference

```yaml
model:
  name: meta-llama/Llama-3.1-8B   # any HF causal LM
  dtype: bfloat16                  # bfloat16 | float16 | float32
  device: auto                     # auto | cuda | mps | cpu  (auto: cuda > mps > cpu)
  trust_remote_code: false

eval:
  dataset: wikitext      # 'wikitext' (needs `datasets` extra) or a .txt path
  seq_len: 1024
  batch_size: 4
  max_batches: 16        # eval cost ∝ this; null = whole corpus
  calibration_batches: 8 # only used by mode: mean

ablation:
  mode: zero             # zero | mean (default for sweep conditions)
  sweep: layers          # layers | heads | mlp | attn | custom
  layers: all            # or [0, 4, 8, ...]
  targets:               # only for sweep: custom — one condition per entry,
    - {kind: head, layer: 14, heads: [3, 7]}        # specs in one entry
    - {kind: layer, layer: 26}                      # are applied jointly

output:
  dir: results/my-experiment
```

## Library use

The hook engine is independent of the CLI — drive it from notebooks or your
own eval harness:

```python
from sparkablate import AblationManager, AblationSpec
from sparkablate.model import load_model

model, tok = load_model("meta-llama/Llama-3.1-8B")
mgr = AblationManager(model)

with mgr.applied(AblationSpec(kind="head", layer=14, heads=(3, 7))):
    out = model.generate(**tok("Hello", return_tensors="pt").to(model.device))
# hooks are gone here; model is bit-for-bit back to baseline
```

`mgr.applied()` accepts a list of specs for joint ablations (e.g. testing
backup-head redundancy). For `mode="mean"`, call
`mgr.calibrate_means(batches)` once first.

## Methodology notes

- **Zero vs. mean:** zero ablation overstates importance by knocking
  layernorm inputs off-distribution; mean ablation is the saner default for
  head-level claims. Both are provided because zero is the standard baseline
  in much of the literature.
- **Single ablations miss redundancy:** backup heads can hide each other.
  Use `sweep: custom` with multi-spec entries to test joint knockouts of the
  top single-ablation candidates.
- **Perplexity is a global metric:** a component critical for a narrow
  capability can look unimportant on generic text. Point `dataset:` at a
  task-specific corpus to probe targeted capabilities.
- **Noise floor:** with small `max_batches`, tiny Δ NLL values are not
  meaningful rankings. Scale the eval set until the effects you care about
  are stable across reruns.

## Tests

```bash
pytest
```

The suite runs on CPU against a tiny randomly-initialized LLaMA (with GQA),
verifying intervention semantics exactly: hook removal restores logits
bit-for-bit, layer-skip matches physically deleting the layer, and zeroing
all heads equals zeroing the attention block.

## Extending

- New architecture not auto-detected → add its module names to
  `LAYER_CONTAINER_PATHS` / `ATTN_NAMES` / `MLP_NAMES` / `OPROJ_NAMES` in
  [hooks.py](src/sparkablate/hooks.py).
- New metric → implement alongside `evaluate_perplexity` in
  [eval.py](src/sparkablate/eval.py) and call it from
  [runner.py](src/sparkablate/runner.py); results rows are free-form JSON.
- New intervention (noise injection, activation patching from a second
  prompt, directional/projection ablation) → add a hook factory on
  `AblationManager` and a `kind` to `AblationSpec`.
