"""Experiment runner: baseline, sweep generation, incremental results."""

from __future__ import annotations

import itertools
import json
import os
import time

from sparkablate.config import ExperimentConfig
from sparkablate.eval import (
    apply_chat_template_prompts,
    evaluate_generations,
    evaluate_perplexity,
    iter_eval_batches,
    load_eval_text,
    load_prompt_lines,
    prompt_nll,
    tokenize_corpus,
)
from sparkablate.hooks import AblationManager, AblationSpec, DirectionSpec
from sparkablate.model import load_model


def _spec_dict(s: AblationSpec | DirectionSpec) -> dict:
    d = dict(vars(s))
    if isinstance(s, AblationSpec):
        d["heads"] = list(s.heads)
    else:
        d["op_kind"] = "direction"
        if s.layers != "all":
            d["layers"] = list(s.layers)
    return d


def _progress(iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def introspect_model(name: str, meta: bool = False, device: str = "cpu",
                     trust_remote_code: bool = False, log=print) -> int:
    """Print the resolved architecture map; returns a shell exit code."""
    from sparkablate.hooks import introspect

    if meta:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    else:
        model, _ = load_model(name, dtype="float32", device=device,
                              trust_remote_code=trust_remote_code)

    try:
        arch = introspect(model, strict=True)
    except RuntimeError as e:
        log(f"FAIL: {e}")
        return 1

    res = arch.resolution
    log(f"model: {name}")
    log(f"layers: {arch.num_layers}  (container: {res['container']})")
    log(f"heads: {arch.num_heads}  kv_heads: {arch.num_kv_heads}  head_dim: {arch.head_dim}")
    log(f"modules: attn={res['attn']}  mlp={res['mlp']}  o_proj={res['o_proj']}")
    log(f"  attn class: {type(arch.layers[0].attn).__name__}")
    log(f"  mlp class:  {type(arch.layers[0].mlp).__name__}")
    for kind in ("head", "attn", "mlp", "layer"):
        if kind == "head" and arch.head_slicing_error:
            log(f"kind {kind!r}: FAIL — {arch.head_slicing_error}")
        else:
            log(f"kind {kind!r}: PASS")
    return 0


def resolve_layers(spec: list[int] | str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers = list(spec)
    bad = [i for i in layers if not 0 <= i < num_layers]
    if bad:
        raise ValueError(f"layer indices {bad} out of range (num_layers={num_layers})")
    return layers


def generate_specs(cfg: ExperimentConfig, num_layers: int, num_heads: int) -> list[list]:
    """Each element is one experimental condition (a list of specs applied together)."""
    ab = cfg.ablation
    direction_conditions = [
        [DirectionSpec(
            op=d.get("op", "project_out"),
            coefficient=d.get("coefficient", 1.0),
            layers="all" if d.get("layers", "all") == "all" else tuple(d["layers"]),
            vector_path=d["path"],
        )]
        for d in ab.directions
    ]

    if ab.sweep == "custom":
        conditions = []
        for t in ab.targets:
            conditions.append([
                AblationSpec(
                    kind=t["kind"],
                    layer=t["layer"],
                    heads=tuple(t.get("heads", ())),
                    mode=t.get("mode", ab.mode),
                )
            ])
        return conditions + direction_conditions

    layers = resolve_layers(ab.layers, num_layers)
    if ab.sweep == "layers":
        conditions = [[AblationSpec(kind="layer", layer=i)] for i in layers]
    elif ab.sweep in ("mlp", "attn"):
        conditions = [[AblationSpec(kind=ab.sweep, layer=i, mode=ab.mode)] for i in layers]
    elif ab.sweep == "heads":
        conditions = [
            [AblationSpec(kind="head", layer=i, heads=(h,), mode=ab.mode)]
            for i, h in itertools.product(layers, range(num_heads))
        ]
    else:
        raise ValueError(f"unknown sweep type {ab.sweep!r}")
    return conditions + direction_conditions


def run_experiment(cfg: ExperimentConfig, log=print) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, "results.jsonl")

    log(f"Loading model {cfg.model.name} ...")
    model, tokenizer = load_model(
        cfg.model.name,
        dtype=cfg.model.dtype,
        device=cfg.model.device,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    mgr = AblationManager(model, strict=cfg.model.strict_introspect)
    log(f"Architecture: {mgr.arch.num_layers} layers, {mgr.arch.num_heads} heads "
        f"({mgr.arch.num_kv_heads} KV), head_dim {mgr.arch.head_dim}, "
        f"resolved via {mgr.arch.resolution}")

    metric = cfg.eval.metric
    if metric not in ("perplexity", "refusal", "both"):
        raise ValueError(f"eval.metric must be perplexity|refusal|both, got {metric!r}")
    need_ppl = metric in ("perplexity", "both")

    conditions = generate_specs(cfg, mgr.arch.num_layers, mgr.arch.num_heads)
    needs_means = any(
        getattr(s, "mode", None) == "mean" for cond in conditions for s in cond
    )

    batches = None
    if need_ppl or needs_means:
        text = load_eval_text(cfg.eval.dataset)
        tokens = tokenize_corpus(tokenizer, text)
        batches = list(iter_eval_batches(tokens, cfg.eval.seq_len, cfg.eval.batch_size,
                                         cfg.eval.max_batches))
        log(f"Eval set: {len(batches)} batches of {cfg.eval.batch_size}x{cfg.eval.seq_len} tokens")

    gen_prompts = None
    if metric in ("refusal", "both"):
        if not cfg.eval.prompts:
            raise ValueError("eval.metric includes 'refusal'; set eval.prompts to a prompt file")
        gen_prompts = load_prompt_lines(cfg.eval.prompts)
        if cfg.eval.chat_template:
            gen_prompts = apply_chat_template_prompts(tokenizer, gen_prompts)
        log(f"Refusal eval: {len(gen_prompts)} prompts, {cfg.eval.max_new_tokens} new tokens")
    patterns = cfg.eval.refusal_patterns or None

    def measure() -> dict:
        out = {}
        if need_ppl:
            out.update(evaluate_perplexity(model, batches))
        if gen_prompts is not None:
            g = evaluate_generations(model, tokenizer, gen_prompts, patterns,
                                     cfg.eval.max_new_tokens, cfg.eval.batch_size)
            out["refusal_rate"] = g["refusal_rate"]
            out["n_prompts"] = g["n"]
        return out

    if needs_means:
        log(f"Calibrating mean activations over {cfg.eval.calibration_batches} batches ...")
        calib = batches[: cfg.eval.calibration_batches]
        mgr.calibrate_means(_progress(calib, "calibrating"))

    def fmt(m: dict) -> str:
        parts = []
        if "perplexity" in m:
            parts.append(f"ppl {m['perplexity']:.3f}")
        if "refusal_rate" in m:
            parts.append(f"refusal {m['refusal_rate']:.2f}")
        return ", ".join(parts)

    log("Measuring baseline ...")
    t0 = time.time()
    baseline = measure()
    log(f"Baseline: {fmt(baseline)} ({time.time() - t0:.1f}s)")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"condition": "baseline", **baseline}) + "\n")
        for n, cond in enumerate(_progress(conditions, "conditions"), 1):
            label = "+".join(s.label() for s in cond)
            t0 = time.time()
            with mgr.applied(cond):
                metrics = measure()
            row = {
                "condition": label,
                "specs": [_spec_dict(s) for s in cond],
                **metrics,
                "seconds": round(time.time() - t0, 2),
            }
            if need_ppl:
                row["delta_nll"] = metrics["nll"] - baseline["nll"]
                row["delta_perplexity"] = metrics["perplexity"] - baseline["perplexity"]
            if gen_prompts is not None:
                row["delta_refusal"] = metrics["refusal_rate"] - baseline["refusal_rate"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            log(f"[{n}/{len(conditions)}] {label}: {fmt(metrics)}")

    summary = {
        "model": cfg.model.name,
        "baseline": baseline,
        "conditions": len(conditions),
        "results": results_path,
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _val_split(prompts: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    n_val = max(1, int(len(prompts) * val_fraction))
    if n_val >= len(prompts):
        raise ValueError(f"val_fraction {val_fraction} leaves no training prompts "
                         f"(n={len(prompts)})")
    return prompts[:-n_val], prompts[-n_val:]


def run_find_direction(cfg: ExperimentConfig, log=print) -> dict:
    """Extract, score, and save a difference-of-means direction (refusal workflow)."""
    from sparkablate.capture import ActivationRecorder
    from sparkablate.directions import candidate_keys, diff_of_means, make_artifact

    if cfg.direction is None:
        raise ValueError("config requires a 'direction' section for find-direction")
    dc = cfg.direction
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, "results.jsonl")

    log(f"Loading model {cfg.model.name} ...")
    model, tokenizer = load_model(
        cfg.model.name, dtype=cfg.model.dtype, device=cfg.model.device,
        device_map=cfg.model.device_map, trust_remote_code=cfg.model.trust_remote_code,
    )
    mgr = AblationManager(model, strict=cfg.model.strict_introspect)
    rec = ActivationRecorder(model, strict=cfg.model.strict_introspect)

    harmful = load_prompt_lines(dc.harmful_prompts)
    harmless = load_prompt_lines(dc.harmless_prompts)
    fmt = ((lambda ps: apply_chat_template_prompts(tokenizer, ps))
           if cfg.eval.chat_template else (lambda ps: ps))
    harmful_train, harmful_val = _val_split(harmful, dc.val_fraction)
    harmless_train, harmless_val = _val_split(harmless, dc.val_fraction)

    def batches_for(prompts):
        texts = fmt(prompts)
        for i in range(0, len(texts), cfg.eval.batch_size):
            yield tokenizer(texts[i : i + cfg.eval.batch_size],
                            return_tensors="pt", padding=True)

    positions = tuple(dc.positions)
    log(f"Capturing activations: {len(harmful_train)} harmful / "
        f"{len(harmless_train)} harmless prompts at positions {list(positions)} ...")
    cap_harmful = rec.record(batches_for(harmful_train), positions)
    cap_harmless = rec.record(batches_for(harmless_train), positions)
    dirs = diff_of_means(cap_harmful, cap_harmless)
    keys = candidate_keys(dirs, mgr.arch.num_layers, dc.layer_fraction)
    log(f"{len(keys)} candidate directions ({len(dirs)} sites, "
        f"layer_fraction {dc.layer_fraction})")

    patterns = cfg.eval.refusal_patterns or None
    val_harmful = fmt(harmful_val)
    val_harmless = fmt(harmless_val)
    base_gen = evaluate_generations(model, tokenizer, val_harmful, patterns,
                                    cfg.eval.max_new_tokens, cfg.eval.batch_size)
    base_nll = prompt_nll(model, tokenizer, val_harmless, cfg.eval.batch_size)
    log(f"Baseline: refusal {base_gen['refusal_rate']:.2f} on {base_gen['n']} harmful "
        f"val prompts, harmless NLL {base_nll['nll']:.4f}")

    rows = []
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "condition": "baseline",
            "refusal_rate": base_gen["refusal_rate"],
            "n_prompts": base_gen["n"],
            "harmless_nll": base_nll["nll"],
        }) + "\n")
        for key in _progress(keys, "candidates"):
            layer_key, off = key
            t0 = time.time()
            mgr.apply_direction(DirectionSpec(op="project_out"), vector=dirs[key])
            try:
                g = evaluate_generations(model, tokenizer, val_harmful, patterns,
                                         cfg.eval.max_new_tokens, cfg.eval.batch_size)
                nll = prompt_nll(model, tokenizer, val_harmless, cfg.eval.batch_size)
            finally:
                mgr.clear()
            row = {
                "condition": f"dir@{layer_key}@{off}",
                "layer_key": layer_key,
                "position": off,
                "refusal_rate": g["refusal_rate"],
                "delta_refusal": g["refusal_rate"] - base_gen["refusal_rate"],
                "harmless_nll": nll["nll"],
                "delta_harmless_nll": nll["nll"] - base_nll["nll"],
                "seconds": round(time.time() - t0, 2),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            log(f"{row['condition']}: refusal {row['refusal_rate']:.2f} "
                f"(d{row['delta_refusal']:+.2f}), harmless dNLL "
                f"{row['delta_harmless_nll']:+.4f}")

    qualified = [r for r in rows if r["delta_harmless_nll"] <= dc.nll_guard]
    if not qualified:
        log(f"WARNING: no candidate kept harmless dNLL <= {dc.nll_guard}; "
            "selecting best refusal drop regardless")
        qualified = rows
    best = min(qualified, key=lambda r: r["refusal_rate"])
    best_key = (best["layer_key"], best["position"])
    log(f"Selected {best['condition']}: refusal {best['refusal_rate']:.2f} "
        f"(baseline {base_gen['refusal_rate']:.2f})")

    artifact_path = dc.artifact or os.path.join(cfg.output_dir, "direction.pt")
    artifact = make_artifact(
        dirs[best_key], cfg.model.name, best_key, harmful, harmless,
        extra_meta={
            "baseline_refusal_rate": base_gen["refusal_rate"],
            "refusal_rate": best["refusal_rate"],
            "delta_harmless_nll": best["delta_harmless_nll"],
            "chat_template": cfg.eval.chat_template,
        },
    )
    artifact.save(artifact_path)
    log(f"wrote {artifact_path}")

    summary = {
        "model": cfg.model.name,
        "artifact": artifact_path,
        "selected": best,
        "baseline_refusal_rate": base_gen["refusal_rate"],
        "candidates": len(rows),
        "results": results_path,
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def compare_generations(model_name: str, specs: list, prompt: str,
                        max_new_tokens: int = 128, dtype: str = "bfloat16",
                        device: str = "auto", trust_remote_code: bool = False,
                        chat_template: bool = False, log=print) -> dict:
    """Qualitative check: greedy generation with and without the intervention."""
    model, tokenizer = load_model(model_name, dtype=dtype, device=device,
                                  trust_remote_code=trust_remote_code)
    mgr = AblationManager(model)
    if chat_template:
        prompt = apply_chat_template_prompts(tokenizer, [prompt])[0]
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    def gen():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    baseline = gen()
    with mgr.applied(specs):
        ablated = gen()

    label = "+".join(s.label() for s in specs)
    log(f"--- baseline ---\n{baseline}\n--- ablated ({label}) ---\n{ablated}")
    return {"prompt": prompt, "baseline": baseline, "ablated": ablated, "specs": label}


def bake_checkpoint(model_name: str, specs: list, out_dir: str, *,
                    dtype: str = "float32", device: str = "cpu",
                    trust_remote_code: bool = False, prune_layers: bool = False,
                    log=print) -> dict:
    """Bake ``specs`` into the model's weights and write a standalone checkpoint.

    Produces a stock Hugging Face checkpoint (no hooks, no custom code) usable by
    transformers / vLLM / TGI, plus provenance metadata and a GGUF conversion
    command for LM Studio. ``specs`` is a flat list of AblationSpec/DirectionSpec.
    """
    from sparkablate.bake import bake_specs, save_baked
    from sparkablate.hooks import introspect

    if dtype != "float32":
        log("WARNING: baking from a half precision can leave a faint residual for "
            "directional ablation; --dtype float32 is recommended.")
    log(f"Loading model {model_name} (dtype={dtype}, device={device}) ...")
    model, tokenizer = load_model(model_name, dtype=dtype, device=device,
                                  trust_remote_code=trust_remote_code)
    arch = introspect(model, strict=True)
    log(f"Architecture: {arch.num_layers} layers, {arch.num_heads} heads, "
        f"head_dim {arch.head_dim}")

    label = "+".join(s.label() for s in specs)
    log(f"Baking: {label}")
    baked = bake_specs(model, arch, specs, prune_layers=prune_layers)

    return save_baked(model, tokenizer, out_dir, base_model=model_name,
                      interventions=baked["interventions"], dtype=dtype,
                      pruned_layers=baked["pruned_layers"], log=log)
