"""Turn results.jsonl into a ranked table and plots."""

from __future__ import annotations

import json
import os


def load_results(results_dir: str) -> tuple[dict, list[dict]]:
    path = os.path.join(results_dir, "results.jsonl")
    baseline, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("condition") == "baseline":
                baseline = row
            else:
                rows.append(row)
    if baseline is None:
        raise ValueError(f"no baseline row found in {path}")
    return baseline, rows


def print_table(results_dir: str, top: int = 25, log=print) -> None:
    baseline, rows = load_results(results_dir)
    if not rows:
        log("no condition rows in results.jsonl")
        return

    # Perplexity rows rank by biggest damage; refusal rows by biggest drop.
    if "delta_nll" in rows[0]:
        rows.sort(key=lambda r: r.get("delta_nll", 0.0), reverse=True)
    else:
        rows.sort(key=lambda r: r.get("delta_refusal", 0.0))

    parts = []
    if "perplexity" in baseline:
        parts.append(f"perplexity {baseline['perplexity']:.3f} ({baseline['tokens']} tokens)")
    if "refusal_rate" in baseline:
        parts.append(f"refusal {baseline['refusal_rate']:.2f} "
                     f"({baseline.get('n_prompts', '?')} prompts)")
    if "harmless_nll" in baseline:
        parts.append(f"harmless NLL {baseline['harmless_nll']:.4f}")
    log("baseline: " + ", ".join(parts))

    cols = []  # (header, format) pairs keyed off the row fields present
    if "perplexity" in rows[0]:
        cols.append(("ppl", "perplexity", "10.3f"))
    if "delta_nll" in rows[0]:
        cols.append(("dNLL", "delta_nll", "+10.4f"))
    if "refusal_rate" in rows[0]:
        cols.append(("refusal", "refusal_rate", "10.2f"))
    if "delta_refusal" in rows[0]:
        cols.append(("dRef", "delta_refusal", "+10.2f"))
    if "delta_harmless_nll" in rows[0]:
        cols.append(("dNLL(ok)", "delta_harmless_nll", "+10.4f"))

    log(f"{'condition':<36} " + " ".join(f"{name:>10}" for name, _, _ in cols))
    for row in rows[:top]:
        cells = " ".join(f"{row[field]:>{fmt}}" for _, field, fmt in cols)
        log(f"{row['condition']:<36} {cells}")
    if len(rows) > top:
        log(f"... {len(rows) - top} more rows in results.jsonl")


def make_plots(results_dir: str, log=print) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("plots require the optional dependency: pip install matplotlib") from e

    baseline, rows = load_results(results_dir)
    written = []

    # find-direction candidate rows: refusal drop per (layer, position).
    candidates = [r for r in rows if "layer_key" in r and "delta_refusal" in r]
    if candidates:
        def layer_x(r):
            return -1 if r["layer_key"] == "emb" else int(r["layer_key"][1:])

        offsets = sorted({r["position"] for r in candidates})
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.8 / len(offsets)
        for k, off in enumerate(offsets):
            sub = sorted((r for r in candidates if r["position"] == off), key=layer_x)
            xs = [layer_x(r) + k * width for r in sub]
            ys = [r["delta_refusal"] for r in sub]
            ax.bar(xs, ys, width=width, label=f"pos {off}")
        ax.set_xlabel("layer (-1 = embedding)")
        ax.set_ylabel("Δ refusal rate vs baseline")
        ax.set_title("Candidate directions: refusal drop under project-out")
        if len(offsets) > 1:
            ax.legend()
        out = os.path.join(results_dir, "direction_candidates.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
        log(f"wrote {out}")

    # Single-spec conditions only (sweeps); grouped by kind.
    singles = [r for r in rows if len(r.get("specs", [])) == 1 and "specs" in r]
    by_kind: dict[str, list[dict]] = {}
    for r in singles:
        kind = r["specs"][0].get("kind")
        if kind:  # direction specs have no component kind; skip
            by_kind.setdefault(kind, []).append(r)

    def metric(r: dict) -> float:
        return r["delta_nll"] if "delta_nll" in r else r["delta_refusal"]

    def metric_label(r: dict) -> str:
        return "Δ NLL vs baseline" if "delta_nll" in r else "Δ refusal rate vs baseline"

    for kind, krows in by_kind.items():
        if kind == "head":
            layers = sorted({r["specs"][0]["layer"] for r in krows})
            heads = sorted({h for r in krows for h in r["specs"][0]["heads"]})
            grid = [[float("nan")] * len(heads) for _ in layers]
            li = {l: i for i, l in enumerate(layers)}
            hi = {h: i for i, h in enumerate(heads)}
            for r in krows:
                s = r["specs"][0]
                if len(s["heads"]) == 1:
                    grid[li[s["layer"]]][hi[s["heads"][0]]] = metric(r)
            fig, ax = plt.subplots(figsize=(max(6, len(heads) * 0.4), max(4, len(layers) * 0.3)))
            im = ax.imshow(grid, aspect="auto", cmap="viridis")
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_xticks(range(len(heads)), heads)
            ax.set_yticks(range(len(layers)), layers)
            fig.colorbar(im, label=metric_label(krows[0]))
            ax.set_title("Per-head ablation impact")
        else:
            krows.sort(key=lambda r: r["specs"][0]["layer"])
            xs = [r["specs"][0]["layer"] for r in krows]
            ys = [metric(r) for r in krows]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(xs, ys)
            ax.set_xlabel("layer")
            ax.set_ylabel(metric_label(krows[0]))
            ax.set_title(f"{kind} ablation impact by layer")
        out = os.path.join(results_dir, f"{kind}_sweep.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
        log(f"wrote {out}")
    return written
