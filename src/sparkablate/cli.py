"""ablate: command-line entry point."""

from __future__ import annotations

import argparse
import sys

from sparkablate import __version__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ablate",
        description="Component-ablation experiments for open-weight causal LMs",
    )
    p.add_argument("--version", action="version", version=f"sparkablate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a sweep from a YAML config")
    run.add_argument("-c", "--config", required=True, help="path to experiment YAML")

    cmp_ = sub.add_parser("compare", help="greedy generation with vs. without an intervention")
    cmp_.add_argument("-m", "--model", required=True)
    cmp_.add_argument("--kind", choices=["head", "attn", "mlp", "layer"])
    cmp_.add_argument("--layer", type=int)
    cmp_.add_argument("--heads", type=int, nargs="*", default=[])
    cmp_.add_argument("--mode", choices=["zero"], default="zero",
                      help="compare supports zero mode (mean needs calibration; use 'run')")
    cmp_.add_argument("--direction", help="path to a direction artifact (.pt)")
    cmp_.add_argument("--op", choices=["project_out", "add"], default="project_out")
    cmp_.add_argument("--coefficient", type=float, default=1.0)
    cmp_.add_argument("--dir-layers", type=int, nargs="*", default=None,
                      help="decoder layers for the direction hook (default: all + embedding)")
    cmp_.add_argument("--chat-template", action="store_true")
    cmp_.add_argument("--prompt", required=True)
    cmp_.add_argument("--max-new-tokens", type=int, default=128)
    cmp_.add_argument("--dtype", default="bfloat16")
    cmp_.add_argument("--device", default="auto")
    cmp_.add_argument("--trust-remote-code", action="store_true")

    find = sub.add_parser("find-direction",
                          help="extract and score a difference-of-means direction")
    find.add_argument("-c", "--config", required=True, help="path to experiment YAML "
                      "with a 'direction' section")

    rep = sub.add_parser("report", help="print ranked results and write plots")
    rep.add_argument("results_dir")
    rep.add_argument("--top", type=int, default=25)
    rep.add_argument("--no-plots", action="store_true")

    intro = sub.add_parser(
        "introspect",
        help="verify a model's architecture resolves before committing to a sweep",
    )
    intro.add_argument("-m", "--model", required=True)
    intro.add_argument("--meta", action="store_true",
                       help="build from config on the meta device (no weight download)")
    intro.add_argument("--device", default="cpu")
    intro.add_argument("--trust-remote-code", action="store_true")

    args = p.parse_args(argv)

    if args.command == "run":
        from sparkablate.config import ExperimentConfig
        from sparkablate.runner import run_experiment

        cfg = ExperimentConfig.from_yaml(args.config)
        summary = run_experiment(cfg)
        print(f"done: {summary['conditions']} conditions -> {summary['results']}")
        return 0

    if args.command == "compare":
        from sparkablate.hooks import AblationSpec, DirectionSpec
        from sparkablate.runner import compare_generations

        specs = []
        if args.kind:
            if args.layer is None:
                p.error("--kind requires --layer")
            specs.append(AblationSpec(kind=args.kind, layer=args.layer,
                                      heads=tuple(args.heads), mode=args.mode))
        if args.direction:
            specs.append(DirectionSpec(
                op=args.op, coefficient=args.coefficient,
                layers="all" if args.dir_layers is None else tuple(args.dir_layers),
                vector_path=args.direction,
            ))
        if not specs:
            p.error("compare needs --kind and/or --direction")
        compare_generations(args.model, specs, args.prompt,
                            max_new_tokens=args.max_new_tokens, dtype=args.dtype,
                            device=args.device, trust_remote_code=args.trust_remote_code,
                            chat_template=args.chat_template)
        return 0

    if args.command == "find-direction":
        from sparkablate.config import ExperimentConfig
        from sparkablate.runner import run_find_direction

        cfg = ExperimentConfig.from_yaml(args.config)
        summary = run_find_direction(cfg)
        print(f"done: {summary['candidates']} candidates -> {summary['artifact']}")
        return 0

    if args.command == "introspect":
        from sparkablate.runner import introspect_model

        return introspect_model(args.model, meta=args.meta, device=args.device,
                                trust_remote_code=args.trust_remote_code)

    if args.command == "report":
        from sparkablate.report import make_plots, print_table

        print_table(args.results_dir, top=args.top)
        if not args.no_plots:
            try:
                make_plots(args.results_dir)
            except RuntimeError as e:
                print(f"(skipping plots: {e})", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
