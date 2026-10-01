"""Command line entry points for reproducible inspection experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from partwise.data import DatasetError, build_manifest


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="partwise")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit MVTec AD and save a split manifest")
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--category", default="metal_nut")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--validation-fraction", type=float, default=0.2)
    audit.add_argument("--output", type=Path, default=Path("artifacts/manifest.json"))

    baseline = commands.add_parser("baseline", help="Run a global-feature anomaly baseline")
    baseline.add_argument("--data-root", type=Path, required=True)
    baseline.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baseline.json"))
    baseline.add_argument("--batch-size", type=int, default=16)
    baseline.add_argument("--false-reject-target", type=float, default=0.05)

    patchcore = commands.add_parser("patchcore", help="Fit PatchCore on the locked normal split")
    patchcore.add_argument("--data-root", type=Path, required=True)
    patchcore.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    patchcore.add_argument("--output", type=Path, default=Path("artifacts/patchcore.json"))
    patchcore.add_argument("--model-output", type=Path, default=Path("artifacts/patchcore_model.pt"))
    patchcore.add_argument("--maps-output", type=Path, default=Path("artifacts/patchcore_maps.npz"))
    patchcore.add_argument("--batch-size", type=int, default=8)
    patchcore.add_argument("--coreset-ratio", type=float, default=0.005)
    patchcore.add_argument("--image-size", type=int, default=224)

    simulate = commands.add_parser("simulate", help="Project inspection outcomes for a scenario")
    simulate.add_argument("--result", type=Path, required=True)
    simulate.add_argument("--scenario", type=Path, required=True)
    simulate.add_argument("--output", type=Path, default=Path("artifacts/projection.json"))
    args = parser.parse_args(argv)

    if args.command == "audit":
        try:
            manifest = build_manifest(
                args.data_root,
                category=args.category,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        except (DatasetError, ValueError) as exc:
            print(f"Dataset audit failed: {exc}", file=sys.stderr)
            return 2
        _save_json(args.output, manifest)
        print(f"Category: {manifest['category']}")
        for key in ("train_normal", "validation_normal", "test_normal", "test_defective"):
            print(f"{key}: {len(manifest[key])}")
        print(f"Manifest: {args.output}")
        return 0

    if args.command == "patchcore":
        from partwise.patchcore import run_patchcore

        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            result = run_patchcore(
                manifest,
                args.data_root,
                batch_size=args.batch_size,
                coreset_ratio=args.coreset_ratio,
                image_size=args.image_size,
                seed=manifest["seed"],
                model_output=args.model_output,
                maps_output=args.maps_output,
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"PatchCore failed: {exc}", file=sys.stderr)
            return 2
        _save_json(args.output, result)
        print(f"Threshold: {result['threshold']:.6f}")
        print(f"Defect recall: {result['metrics']['defect_recall']:.3f}")
        print(f"False reject rate: {result['metrics']['false_reject_rate']:.3f}")
        print(f"Image AUROC: {result['metrics']['image_auroc']:.3f}")
        print(f"Pixel AUROC: {result['metrics']['pixel_auroc']:.3f}")
        print(f"Results: {args.output}")
        return 0

    if args.command == "simulate":
        from partwise.decisions import Scenario, project_decisions

        try:
            measured = json.loads(args.result.read_text(encoding="utf-8"))
            scenario_values = json.loads(args.scenario.read_text(encoding="utf-8"))
            scenario = Scenario.from_dict(scenario_values)
            metrics = measured["metrics"]
            projection = project_decisions(
                metrics["defect_recall"], metrics["false_reject_rate"], scenario
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"Simulation failed: {exc}", file=sys.stderr)
            return 2
        _save_json(args.output, {
            "basis": {
                "method": measured["method"],
                "threshold": measured["threshold"],
                "defect_recall": metrics["defect_recall"],
                "false_reject_rate": metrics["false_reject_rate"],
                "test_good_count": metrics["n_good"],
                "test_defective_count": metrics["n_defective"],
            },
            "scenario": scenario_values,
            "projection": projection,
        })
        print(f"Projected missed defects: {projection['missed_defective']:.1f}")
        print(f"Projected good parts rejected: {projection['good_rejected']:.1f}")
        print(f"Projected review workload: {projection['reviewed']:.1f}")
        print(f"Projected cost units: {projection['total_cost']:.2f}")
        print(f"Projection: {args.output}")
        return 0

    from partwise.baseline import score_manifest
    from partwise.evaluation import evaluate_result

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = score_manifest(
            manifest,
            args.data_root,
            batch_size=args.batch_size,
            false_reject_target=args.false_reject_target,
        )
        result["metrics"] = evaluate_result(result)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Baseline failed: {exc}", file=sys.stderr)
        return 2
    _save_json(args.output, result)
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Defect recall: {result['metrics']['defect_recall']:.3f}")
    print(f"False reject rate: {result['metrics']['false_reject_rate']:.3f}")
    print(f"Image AUROC: {result['metrics']['image_auroc']:.3f}")
    print(f"Results: {args.output}")
    return 0
