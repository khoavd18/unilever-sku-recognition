"""Generic validation and metrics reporting tool for YOLO detectors.

Supports any number of classes/brands dynamically without hardcoding class names.
Can be used via CLI or imported into notebooks/scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    """Resolve path relative to project root if not absolute."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def evaluate_detector_metrics(
    weights: str | Path,
    data: str | Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    output_dir: str | Path | None = None,
    save_plots: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run validation and extract comprehensive metrics for all detected classes dynamically.

    Args:
        weights: Path to YOLO weights (.pt).
        data: Path to dataset YAML (e.g., data.yaml).
        split: Split to evaluate on ('val' or 'test').
        imgsz: Inference image size.
        batch: Batch size.
        device: Device ID ('0', 'cpu', 'cuda:0', etc.).
        output_dir: Optional path to save output CSV and JSON reports.
        save_plots: Whether to generate and save PR/confusion matrix plots.
        verbose: Whether to print formatted table to stdout.

    Returns:
        Dict containing model info, speed, overall metrics, and per-class metrics.
    """
    weights_path = resolve_path(weights)
    data_path = resolve_path(data)

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    from ultralytics import YOLO, __version__ as ultralytics_version

    model = YOLO(str(weights_path))

    # Retrieve fused model parameters and FLOPs
    try:
        model.fuse()
        model_info = model.info(detailed=False)
        layers, params, gradients, flops = model_info if model_info else (0, 0, 0, 0)
    except Exception:
        layers, params, gradients, flops = 0, 0, 0, 0

    val_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "plots": save_plots,
        "verbose": False,
    }

    if output_dir is not None:
        out_path = resolve_path(output_dir)
        val_kwargs["project"] = str(out_path.parent)
        val_kwargs["name"] = out_path.name
        val_kwargs["exist_ok"] = True

    metrics = model.val(**val_kwargs)

    # Class names mapping from model / validator
    names: Mapping[int, str] = getattr(metrics, "names", getattr(model, "names", {}))

    # Speed metrics
    speed = getattr(metrics, "speed", {})
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    loss_ms = float(speed.get("loss", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms

    # Total images seen
    total_images = int(getattr(metrics, "seen", 0))
    if total_images == 0:
        try:
            import yaml
            data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
            split_dir = data_cfg.get(split, f"images/{split}")
            base_dir = Path(data_cfg.get("path", data_path.parent))
            img_dir = Path(split_dir) if Path(split_dir).is_absolute() else base_dir / split_dir
            if img_dir.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                total_images = len([f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts])
        except Exception:
            total_images = 0

    # Overall / macro metrics
    mean_p = float(metrics.box.mp)
    mean_r = float(metrics.box.mr)
    mean_map50 = float(metrics.box.map50)
    mean_map50_95 = float(metrics.box.map)

    # Instance counts
    nt_per_image = getattr(metrics, "nt_per_image", [0] * len(names))
    nt_per_class = getattr(metrics, "nt_per_class", [0] * len(names))
    total_instances = int(nt_per_class.sum()) if hasattr(nt_per_class, "sum") else int(sum(nt_per_class))
    mean_f1 = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) > 0 else 0.0

    # Per-class breakdown
    ap_classes = getattr(metrics.box, "ap_class_index", list(names.keys()))

    per_class_rows: list[dict[str, Any]] = []
    for idx, class_id in enumerate(ap_classes):
        class_name = names.get(class_id, f"class_{class_id}")
        p, r, ap50, ap = metrics.box.class_result(idx)
        p, r, ap50, ap = float(p), float(r), float(ap50), float(ap)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        n_images = int(nt_per_image[class_id]) if class_id < len(nt_per_image) else 0
        n_instances = int(nt_per_class[class_id]) if class_id < len(nt_per_class) else 0

        per_class_rows.append(
            {
                "class_id": int(class_id),
                "class": class_name,
                "images": n_images,
                "instances": n_instances,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "map50": round(ap50, 4),
                "map50_95": round(ap, 4),
            }
        )

    summary: dict[str, Any] = {
        "ultralytics_version": str(ultralytics_version),
        "weights": str(weights_path),
        "data": str(data_path),
        "split": split,
        "model_summary": {
            "layers": layers,
            "parameters": params,
            "gradients": gradients,
            "gflops": round(float(flops), 2) if flops else 0.0,
        },
        "speed": {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "loss_ms": round(loss_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_ms_per_image": round(total_latency_ms, 2),
            "fps": round(1000.0 / total_latency_ms, 1) if total_latency_ms > 0 else 0.0,
        },
        "all": {
            "images": total_images,
            "instances": total_instances,
            "precision": round(mean_p, 4),
            "recall": round(mean_r, 4),
            "f1_score": round(mean_f1, 4),
            "map50": round(mean_map50, 4),
            "map50_95": round(mean_map50_95, 4),
        },
        "per_class": per_class_rows,
    }

    # Print formatted output
    if verbose:
        print("\n" + "=" * 80)
        print(f"Model summary: {layers} layers, {params:,} parameters, {flops:.1f} GFLOPs")
        print("=" * 80)
        header = f"{'Class':<14} {'Images':>8} {'Instances':>11} {'Box(P)':>10} {'R':>10} {'F1':>10} {'mAP50':>10} {'mAP50-95':>10}"
        print(header)
        print("-" * 80)
        # All row
        all_row = (
            f"{'all':<14} {total_images:>8} {total_instances:>11} "
            f"{mean_p:>10.3f} {mean_r:>10.3f} {mean_f1:>10.3f} {mean_map50:>10.3f} {mean_map50_95:>10.3f}"
        )
        print(all_row)
        for row in per_class_rows:
            line = (
                f"{row['class']:<14} {row['images']:>8} {row['instances']:>11} "
                f"{row['precision']:>10.3f} {row['recall']:>10.3f} {row['f1_score']:>10.3f} "
                f"{row['map50']:>10.3f} {row['map50_95']:>10.3f}"
            )
            print(line)
        print("-" * 80)
        print(
            f"Speed: {preprocess_ms:.1f}ms preprocess, {inference_ms:.1f}ms inference, "
            f"{loss_ms:.1f}ms loss, {postprocess_ms:.1f}ms postprocess per image "
            f"(Total: {total_latency_ms:.1f}ms ~ {1000.0/total_latency_ms:.1f} FPS)"
        )
        print("=" * 80 + "\n")

    # Export to files if output_dir provided
    if output_dir is not None:
        out_path = resolve_path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # JSON summary
        json_file = out_path / "metrics_summary.json"
        json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        # CSV table
        csv_file = out_path / "metrics_table.csv"
        csv_rows = [
            {
                "class_id": "all",
                "class": "all",
                "images": total_images,
                "instances": total_instances,
                "precision": round(mean_p, 4),
                "recall": round(mean_r, 4),
                "f1_score": round(mean_f1, 4),
                "map50": round(mean_map50, 4),
                "map50_95": round(mean_map50_95, 4),
            },
            *per_class_rows,
        ]
        with csv_file.open("w", encoding="utf-8", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        if verbose:
            print(f"Results saved to: {out_path}")

    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights file")
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml file")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"], help="Split to evaluate")
    parser.add_argument("--output-dir", default=None, help="Directory to save JSON and CSV reports")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--device", default="0", help="Computation device (e.g. '0' or 'cpu')")
    parser.add_argument("--plots", action="store_true", help="Generate and save Ultralytics plots")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_detector_metrics(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=args.output_dir,
        save_plots=args.plots,
        verbose=True,
    )


if __name__ == "__main__":
    main()
