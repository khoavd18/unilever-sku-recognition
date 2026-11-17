"""Create or validate a versioned checkpoint-to-product identity mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.catalog import load_catalog
from src.catalog.mapping import build_mapping
from src.catalog.profiles import load_product_profile


def finalize_mapping(
    mapping: dict,
    output: Path,
    *,
    dry_run: bool,
) -> dict:
    """Validate or write one mapping; dry-run is strictly read-only."""

    existing_matches: bool | None = None
    if output.exists():
        if dry_run:
            existing = json.loads(output.read_text(encoding="utf-8"))
            existing_matches = existing == mapping
        else:
            raise FileExistsError(f"Mapping artifact already exists: {output}")
    report = {
        "operation": "mapping_validation_dry_run" if dry_run else "mapping_migration",
        "status": mapping["status"],
        "output": str(output.resolve()),
        "output_written": not dry_run,
        "existing_output_found": output.exists(),
        "existing_output_matches": existing_matches,
        "checkpoint_modified": False,
        "folders_renamed": False,
        "class_order": [row["legacy_class_name"] for row in mapping["mappings"]],
        "issues": mapping["issues"],
    }
    if dry_run:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output.with_name(output.stem + ".dry_run_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", default="artifacts/catalog/axe5_demo_v3/product_catalog.json"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/axe_5sku/classification_100ep_s42/best_model.pth",
    )
    parser.add_argument(
        "--output",
        default="artifacts/mappings/axe5_classification_100ep_s42_axe5_demo_mapping.v3.json",
    )
    parser.add_argument(
        "--product-profile",
        default="axe5",
        help="Validated profile name under configs/products/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate in memory and compare an existing output; write no files.",
    )
    args = parser.parse_args()

    import torch

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "class_to_idx" not in checkpoint:
        raise ValueError("Checkpoint has no class_to_idx metadata")
    mapping = build_mapping(
        load_catalog(args.catalog),
        checkpoint["class_to_idx"],
        checkpoint_path,
        product_profile=load_product_profile(args.product_profile),
    )
    output = Path(args.output)
    report = finalize_mapping(mapping, output, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mapping["status"] != "verified" or report["existing_output_matches"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
