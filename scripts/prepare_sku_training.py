"""Verify SKU crops and prepare complete, group-disjoint MobileNet manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.paths import resolve_repo_path  # noqa: E402
from src.data.sku_training import prepare_training_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/sku469_v33")
    parser.add_argument("--output-dir", default="data/sku469_v33_training_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quarantine-conflicts", action="store_true", help="Exclude all exact-content label conflicts from manifests and write a review CSV; never guess the correct SKU.")
    args = parser.parse_args()
    report = prepare_training_dataset(resolve_repo_path(args.source_dir), resolve_repo_path(args.output_dir), seed=args.seed, quarantine_conflicts=args.quarantine_conflicts)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
