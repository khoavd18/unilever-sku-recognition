"""Validate ITEMCODE_SKU.xlsx and create a versioned item-code catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.catalog import create_scoped_catalog, import_workbook, write_catalog


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default="ITEMCODE_SKU.xlsx")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--output-dir", default="artifacts/catalog/v1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--scope-name", default=None)
    parser.add_argument("--scope-item-codes", nargs="+", default=None)
    args = parser.parse_args()
    if bool(args.scope_name) != bool(args.scope_item_codes):
        parser.error("--scope-name and --scope-item-codes must be supplied together")
    full_bundle = import_workbook(args.workbook, args.sheet)
    bundle = (
        create_scoped_catalog(full_bundle, args.scope_item_codes, args.scope_name)
        if args.scope_name else full_bundle
    )
    if not args.validate_only:
        if bundle["summary"]["errors"]:
            print(json.dumps({
                "status": bundle["status"], "errors": bundle["summary"]["errors"],
                "message": "Invalid catalog was not written; resolve blocking issues first.",
            }, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        write_catalog(bundle, args.output_dir)
    print(json.dumps({
        "status": bundle["status"],
        "catalog_version": bundle["catalog_version"],
        "records": bundle["record_count"],
        "numeric_item_code_rows": bundle["numeric_item_code_rows"],
        "scope": ({
            "scope_name": bundle["scope"]["scope_name"],
            "selected_item_codes": bundle["scope"]["selected_item_codes"],
            "occurrence_status": [
                {"item_code": row["item_code"], "status": row["status"],
                 "workbook_occurrence_count": row["workbook_occurrence_count"]}
                for row in bundle["scope"]["occurrence_validation"]
            ],
        } if bundle.get("scope") else None),
        "source_full_catalog_status": bundle.get("source_full_catalog", {}).get("status"),
        **bundle["summary"],
        "output_dir": None if args.validate_only else str(Path(args.output_dir).resolve()),
    }, ensure_ascii=False, indent=2))
    if bundle["summary"]["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
