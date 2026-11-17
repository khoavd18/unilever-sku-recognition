"""Versioned identity mapping between legacy logits and item-code catalog rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.catalog.profiles import ProductIdentityProfile, load_product_profile


_AXE5_PROFILE = load_product_profile("axe5")
EXPECTED_AXE5 = _AXE5_PROFILE.expected_mapping
_BLUE_LAVENDER_IDENTITIES = _AXE5_PROFILE.identity_guard("blue_lavender")


def ordered_classes(class_to_idx: dict[str, int]) -> list[str]:
    normalized = {str(name): int(index) for name, index in class_to_idx.items()}
    indexes = list(normalized.values())
    if len(indexes) != len(set(indexes)) or sorted(indexes) != list(range(len(indexes))):
        raise ValueError("Checkpoint class indexes must be unique and contiguous from zero")
    return [name for name, _ in sorted(normalized.items(), key=lambda pair: pair[1])]


def build_mapping(
    catalog: dict[str, Any],
    class_to_idx: dict[str, int],
    checkpoint_path: str | Path | None = None,
    *,
    product_profile: ProductIdentityProfile | None = None,
) -> dict[str, Any]:
    """Build exact mappings only; name similarity is never accepted as identity proof."""

    selected_profile = product_profile or _AXE5_PROFILE
    expected_mapping = selected_profile.expected_mapping
    classes = ordered_classes(class_to_idx)
    expected_classes = list(selected_profile.legacy_class_names)
    records = catalog["records"]
    by_legacy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        legacy = record.get("legacy_class_name")
        if legacy:
            by_legacy.setdefault(legacy, []).append(record)

    mappings: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if classes != expected_classes:
        issues.append(
            {
                "severity": "error",
                "code": "checkpoint_product_profile_class_mismatch",
                "product_profile": selected_profile.name,
                "expected_class_order": expected_classes,
                "checkpoint_class_order": classes,
                "missing_classes": [name for name in expected_classes if name not in classes],
                "unexpected_classes": [name for name in classes if name not in expected_classes],
            }
        )
    for index, legacy_name in enumerate(classes):
        candidates = by_legacy.get(legacy_name, [])
        if len(candidates) == 1:
            record = candidates[0]
            expected_code = expected_mapping.get(legacy_name)
            verified = expected_code is None or record["item_code"] == expected_code
            if not verified:
                issues.append({
                    "severity": "error",
                    "code": "mentor_identity_mismatch",
                    "class_index": index,
                    "legacy_class_name": legacy_name,
                    "expected_item_code": expected_code,
                    "catalog_item_code": record["item_code"],
                })
            mappings.append({
                "class_index": index,
                "item_code": record["item_code"],
                "item_name": record["item_name"],
                "class_name": record["class_name"],
                "legacy_class_name": legacy_name,
                "identity_evidence": "exact_catalog_legacy_class_label",
                "verified": verified,
            })
        elif not candidates:
            issues.append({
                "severity": "error", "code": "legacy_class_not_found",
                "class_index": index, "legacy_class_name": legacy_name,
            })
        else:
            issues.append({
                "severity": "error", "code": "ambiguous_legacy_class_match",
                "class_index": index, "legacy_class_name": legacy_name,
                "candidate_item_codes": [row["item_code"] for row in candidates],
            })

    mapped_codes = {row["item_code"] for row in mappings}
    explicit_scope = catalog.get("scope", {}).get("selected_item_codes")
    blue_lavender_guard: dict[str, Any] | None = None
    generic_guards: dict[str, Any] | None = None
    if selected_profile.name == "axe5":
        # Preserve the deployed v1 payload exactly, including its named guard.
        spray_code = _BLUE_LAVENDER_IDENTITIES["spray"]
        wash_code = _BLUE_LAVENDER_IDENTITIES["body_wash"]
        spray = next((row for row in records if row["item_code"] == spray_code), None)
        wash = next((row for row in records if row["item_code"] == wash_code), None)
        spray_explicitly_out_of_scope = bool(explicit_scope) and spray_code not in explicit_scope
        identity_guard_verified = bool(wash) and (
            (bool(spray) and spray["item_code"] != wash["item_code"])
            or (spray is None and spray_explicitly_out_of_scope)
        )
        blue_lavender_guard = {
            "spray_item_code": spray["item_code"] if spray else None,
            "body_wash_item_code": wash["item_code"] if wash else None,
            "spray_expected_item_code": spray_code,
            "body_wash_expected_item_code": wash_code,
            "spray_explicitly_out_of_scope": spray_explicitly_out_of_scope,
            "distinct": bool(spray and wash and spray["item_code"] != wash["item_code"]),
            "identity_guard_verified": identity_guard_verified,
            "spray_is_trained": spray_code in mapped_codes,
            "body_wash_is_trained": wash_code in mapped_codes,
        }
        if not identity_guard_verified:
            issues.append({"severity": "error", "code": "blue_lavender_identity_guard_failed"})
    else:
        generic_guards = {}
        records_by_code = {str(row["item_code"]): row for row in records}
        for guard_name, identities in selected_profile.identity_guards.items():
            codes = list(identities.values())
            states = {
                identity: {
                    "expected_item_code": code,
                    "present_in_catalog": code in records_by_code,
                    "explicitly_out_of_scope": bool(explicit_scope) and code not in explicit_scope,
                    "trained": code in mapped_codes,
                }
                for identity, code in identities.items()
            }
            verified = len(codes) == len(set(codes)) and all(
                state["present_in_catalog"] or state["explicitly_out_of_scope"]
                for state in states.values()
            )
            generic_guards[guard_name] = {
                "identities": states,
                "identity_guard_verified": verified,
            }
            if not verified:
                issues.append(
                    {
                        "severity": "error",
                        "code": "product_identity_guard_failed",
                        "guard": guard_name,
                    }
                )

    checkpoint_fingerprint = None
    if checkpoint_path:
        path = Path(checkpoint_path).resolve()
        checkpoint_fingerprint = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "catalog_version": catalog["catalog_version"],
        "catalog_validation_status": catalog.get("status"),
        "catalog_scope": catalog.get("scope"),
        "catalog_scope_note": (
            "This mapping verifies only the listed trained classes by exact workbook label. "
            "An explicit scoped catalog does not change the full catalog's validation status."
        ),
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_class_order_source": "checkpoint.class_to_idx sorted by stored index",
        "mapping_policy": "exact legacy class label only; no fuzzy identity assignment",
        "mappings": mappings,
        "coverage": {
            "catalog_item_count": len(records),
            "trained_class_count": len(classes),
            "mapped_trained_class_count": len(mappings),
            "gallery_item_count": None,
            "note": "Catalog, trained classifier, and gallery coverage are independent.",
        },
        "compatibility": {
            "weights_reusable": not issues and len(mappings) == len(classes),
            "change_type": "identity_only_label_mapping" if not issues else "incompatible_or_unverified",
            "checkpoint_modified": False,
            "source_folders_renamed": False,
        },
        "issues": issues,
    }
    if blue_lavender_guard is not None:
        payload["blue_lavender_guard"] = blue_lavender_guard
    else:
        payload["product_profile"] = selected_profile.name
        payload["identity_guards"] = generic_guards
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["mapping_version"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["status"] = "verified" if not issues else "invalid"
    return payload


def build_exact_catalog_mapping(
    catalog: dict[str, Any],
    class_to_idx: dict[str, int],
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Map every checkpoint class to one scoped catalog record by exact label.

    This is the deployable counterpart to a hand-curated product profile.  It
    deliberately accepts no fuzzy matching: a class must occur exactly once in
    the scoped catalog and every selected catalog row must be represented by
    the checkpoint.
    """

    classes = ordered_classes(class_to_idx)
    records = catalog["records"]
    by_legacy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        legacy = record.get("legacy_class_name")
        if isinstance(legacy, str) and legacy:
            by_legacy.setdefault(legacy, []).append(record)

    mappings: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, legacy_name in enumerate(classes):
        candidates = by_legacy.get(legacy_name, [])
        if len(candidates) != 1:
            issues.append({
                "severity": "error",
                "code": "checkpoint_class_requires_one_exact_catalog_record",
                "class_index": index,
                "legacy_class_name": legacy_name,
                "matching_catalog_records": len(candidates),
            })
            continue
        record = candidates[0]
        mappings.append({
            "class_index": index,
            "item_code": record["item_code"],
            "item_name": record["item_name"],
            "class_name": record["class_name"],
            "legacy_class_name": legacy_name,
            "identity_evidence": "exact_catalog_legacy_class_label",
            "verified": True,
        })

    mapped_legacy = {row["legacy_class_name"] for row in mappings}
    extra_catalog_classes = sorted(set(by_legacy) - set(classes))
    if extra_catalog_classes:
        issues.append({
            "severity": "error",
            "code": "scoped_catalog_contains_classes_absent_from_checkpoint",
            "legacy_class_names": extra_catalog_classes,
        })

    checkpoint_fingerprint = None
    if checkpoint_path:
        path = Path(checkpoint_path).resolve()
        checkpoint_fingerprint = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "catalog_version": catalog["catalog_version"],
        "catalog_validation_status": catalog.get("status"),
        "catalog_scope": catalog.get("scope"),
        "catalog_scope_note": "The scoped catalog contains exactly the checkpoint's verified classes.",
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_class_order_source": "checkpoint.class_to_idx sorted by stored index",
        "mapping_policy": "exact legacy class label only; no fuzzy identity assignment",
        "mappings": mappings,
        "coverage": {
            "catalog_item_count": len(records),
            "trained_class_count": len(classes),
            "mapped_trained_class_count": len(mappings),
            "gallery_item_count": None,
            "note": "Catalog, trained classifier, and gallery coverage are independently versioned.",
        },
        "compatibility": {
            "weights_reusable": not issues and len(mappings) == len(classes),
            "change_type": "exact_catalog_identity_mapping" if not issues else "incompatible_or_unverified",
            "checkpoint_modified": False,
            "source_folders_renamed": False,
        },
        "product_profile": None,
        "identity_guards": {},
        "issues": issues,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["mapping_version"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    payload["status"] = "verified" if not issues else "invalid"
    return payload


def load_mapping(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("status") != "verified":
        raise ValueError(f"Mapping is not a verified schema-v1 mapping: {path}")
    mappings = value.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("Mapping must contain at least one checkpoint class")
    required = {"class_index", "item_code", "legacy_class_name", "verified"}
    for index, row in enumerate(mappings):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Mapping row {index} is missing required fields")
        if not isinstance(row["item_code"], str) or not row["item_code"].strip():
            raise ValueError(f"Mapping row {index} item_code must be a non-empty string")
        if row["verified"] is not True:
            raise ValueError(f"Mapping row {index} is not identity-verified")
    indexes = [int(row["class_index"]) for row in mappings]
    if indexes != list(range(len(indexes))):
        raise ValueError("Mapping order no longer matches explicit checkpoint class order")
    item_codes = [row["item_code"] for row in mappings]
    if len(item_codes) != len(set(item_codes)):
        raise ValueError("Mapping contains duplicate item codes")
    legacy_names = [str(row["legacy_class_name"]) for row in mappings]
    if len(legacy_names) != len(set(legacy_names)):
        raise ValueError("Mapping contains duplicate legacy class names")
    guard = value.get("blue_lavender_guard")
    if guard is not None:
        legacy_guard_verified = bool(
            isinstance(guard, dict)
            and guard.get("distinct") is True
            and guard.get("spray_item_code")
            and guard.get("body_wash_item_code")
            and guard.get("spray_item_code") != guard.get("body_wash_item_code")
        )
        if not isinstance(guard, dict) or not (
            guard.get("identity_guard_verified") is True or legacy_guard_verified
        ):
            raise ValueError("Mapping does not verify the Blue Lavender identity guard")
        if (
            guard.get("spray_expected_item_code") is not None
            and guard.get("spray_expected_item_code")
            == guard.get("body_wash_expected_item_code")
        ):
            raise ValueError("Mapping merges Blue Lavender spray and body wash identities")
    else:
        generic_guards = value.get("identity_guards")
        if not isinstance(generic_guards, dict) or any(
            not isinstance(row, dict) or row.get("identity_guard_verified") is not True
            for row in generic_guards.values()
        ):
            raise ValueError("Mapping contains an unverified product identity guard")
    fingerprint = value.get("checkpoint_fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str)
        or not fingerprint.startswith("sha256:")
        or len(fingerprint) != 71
    ):
        raise ValueError("Mapping checkpoint_fingerprint is not a SHA-256 identity")
    return value
