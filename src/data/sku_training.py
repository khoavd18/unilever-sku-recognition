"""Prepare versioned MobileNet manifests from SKU crops without copying images.

Source groups stay together. Groups linked by byte-identical crops also stay
together, even when Roboflow gave them different filenames. Split repair uses
labels and group membership only, before any model/threshold evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from src.data.sku_gallery import is_augmented
from src.data.yolo_archive import write_csv, write_json

SPLITS = ("train", "val", "test")


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_splits(
    component_labels: dict[str, set[str]],
    original_votes: dict[str, Counter],
    *,
    seed: int = 42,
) -> dict[str, str]:
    """Preserve original membership where possible; cover every class in each split.

    Never fix one missing class by removing another class's last component from
    a split. A constrained dataset that this greedy repair cannot cover fails
    explicitly rather than silently emitting incomplete validation/test sets.
    """
    by_class: dict[str, set[str]] = defaultdict(set)
    for component, labels in component_labels.items():
        for label in labels:
            by_class[label].add(component)
    insufficient = {label: len(groups) for label, groups in by_class.items() if len(groups) < 3}
    if insufficient:
        raise ValueError(f"Need at least 3 independent components per SKU: {insufficient}")
    assigned = {
        component: max(SPLITS, key=lambda split: (votes[split], -SPLITS.index(split)))
        for component, votes in original_votes.items()
    }
    counts = Counter((label, assigned[group]) for group, labels in component_labels.items() for label in labels)
    requirements = sorted(
        ((label, split) for label in by_class for split in SPLITS),
        key=lambda pair: (len(by_class[pair[0]]), pair[0], SPLITS.index(pair[1])),
    )
    for label, split in requirements:
        if counts[label, split]:
            continue
        candidates = [
            group for group in by_class[label]
            if assigned[group] != split
            and all(counts[other, assigned[group]] > 1 for other in component_labels[group])
        ]
        if not candidates:
            raise ValueError(
                f"Cannot repair {label}/{split} without removing another SKU's coverage. "
                "Review the linked source groups; no prepared dataset was written."
            )

        def cost(group: str) -> tuple:
            vote = original_votes[group]
            missing_covered = sum(counts[other, split] == 0 for other in component_labels[group])
            tie = hashlib.sha256(f"{seed}:{group}:{split}".encode()).hexdigest()
            return (-missing_covered, sum(vote.values()) - vote[split], tie)

        chosen = min(candidates, key=cost)
        previous = assigned[chosen]
        for other in component_labels[chosen]:
            counts[other, previous] -= 1
            counts[other, split] += 1
        assigned[chosen] = split
    return assigned


def _preview(rows: list[dict], path: Path) -> None:
    """One crop per brand for a small, labelled visual inspection sheet."""
    brands = sorted({row["brand_name"] for row in rows})
    canvas = Image.new("RGB", (1200, ((len(brands) + 4) // 5) * 240), "white")
    draw = ImageDraw.Draw(canvas)
    for index, brand in enumerate(brands):
        candidates = [row for row in rows if row["brand_name"] == brand]
        row = min(candidates, key=lambda r: (r["source_is_augmented"], -min(r["width"], r["height"]), r["image_path"]))
        x, y = (index % 5) * 240, (index // 5) * 240
        with Image.open(row["image_path"]) as image:
            tile = ImageOps.contain(image.convert("RGB"), (220, 185))
            canvas.paste(tile, (x + (240 - tile.width) // 2, y + 5 + (185 - tile.height) // 2))
        draw.text((x + 8, y + 195), brand, fill="black")
        draw.text((x + 8, y + 210), f"{row['item_code']} / {row['split']}", fill="black")
        draw.text((x + 8, y + 225), "augmented source" if row["source_is_augmented"] else "unmarked source", fill="black")
    canvas.save(path, quality=90)


def prepare_training_dataset(source_dir: Path, output_dir: Path, *, seed: int = 42, quarantine_conflicts: bool = False) -> dict:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    if output_dir.is_relative_to(source_dir):
        raise ValueError("Training output must be separate from the source crop dataset")
    raw = read_rows(source_dir / "crops_manifest.csv", {
        "crop_path", "gallery_split", "group_id", "source_image", "annotation_index",
        "sku_class_name", "item_code", "brand_name", "match_method",
    })
    source_audit = source_dir / "audit_report.json"
    source_zip = json.loads(source_audit.read_text(encoding="utf-8")).get("zip", {}).get("path", "") if source_audit.is_file() else ""
    class_map = read_rows(source_dir / "class_map.csv", {
        "sku_class_name", "item_code", "brand_name", "product_name", "match_method",
    })
    selected = [row for row in class_map if row["match_method"] == "exact" and row["item_code"] and row["brand_name"]]
    identities = {row["sku_class_name"]: row for row in selected}
    if len(identities) != len(selected) or len({r["item_code"] for r in selected}) != len(selected):
        raise ValueError("Exact class mapping must have unique SKU names and item codes")
    class_to_idx = {name: index for index, name in enumerate(sorted(identities))}
    parents: dict[str, str] = {}

    def find(group: str) -> str:
        parents.setdefault(group, group)
        root = group
        while parents[root] != root:
            root = parents[root]
        while parents[group] != root:
            parent = parents[group]
            parents[group] = root
            group = parent
        return root

    def union(left: str, right: str) -> None:
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    rows, paths = [], set()
    for index, row in enumerate(raw):
        identity = identities.get(row["sku_class_name"])
        if (row["match_method"] != "exact" or identity is None
                or row["item_code"] != identity["item_code"] or row["brand_name"] != identity["brand_name"]):
            raise ValueError(f"Unverified class/item/brand identity at crop row {index + 2}")
        split = "val" if row["gallery_split"] == "valid" else row["gallery_split"]
        if split not in SPLITS or not row["group_id"]:
            raise ValueError(f"Invalid group/split at crop row {index + 2}")
        image_path = (source_dir / row["crop_path"]).resolve()
        if not image_path.is_relative_to(source_dir) or image_path in paths:
            raise ValueError(f"Crop path escapes source or is duplicated: {image_path}")
        paths.add(image_path)
        contents = image_path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        with Image.open(io.BytesIO(contents)) as image:
            width, height = image.size
            image.verify()
        group = row["group_id"]
        rows.append({
            "image_path": str(image_path), "class_name": row["sku_class_name"],
            "item_code": row["item_code"], "brand_name": row["brand_name"],
            "group_id": group, "original_gallery_split": split,
            "source_image_id": row["source_image"], "source_annotation_id": row["annotation_index"],
            "source_zip": source_zip,
            "generated_crop_path": str(image_path), "image_sha256": digest,
            "source_is_augmented": is_augmented(Path(row["source_image"]).stem),
            "width": width, "height": height,
        })
        if (index + 1) % 5000 == 0:
            print(f"Verified {index + 1}/{len(raw)} crops", flush=True)
    hash_codes = defaultdict(set)
    for row in rows:
        hash_codes[row["image_sha256"]].add(row["item_code"])
    conflicts = {digest: sorted(codes) for digest, codes in hash_codes.items() if len(codes) > 1}
    if conflicts and not quarantine_conflicts:
        raise ValueError(
            f"{len(conflicts)} identical crop hashes have conflicting item codes. "
            "Review them or explicitly choose --quarantine-conflicts; no output was written."
        )
    quarantined = [dict(row, reason="identical_content_conflicting_item_codes") for row in rows if row["image_sha256"] in conflicts]
    rows = [row for row in rows if row["image_sha256"] not in conflicts]
    seen_hashes = {}
    for row in rows:
        group, digest = row["group_id"], row["image_sha256"]
        find(group)
        if digest in seen_hashes:
            union(group, seen_hashes[digest]["group_id"])
        else:
            seen_hashes[digest] = row
    if {row["class_name"] for row in rows} != set(class_to_idx):
        raise ValueError("One or more exact-mapped SKU classes have no usable crops")
    component_labels, votes = defaultdict(set), defaultdict(Counter)
    original_group_splits = defaultdict(set)
    for row in rows:
        component = find(row["group_id"])
        row["family_id"] = component
        component_labels[component].add(row["class_name"])
        votes[component][row["original_gallery_split"]] += 1
        original_group_splits[row["group_id"]].add(row["original_gallery_split"])
    if any(len(splits) != 1 for splits in original_group_splits.values()):
        raise ValueError("A source group has inconsistent original gallery_split values")
    assignments = repair_splits(component_labels, votes, seed=seed)
    for row in rows:
        row["split"] = assignments[row["family_id"]]
    counts = Counter((row["class_name"], row["split"]) for row in rows)
    group_counts = Counter((label, assignments[group]) for group, labels in component_labels.items() for label in labels)
    classes = [{
        "class_index": index, "class_name": name, "item_code": identities[name]["item_code"],
        "brand_name": identities[name]["brand_name"], "product_name": identities[name]["product_name"],
    } for name, index in class_to_idx.items()]
    # Everything above is read-only. All mutations below target a fresh output.
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    for split in SPLITS:
        write_csv(output_dir / "manifests" / f"{split}.csv", fields, (row for row in rows if row["split"] == split))
    write_csv(output_dir / "crop_manifest.csv", fields, rows)
    if quarantined:
        write_csv(output_dir / "quarantined_crops.csv", list(quarantined[0]), quarantined)
    write_csv(output_dir / "sku_catalog.csv", list(classes[0]), classes)
    write_json(output_dir / "class_mapping.json", {"schema_version": 1, "identity": "item_code", "class_to_idx": class_to_idx, "classes": classes})
    write_csv(output_dir / "sku_split_counts.csv", ["item_code", "class_name", *SPLITS, "train_groups", "val_groups", "test_groups"], [
        {"item_code": row["item_code"], "class_name": row["class_name"],
         **{split: counts[row["class_name"], split] for split in SPLITS},
         **{f"{split}_groups": group_counts[row["class_name"], split] for split in SPLITS}}
        for row in classes
    ])
    write_csv(output_dir / "group_assignments.csv", ["group_id", "family_id", "original_split", "split"], [
        {"group_id": group, "family_id": find(group), "original_split": next(iter(original_group_splits[group])), "split": assignments[find(group)]}
        for group in sorted(parents)
    ])
    hashes = {name: file_hash(output_dir / name) for name in ("class_mapping.json", "manifests/train.csv", "manifests/val.csv", "manifests/test.csv")}
    training_fingerprint = hashlib.sha256("\n".join(hashes[name] for name in ("class_mapping.json", "manifests/train.csv", "manifests/val.csv")).encode()).hexdigest()
    report = {
        "schema_version": 1, "status": "ready", "source_dir": str(source_dir), "seed": seed,
        "source_manifest_sha256": file_hash(source_dir / "crops_manifest.csv"),
        "source_class_map_sha256": file_hash(source_dir / "class_map.csv"),
        "class_count": len(classes), "brand_count": len({row["brand_name"] for row in classes}),
        "source_crop_count": len(raw), "quarantined_crop_count": len(quarantined),
        "conflicting_hash_count": len(conflicts), "conflicting_hash_item_codes": conflicts,
        "quarantined_crops_per_item_code": dict(Counter(row["item_code"] for row in quarantined)),
        "crop_counts": dict(Counter(row["split"] for row in rows)),
        "sku_counts": {split: sum(counts[name, split] > 0 for name in class_to_idx) for split in SPLITS},
        "source_group_count": len(parents), "independent_component_count": len(assignments),
        "duplicate_crop_count": len(rows) - len(seen_hashes),
        "moved_source_groups": sum(assignments[find(group)] not in old for group, old in original_group_splits.items()),
        "moved_crops": sum(row["split"] != row["original_gallery_split"] for row in rows),
        "cross_split_group_leakage": 0, "cross_split_exact_hash_leakage": 0,
        "minimum_components_per_sku_per_split": min(group_counts[name, split] for name in class_to_idx for split in SPLITS),
        "augmented_source_crops": dict(Counter(row["split"] for row in rows if row["source_is_augmented"])),
        "manifest_sha256": hashes, "training_fingerprint": training_fingerprint,
        "images_copied": False, "training_started": False, "test_evaluation_run": False,
        "remaining_uncertainty": "Filename groups and exact-byte hashes do not prove capture-session or near-duplicate independence. Some representatives are augmented; evaluate real shelf photos separately.",
    }
    _preview(rows, output_dir / "crop_preview.jpg")
    write_json(output_dir / "preparation_report.json", report)
    return report
