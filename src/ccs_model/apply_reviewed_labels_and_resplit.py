#!/usr/bin/env python3
"""Apply blinded doctor reviews and rebuild a patient-disjoint development split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageOps


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def review_hash(path: Path) -> str:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    width, height = image.size
    ImageDraw.Draw(image).rectangle(
        (0, 0, round(width * 0.35), round(height * 0.06)), fill=(0, 0, 0)
    )
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95, optimize=True, exif=b"")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--target-validation", type=int, default=177)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    review_images = sorted(args.review_dir.glob("CCS12-*.jpg"))
    review_hash_to_id = {hashlib.sha256(path.read_bytes()).hexdigest(): path.stem for path in review_images}
    if len(review_hash_to_id) != 23:
        raise ValueError("expected 23 unique review images")

    review_id_to_index = {}
    for index, row in enumerate(rows):
        source = Path(str(row["local_image_path"]))
        if not source.is_absolute():
            source = args.data_root / source
        digest = review_hash(source)
        review_id = review_hash_to_id.get(digest)
        if review_id:
            review_id_to_index[review_id] = index
    if len(review_id_to_index) != 23:
        raise ValueError(f"matched {len(review_id_to_index)} of 23 reviewed images")

    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    if len(reviews) != 23:
        raise ValueError("review CSV must contain 23 rows")
    changes = Counter()
    reviewed_indices = set()
    for review in reviews:
        review_id = str(review["复核编号"]).strip()
        if review_id not in review_id_to_index:
            raise ValueError(f"unknown review id: {review_id}")
        grade_text = str(review["医生复核等级"]).strip()
        if grade_text not in {"0", "1", "2", "3"}:
            raise ValueError(f"invalid reviewed grade for {review_id}: {grade_text}")
        index = review_id_to_index[review_id]
        old_grade = int(rows[index]["ccs_grade"])
        new_grade = int(grade_text)
        rows[index] = {
            **rows[index],
            "ccs_grade": new_grade,
            "label_review_status": "single_clinician_reviewed_20260928",
            "previous_ccs_grade": old_grade,
        }
        reviewed_indices.add(index)
        changes[f"{old_grade}_to_{new_grade}"] += 1

    forced_train_groups = {str(rows[index]["patient_group_id"]) for index in reviewed_indices}
    by_group = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[str(row["patient_group_id"])].append(index)
    eligible_groups = [group for group in by_group if group not in forced_train_groups]
    overall = Counter(int(row["ccs_grade"]) for row in rows)
    overall_proportion = {grade: overall[grade] / len(rows) for grade in range(4)}

    best = None
    for seed in range(10000):
        candidates = eligible_groups.copy()
        random.Random(seed).shuffle(candidates)
        validation_groups = set()
        validation_count = 0
        for group in candidates:
            group_size = len(by_group[group])
            if validation_count < args.target_validation or abs(validation_count + group_size - args.target_validation) < abs(validation_count - args.target_validation):
                validation_groups.add(group)
                validation_count += group_size
            if validation_count >= args.target_validation:
                break
        validation_indices = [index for group in validation_groups for index in by_group[group]]
        grade_counts = Counter(int(rows[index]["ccs_grade"]) for index in validation_indices)
        if len(grade_counts) < 4:
            continue
        proportion_error = sum(
            abs(grade_counts[grade] / len(validation_indices) - overall_proportion[grade])
            for grade in range(4)
        )
        score = abs(len(validation_indices) - args.target_validation) / args.target_validation + proportion_error
        if best is None or score < best[0]:
            best = (score, seed, validation_groups, validation_indices)
    if best is None:
        raise RuntimeError("failed to build a stratified patient-group split")
    _, split_seed, validation_groups, validation_indices = best
    for row in rows:
        row["split"] = "val" if str(row["patient_group_id"]) in validation_groups else "train"

    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    train_patients = {row["patient_group_id"] for row in train}
    val_patients = {row["patient_group_id"] for row in val}
    train_eyes = {row["eye_group_id"] for row in train}
    val_eyes = {row["eye_group_id"] for row in val}
    if train_patients & val_patients or train_eyes & val_eyes:
        raise ValueError("rebuilt split has patient or eye leakage")
    if any(rows[index]["split"] != "train" for index in reviewed_indices):
        raise ValueError("reviewed samples were not all assigned to training")

    write_jsonl(args.output, rows)
    audit = {
        "schema_version": "ccs.reviewed-resplit.v1",
        "input_records": len(rows),
        "reviewed_records": len(reviewed_indices),
        "reviewed_patient_groups_forced_to_train": len(forced_train_groups),
        "label_changes": dict(sorted(changes.items())),
        "changed_label_records": sum(count for key, count in changes.items() if key.split("_to_")[0] != key.split("_to_")[1]),
        "split_seed": split_seed,
        "split_counts": {"train": len(train), "val": len(val)},
        "grade_counts": {
            "train": dict(sorted(Counter(int(row["ccs_grade"]) for row in train).items())),
            "val": dict(sorted(Counter(int(row["ccs_grade"]) for row in val).items())),
        },
        "wtw_source_counts": dict(sorted(Counter(row["wtw_source"] for row in rows).items())),
        "patient_overlap": len(train_patients & val_patients),
        "eye_overlap": len(train_eyes & val_eyes),
        "test_records_loaded": 0,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
