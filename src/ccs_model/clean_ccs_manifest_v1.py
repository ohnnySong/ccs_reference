#!/usr/bin/env python3
"""Create a clean CCS development manifest without the anomalous group or duplicates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


SAME_FIELDS = (
    "ccs_grade",
    "opening_area_mm2",
    "wtw_mm",
    "wtw_source",
    "wtw_match_status",
    "wtw_match_basis",
    "patient_group_id",
    "split",
    "local_image_path",
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = [row for row in rows if row.get("split") == "val"]
    validation_groups = Counter(str(row["patient_group_id"]) for row in validation)
    anomalous_group, anomalous_count = validation_groups.most_common(1)[0]
    if anomalous_count != 134:
        raise ValueError(f"expected 134-record anomalous validation group, found {anomalous_count}")

    without_anomalous = [
        row for row in rows if str(row["patient_group_id"]) != anomalous_group
    ]
    by_path: dict[str, list[dict]] = defaultdict(list)
    for row in without_anomalous:
        by_path[str(row["local_image_path"])].append(row)

    retained_sample_ids: set[str] = set()
    duplicate_groups = 0
    removed_duplicates = 0
    for path, candidates in by_path.items():
        if len(candidates) == 1:
            retained_sample_ids.add(str(candidates[0]["sample_id"]))
            continue
        if len(candidates) != 2:
            raise ValueError(f"unexpected duplicate count for one image path: {len(candidates)}")
        sources = {str(row["source_id"]) for row in candidates}
        if sources != {"ccs-summary-v1", "ccs-augmentation-20260823"}:
            raise ValueError(f"unexpected duplicate source combination: {sorted(sources)}")
        if any(candidates[0].get(field) != candidates[1].get(field) for field in SAME_FIELDS):
            raise ValueError(f"duplicate records disagree on clinical fields: {path}")
        authoritative = next(
            row for row in candidates if row["source_id"] == "ccs-summary-v1"
        )
        retained_sample_ids.add(str(authoritative["sample_id"]))
        duplicate_groups += 1
        removed_duplicates += 1

    cleaned = [
        row for row in without_anomalous if str(row["sample_id"]) in retained_sample_ids
    ]
    if duplicate_groups != 383 or len(cleaned) != 954:
        raise ValueError(
            f"unexpected cleaning result: duplicate_groups={duplicate_groups}, records={len(cleaned)}"
        )

    train = [row for row in cleaned if row["split"] == "train"]
    val = [row for row in cleaned if row["split"] == "val"]
    train_patients = {row["patient_group_id"] for row in train}
    val_patients = {row["patient_group_id"] for row in val}
    train_eyes = {row["eye_group_id"] for row in train}
    val_eyes = {row["eye_group_id"] for row in val}
    unique_paths = {row["local_image_path"] for row in cleaned}
    unique_samples = {row["sample_id"] for row in cleaned}
    if len(unique_paths) != len(cleaned) or len(unique_samples) != len(cleaned):
        raise ValueError("cleaned manifest still contains duplicate paths or sample IDs")
    if train_patients & val_patients or train_eyes & val_eyes:
        raise ValueError("cleaned manifest has patient or eye overlap between splits")

    write_jsonl(args.output, cleaned)
    audit = {
        "schema_version": "ccs.development-clean.v1",
        "input_records": len(rows),
        "excluded_anomalous_validation_group_records": anomalous_count,
        "duplicate_path_groups_removed": duplicate_groups,
        "duplicate_records_removed": removed_duplicates,
        "output_records": len(cleaned),
        "split_counts": dict(sorted(Counter(row["split"] for row in cleaned).items())),
        "opacity_grade_counts": {
            split: dict(
                sorted(
                    Counter(
                        int(row["ccs_grade"])
                        for row in cleaned
                        if row["split"] == split
                    ).items()
                )
            )
            for split in ("train", "val")
        },
        "wtw_source_counts": dict(
            sorted(Counter(row["wtw_source"] for row in cleaned).items())
        ),
        "source_counts": dict(
            sorted(Counter(row["source_id"] for row in cleaned).items())
        ),
        "unique_image_paths": len(unique_paths),
        "unique_sample_ids": len(unique_samples),
        "patient_group_overlap_count": len(train_patients & val_patients),
        "eye_group_overlap_count": len(train_eyes & val_eyes),
        "test_records_loaded": 0,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
