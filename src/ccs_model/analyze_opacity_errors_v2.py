#!/usr/bin/env python3
"""Aggregate opacity-grading error distributions without patient identifiers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from ccs_model.train_opacity_features_v2 import Head
from ccs_model.metrics_v4 import multiclass_metrics
from ccs_model.train_document_v1 import OPACITY_NAMES


def area_band(value: float) -> str:
    if value > 20.0:
        return "A_gt_20"
    if value > 16.0:
        return "16_lt_A_le_20"
    return "A_le_16"


def grouped(rows, correct):
    total = Counter(); errors = Counter()
    for value, ok in rows:
        total[str(value)] += 1
        if not ok:
            errors[str(value)] += 1
    return {
        key: {
            "records": total[key],
            "errors": errors[key],
            "error_rate": errors[key] / total[key],
        }
        for key in sorted(total)
    }


def cross_grouped(rows, correct):
    return grouped([(f"{left}|grade_{right}", ok) for left, right, ok in rows], correct)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    chunks = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(args.feature_dir.glob("chunk_*.pt"))
    ]
    features = torch.cat([chunk["features"] for chunk in chunks])
    labels = torch.cat([chunk["labels"] for chunk in chunks])
    splits = sum([chunk["splits"] for chunk in chunks], [])
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.source_manifest is not None:
        source_rows = [
            json.loads(line)
            for line in args.source_manifest.read_text().splitlines()
            if line.strip()
        ]
        source_by_id = {str(row["sample_id"]): row for row in source_rows}
        rows = [dict(source_by_id.get(str(row["sample_id"]), {}), **row) for row in rows]
    if len(rows) != len(features):
        raise ValueError("feature and manifest record counts differ")

    selected = torch.tensor([split == "val" for split in splits])
    features = features[selected]
    labels = labels[selected]
    val_rows = [row for row, keep in zip(rows, selected.tolist(), strict=True) if keep]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Head()
    model.load_state_dict(checkpoint["head_state_dict"])
    model.eval()
    normalized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.no_grad():
        probabilities = model(normalized).softmax(1)
    predictions = probabilities.argmax(1)
    confidence = probabilities.max(1).values
    correct = predictions.eq(labels)

    matrix = [[0] * 4 for _ in range(4)]
    transition = Counter()
    distance = Counter()
    true_total = Counter()
    true_error = Counter()
    predicted_total = Counter()
    confidence_correct = []
    confidence_error = []
    for expected, predicted, conf, ok in zip(
        labels.tolist(), predictions.tolist(), confidence.tolist(), correct.tolist(), strict=True
    ):
        matrix[expected][predicted] += 1
        true_total[expected] += 1
        predicted_total[predicted] += 1
        (confidence_correct if ok else confidence_error).append(conf)
        if not ok:
            true_error[expected] += 1
            transition[f"{expected}_to_{predicted}"] += 1
            distance[abs(expected - predicted)] += 1

    grouping_values = {
        "wtw_source": [(row["wtw_source"], bool(ok)) for row, ok in zip(val_rows, correct.tolist(), strict=True)],
        "data_source": [(row.get("source_id", "unknown"), bool(ok)) for row, ok in zip(val_rows, correct.tolist(), strict=True)],
        "area_band": [(area_band(float(row["opening_area_mm2"])), bool(ok)) for row, ok in zip(val_rows, correct.tolist(), strict=True)],
        "image_suffix": [(Path(row["local_image_path"]).suffix.lower(), bool(ok)) for row, ok in zip(val_rows, correct.tolist(), strict=True)],
    }
    patient_total = Counter()
    patient_errors = Counter()
    for row, ok in zip(val_rows, correct.tolist(), strict=True):
        patient = str(row["patient_group_id"])
        patient_total[patient] += 1
        if not ok:
            patient_errors[patient] += 1
    errors_per_patient = Counter(patient_errors.values())
    patient_rows = defaultdict(list)
    for row, expected, predicted, ok in zip(
        val_rows, labels.tolist(), predictions.tolist(), correct.tolist(), strict=True
    ):
        patient_rows[str(row["patient_group_id"])].append(
            (row, int(expected), int(predicted), bool(ok))
        )
    ranked_patient_groups = sorted(
        patient_rows.items(),
        key=lambda pair: (-sum(not item[3] for item in pair[1]), -len(pair[1])),
    )
    top_error_patient_groups = []
    for rank, (_, items) in enumerate(ranked_patient_groups[:10], 1):
        errors = sum(not item[3] for item in items)
        top_error_patient_groups.append(
            {
                "rank": rank,
                "records": len(items),
                "errors": errors,
                "error_rate": errors / len(items),
                "true_grade_counts": dict(sorted(Counter(item[1] for item in items).items())),
                "predicted_grade_counts": dict(sorted(Counter(item[2] for item in items).items())),
                "data_source_counts": dict(
                    sorted(Counter(item[0].get("source_id", "unknown") for item in items).items())
                ),
                "wtw_source_counts": dict(
                    sorted(Counter(item[0]["wtw_source"] for item in items).items())
                ),
            }
        )
    largest_patient_id = ranked_patient_groups[0][0]
    diagnostic_keep = [
        str(row["patient_group_id"]) != largest_patient_id for row in val_rows
    ]
    diagnostic_reference = [
        value for value, keep in zip(labels.tolist(), diagnostic_keep, strict=True) if keep
    ]
    diagnostic_prediction = [
        value for value, keep in zip(predictions.tolist(), diagnostic_keep, strict=True) if keep
    ]
    error_count = int((~correct).sum())
    report = {
        "schema_version": "ccs.opacity-error-distribution.v1",
        "scope": "internal_validation_aggregate_only",
        "records": len(labels),
        "correct": int(correct.sum()),
        "errors": error_count,
        "error_rate": error_count / len(labels),
        "confusion_matrix": matrix,
        "true_grade_distribution": {
            str(grade): {
                "records": true_total[grade],
                "errors": true_error[grade],
                "error_rate": true_error[grade] / true_total[grade],
            }
            for grade in range(4)
        },
        "predicted_grade_distribution": {str(grade): predicted_total[grade] for grade in range(4)},
        "error_transitions": dict(sorted(transition.items(), key=lambda item: (-item[1], item[0]))),
        "error_distance": {
            str(key): {"errors": value, "share_of_errors": value / error_count}
            for key, value in sorted(distance.items())
        },
        "grouped_error_distribution": {
            name: grouped(values, correct) for name, values in grouping_values.items()
        },
        "cross_grouped_error_distribution": {
            "wtw_source_by_true_grade": cross_grouped(
                [
                    (row["wtw_source"], int(expected), bool(ok))
                    for row, expected, ok in zip(val_rows, labels.tolist(), correct.tolist(), strict=True)
                ],
                correct,
            ),
            "data_source_by_true_grade": cross_grouped(
                [
                    (row.get("source_id", "unknown"), int(expected), bool(ok))
                    for row, expected, ok in zip(val_rows, labels.tolist(), correct.tolist(), strict=True)
                ],
                correct,
            ),
        },
        "patient_group_distribution": {
            "patient_groups": len(patient_total),
            "patient_groups_with_errors": len(patient_errors),
            "errors_per_affected_patient_group": {
                str(count): groups for count, groups in sorted(errors_per_patient.items())
            },
            "top_error_patient_groups_anonymized": top_error_patient_groups,
            "diagnostic_excluding_largest_group": {
                "posthoc_diagnostic_only": True,
                "excluded_records": len(ranked_patient_groups[0][1]),
                "remaining_records": len(diagnostic_reference),
                "metrics": multiclass_metrics(
                    diagnostic_reference, diagnostic_prediction, OPACITY_NAMES
                ),
            },
        },
        "confidence": {
            "correct_mean": sum(confidence_correct) / len(confidence_correct),
            "error_mean": sum(confidence_error) / len(confidence_error),
            "errors_confidence_ge_0_70": sum(value >= 0.70 for value in confidence_error),
            "errors_confidence_ge_0_80": sum(value >= 0.80 for value in confidence_error),
        },
        "contains_patient_identifiers": False,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
