"""Aggregate CCS v4 metrics without patient identifiers or predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence


SEVERITY_NAMES = ("light", "moderate", "severe")


def _confusion(reference: Sequence[int], prediction: Sequence[int], classes: int) -> list[list[int]]:
    if len(reference) != len(prediction) or not reference:
        raise ValueError("reference and prediction must be non-empty and equal length")
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for expected, observed in zip(reference, prediction, strict=True):
        if not 0 <= expected < classes or not 0 <= observed < classes:
            raise ValueError("class index is outside the configured range")
        matrix[expected][observed] += 1
    return matrix


def _binary_ranking_metrics(
    reference: Sequence[int], probability: Sequence[float]
) -> dict[str, float]:
    if len(reference) != len(probability):
        raise ValueError("binary reference and probability must have equal length")
    positives = sum(reference)
    negatives = len(reference) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC/AUPRC require both binary classes")
    ranked = sorted(zip(probability, reference, strict=True), reverse=True)
    positive_scores = [score for score, label in zip(probability, reference, strict=True) if label]
    negative_scores = [score for score, label in zip(probability, reference, strict=True) if not label]
    concordance = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores
    )
    auroc = concordance / (positives * negatives)
    true_positives = 0
    average_precision = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label == 1:
            true_positives += 1
            average_precision += true_positives / rank
    return {"auroc": auroc, "auprc": average_precision / positives}


def severity_metrics(
    reference: Sequence[int],
    prediction: Sequence[int],
    collapsed_probability: Sequence[float] | None = None,
) -> dict[str, object]:
    """Report three-class performance and light vs moderate+severe performance."""

    matrix = _confusion(reference, prediction, 3)
    total = len(reference)
    per_class: dict[str, dict[str, float]] = {}
    f1_values = []
    recalls = []
    for index, name in enumerate(SEVERITY_NAMES):
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(row[index] for row in matrix) - tp
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": sum(matrix[index]),
        }
        f1_values.append(f1)
        recalls.append(recall)

    collapsed_reference = [int(value >= 1) for value in reference]
    collapsed_prediction = [int(value >= 1) for value in prediction]
    collapsed = _confusion(collapsed_reference, collapsed_prediction, 2)
    tn, fp = collapsed[0]
    fn, tp = collapsed[1]
    collapsed_metrics: dict[str, object] = {
        "definition": "reference severity is moderate or severe",
        "confusion_matrix": collapsed,
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / total,
    }
    if collapsed_probability is not None:
        collapsed_metrics.update(
            _binary_ranking_metrics(collapsed_reference, collapsed_probability)
        )
    return {
        "labels": list(SEVERITY_NAMES),
        "confusion_matrix": matrix,
        "exact_accuracy": sum(matrix[i][i] for i in range(3)) / total,
        "balanced_accuracy": sum(recalls) / 3,
        "macro_f1": sum(f1_values) / 3,
        "per_class": per_class,
        "moderate_or_severe": collapsed_metrics,
    }


def multiclass_metrics(
    reference: Sequence[int],
    prediction: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Report aggregate and per-class metrics for an arbitrary class mapping."""

    names = tuple(class_names)
    matrix = _confusion(reference, prediction, len(names))
    total = len(reference)
    per_class: dict[str, dict[str, float]] = {}
    recalls = []
    f1_values = []
    for index, name in enumerate(names):
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(row[index] for row in matrix) - tp
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": sum(matrix[index]),
        }
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "labels": list(names),
        "confusion_matrix": matrix,
        "exact_accuracy": sum(matrix[i][i] for i in range(len(names))) / total,
        "balanced_accuracy": sum(recalls) / len(names),
        "macro_f1": sum(f1_values) / len(names),
        "per_class": per_class,
    }


def binary_metrics(
    reference: Sequence[int],
    prediction: Sequence[int],
    probability: Sequence[float],
) -> dict[str, object]:
    """Evaluate light (0) versus moderate-or-severe (1)."""

    matrix = _confusion(reference, prediction, 2)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = len(reference)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "labels": ["light", "moderate_or_severe"],
        "confusion_matrix": matrix,
        "accuracy": (tp + tn) / total,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision,
        "f1": f1,
        **_binary_ranking_metrics(reference, probability),
    }


def area_metrics(reference_mm2: Sequence[float], prediction_mm2: Sequence[float]) -> dict[str, float]:
    """Measure opening-area agreement in physical square millimetres."""

    if len(reference_mm2) != len(prediction_mm2) or not reference_mm2:
        raise ValueError("area arrays must be non-empty and equal length")
    errors = [float(p) - float(r) for r, p in zip(reference_mm2, prediction_mm2, strict=True)]
    if not all(math.isfinite(value) for value in errors):
        raise ValueError("area values must be finite")
    count = len(errors)
    bias = sum(errors) / count
    variance = sum((value - bias) ** 2 for value in errors) / max(count - 1, 1)
    sd = math.sqrt(variance)
    return {
        "mae_mm2": sum(abs(value) for value in errors) / count,
        "rmse_mm2": math.sqrt(sum(value * value for value in errors) / count),
        "signed_bias_mm2": bias,
        "bland_altman_lower_mm2": bias - 1.96 * sd,
        "bland_altman_upper_mm2": bias + 1.96 * sd,
        "within_1_mm2": sum(abs(value) <= 1.0 for value in errors) / count,
        "within_2_mm2": sum(abs(value) <= 2.0 for value in errors) / count,
    }
