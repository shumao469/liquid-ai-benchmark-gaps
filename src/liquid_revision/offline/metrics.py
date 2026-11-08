"""Dependency-light predictive and synchronized deployment metrics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn


def accuracy_macro_f1(
    labels: np.ndarray, predictions: np.ndarray, *, num_classes: int
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("labels and predictions must be nonempty vectors of equal length")
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(int(num_classes)):
        true_positive = int(np.sum((truth == class_index) & (predicted == class_index)))
        false_positive = int(np.sum((truth != class_index) & (predicted == class_index)))
        false_negative = int(np.sum((truth == class_index) & (predicted != class_index)))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(float(recall))
        f1_scores.append(float(f1))
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_recall": recalls,
        "n": int(truth.size),
    }


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape or truth.size == 0:
        raise ValueError("labels and probabilities must be nonempty vectors of equal length")
    positives = truth == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both outcome classes")
    ranks = _average_tie_ranks(scores)
    rank_sum = float(ranks[positives].sum())
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def binary_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape or truth.size == 0:
        raise ValueError("labels and probabilities must be nonempty vectors of equal length")
    positive_count = int((truth == 1).sum())
    if positive_count == 0:
        raise ValueError("AUPRC requires at least one positive outcome")
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = (truth[order] == 1).astype(np.int64)
    cumulative_positive = np.cumsum(sorted_truth)
    precision = cumulative_positive / np.arange(1, len(sorted_truth) + 1)
    return float(precision[sorted_truth == 1].sum() / positive_count)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if bins <= 1 or truth.shape != scores.shape or truth.size == 0:
        raise ValueError("invalid calibration inputs")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(truth))
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (scores >= boundaries[index]) & (scores <= boundaries[index + 1])
        else:
            selected = (scores >= boundaries[index]) & (scores < boundaries[index + 1])
        if selected.any():
            result += selected.sum() / total * abs(scores[selected].mean() - truth[selected].mean())
    return float(result)


def calibration_intercept_slope(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float | None, float | None]:
    """Fit y ~ intercept + slope*logit(p) by Newton-Raphson."""

    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape or truth.size == 0 or len(np.unique(truth)) < 2:
        return None, None
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    predictor = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones_like(predictor), predictor])
    coefficients = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(50):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (truth - fitted)
        try:
            delta = np.linalg.solve(hessian + np.eye(2) * 1e-8, gradient)
        except np.linalg.LinAlgError:
            return None, None
        coefficients += delta
        if np.linalg.norm(delta) < 1e-8:
            break
    if not np.isfinite(coefficients).all():
        return None, None
    return float(coefficients[0]), float(coefficients[1])


def binary_classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape or truth.size == 0:
        raise ValueError("labels and probabilities must be nonempty vectors of equal length")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    predicted = scores >= threshold
    positive = truth == 1
    sensitivity = float(np.mean(predicted[positive])) if positive.any() else 0.0
    specificity = float(np.mean(~predicted[~positive])) if (~positive).any() else 0.0
    intercept, slope = calibration_intercept_slope(truth, scores)
    return {
        "auroc": binary_auroc(truth, scores),
        "auprc": binary_auprc(truth, scores),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "brier_score": float(np.mean((scores - truth) ** 2)),
        "expected_calibration_error_10": expected_calibration_error(truth, scores, bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
        "n": int(truth.size),
        "positives": int(positive.sum()),
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_batch1_latency(
    callable_model: Callable[[], torch.Tensor],
    *,
    device: torch.device | str,
    warmup: int = 1000,
    repeats: int = 10000,
) -> dict[str, Any]:
    """Measure synchronized end-to-end model-call latency.

    The caller closes over already resident batch-size-one tensors.  Host-to-device
    transfer is therefore excluded and this fact is recorded in the result.
    """

    target = torch.device(device)
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    measurements = np.empty(repeats, dtype=np.float64)
    with torch.inference_mode():
        for _ in range(warmup):
            output = callable_model()
            if not torch.isfinite(output).all():
                raise FloatingPointError("nonfinite output during latency warm-up")
        _synchronize(target)
        for index in range(repeats):
            _synchronize(target)
            started = time.perf_counter_ns()
            output = callable_model()
            _synchronize(target)
            measurements[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    if not np.isfinite(measurements).all():
        raise FloatingPointError("nonfinite latency measurement")
    return {
        "median_ms": float(np.median(measurements)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "mean_ms": float(np.mean(measurements)),
        "deadline_miss_fraction_20ms": float(np.mean(measurements > 20.0)),
        "warmup_calls": int(warmup),
        "timed_calls": int(repeats),
        "batch_size": 1,
        "synchronized": target.type == "cuda",
        "device": str(target),
        "host_to_device_transfer_included": False,
    }

