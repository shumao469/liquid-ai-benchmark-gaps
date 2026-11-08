from __future__ import annotations

import numpy as np
import torch

from liquid_revision.offline.metrics import (
    accuracy_macro_f1,
    binary_classification_metrics,
    count_trainable_parameters,
    measure_batch1_latency,
)


def test_har_metrics_include_accuracy_and_macro_f1() -> None:
    result = accuracy_macro_f1(
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 1, 1, 1, 2, 0]),
        num_classes=3,
    )
    assert result["accuracy"] == 4 / 6
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert len(result["per_class_recall"]) == 3


def test_physionet_metrics_include_discrimination_and_calibration() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    result = binary_classification_metrics(labels, probabilities)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
    assert "calibration_slope" in result


def test_latency_measurement_is_synchronized_and_batch1() -> None:
    layer = torch.nn.Linear(3, 2).eval()
    values = torch.ones(1, 3)
    result = measure_batch1_latency(
        lambda: layer(values), device="cpu", warmup=1, repeats=3
    )
    assert result["batch_size"] == 1
    assert result["timed_calls"] == 3
    assert result["median_ms"] >= 0
    assert count_trainable_parameters(layer) == 8

