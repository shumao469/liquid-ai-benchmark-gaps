"""Pre-registered paired hierarchical inference for offline robustness curves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from liquid_revision.io_utils import atomic_write_json, sha256_file

from .metrics import accuracy_macro_f1, binary_auroc


DEFAULT_COMPARISONS = (
    ("ltc", "lstm"),
    ("ltc", "ode-rnn"),
    ("cfc", "lstm"),
    ("cfc", "ode-rnn"),
)


@dataclass(frozen=True)
class PredictionBundle:
    ids: np.ndarray
    group_ids: np.ndarray
    labels: np.ndarray
    logits: np.ndarray


def _load_prediction(path: str | Path, expected_sha256: str | None = None) -> PredictionBundle:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError(f"prediction artifact hash mismatch: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {"ids", "group_ids", "labels", "logits"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"prediction artifact lacks {sorted(missing)}: {source}")
        ids = np.asarray(payload["ids"]).astype(str)
        group_ids = np.asarray(payload["group_ids"]).astype(str)
        labels = np.asarray(payload["labels"]).reshape(-1)
        logits = np.asarray(payload["logits"])
    n = len(ids)
    if group_ids.shape != (n,) or labels.shape != (n,) or logits.shape[0] != n:
        raise ValueError(f"prediction arrays are misaligned: {source}")
    if len(set(ids.tolist())) != n:
        raise ValueError(f"prediction sample IDs are duplicated: {source}")
    return PredictionBundle(ids=ids, group_ids=group_ids, labels=labels, logits=logits)


def _metric(dataset: str, labels: np.ndarray, logits: np.ndarray) -> float:
    if dataset == "har":
        if logits.ndim != 2 or logits.shape[1] != 6:
            raise ValueError("HAR logits must have six columns")
        return float(
            accuracy_macro_f1(
                labels, np.argmax(logits, axis=1), num_classes=6
            )["macro_f1"]
        )
    if dataset == "physionet2012":
        flat = logits.reshape(len(labels), -1)
        if flat.shape[1] != 1:
            raise ValueError("PhysioNet logits must have one column")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(flat[:, 0], -40, 40)))
        return float(binary_auroc(labels, probabilities))
    raise ValueError(f"unsupported analysis dataset: {dataset}")


def _curve_summary(
    dataset: str,
    curve: Mapping[float, PredictionBundle],
    indices: np.ndarray,
) -> dict[str, float]:
    severities = np.asarray(sorted(curve), dtype=np.float64)
    if len(severities) < 2 or severities[-1] <= severities[0]:
        raise ValueError("robustness AUC requires at least two distinct severities")
    values = np.asarray(
        [
            _metric(dataset, curve[float(level)].labels[indices], curve[float(level)].logits[indices])
            for level in severities
        ],
        dtype=np.float64,
    )
    span = float(severities[-1] - severities[0])
    area = float(np.trapezoid(values, severities))
    normalized_average = area / span
    return {
        "area_under_robustness_curve": area,
        "severity_span": span,
        "normalized_average_metric": normalized_average,
        "nominal_metric": float(values[0]),
        "degradation_from_nominal": float(values[0] - normalized_average),
    }


def _validate_aligned(reference: PredictionBundle, observed: PredictionBundle) -> None:
    for name in ("ids", "group_ids", "labels"):
        if not np.array_equal(getattr(reference, name), getattr(observed, name)):
            raise ValueError(f"paired prediction identity mismatch at {name}")


def _sample_group_indices(
    bundle: PredictionBundle,
    *,
    dataset: str,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.unique(bundle.group_ids)
    by_group = {
        group: np.flatnonzero(bundle.group_ids == group) for group in groups
    }
    if dataset == "physionet2012":
        strata: dict[int, list[str]] = {}
        for group in groups:
            labels = np.unique(bundle.labels[by_group[group]])
            if len(labels) != 1:
                raise ValueError("each PhysioNet patient must have exactly one outcome")
            strata.setdefault(int(labels[0]), []).append(str(group))
        if len(strata) < 2:
            raise ValueError("PhysioNet paired bootstrap requires both outcome strata")
        sampled_groups: list[str] = []
        for label in sorted(strata):
            values = np.asarray(sorted(strata[label]))
            sampled_groups.extend(rng.choice(values, size=len(values), replace=True).tolist())
    else:
        values = np.asarray(sorted(groups.tolist()))
        sampled_groups = rng.choice(values, size=len(values), replace=True).tolist()
    return np.concatenate([by_group[str(group)] for group in sampled_groups]).astype(np.int64)


def holm_adjust(raw_p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values in original comparison order."""

    raw = np.asarray(raw_p_values, dtype=np.float64)
    if raw.ndim != 1 or np.any(~np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
        raise ValueError("raw p-values must be finite values in [0,1]")
    order = np.argsort(raw, kind="stable")
    adjusted_sorted = np.empty(len(raw), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(raw) - rank) * float(raw[index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(raw), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _parse_comparisons(values: Iterable[Any] | None) -> list[tuple[str, str]]:
    source = DEFAULT_COMPARISONS if values is None else values
    comparisons: list[tuple[str, str]] = []
    for value in source:
        if isinstance(value, Mapping):
            pair = (str(value["first"]), str(value["second"]))
        else:
            pair_values = list(value)
            if len(pair_values) != 2:
                raise ValueError("each comparison must contain exactly two model names")
            pair = (str(pair_values[0]), str(pair_values[1]))
        comparisons.append(pair)
    if not comparisons or len(comparisons) != len(set(comparisons)):
        raise ValueError("comparisons must be non-empty and unique")
    return comparisons


def hierarchical_paired_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    analysis_seed: int = 20_261_105,
    comparisons: Iterable[Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze paired robustness-AUC contrasts with seed->subject/patient resampling.

    The same sampled groups are retained across both models and every severity.
    HAR resamples subjects with all their windows; PhysioNet resamples patients
    within outcome strata so every AUROC replicate remains defined.
    """

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    parsed_comparisons = _parse_comparisons(comparisons)
    curves: dict[tuple[str, int, str, str], dict[float, PredictionBundle]] = {}
    for record in records:
        dataset = str(record["dataset"])
        seed = int(record["seed"])
        model = str(record["model"])
        kind = str(record["condition_kind"])
        severity = float(record["severity"])
        key = (dataset, seed, model, kind)
        curve = curves.setdefault(key, {})
        if severity in curve:
            raise ValueError(f"duplicate prediction condition: {key}/{severity}")
        curve[severity] = _load_prediction(
            record["predictions"], record.get("predictions_sha256")
        )
    datasets = sorted({key[0] for key in curves})
    kinds = sorted({key[3] for key in curves})
    results: list[dict[str, Any]] = []
    master = np.random.SeedSequence(int(analysis_seed))
    family_specs = [(dataset, kind) for dataset in datasets for kind in kinds]
    children = master.spawn(len(family_specs) * len(parsed_comparisons))
    child_index = 0
    for dataset, kind in family_specs:
        family_results: list[dict[str, Any]] = []
        for first, second in parsed_comparisons:
            seeds_first = {
                key[1] for key in curves if key[0] == dataset and key[2] == first and key[3] == kind
            }
            seeds_second = {
                key[1] for key in curves if key[0] == dataset and key[2] == second and key[3] == kind
            }
            if not seeds_first or seeds_first != seeds_second:
                raise ValueError(
                    f"paired seed set mismatch for {dataset}/{kind}/{first}-{second}"
                )
            seeds = sorted(seeds_first)
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                first_curve = curves[(dataset, seed, first, kind)]
                second_curve = curves[(dataset, seed, second, kind)]
                if sorted(first_curve) != sorted(second_curve):
                    raise ValueError("paired severity grids differ")
                reference = next(iter(first_curve.values()))
                for bundle in list(first_curve.values()) + list(second_curve.values()):
                    _validate_aligned(reference, bundle)
                indices = np.arange(len(reference.labels), dtype=np.int64)
                first_summary = _curve_summary(dataset, first_curve, indices)
                second_summary = _curve_summary(dataset, second_curve, indices)
                first_auc = first_summary["area_under_robustness_curve"]
                second_auc = second_summary["area_under_robustness_curve"]
                per_seed.append(
                    {
                        "seed": seed,
                        "first_robustness_auc": first_auc,
                        "second_robustness_auc": second_auc,
                        "first_nominal_metric": first_summary["nominal_metric"],
                        "second_nominal_metric": second_summary["nominal_metric"],
                        "first_degradation_from_nominal": first_summary[
                            "degradation_from_nominal"
                        ],
                        "second_degradation_from_nominal": second_summary[
                            "degradation_from_nominal"
                        ],
                        "contrast_first_minus_second": first_auc - second_auc,
                    }
                )
            estimate = float(
                np.mean([value["contrast_first_minus_second"] for value in per_seed])
            )
            rng = np.random.default_rng(children[child_index])
            child_index += 1
            replicates = np.empty(bootstrap_replicates, dtype=np.float64)
            seed_values = np.asarray(seeds, dtype=np.int64)
            for replicate_index in range(bootstrap_replicates):
                sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
                differences: list[float] = []
                for sampled_seed in sampled_seeds:
                    first_curve = curves[(dataset, int(sampled_seed), first, kind)]
                    second_curve = curves[(dataset, int(sampled_seed), second, kind)]
                    reference = next(iter(first_curve.values()))
                    indices = _sample_group_indices(reference, dataset=dataset, rng=rng)
                    differences.append(
                        _curve_summary(dataset, first_curve, indices)[
                            "area_under_robustness_curve"
                        ]
                        - _curve_summary(dataset, second_curve, indices)[
                            "area_under_robustness_curve"
                        ]
                    )
                replicates[replicate_index] = float(np.mean(differences))
            lower, upper = np.percentile(replicates, [2.5, 97.5]).tolist()
            lower_tail = (int(np.sum(replicates <= 0.0)) + 1) / (bootstrap_replicates + 1)
            upper_tail = (int(np.sum(replicates >= 0.0)) + 1) / (bootstrap_replicates + 1)
            raw_p = min(1.0, 2.0 * min(lower_tail, upper_tail))
            family_results.append(
                {
                    "dataset": dataset,
                    "corruption_family": kind,
                    "first_model": first,
                    "second_model": second,
                    "estimand": "mean_over_training_seeds_of_trapezoidal_robustness_auc_first_minus_second",
                    "favourable_direction": "positive_favours_first_model",
                    "estimate": estimate,
                    "confidence_interval_95_percentile": [float(lower), float(upper)],
                    "raw_two_sided_bootstrap_p": float(raw_p),
                    "bootstrap_replicates": int(bootstrap_replicates),
                    "training_seed_count": len(seeds),
                    "per_seed": per_seed,
                }
            )
        adjusted = holm_adjust(
            [value["raw_two_sided_bootstrap_p"] for value in family_results]
        )
        for value, adjusted_p in zip(family_results, adjusted, strict=True):
            value["holm_adjusted_p"] = float(adjusted_p)
            value["holm_family_size"] = len(family_results)
            value["holm_family"] = f"{dataset}:{kind}"
        results.extend(family_results)
    payload = {
        "schema_version": 1,
        "analysis_seed": int(analysis_seed),
        "bootstrap_replicates": int(bootstrap_replicates),
        "confidence_interval": "paired hierarchical percentile 95%",
        "hierarchy": "training_seed_then_subject" if datasets == ["har"] else "training_seed_then_group",
        "pairing": "same groups retained across models and severities",
        "har_resampling_unit": "subject_all_windows",
        "physionet_resampling_unit": "patient_stratified_by_outcome",
        "multiplicity": "Holm step-down within each dataset x corruption-family over four preregistered contrasts",
        "comparisons": [list(value) for value in parsed_comparisons],
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    return payload
