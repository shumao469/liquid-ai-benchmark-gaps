"""Static, valid-length compute accounting for offline sequence models.

The functions in this module never execute a model or inspect predictions.
They describe the registered numerical work implied by ``dt`` and ``lengths``
and distinguish it from padded work performed by the current dense forward.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import LSTMEncoder, LiquidEncoder, ODERNNEncoder, OfflineSequenceClassifier


def _integer_summary(values: np.ndarray) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError("compute-accounting summaries require at least one sample")
    return {
        "sample_count": int(array.size),
        "total": int(array.sum()),
        "mean_per_sample": float(array.mean()),
        "median_per_sample": float(np.median(array)),
        "minimum_per_sample": int(array.min()),
        "maximum_per_sample": int(array.max()),
    }


def sequence_compute_accounting(
    model: OfflineSequenceClassifier,
    dt: np.ndarray,
    lengths: np.ndarray | None,
) -> dict[str, Any]:
    """Return solver/update counts without running the model.

    ``valid_*`` fields exclude padded time points according to ``lengths``.
    ``padded_*`` fields describe the dense sequence tensor consumed by the
    current forward implementation.  ODE-RNN valid NFE uses four vector-field
    evaluations per classical RK4 substep.  LTC valid unfolds use the frozen
    ``ode_unfolds`` setting per valid recurrent update.
    """

    elapsed = np.asarray(dt, dtype=np.float64)
    if elapsed.ndim == 3 and elapsed.shape[-1] == 1:
        elapsed = elapsed[..., 0]
    if elapsed.ndim != 2 or elapsed.shape[0] == 0 or elapsed.shape[1] == 0:
        raise ValueError("dt must have non-empty shape [sample,time]")
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError("dt must be finite and strictly positive")
    sample_count, padded_length = elapsed.shape
    if lengths is None:
        valid_lengths = np.full(sample_count, padded_length, dtype=np.int64)
    else:
        valid_lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if valid_lengths.shape != (sample_count,):
        raise ValueError("lengths must have one entry per sample")
    if np.any(valid_lengths <= 0) or np.any(valid_lengths > padded_length):
        raise ValueError("lengths must lie within the padded sequence")

    valid_mask = np.arange(padded_length)[None, :] < valid_lengths[:, None]
    valid_updates = valid_mask.sum(axis=1, dtype=np.int64)
    padded_updates = np.full(sample_count, padded_length, dtype=np.int64)
    encoder = model.encoder
    report: dict[str, Any] = {
        "schema_version": 1,
        "accounting_scope": "static_from_dt_and_valid_lengths_no_model_execution",
        "model_family": (
            "ode-rnn"
            if isinstance(encoder, ODERNNEncoder)
            else encoder.kind
            if isinstance(encoder, LiquidEncoder)
            else "lstm"
            if isinstance(encoder, LSTMEncoder)
            else type(encoder).__name__
        ),
        "padded_sequence_length": int(padded_length),
        "valid_updates": _integer_summary(valid_updates),
        "padded_updates_executed_by_current_dense_forward": _integer_summary(
            padded_updates
        ),
        "padding_updates_total": int(padded_updates.sum() - valid_updates.sum()),
        "valid_length_accounting": True,
        "current_forward_masks_loss_selection_but_does_not_skip_padded_recurrence": True,
    }

    if isinstance(encoder, ODERNNEncoder):
        substeps = np.ceil(elapsed / float(encoder.max_step)).astype(np.int64)
        valid_nfe_per_sample = (4 * substeps * valid_mask).sum(
            axis=1, dtype=np.int64
        )
        padded_individual_nfe = (4 * substeps).sum(axis=1, dtype=np.int64)
        # The implementation vectorizes a batch using the largest dt at each
        # time point.  These two fields describe a hypothetical single forward
        # over the supplied arrays; actual minibatch grouping is logged
        # separately by the training pipeline and is not inferred here.
        batched_substeps = np.ceil(
            elapsed.max(axis=0) / float(encoder.max_step)
        ).astype(np.int64)
        report["ode_nfe"] = {
            "solver": "fixed_step_classical_rk4",
            "max_step": float(encoder.max_step),
            "vector_field_evaluations_per_substep": 4,
            "valid_sample_equivalent": _integer_summary(valid_nfe_per_sample),
            "padded_individual_dt_sample_equivalent": _integer_summary(
                padded_individual_nfe
            ),
            "single_supplied_batch_vector_field_call_count": int(
                4 * batched_substeps.sum()
            ),
            "single_supplied_batch_padded_sample_equivalent": int(
                sample_count * 4 * batched_substeps.sum()
            ),
            "batch_vectorization_uses_max_dt_per_timepoint": True,
            "single_batch_fields_are_not_claimed_as_actual_training_minibatch_counts": True,
        }
    elif isinstance(encoder, LiquidEncoder) and encoder.kind == "ltc":
        unfolds = int(encoder.ode_unfolds)
        report["ltc_internal_unfolds"] = {
            "unfolds_per_recurrent_update": unfolds,
            "valid_sample_equivalent": _integer_summary(valid_updates * unfolds),
            "padded_sample_equivalent": _integer_summary(padded_updates * unfolds),
            "solver_evaluations_not_equated_with_unfolds": True,
        }
    elif isinstance(encoder, LiquidEncoder) and encoder.kind == "cfc":
        report["solver_accounting"] = {
            "solver_free_closed_form_update": True,
            "valid_solver_evaluations_total": 0,
            "valid_recurrent_updates": _integer_summary(valid_updates),
        }
    elif isinstance(encoder, LSTMEncoder):
        report["solver_accounting"] = {
            "continuous_time_solver_not_applicable": True,
            "valid_solver_evaluations_total": 0,
            "valid_recurrent_updates": _integer_summary(valid_updates),
        }
    else:  # pragma: no cover - defensive for future registered encoders
        raise TypeError(f"Unsupported offline encoder for accounting: {type(encoder)!r}")
    return report
