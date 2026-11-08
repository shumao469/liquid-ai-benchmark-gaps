from __future__ import annotations

import numpy as np

from liquid_revision.offline.compute_accounting import sequence_compute_accounting
from liquid_revision.offline.models import build_offline_model


def test_ode_nfe_uses_dt_substeps_and_valid_lengths() -> None:
    model = build_offline_model(
        "ode-rnn", input_dim=3, hidden_dim=4, output_dim=2, ode_max_step=1.0
    )
    dt = np.asarray([[0.5, 1.5, 2.1], [1.0, 0.25, 3.0]], dtype=np.float32)
    report = sequence_compute_accounting(model, dt, np.asarray([2, 1]))
    # sample 1: 4*(ceil(.5)+ceil(1.5)) = 12; sample 2: 4*ceil(1) = 4
    assert report["ode_nfe"]["valid_sample_equivalent"]["total"] == 16
    assert report["ode_nfe"]["valid_sample_equivalent"]["maximum_per_sample"] == 12
    assert report["valid_updates"]["total"] == 3
    assert report["padded_updates_executed_by_current_dense_forward"]["total"] == 6
    assert report["padding_updates_total"] == 3


def test_ltc_unfolds_are_counted_only_over_valid_updates() -> None:
    model = build_offline_model(
        "ltc", input_dim=3, hidden_dim=4, output_dim=2, ltc_ode_unfolds=6
    )
    dt = np.ones((2, 4), dtype=np.float32)
    report = sequence_compute_accounting(model, dt, np.asarray([4, 2]))
    assert report["ltc_internal_unfolds"]["unfolds_per_recurrent_update"] == 6
    assert report["ltc_internal_unfolds"]["valid_sample_equivalent"]["total"] == 36
    assert report["ltc_internal_unfolds"]["padded_sample_equivalent"]["total"] == 48
    assert report["ltc_internal_unfolds"][
        "solver_evaluations_not_equated_with_unfolds"
    ]


def test_physionet_registered_counts_are_four_nfe_and_six_unfolds_per_update() -> None:
    dt = np.ones((3, 48), dtype=np.float32)
    lengths = np.full(3, 48, dtype=np.int64)
    ode = build_offline_model(
        "ode-rnn", input_dim=2, hidden_dim=3, output_dim=1, ode_max_step=1.0
    )
    ltc = build_offline_model(
        "ltc", input_dim=2, hidden_dim=3, output_dim=1, ltc_ode_unfolds=6
    )
    ode_report = sequence_compute_accounting(ode, dt, lengths)
    ltc_report = sequence_compute_accounting(ltc, dt, lengths)
    assert ode_report["ode_nfe"]["valid_sample_equivalent"]["total"] == 3 * 48 * 4
    assert ltc_report["ltc_internal_unfolds"]["valid_sample_equivalent"]["total"] == 3 * 48 * 6
