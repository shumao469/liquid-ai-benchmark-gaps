from __future__ import annotations

import numpy as np
import pytest

from liquid_revision.offline.corruption import (
    CorruptionSpec,
    apply_corruption,
    default_corruption_specs,
    recompute_age,
)


def test_physionet_corruption_never_relabels_natural_missingness() -> None:
    values = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    original = np.ones_like(values, dtype=bool)
    original[:, ::2, 1] = False
    values[~original] = 0.0
    dt = np.ones((2, 8), dtype=np.float32)
    result = apply_corruption(
        values, original, dt, CorruptionSpec("mcar", 0.5, seed=123, packetwise=False)
    )
    assert not np.any(result.mask & ~original)
    assert np.all(result.values[~result.mask] == 0)
    assert np.all(result.age >= 0)


def test_har_packetwise_corruption_removes_all_nine_channels_together() -> None:
    values = np.ones((3, 20, 9), dtype=np.float32)
    mask = np.ones_like(values, dtype=bool)
    dt = np.full((3, 20), 0.02, dtype=np.float32)
    result = apply_corruption(
        values, mask, dt, CorruptionSpec("mcar", 0.4, seed=9, packetwise=True)
    )
    per_time = result.mask.sum(axis=-1)
    assert set(np.unique(per_time)).issubset({0, 9})


def test_temporal_block_has_requested_contiguous_length() -> None:
    values = np.ones((4, 10, 2), dtype=np.float32)
    mask = np.ones_like(values, dtype=bool)
    dt = np.ones((4, 10), dtype=np.float32)
    result = apply_corruption(
        values,
        mask,
        dt,
        CorruptionSpec("temporal_block", 0.3, seed=31, packetwise=False),
    )
    for sample in result.mask:
        missing_times = np.flatnonzero(~sample.any(axis=1))
        assert len(missing_times) == 3
        assert np.all(np.diff(missing_times) == 1)


def test_age_accumulates_and_resets_after_observation() -> None:
    mask = np.array([[[True], [False], [False], [True], [False]]])
    dt = np.array([[0.5, 1.0, 2.0, 0.5, 1.5]], dtype=np.float32)
    age = recompute_age(mask, dt)
    np.testing.assert_allclose(age[0, :, 0], [0.0, 1.0, 3.0, 0.0, 1.5])


def test_default_grid_matches_registered_severities() -> None:
    specs = default_corruption_specs(base_seed=5, dataset="har", packetwise=True)
    assert [item.severity for item in specs if item.kind == "mcar"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    assert [item.severity for item in specs if item.kind == "temporal_block"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]


def test_explicit_pilot_grid_is_honoured_exactly() -> None:
    specs = default_corruption_specs(
        base_seed=5,
        dataset="har",
        packetwise=True,
        mcar_levels=[0.0, 0.3],
        temporal_block_levels=[0.0, 0.4],
    )
    assert [(item.kind, item.severity) for item in specs] == [
        ("mcar", 0.0),
        ("mcar", 0.3),
        ("temporal_block", 0.0),
        ("temporal_block", 0.4),
    ]


def test_duplicate_or_out_of_range_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        default_corruption_specs(
            base_seed=5,
            dataset="har",
            packetwise=True,
            mcar_levels=[0.1, 0.1],
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        default_corruption_specs(
            base_seed=5,
            dataset="har",
            packetwise=True,
            temporal_block_levels=[1.1],
        )

