import json

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS
from env.articulated_action_mask import (
    ArticulatedActionMask,
    default_action_mask_path,
    generate_sweep_tables,
    save_sweep_tables,
)


def test_action_mask_npz_loads_required_arrays():
    mask = ArticulatedActionMask.load(default_action_mask_path())
    assert mask.sweep_table_front.shape == (13, 2, 11, 8, 54)
    assert mask.sweep_table_rear.shape == (13, 2, 11, 8, 54)
    assert mask.feature_dim == 22
    assert mask.metadata["online_algorithm"] == "lidar_vs_precomputed_sweep_matrix_compare"


def test_generated_npz_round_trip(tmp_path):
    tables = generate_sweep_tables(trace_samples=1)
    path = save_sweep_tables(str(tmp_path / "mask.npz"), tables)
    loaded = ArticulatedActionMask.load(path)
    assert loaded.sweep_table_front.shape == (13, 2, 11, 8, 54)


def test_online_mask_is_matrix_comparison(synthetic_action_mask):
    clear = np.full(54, 2.0, dtype=np.float32)
    blocked = np.full(54, 0.5, dtype=np.float32)
    clear_mask = synthetic_action_mask.compute_mask(0.0, clear, clear)
    blocked_mask = synthetic_action_mask.compute_mask(0.0, blocked, clear)
    assert np.allclose(clear_mask, 1.0)
    assert np.allclose(blocked_mask, 0.0)


def test_filter_and_clip_action_ratios_and_articulation_limit(synthetic_action_mask):
    p = DEFAULT_VEHICLE_PARAMS
    full = np.ones((2, 11), dtype=np.float32)
    half = np.full((2, 11), 0.5, dtype=np.float32)
    zero = np.zeros((2, 11), dtype=np.float32)

    unchanged = synthetic_action_mask.filter_and_clip_action([0.5, 0.0], full, 0.0)
    assert np.isclose(unchanged.executed_action[0], 0.5 * p.parking_v_forward_max)
    assert unchanged.speed_clipped is False

    clipped = synthetic_action_mask.filter_and_clip_action([1.0, 0.0], half, 0.0)
    assert np.isclose(clipped.executed_action[0], 0.5 * p.parking_v_forward_max)
    assert clipped.speed_clipped is True

    invalid = synthetic_action_mask.filter_and_clip_action([1.0, 0.0], zero, 0.0)
    assert invalid.executed_action[0] == 0.0
    assert invalid.invalid is True

    limited = synthetic_action_mask.filter_and_clip_action(
        [0.0, 1.0],
        full,
        p.phi_max - 0.01,
    )
    assert p.phi_max - 0.01 + limited.executed_action[1] * p.dt <= p.phi_max + 1e-7
