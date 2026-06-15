import numpy as np

from config import DEFAULT_ENV_CONFIG, DEFAULT_VEHICLE_PARAMS
from env.articulated_action_mask import (
    ArticulatedActionMask,
    default_action_mask_path,
    generate_sweep_tables,
    save_sweep_tables,
)
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


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


def test_filter_and_clip_action_ratios_and_articulation_mapping(synthetic_action_mask):
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

    phi = p.phi_max - 0.01
    expected_lower = -p.phi_dot_max
    expected_upper = 0.01 / p.dt
    for normalized, expected in (
        (-1.0, expected_lower),
        (0.0, 0.5 * (expected_lower + expected_upper)),
        (1.0, expected_upper),
    ):
        mapped = synthetic_action_mask.filter_and_clip_action(
            [0.0, normalized],
            full,
            phi,
        )
        assert np.isclose(mapped.decoded_action[1], expected)
        assert np.isclose(mapped.executed_action[1], expected)
        assert phi + mapped.executed_action[1] * p.dt <= p.phi_max + 1e-7


def test_phi_dot_mapping_preserves_full_symmetric_range_away_from_limit(
    synthetic_action_mask,
):
    p = DEFAULT_VEHICLE_PARAMS
    full = np.ones((2, 11), dtype=np.float32)
    for normalized in (-1.0, 0.0, 1.0):
        decoded = synthetic_action_mask.filter_and_clip_action(
            [0.0, normalized],
            full,
            0.0,
        )
        assert np.isclose(
            decoded.executed_action[1],
            normalized * p.phi_dot_max,
        )


def test_decode_uses_mapped_phi_dot_for_action_mask_lookup(synthetic_action_mask):
    p = DEFAULT_VEHICLE_PARAMS
    phi = p.phi_max - 0.01
    expected_phi_dot = 0.5 * (-p.phi_dot_max + 0.01 / p.dt)
    increasing_ratio = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    mask = np.stack([increasing_ratio, increasing_ratio])

    decoded = synthetic_action_mask.decode_safe_speed_and_cost(
        [1.0, 0.0],
        mask,
        phi,
        dt=p.dt,
        prev_motion_gear=None,
        config=DEFAULT_ENV_CONFIG,
    )

    expected_ratio = np.interp(
        expected_phi_dot,
        synthetic_action_mask.phi_dot_bins,
        increasing_ratio,
    )
    assert np.isclose(decoded["phi_dot_exec"], expected_phi_dot)
    assert np.isclose(decoded["r_raw"], expected_ratio)
    assert np.isclose(
        decoded["v_exec"],
        expected_ratio * p.parking_v_forward_max,
    )


def test_mapped_phi_dot_respects_articulated_vehicle_dynamics(synthetic_action_mask):
    p = DEFAULT_VEHICLE_PARAMS
    model = ArticulatedVehicleModel(p)
    full = np.ones((2, 11), dtype=np.float32)

    for phi in (-p.phi_max, -0.2, 0.0, 0.2, p.phi_max):
        for normalized in (-1.0, -0.25, 0.0, 0.75, 1.0):
            decoded = synthetic_action_mask.filter_and_clip_action(
                [0.0, normalized],
                full,
                phi,
            )
            state = ArticulatedState(
                x_front=0.0,
                y_front=0.0,
                theta_front=phi,
                theta_rear=0.0,
            )
            next_state = model.step(state, decoded.executed_action)
            expected_phi = phi + float(decoded.executed_action[1]) * p.dt

            assert np.isclose(next_state.phi, expected_phi, atol=1e-6)
            assert abs(next_state.phi) <= p.phi_max + 1e-7


def test_invalid_articulation_maps_to_fastest_recovery_rate(synthetic_action_mask):
    p = DEFAULT_VEHICLE_PARAMS
    full = np.ones((2, 11), dtype=np.float32)

    for phi, expected in (
        (-np.pi, p.phi_dot_max),
        (np.pi, -p.phi_dot_max),
    ):
        for normalized in (-1.0, 0.0, 1.0):
            decoded = synthetic_action_mask.filter_and_clip_action(
                [0.0, normalized],
                full,
                phi,
            )
            assert np.isclose(decoded.executed_action[1], expected)
