from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from config import DEFAULT_ENV_CONFIG, DEFAULT_VEHICLE_PARAMS
from env.articulated_action_mask import FORWARD_GEAR
from env.dwa_recovery import DWARecoveryController, DWAResult
from env.geometry import DirectedParkingSlot
from env.local_parking_env import LocalParkingEnv
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


class _EmptyPrepared:
    def intersects(self, _polygon):
        return False


class _FakeLidar:
    def observe(self, state, vehicle_model, scene, normalize=False):
        del state, vehicle_model, scene, normalize
        beams = DEFAULT_VEHICLE_PARAMS.lidar_beams
        return (
            np.full(beams, 10.0, dtype=np.float32),
            np.full(beams, 10.0, dtype=np.float32),
        )


class _FakeDWA:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, mode, *args, **kwargs):
        self.calls.append((mode, kwargs.get("reason", "")))
        return self.result


def _simple_dwa_context(synthetic_action_mask):
    p = DEFAULT_VEHICLE_PARAMS
    vehicle_model = ArticulatedVehicleModel(p)
    scene = SimpleNamespace(
        world_bounds=(-100.0, -100.0, 100.0, 100.0),
        prepared_obstacles=_EmptyPrepared(),
    )
    slot = DirectedParkingSlot(
        x_goal=0.0,
        y_goal=0.0,
        theta_goal=0.0,
        front_body_length=p.front_body_length,
        front_body_width=p.front_body_width,
    )
    state = ArticulatedState(
        x_front=-4.0,
        y_front=0.0,
        theta_front=0.0,
        theta_rear=0.0,
    )
    return state, slot, scene, vehicle_model, _FakeLidar(), synthetic_action_mask


def test_dwa_unlock_all_zero_outputs_phi_only_action(synthetic_action_mask):
    cfg = replace(
        DEFAULT_ENV_CONFIG,
        dwa_horizon_steps=2,
        dwa_unlock_safe_ratio=0.08,
    )
    state, slot, scene, vehicle_model, lidar, action_mask = _simple_dwa_context(
        synthetic_action_mask
    )

    def compute_mask(phi, front_lidar, rear_lidar):
        del front_lidar, rear_lidar
        if abs(float(phi)) >= 0.20:
            return np.full((2, 11), 0.10, dtype=np.float32)
        return np.zeros((2, 11), dtype=np.float32)

    action_mask.compute_mask = compute_mask
    controller = DWARecoveryController(cfg)
    result = controller.run_unlock(
        state,
        slot,
        scene,
        vehicle_model,
        lidar,
        action_mask,
        np.zeros((2, 11), dtype=np.float32),
        None,
        None,
        None,
        cfg,
    )

    assert result.used is True
    assert result.mode == "unlock"
    assert result.unlock_success is True
    assert result.raw_action.shape == (2,)
    assert result.raw_action[0] == 0.0
    assert result.executed_action_preview[0] == 0.0
    assert result.valid_candidate_count >= 1
    assert result.final_max_safe_ratio >= cfg.dwa_unlock_safe_ratio


def test_dwa_unlock_deadlocks_when_safe_ratio_never_recovers(synthetic_action_mask):
    cfg = replace(DEFAULT_ENV_CONFIG, dwa_horizon_steps=2, dwa_unlock_safe_ratio=0.08)
    state, slot, scene, vehicle_model, lidar, action_mask = _simple_dwa_context(
        synthetic_action_mask
    )
    action_mask.compute_mask = lambda phi, front, rear: np.zeros((2, 11), dtype=np.float32)
    controller = DWARecoveryController(cfg)

    result = controller.run_unlock(
        state,
        slot,
        scene,
        vehicle_model,
        lidar,
        action_mask,
        np.zeros((2, 11), dtype=np.float32),
        None,
        None,
        None,
        cfg,
    )

    assert result.used is False
    assert result.valid_candidate_count == 0
    assert result.deadlock is True
    assert result.raw_action is None


def test_env_default_config_keeps_dwa_disabled(synthetic_action_mask):
    env = LocalParkingEnv(
        config=DEFAULT_ENV_CONFIG,
        action_mask=synthetic_action_mask,
        seed=13,
    )
    obs, _ = env.reset()
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(
        np.zeros((2, 11), dtype=np.float32)
    )

    next_obs, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert obs.shape == (149,)
    assert next_obs.shape == (149,)
    assert info["dwa_enabled"] is False
    assert info["dwa_triggered"] is False
    assert info["dwa_used"] is False
    assert np.allclose(info["raw_action"], [1.0, 0.0])


def test_env_dwa_diagnostics_do_not_override_when_disabled(synthetic_action_mask):
    cfg = replace(
        DEFAULT_ENV_CONFIG,
        enable_dwa_recovery=True,
        dwa_override_policy_action=False,
    )
    env = LocalParkingEnv(config=cfg, action_mask=synthetic_action_mask, seed=17)
    env.reset()
    env.prev_motion_gear = FORWARD_GEAR
    env.prev_gear_in_obs = 1.0
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(
        np.zeros((2, 11), dtype=np.float32)
    )
    env.dwa_recovery = _FakeDWA(
        DWAResult(
            used=True,
            mode="unlock",
            reason="test",
            raw_action=np.asarray([0.0, 1.0], dtype=np.float32),
            executed_action_preview=np.asarray([0.0, 0.5], dtype=np.float32),
            candidate_count=11,
            valid_candidate_count=1,
            final_max_safe_ratio=0.1,
            unlock_success=True,
        )
    )

    _, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert info["dwa_triggered"] is True
    assert info["dwa_used"] is True
    assert info["dwa_override_policy_action"] is False
    assert np.allclose(info["policy_raw_action"], [1.0, 0.0])
    assert np.allclose(info["raw_action"], [1.0, 0.0])
    assert np.allclose(info["executed_action"], [0.0, 0.0])


def test_env_dwa_override_replaces_raw_action(synthetic_action_mask):
    cfg = replace(
        DEFAULT_ENV_CONFIG,
        enable_dwa_recovery=True,
        dwa_override_policy_action=True,
    )
    env = LocalParkingEnv(config=cfg, action_mask=synthetic_action_mask, seed=19)
    env.reset()
    env.prev_motion_gear = FORWARD_GEAR
    env.prev_gear_in_obs = 1.0
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(
        np.zeros((2, 11), dtype=np.float32)
    )
    env.dwa_recovery = _FakeDWA(
        DWAResult(
            used=True,
            mode="unlock",
            reason="test",
            raw_action=np.asarray([0.0, 1.0], dtype=np.float32),
            executed_action_preview=np.asarray([0.0, 0.5], dtype=np.float32),
            candidate_count=11,
            valid_candidate_count=1,
            final_max_safe_ratio=0.1,
            unlock_success=True,
        )
    )

    _, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert info["dwa_triggered"] is True
    assert info["dwa_override_policy_action"] is True
    assert np.allclose(info["policy_raw_action"], [1.0, 0.0])
    assert np.allclose(info["raw_action"], [0.0, 1.0])
    assert np.isclose(info["executed_action"][0], 0.0)
    assert abs(float(info["executed_action"][1])) > 0.0
    assert env.prev_motion_gear == FORWARD_GEAR
    assert env.prev_gear_in_obs == 1.0


def test_env_dwa_deadlock_can_terminate_before_timeout(synthetic_action_mask):
    cfg = replace(
        DEFAULT_ENV_CONFIG,
        enable_dwa_recovery=True,
        dwa_enable_deadlock_termination=True,
        dwa_deadlock_patience=1,
        max_steps=50,
    )
    env = LocalParkingEnv(config=cfg, action_mask=synthetic_action_mask, seed=23)
    env.reset()
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(
        np.zeros((2, 11), dtype=np.float32)
    )
    env.dwa_recovery = _FakeDWA(
        DWAResult(
            used=False,
            mode="unlock",
            reason="test_deadlock",
            candidate_count=11,
            valid_candidate_count=0,
            deadlock=True,
        )
    )

    _, _, terminated, truncated, info = env.step(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert info["deadlock"] is True
    assert info["failure_type"] == "deadlock"
    assert info["timeout"] is False
