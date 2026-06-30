from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box

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


def test_dwa_unlock_all_zero_outputs_sparse_recovery_mask(synthetic_action_mask):
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
    assert abs(float(result.raw_action[0])) == 1.0
    assert 0.0 < abs(float(result.executed_action_preview[0])) <= (
        cfg.dwa_recovery_max_speed_ratio
        * synthetic_action_mask.vehicle_params.parking_v_forward_max
    )
    assert result.recovery_mask.shape == (2, 11)
    assert np.count_nonzero(result.recovery_mask) >= 1
    assert np.max(result.recovery_mask) <= cfg.dwa_recovery_max_speed_ratio + 1e-6
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


def test_dwa_unlock_accepts_prefix_before_later_invalid(synthetic_action_mask):
    cfg = replace(DEFAULT_ENV_CONFIG, dwa_horizon_steps=3, dwa_unlock_safe_ratio=0.08)
    scene = SimpleNamespace(
        world_bounds=(-10.0, -10.0, 10.0, 10.0),
        prepared_obstacles=_EmptyPrepared(),
    )
    slot = SimpleNamespace()
    initial_state = SimpleNamespace(phi=0.0, sim_step=0)

    class PrefixVehicleModel:
        params = SimpleNamespace(
            phi_max=10.0,
            dt=1.0,
            parking_v_forward_max=1.0,
            parking_v_reverse_max=1.0,
        )

        def step(self, state, action):
            return SimpleNamespace(
                phi=float(state.phi) + float(action[1]),
                sim_step=int(getattr(state, "sim_step", 0)) + 1,
            )

        def body_boxes(self, state):
            if int(getattr(state, "sim_step", 0)) >= 2:
                return box(20.0, 0.0, 21.0, 1.0), box(20.0, 0.0, 21.0, 1.0)
            return box(0.0, 0.0, 1.0, 1.0), box(0.0, 0.0, 1.0, 1.0)

    def compute_mask(phi, front_lidar, rear_lidar):
        del front_lidar, rear_lidar
        if abs(float(phi)) > 0.0:
            return np.full((2, 11), 0.10, dtype=np.float32)
        return np.zeros((2, 11), dtype=np.float32)

    synthetic_action_mask.compute_mask = compute_mask
    controller = DWARecoveryController(cfg)
    result = controller.run_unlock(
        initial_state,
        slot,
        scene,
        PrefixVehicleModel(),
        _FakeLidar(),
        synthetic_action_mask,
        np.zeros((2, 11), dtype=np.float32),
        None,
        None,
        None,
        cfg,
    )

    assert result.used is True
    assert result.unlock_success is True
    assert result.deadlock is False
    assert result.unlock_step == 1
    assert result.valid_candidate_count >= 1
    assert result.final_max_safe_ratio >= cfg.dwa_unlock_safe_ratio
    assert np.count_nonzero(result.recovery_mask) >= 1


def test_env_default_config_keeps_dwa_disabled(synthetic_action_mask):
    cfg = replace(
        DEFAULT_ENV_CONFIG,
        enable_dwa_recovery=False,
        dwa_override_policy_action=False,
    )
    env = LocalParkingEnv(
        config=cfg,
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
            raw_action=np.asarray([1.0, 1.0], dtype=np.float32),
            executed_action_preview=np.asarray([0.1, 0.5], dtype=np.float32),
            recovery_mask=np.full((2, 11), 0.1, dtype=np.float32),
            candidate_count=11,
            valid_candidate_count=1,
            final_max_safe_ratio=0.1,
            teacher_action_valid=True,
            unlock_success=True,
        )
    )

    _, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert info["dwa_triggered"] is True
    assert info["dwa_used"] is True
    assert info["dwa_override_policy_action"] is False
    assert info["recovery_mask_applied"] is True
    assert np.allclose(info["policy_raw_action"], [1.0, 0.0])
    assert np.allclose(info["raw_action"], [1.0, 0.0])
    assert info["executed_action"][0] > 0.0


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
            raw_action=np.asarray([1.0, 1.0], dtype=np.float32),
            executed_action_preview=np.asarray([0.1, 0.5], dtype=np.float32),
            recovery_mask=np.full((2, 11), 0.1, dtype=np.float32),
            candidate_count=11,
            valid_candidate_count=1,
            final_max_safe_ratio=0.1,
            teacher_action_valid=True,
            unlock_success=True,
        )
    )

    _, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert info["dwa_triggered"] is True
    assert info["dwa_override_policy_action"] is True
    assert np.allclose(info["policy_raw_action"], [1.0, 0.0])
    assert np.allclose(info["raw_action"], [1.0, 1.0])
    assert info["executed_action"][0] > 0.0
    assert abs(float(info["executed_action"][1])) > 0.0
    assert env.prev_motion_gear == FORWARD_GEAR
    assert env.prev_gear_in_obs == 1.0
    assert info["dwa_policy_loss_weight"] == cfg.dwa_override_policy_loss_weight


def test_env_recovery_mask_only_never_overrides_policy_action(synthetic_action_mask):
    cfg = replace(
        DEFAULT_ENV_CONFIG,
        enable_dwa_recovery=True,
        dwa_recovery_mode="recovery_mask_only",
        dwa_override_policy_action=True,
    )
    env = LocalParkingEnv(config=cfg, action_mask=synthetic_action_mask, seed=21)
    env.reset()
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(
        np.zeros((2, 11), dtype=np.float32),
        allow_floor=False,
    )
    env.current_normal_mask = np.zeros((2, 11), dtype=np.float32)
    recovery_mask = np.zeros((2, 11), dtype=np.float32)
    recovery_mask[FORWARD_GEAR, :] = 0.1
    env.dwa_recovery = _FakeDWA(
        DWAResult(
            used=True,
            mode="unlock",
            reason="test",
            raw_action=np.asarray([1.0, 0.0], dtype=np.float32),
            executed_action_preview=np.asarray([0.1, 0.0], dtype=np.float32),
            recovery_mask=recovery_mask,
            candidate_count=11,
            valid_candidate_count=1,
            final_max_safe_ratio=0.1,
            teacher_action_valid=True,
            unlock_success=True,
        )
    )

    _, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert info["dwa_triggered"] is True
    assert info["dwa_teacher_action_valid"] is True
    assert info["dwa_override_policy_action"] is False
    assert info["recovery_mask_applied"] is True
    assert np.allclose(info["policy_raw_action"], [1.0, 0.0])
    assert np.allclose(info["raw_action"], [1.0, 0.0])
    assert info["executed_action"][0] > 0.0
    assert info["dwa_policy_loss_weight"] == 1.0


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
