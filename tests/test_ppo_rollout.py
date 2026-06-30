from dataclasses import replace

import numpy as np
import torch

from config import DEFAULT_ENV_CONFIG
from config import DEFAULT_PPO_CONFIG
from env.local_parking_env import LocalParkingEnv
from model.continuous_ppo import ContinuousPPOAgent, RolloutBuffer


def test_ppo_rollout_shapes_and_raw_action_log_prob(synthetic_action_mask):
    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, curriculum_stage=1),
        action_mask=synthetic_action_mask,
        seed=9,
    )
    obs, _ = env.reset()
    agent = ContinuousPPOAgent(device="cpu")
    raw_action, pre_tanh_action, log_prob, value = agent.act_with_pre_tanh(obs)
    next_obs, reward, terminated, truncated, info = env.step(raw_action)

    assert obs.shape == (149,)
    assert raw_action.shape == (2,)
    assert info["executed_action"].shape == (2,)
    assert next_obs.shape == (149,)

    buffer = RolloutBuffer()
    buffer.add(
        obs,
        raw_action,
        info["executed_action"],
        log_prob,
        reward,
        terminated or truncated,
        value,
        pre_tanh_action=pre_tanh_action,
        task_family="head_in",
    )
    assert buffer.log_prob_action_source == "pre_tanh_action_and_raw_action"
    assert np.allclose(buffer.pre_tanh_actions[0], pre_tanh_action)
    assert np.allclose(buffer.raw_actions[0], raw_action)
    assert np.allclose(buffer.executed_actions[0], info["executed_action"])
    assert buffer.task_families == ["head_in"]

    obs_t = torch.as_tensor(obs).unsqueeze(0)
    pre_tanh_t = torch.as_tensor(pre_tanh_action).unsqueeze(0)
    raw_t = torch.as_tensor(raw_action).unsqueeze(0)
    with torch.no_grad():
        recomputed, _, _ = agent.network.evaluate_actions(
            obs_t,
            pre_tanh_t,
            raw_t,
        )
    assert np.isclose(log_prob, float(recomputed.item()), atol=1e-4)


def test_env_step_lifts_degenerate_mask_before_executing_action(synthetic_action_mask):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            enable_mask_floor_fallback=True,
            enable_dwa_recovery=False,
            dwa_override_policy_action=False,
        ),
        action_mask=synthetic_action_mask,
        seed=3,
    )
    env.reset()
    zero_mask = np.zeros((2, 11), dtype=np.float32)
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(zero_mask)
    seen = {}
    original_step = env.vehicle_model.step

    def recording_step(state, action, dt=None):
        seen["action"] = np.asarray(action).copy()
        return original_step(state, action, dt=dt)

    env.vehicle_model.step = recording_step
    _, _, _, _, info = env.step(np.asarray([1.0, 0.25], dtype=np.float32))
    assert info["raw_action"][0] == 1.0
    assert info["degenerate_mask"] is True
    assert info["mask_floor_applied"] is True
    assert info["forced_stop"] is False
    assert np.isclose(info["mask_max_before_floor"], 0.0)
    assert np.isclose(info["raw_safe_ratio"], DEFAULT_ENV_CONFIG.mask_floor_value)
    expected_speed = (
        DEFAULT_ENV_CONFIG.mask_floor_value
        * synthetic_action_mask.vehicle_params.parking_v_forward_max
    )
    assert np.isclose(info["executed_action"][0], expected_speed)
    assert np.isclose(seen["action"][0], expected_speed)


def test_env_step_can_disable_mask_floor_fallback(synthetic_action_mask):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            enable_mask_floor_fallback=False,
            enable_dwa_recovery=False,
            dwa_override_policy_action=False,
        ),
        action_mask=synthetic_action_mask,
        seed=3,
    )
    env.reset()
    previous_state = env.state
    env.prev_motion_gear = 0
    env.prev_gear_in_obs = 1.0
    zero_mask = np.zeros((2, 11), dtype=np.float32)
    env.current_mask, env.current_mask_floor_info = env._mask_floor_state(zero_mask)

    _, _, _, _, info = env.step(np.asarray([-1.0, 1.0], dtype=np.float32))

    assert info["degenerate_mask"] is True
    assert info["mask_floor_applied"] is False
    assert info["forced_stop"] is True
    assert np.allclose(info["executed_action"], [0.0, 0.0])
    assert np.isclose(env.state.x_front, previous_state.x_front)
    assert np.isclose(env.state.y_front, previous_state.y_front)
    assert np.isclose(env.state.theta_front, previous_state.theta_front)
    assert np.isclose(env.state.theta_rear, previous_state.theta_rear)
    assert np.isclose(env.state.phi, previous_state.phi)
    assert env.prev_motion_gear == 0
    assert env.prev_gear_in_obs == 1.0


def test_env_can_reset_from_viable_hard_case_replay_state(synthetic_action_mask):
    config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=3,
        hard_case_replay_attempts=1,
        hard_case_replay_xy_std=0.0,
        hard_case_replay_heading_std_deg=0.0,
        hard_case_replay_phi_std_deg=0.0,
    )
    env = LocalParkingEnv(
        config=config,
        action_mask=synthetic_action_mask,
        seed=31,
    )
    _, reset_info = env.reset(seed=31)
    replay_case = {
        "scene": env.scene,
        "slot": env.slot,
        "state": env.state,
        "stage": 3,
        "episode": 4,
        "scene_seed": reset_info["scene_seed"],
        "scenario_type": reset_info["scenario_type"],
        "task_family": reset_info["task_family"],
        "failure_type": "timeout",
    }

    obs, info = env.reset(replay_case=replay_case)

    assert obs.shape == (149,)
    assert info["hard_case_replay_attempted"] is True
    assert info["hard_case_replay_used"] is True
    assert info["hard_case_replay_source_episode"] == 4
    assert info["scenario_type"].endswith("_hard_case_replay")


def test_actor_uses_bounded_global_log_std():
    agent = ContinuousPPOAgent(device="cpu")
    observations = torch.randn(4, 149)
    distribution = agent.network.distribution(observations)

    assert tuple(agent.network.actor_log_std.shape) == (2,)
    assert np.allclose(
        agent.global_log_std(),
        np.asarray([-0.7, -0.7], dtype=np.float32),
    )
    assert torch.allclose(
        distribution.stddev[0],
        distribution.stddev[1],
    )

    with torch.no_grad():
        agent.network.actor_log_std.fill_(1.0)
    agent.network.project_log_std()
    assert np.allclose(
        agent.global_log_std(),
        np.asarray([-0.3, -0.3], dtype=np.float32),
    )


def test_ppo_early_stops_after_epoch_when_target_kl_is_exceeded():
    config = replace(
        DEFAULT_PPO_CONFIG,
        actor_lr=5e-2,
        critic_lr=3e-4,
        batch_size=32,
        ppo_epochs=4,
        target_kl=1e-8,
    )
    agent = ContinuousPPOAgent(config=config, device="cpu")
    buffer = RolloutBuffer()
    rng = np.random.default_rng(12)
    for index in range(32):
        observation = rng.normal(size=149).astype(np.float32)
        raw_action, pre_tanh_action, log_prob, value = agent.act_with_pre_tanh(
            observation
        )
        buffer.add(
            observation=observation,
            raw_action=raw_action,
            executed_action=np.zeros(2, dtype=np.float32),
            log_prob=log_prob,
            reward=float(index % 5),
            done=index == 31,
            value=value,
            pre_tanh_action=pre_tanh_action,
            task_family="head_in",
        )

    stats = agent.update(
        buffer,
        last_observation=np.zeros(149, dtype=np.float32),
        last_done=True,
    )

    assert stats["kl_early_stopped"] is True
    assert stats["ppo_epochs_completed"] == 1
    assert stats["approx_kl_max"] > config.target_kl
    assert agent._policy_loss_weight("head_in") == 1.0


def test_ppo_update_reports_dwa_bc_and_policy_weight():
    config = replace(
        DEFAULT_PPO_CONFIG,
        batch_size=8,
        ppo_epochs=1,
        target_kl=10.0,
    )
    agent = ContinuousPPOAgent(config=config, device="cpu")
    buffer = RolloutBuffer()
    rng = np.random.default_rng(23)
    teacher_count = 0
    for index in range(8):
        observation = rng.normal(size=149).astype(np.float32)
        raw_action, pre_tanh_action, log_prob, value = agent.act_with_pre_tanh(
            observation
        )
        teacher_valid = bool(index % 2 == 0)
        if teacher_valid:
            teacher_count += 1
        buffer.add(
            observation=observation,
            raw_action=raw_action,
            executed_action=np.zeros(2, dtype=np.float32),
            log_prob=log_prob,
            reward=float(index % 3),
            done=index == 7,
            value=value,
            pre_tanh_action=pre_tanh_action,
            task_family="head_in",
            dwa_raw_action=np.asarray([1.0, 0.0], dtype=np.float32),
            dwa_teacher_action_valid=teacher_valid,
            dwa_used=teacher_valid,
            dwa_policy_loss_weight=0.0 if teacher_valid else 1.0,
            recovery_mask_applied=teacher_valid,
            recovery_mask_nonzero_count=1 if teacher_valid else 0,
            recovery_mask_max=0.1 if teacher_valid else 0.0,
        )

    stats = agent.update(
        buffer,
        last_observation=np.zeros(149, dtype=np.float32),
        last_done=True,
        dwa_bc_coef=0.5,
    )

    assert np.isclose(stats["dwa_teacher_fraction"], teacher_count / len(buffer))
    assert stats["dwa_override_policy_weight_mean"] < 1.0
    assert stats["dwa_bc_coef"] == 0.5
    assert stats["dwa_bc_loss"] >= 0.0


def test_checkpoint_preserves_log_std_bounds_and_ppo_config(tmp_path):
    config = replace(
        DEFAULT_PPO_CONFIG,
        log_std_max=0.0,
    )
    agent = ContinuousPPOAgent(config=config, device="cpu")
    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = ContinuousPPOAgent(device="cpu")
    restored.network.load_state_dict(payload["network"])

    assert payload["ppo_config"]["log_std_max"] == 0.0
    assert float(restored.network.actor_log_std_max.item()) == 0.0
