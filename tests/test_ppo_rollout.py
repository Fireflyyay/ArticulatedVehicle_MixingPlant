from dataclasses import replace

import numpy as np
import torch

from config import DEFAULT_ENV_CONFIG
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
    raw_action, log_prob, value = agent.act(obs)
    next_obs, reward, terminated, truncated, info = env.step(raw_action)

    assert obs.shape == (148,)
    assert raw_action.shape == (2,)
    assert info["executed_action"].shape == (2,)
    assert next_obs.shape == (148,)

    buffer = RolloutBuffer()
    buffer.add(
        obs,
        raw_action,
        info["executed_action"],
        log_prob,
        reward,
        terminated or truncated,
        value,
    )
    assert buffer.log_prob_action_source == "raw_action"
    assert np.allclose(buffer.raw_actions[0], raw_action)
    assert np.allclose(buffer.executed_actions[0], info["executed_action"])

    obs_t = torch.as_tensor(obs).unsqueeze(0)
    raw_t = torch.as_tensor(raw_action).unsqueeze(0)
    with torch.no_grad():
        recomputed, _, _ = agent.network.evaluate_raw_actions(obs_t, raw_t)
    assert np.isclose(log_prob, float(recomputed.item()), atol=1e-4)


def test_env_step_advances_with_executed_action(synthetic_action_mask):
    env = LocalParkingEnv(
        action_mask=synthetic_action_mask,
        seed=3,
    )
    env.reset()
    env.current_mask[:] = 0.0
    seen = {}
    original_step = env.vehicle_model.step

    def recording_step(state, action, dt=None):
        seen["action"] = np.asarray(action).copy()
        return original_step(state, action, dt=dt)

    env.vehicle_model.step = recording_step
    _, _, _, _, info = env.step(np.asarray([1.0, 0.25], dtype=np.float32))
    assert info["raw_action"][0] == 1.0
    assert info["executed_action"][0] == 0.0
    assert seen["action"][0] == 0.0
