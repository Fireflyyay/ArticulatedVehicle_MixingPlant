from argparse import Namespace
from dataclasses import replace
from datetime import datetime
import os

from config import DEFAULT_ENV_CONFIG, DEFAULT_PPO_CONFIG, DEFAULT_SCENE_CONFIG
from env.local_parking_env import LocalParkingEnv
from model.continuous_ppo import ContinuousPPOAgent
from train.train_local_parking import (
    REPO_ROOT,
    _add_config_bool_argument,
    _checkpoint_selection_score,
    _evaluate_policy_by_family,
    _resolve_output_dir,
    _update_reward_plot,
    _weighted_checkpoint_score,
    _write_config_snapshot,
)
import numpy as np


def test_default_training_output_is_timestamped_under_workspace_runs():
    output = _resolve_output_dir(
        output_dir=None,
        seed=7,
        timestamp=datetime(2026, 6, 9, 12, 34, 56),
    )
    assert output == os.path.join(
        REPO_ROOT,
        "runs",
        "local_parking_20260609_123456_seed7",
    )


def test_training_config_snapshot_contains_effective_sections(tmp_path):
    args = Namespace(
        total_episodes=20,
        rollout_steps=64,
        batch_size=32,
        stage=2,
        use_hybrid_astar=False,
        seed=3,
        device="cpu",
        checkpoint_interval=10,
        output_dir=str(tmp_path),
    )
    env_config = replace(DEFAULT_ENV_CONFIG, curriculum_stage=2)
    ppo_config = replace(DEFAULT_PPO_CONFIG, rollout_steps=64, batch_size=32)
    path = tmp_path / "config.txt"
    _write_config_snapshot(str(path), args, env_config, ppo_config)
    contents = path.read_text(encoding="utf-8")
    assert "[training_arguments]" in contents
    assert "total_episodes = 20" in contents
    assert "[vehicle]" in contents
    assert "[scene]" in contents
    assert "[environment]" in contents
    assert "curriculum_stage = 2" in contents
    assert "[ppo]" in contents


def test_observation_ablation_flags_zero_only_policy_slices(synthetic_action_mask):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=1,
            scene_pool_size=1,
            disable_mask_observation=True,
            rear_lidar_observation_mode="zero",
        ),
        action_mask=synthetic_action_mask,
        seed=23,
    )

    observation, _ = env.reset(seed=23)
    lidar_slice = LocalParkingEnv.OBS_SLICES["lidar"]
    mask_slice = LocalParkingEnv.OBS_SLICES["mask"]
    lidar = observation[lidar_slice]
    rear_lidar = lidar[LocalParkingEnv.LIDAR_FEATURE_DIM // 2 :]

    assert np.allclose(observation[mask_slice], 0.0)
    assert np.allclose(rear_lidar, 0.0)
    assert env.current_mask is not None
    assert not np.allclose(env.current_mask.reshape(-1), 0.0)
    assert env.last_rear_lidar_m is not None
    assert not np.allclose(env.last_rear_lidar_m, 0.0)


def test_dwa_cli_booleans_inherit_config_defaults():
    import argparse

    parser = argparse.ArgumentParser()
    _add_config_bool_argument(
        parser,
        "enable-dwa-recovery",
        DEFAULT_ENV_CONFIG.enable_dwa_recovery,
        "enable",
        "disable",
    )
    _add_config_bool_argument(
        parser,
        "dwa-override-policy-action",
        DEFAULT_ENV_CONFIG.dwa_override_policy_action,
        "enable",
        "disable",
    )

    defaults = parser.parse_args([])
    assert defaults.enable_dwa_recovery is DEFAULT_ENV_CONFIG.enable_dwa_recovery
    assert (
        defaults.dwa_override_policy_action
        is DEFAULT_ENV_CONFIG.dwa_override_policy_action
    )

    disabled = parser.parse_args(
        ["--disable-dwa-recovery", "--disable-dwa-override-policy-action"]
    )
    assert disabled.enable_dwa_recovery is False
    assert disabled.dwa_override_policy_action is False


def test_reward_plot_is_written_from_episode_rewards(tmp_path):
    path = tmp_path / "reward_curve.png"
    _update_reward_plot(str(path), [float(index) for index in range(1, 11)])
    assert path.is_file()
    assert path.stat().st_size > 0


def test_weighted_checkpoint_score_uses_head_in_success():
    score = _weighted_checkpoint_score(
        {
            "head_in": 1.0,
        },
        DEFAULT_PPO_CONFIG,
    )
    assert score == 1.0


def test_checkpoint_selection_score_uses_stage3_and_stage4_failure_slices():
    score = _checkpoint_selection_score(
        {
            "stage3_no_latch_success": 0.72,
            "stage4_recovery_success": 0.61,
        }
    )

    assert score == 0.61


def test_checkpoint_selection_score_prefers_scene_type_equal_slices():
    score = _checkpoint_selection_score(
        {
            "stage3_no_latch_success": 0.95,
            "stage4_recovery_success": 0.94,
            "stage3_scene_type_equal_no_latch_success": 0.66,
            "stage4_scene_type_equal_recovery_success": 0.58,
        }
    )

    assert score == 0.58


def test_evaluation_uses_eval_split_seed_namespace():
    agent = ContinuousPPOAgent(device="cpu")
    env_config = replace(
        DEFAULT_ENV_CONFIG,
        max_steps=1,
        scene_pool_size=1,
        scene_refresh_enabled=True,
        scene_refresh_interval=1,
        scene_refresh_count=1,
    )

    result = _evaluate_policy_by_family(
        agent=agent,
        env_config=env_config,
        scene_config=DEFAULT_SCENE_CONFIG,
        stage=1,
        seed=5,
        episodes_per_family=20,
        eval_modes=("no_guide",),
        scene_type_schedule=(DEFAULT_SCENE_CONFIG.scene_type,),
    )

    assert result["split"] == "eval"
    assert result["seed_base"] == 100_005
    assert result["test_seed_base"] == 1_000_005
    assert result["deterministic_episodes"]
    assert all(item["split"] == "eval" for item in result["deterministic_episodes"])
    assert all(
        int(item["seed_base"]) == 100_005
        for item in result["deterministic_episodes"]
    )


def test_default_ppo_stability_configuration():
    assert DEFAULT_PPO_CONFIG.log_std_init == -0.7
    assert DEFAULT_PPO_CONFIG.log_std_min == -2.5
    assert DEFAULT_PPO_CONFIG.log_std_max == -0.3
    assert DEFAULT_PPO_CONFIG.target_kl == 0.03
    assert DEFAULT_PPO_CONFIG.ppo_epochs == 6
    assert DEFAULT_PPO_CONFIG.clip_range == 0.2
    assert DEFAULT_PPO_CONFIG.actor_lr == 3e-4
    assert DEFAULT_PPO_CONFIG.critic_lr == 1e-3
    assert DEFAULT_PPO_CONFIG.entropy_coef == 0.0
    assert DEFAULT_PPO_CONFIG.policy_loss_weight_head_in == 1.0
