from argparse import Namespace
from dataclasses import replace
from datetime import datetime
import os

from config import DEFAULT_ENV_CONFIG, DEFAULT_PPO_CONFIG
from train.train_local_parking import (
    REPO_ROOT,
    _resolve_output_dir,
    _update_reward_plot,
    _weighted_checkpoint_score,
    _write_config_snapshot,
)


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


def test_reward_plot_is_written_from_episode_rewards(tmp_path):
    path = tmp_path / "reward_curve.png"
    _update_reward_plot(str(path), [float(index) for index in range(1, 11)])
    assert path.is_file()
    assert path.stat().st_size > 0


def test_weighted_checkpoint_score_prioritizes_parallel_reverse():
    score = _weighted_checkpoint_score(
        {
            "head_in": 1.0,
            "parallel_fwd": 0.5,
            "parallel_rev": 0.25,
        },
        DEFAULT_PPO_CONFIG,
    )
    assert score == (1.0 + 2.0 * 0.5 + 4.0 * 0.25) / 7.0


def test_default_ppo_stability_configuration():
    assert DEFAULT_PPO_CONFIG.log_std_init == -0.7
    assert DEFAULT_PPO_CONFIG.log_std_min == -2.5
    assert DEFAULT_PPO_CONFIG.log_std_max == -0.3
    assert DEFAULT_PPO_CONFIG.target_kl == 0.03
    assert DEFAULT_PPO_CONFIG.ppo_epochs == 4
    assert DEFAULT_PPO_CONFIG.clip_range == 0.15
    assert DEFAULT_PPO_CONFIG.actor_lr == 1e-4
    assert DEFAULT_PPO_CONFIG.critic_lr == 3e-4
    assert DEFAULT_PPO_CONFIG.entropy_coef == 0.0
    assert DEFAULT_PPO_CONFIG.policy_loss_weight_parallel_rev == 0.2
