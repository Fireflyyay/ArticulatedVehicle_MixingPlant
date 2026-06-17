import importlib.util
import os
from dataclasses import replace

import numpy as np

from config import DEFAULT_ENV_CONFIG
from env.local_parking_env import LocalParkingEnv


def _hope_paths():
    hope_dir = "/home/cyberbus/Public/HOPE"
    return (
        hope_dir,
        os.path.join(hope_dir, "src", "model", "ckpt", "HOPE_PPO.pt"),
    )


def test_hope_teacher_disabled_preserves_observation_shape(synthetic_action_mask):
    config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=1,
        scene_pool_size=1,
        scene_family_schedule=("head_in",),
        rs_potential_enabled=False,
        enable_hope_teacher=False,
        use_teacher_reward=False,
    )
    env = LocalParkingEnv(config=config, action_mask=synthetic_action_mask, seed=3)
    obs, reset_info = env.reset()
    next_obs, _reward, _terminated, _truncated, info = env.step(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )

    assert obs.shape == (149,)
    assert next_obs.shape == (149,)
    assert reset_info["hope_teacher_enabled"] is False
    assert info["hope_teacher_enabled"] is False
    assert info["reward_components"]["hope_teacher"] == 0.0


def test_hope_teacher_failure_degrades_without_crashing(tmp_path, synthetic_action_mask):
    config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=1,
        max_steps=2,
        scene_pool_size=1,
        scene_family_schedule=("head_in",),
        rs_potential_enabled=False,
        enable_hope_teacher=True,
        hope_code_dir=str(tmp_path / "missing_hope"),
        hope_weight_path=str(tmp_path / "missing.pt"),
        hope_cache_dir=str(tmp_path / "cache"),
        use_teacher_reward=True,
        guide_dropout_initial=0.0,
        guide_dropout_final=0.0,
    )
    env = LocalParkingEnv(config=config, action_mask=synthetic_action_mask, seed=4)
    _obs, reset_info = env.reset()
    _next_obs, _reward, _terminated, _truncated, info = env.step(
        np.asarray([0.5, 0.0], dtype=np.float32)
    )

    assert reset_info["hope_teacher_enabled"] is True
    assert reset_info["hope_teacher_available"] is False
    assert reset_info["hope_plan_success"] is False
    assert "hope_code_unavailable" in reset_info["hope_plan_fail_reason"]
    assert info["guide_reward"] == 0.0


def test_hope_teacher_cache_records_and_hits(tmp_path, synthetic_action_mask):
    hope_dir, hope_weight = _hope_paths()
    config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=1,
        scene_pool_size=1,
        scene_family_schedule=("head_in",),
        rs_potential_enabled=False,
        enable_hope_teacher=True,
        hope_code_dir=hope_dir,
        hope_weight_path=hope_weight,
        hope_cache_dir=str(tmp_path / "cache"),
        use_teacher_reward=True,
        guide_weight_initial=0.5,
        guide_weight_final=0.5,
        guide_anneal_start_episode=0,
        guide_anneal_end_episode=1,
        guide_dropout_initial=0.0,
        guide_dropout_final=0.0,
    )
    env1 = LocalParkingEnv(config=config, action_mask=synthetic_action_mask, seed=5)
    _obs1, info1 = env1.reset(seed=5)
    env2 = LocalParkingEnv(config=config, action_mask=synthetic_action_mask, seed=5)
    _obs2, info2 = env2.reset(seed=5)

    assert info1["hope_teacher_enabled"] is True
    assert info1["hope_code_loaded"] is True
    assert info1["hope_weight_loaded"] is True, info1["hope_weight_load_error"]
    assert info2["hope_cache_hit"] is True
    assert any((tmp_path / "cache").glob("*.json"))


def test_no_guide_eval_config_disables_hope():
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "evaluate_checkpoint_stages.py",
    )
    spec = importlib.util.spec_from_file_location("eval_checkpoint_stages", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    no_guide = module._eval_config_for_mode(
        "no_guide",
        hope_code_dir="/does/not/matter",
        hope_weight_path="/does/not/matter.pt",
        hope_cache_dir="/tmp/does-not-matter",
        use_teacher_reward=True,
    )
    deployment = module._eval_config_for_mode(
        "deployment",
        hope_code_dir="/does/not/matter",
        hope_weight_path="/does/not/matter.pt",
        hope_cache_dir="/tmp/does-not-matter",
        use_teacher_reward=True,
    )
    guided = module._eval_config_for_mode(
        "guided",
        hope_code_dir="/does/not/matter",
        hope_weight_path="/does/not/matter.pt",
        hope_cache_dir="/tmp/does-not-matter",
        use_teacher_reward=True,
    )

    assert no_guide.enable_hope_teacher is False
    assert no_guide.use_teacher_reward is False
    assert deployment.enable_hope_teacher is False
    assert guided.enable_hope_teacher is True
    assert guided.use_teacher_reward is True


def test_failure_aggregation_writes_failure_record(tmp_path, synthetic_action_mask):
    config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=1,
        max_steps=1,
        scene_pool_size=1,
        scene_family_schedule=("head_in",),
        rs_potential_enabled=False,
        enable_hope_teacher=True,
        hope_code_dir=str(tmp_path / "missing_hope"),
        hope_weight_path=str(tmp_path / "missing.pt"),
        hope_cache_dir=str(tmp_path / "cache"),
        use_teacher_reward=True,
        enable_failure_aggregation=True,
    )
    env = LocalParkingEnv(config=config, action_mask=synthetic_action_mask, seed=6)
    env.reset()
    _obs, _reward, _terminated, truncated, info = env.step(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )

    assert truncated is True
    assert info["hope_failure_aggregation_recorded"] is True
    assert (tmp_path / "cache" / "failure_aggregation.jsonl").is_file()
