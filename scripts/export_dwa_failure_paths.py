#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from model.continuous_ppo import ContinuousPPOAgent  # noqa: E402
from train.curriculum import MultiStageScenePool  # noqa: E402
from visualize_local_parking_paths import (  # noqa: E402
    PathRollout,
    _plot_scene_and_path,
    _status_label,
)


TASK_FAMILIES = ("head_in",)


def _load_agent(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()
    return agent, dict(payload.get("extra", {}))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _compact_info(info):
    keys = (
        "success",
        "collision",
        "timeout",
        "deadlock",
        "failure_type",
        "out_of_bounds",
        "articulation_limit_violation",
        "front_overlap",
        "rear_body_overlap",
        "heading_error_deg",
        "rear_heading_error_deg",
        "distance_to_goal",
        "min_lidar_distance",
        "max_safe_ratio",
        "forced_stop",
        "policy_forced_stop",
        "dwa_enabled",
        "dwa_triggered",
        "dwa_used",
        "dwa_mode",
        "dwa_reason",
        "dwa_candidate_count",
        "dwa_valid_candidate_count",
        "dwa_unlock_success",
        "dwa_deadlock",
        "dwa_final_max_safe_ratio",
        "dwa_override_policy_action",
        "mask_all_zero_before_floor",
        "mask_max_before_floor",
        "scenario_type",
        "scene_seed",
        "task_family",
    )
    return {key: _json_safe(info.get(key)) for key in keys if key in info}


def _env_config(stage, max_steps):
    kwargs = dict(
        curriculum_stage=int(stage),
        scene_family_schedule=TASK_FAMILIES,
        use_hybrid_astar=False,
        rs_potential_enabled=False,
        enable_hope_teacher=False,
        use_teacher_reward=False,
        enable_offpath_reset=False,
        enable_failure_aggregation=False,
        enable_dwa_recovery=True,
        dwa_override_policy_action=True,
        dwa_enable_deadlock_termination=False,
    )
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)
    return replace(DEFAULT_ENV_CONFIG, **kwargs)


def _rollout(env, agent, deterministic):
    obs, reset_info = env.reset()
    states = [replace(env.state)]
    scene = env.scene
    slot = env.slot
    done = False
    total_reward = 0.0
    final_info = dict(reset_info)
    while not done:
        raw_action, _, _ = agent.act(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(raw_action)
        states.append(replace(env.state))
        total_reward += float(reward)
        final_info = info
        done = bool(terminated or truncated)
    return PathRollout(
        seed=int(final_info.get("scene_seed", -1)),
        scene=scene,
        slot=slot,
        reset_info=dict(reset_info),
        states=states,
        final_info=dict(final_info),
        total_reward=float(total_reward),
    )


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def export_failures(args):
    checkpoint_path = os.path.abspath(args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    agent, extra = _load_agent(checkpoint_path, args.device)
    deterministic = not bool(args.stochastic)

    all_records = []
    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_extra": extra,
        "seed": int(args.seed),
        "episodes_per_stage": int(args.episodes_per_stage),
        "max_failure_images_per_stage": int(args.max_failure_images_per_stage),
        "deterministic": bool(deterministic),
        "dwa_assisted": True,
        "stages": {},
    }

    for stage in args.stages:
        stage = int(stage)
        config = _env_config(stage, args.max_steps)
        multi_pool = MultiStageScenePool(
            pool_size=int(config.scene_pool_size),
            base_seed=int(args.seed),
            scene_config=DEFAULT_SCENE_CONFIG,
            family_schedule=config.scene_family_schedule,
        )
        env = LocalParkingEnv(config=config, multi_stage_pool=multi_pool, seed=int(args.seed))
        env.set_active_stage(stage)

        stage_dir = os.path.join(output_dir, "stage{}".format(stage))
        os.makedirs(stage_dir, exist_ok=True)
        status_counter = Counter()
        failure_type_counter = Counter()
        scenario_counter = Counter()
        rendered_failures = 0

        for episode_index in range(int(args.episodes_per_stage)):
            rollout = _rollout(env, agent, deterministic)
            info = rollout.final_info
            status = _status_label(info)
            scenario = str(rollout.reset_info.get("scenario_type", "unknown"))
            status_counter[status] += 1
            scenario_counter[scenario] += 1
            failure_type_counter[str(info.get("failure_type", status))] += 1

            image_path = ""
            is_failure = not bool(info.get("success", False))
            if (
                is_failure
                and rendered_failures < int(args.max_failure_images_per_stage)
            ):
                image_name = (
                    "stage{stage:01d}_fail{idx:03d}_ep{episode:04d}_"
                    "{status}_{scenario}_seed{seed}.png"
                ).format(
                    stage=stage,
                    idx=rendered_failures + 1,
                    episode=episode_index + 1,
                    status=status,
                    scenario=scenario,
                    seed=int(info.get("scene_seed", -1)),
                )
                image_path = os.path.join(stage_dir, image_name)
                _plot_scene_and_path(
                    env=env,
                    rollout=rollout,
                    checkpoint_path=checkpoint_path,
                    output=image_path,
                    stage=stage,
                    deterministic=deterministic,
                    path_index=rendered_failures,
                    total_paths=max(1, int(args.max_failure_images_per_stage)),
                )
                rendered_failures += 1

            record = {
                "stage": stage,
                "episode_index": int(episode_index),
                "status": status,
                "is_failure": bool(is_failure),
                "scenario_type": scenario,
                "image_path": image_path,
                "reset_info": {
                    key: _json_safe(value)
                    for key, value in rollout.reset_info.items()
                    if key
                    in (
                        "scene_seed",
                        "scenario_type",
                        "task_family",
                        "goal_orientation_mode",
                        "fallback_used",
                        "initial_mask_max_before_floor",
                        "initial_mask_all_zero_before_floor",
                        "min_lidar_distance",
                    )
                },
                "final_info": _compact_info(info),
                "steps": max(0, len(rollout.states) - 1),
                "total_reward": float(rollout.total_reward),
            }
            all_records.append(record)

            if (episode_index + 1) % int(args.progress_interval) == 0:
                print(
                    "stage {}: {}/{} episodes, success {:.1f}%, failures {}, rendered {}".format(
                        stage,
                        episode_index + 1,
                        int(args.episodes_per_stage),
                        100.0 * status_counter["success"] / float(episode_index + 1),
                        (episode_index + 1) - status_counter["success"],
                        rendered_failures,
                    ),
                    flush=True,
                )

        env.close() if hasattr(env, "close") else None
        total = sum(status_counter.values())
        summary["stages"][str(stage)] = {
            "episodes": int(total),
            "success_rate": (
                float(status_counter["success"]) / float(total) if total else 0.0
            ),
            "status_counts": dict(status_counter),
            "failure_type_counts": dict(failure_type_counter),
            "scenario_counts": dict(scenario_counter),
            "rendered_failure_images": int(rendered_failures),
            "stage_output_dir": stage_dir,
        }
        print(
            "stage {} done: success {:.1f}% status={} rendered={}".format(
                stage,
                100.0 * summary["stages"][str(stage)]["success_rate"],
                dict(status_counter),
                rendered_failures,
            ),
            flush=True,
        )

    summary_path = os.path.join(output_dir, "summary.json")
    records_path = os.path.join(output_dir, "episodes.jsonl")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _write_jsonl(records_path, all_records)
    print("wrote {}".format(summary_path))
    print("wrote {}".format(records_path))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run DWA-assisted checkpoint evaluation and render failed paths."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "dwa_failures"),
    )
    parser.add_argument("--episodes-per-stage", type=int, default=100)
    parser.add_argument("--max-failure-images-per-stage", type=int, default=30)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    if int(args.episodes_per_stage) <= 0:
        raise ValueError("--episodes-per-stage must be positive")
    if int(args.max_failure_images_per_stage) < 0:
        raise ValueError("--max-failure-images-per-stage must be non-negative")
    if int(args.progress_interval) <= 0:
        raise ValueError("--progress-interval must be positive")
    export_failures(args)


if __name__ == "__main__":
    main()
