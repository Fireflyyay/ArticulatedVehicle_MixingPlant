#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_SCENE_CONFIG  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from env.mixing_plant_scene import SUPPORTED_SCENE_TYPES  # noqa: E402
from train.curriculum import MultiStageScenePool  # noqa: E402

from analyze_dwa_checkpoints import (  # noqa: E402
    _aggregate,
    _checkpoint_episode,
    _compact_step_info,
    _env_config,
    _json_safe,
    _load_agent,
    _plot_failure,
    _rollout,
    _scalar_dict,
    _select_checkpoints,
    _status_label,
)


def _parse_config_value(path, key):
    if not os.path.isfile(path):
        return None
    pattern = re.compile(r"^\s*{}\s*=\s*(.+?)\s*$".format(re.escape(key)))
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line)
            if not match:
                continue
            raw = match.group(1).strip()
            try:
                return ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                return raw.strip("'\"")
    return None


def _scene_types(args):
    if args.scene_types:
        values = tuple(str(item) for item in args.scene_types)
    else:
        configured = _parse_config_value(
            os.path.join(os.path.abspath(args.run_dir), "config.txt"),
            "curriculum_scene_types",
        )
        if configured:
            values = tuple(part.strip() for part in str(configured).split(",") if part.strip())
        else:
            values = tuple(SUPPORTED_SCENE_TYPES)
    unknown = [item for item in values if item not in SUPPORTED_SCENE_TYPES]
    if unknown:
        raise ValueError("unsupported scene types: {}".format(",".join(unknown)))
    if not values:
        raise ValueError("no scene types selected")
    return values


def _episode_summary(checkpoint_path, checkpoint_extra, scene_type, stage, episode_index, rollout, image_path):
    final_info = rollout["final_info"]
    steps = rollout["steps"]
    speeds = [abs(float(step["executed_action"][0])) for step in steps]
    phi_rates = [abs(float(step["executed_action"][1])) for step in steps]
    dwa_trigger_steps = sum(1 for step in steps if bool(step["info"].get("dwa_triggered", False)))
    dwa_used_steps = sum(1 for step in steps if bool(step["info"].get("dwa_used", False)))
    forced_stop_steps = sum(1 for step in steps if bool(step["info"].get("forced_stop", False)))
    policy_forced_stop_steps = sum(1 for step in steps if bool(step["info"].get("policy_forced_stop", False)))
    mask_zero_steps = sum(1 for step in steps if bool(step["info"].get("mask_all_zero_before_floor", False)))
    valid_counts = [
        int(step["info"].get("dwa_valid_candidate_count", 0))
        for step in steps
        if bool(step["info"].get("dwa_triggered", False))
    ]
    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "checkpoint_episode": _checkpoint_episode(checkpoint_path),
        "checkpoint_extra_episode": checkpoint_extra.get("episode"),
        "checkpoint_extra_stage": checkpoint_extra.get("stage"),
        "scene_type": str(scene_type),
        "actual_scene_type": str(rollout["scene_meta"].get("scene_type", "")),
        "requested_scene_type": str(rollout["scene_meta"].get("requested_scene_type", "")),
        "stage": int(stage),
        "episode_index": int(episode_index),
        "status": _status_label(final_info),
        "success": bool(final_info.get("success", False)),
        "failure_type": str(final_info.get("failure_type", _status_label(final_info))),
        "scenario_type": str(rollout["reset_info"].get("scenario_type", "")),
        "scene_seed": int(final_info.get("scene_seed", rollout["reset_info"].get("scene_seed", -1))),
        "steps": max(0, len(rollout["states"]) - 1),
        "total_reward": float(rollout["total_reward"]),
        "front_overlap": float(final_info.get("front_overlap", 0.0)),
        "rear_body_overlap": float(final_info.get("rear_body_overlap", 0.0)),
        "heading_error_deg": float(final_info.get("heading_error_deg", 0.0)),
        "rear_heading_error_deg": float(final_info.get("rear_heading_error_deg", 0.0)),
        "distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
        "min_lidar_distance": float(final_info.get("min_lidar_distance", 0.0)),
        "avg_abs_v_exec": float(np.mean(speeds)) if speeds else 0.0,
        "max_abs_v_exec": float(np.max(speeds)) if speeds else 0.0,
        "avg_abs_phi_dot_exec": float(np.mean(phi_rates)) if phi_rates else 0.0,
        "forced_stop_steps": int(forced_stop_steps),
        "policy_forced_stop_steps": int(policy_forced_stop_steps),
        "mask_zero_steps": int(mask_zero_steps),
        "dwa_trigger_steps": int(dwa_trigger_steps),
        "dwa_used_steps": int(dwa_used_steps),
        "dwa_valid_candidate_steps": int(sum(1 for value in valid_counts if value > 0)),
        "dwa_avg_valid_candidates": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "image_path": image_path,
        "reset_info": _scalar_dict(rollout["reset_info"]),
        "scene_meta": _scalar_dict(rollout["scene_meta"]),
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def _combo_aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        key = (record["checkpoint"], record["scene_type"], record["stage"])
        grouped[key].append(record)
    rows = []
    for key, entries in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        checkpoint, scene_type, stage = key
        n = len(entries)
        failures = [item for item in entries if not item["success"]]
        rows.append(
            {
                "checkpoint": checkpoint,
                "scene_type": scene_type,
                "stage": int(stage),
                "episodes": int(n),
                "success_rate": sum(1 for item in entries if item["success"]) / float(n) if n else 0.0,
                "status_counts": dict(Counter(item["status"] for item in entries)),
                "failure_type_counts": dict(Counter(item["failure_type"] for item in failures)),
                "scenario_success": {
                    scenario: sum(1 for item in sub if item["success"]) / float(len(sub))
                    for scenario, sub in _group_by(entries, "scenario_type").items()
                },
                "scenario_counts": {
                    scenario: len(sub)
                    for scenario, sub in _group_by(entries, "scenario_type").items()
                },
                "avg_steps": float(np.mean([item["steps"] for item in entries])) if entries else 0.0,
                "avg_front_overlap": float(np.mean([item["front_overlap"] for item in entries])) if entries else 0.0,
                "avg_heading_error_deg": float(np.mean([item["heading_error_deg"] for item in entries])) if entries else 0.0,
                "dwa_trigger_episode_rate": sum(1 for item in entries if item["dwa_trigger_steps"] > 0) / float(n) if n else 0.0,
                "dwa_used_episode_rate": sum(1 for item in entries if item["dwa_used_steps"] > 0) / float(n) if n else 0.0,
                "avg_forced_stop_steps": float(np.mean([item["forced_stop_steps"] for item in entries])) if entries else 0.0,
                "avg_mask_zero_steps": float(np.mean([item["mask_zero_steps"] for item in entries])) if entries else 0.0,
            }
        )
    return rows


def _group_by(records, key):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record.get(key, ""))].append(record)
    return grouped


def run(args):
    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir)
    failures_dir = os.path.join(output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    scene_types = _scene_types(args)
    checkpoints = _select_checkpoints(
        run_dir,
        args.checkpoint_mode,
        args.checkpoint_stride,
        args.checkpoints,
    )
    if not checkpoints:
        raise ValueError("no checkpoints selected from {}".format(run_dir))

    episode_records = []
    failure_records = []
    checkpoint_extras = {}
    print(
        "selected {} checkpoints; scene_types={}; stages={}; episodes_per_combo={}".format(
            len(checkpoints),
            ",".join(scene_types),
            ",".join(str(stage) for stage in args.stages),
            int(args.episodes_per_combo),
        ),
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = os.path.basename(checkpoint_path)
        agent, extra = _load_agent(checkpoint_path, args.device)
        checkpoint_extras[checkpoint_name] = _json_safe(extra)
        print("[{}/{}] {}".format(checkpoint_index + 1, len(checkpoints), checkpoint_name), flush=True)
        for scene_type in scene_types:
            scene_config = replace(DEFAULT_SCENE_CONFIG, scene_type=str(scene_type))
            for stage in args.stages:
                stage = int(stage)
                config = _env_config(stage, args.max_steps)
                pool = MultiStageScenePool(
                    pool_size=int(config.scene_pool_size),
                    base_seed=int(args.seed),
                    scene_config=scene_config,
                    family_schedule=config.scene_family_schedule,
                    scene_type_schedule=(str(scene_type),),
                )
                env = LocalParkingEnv(
                    config=config,
                    scene_config=scene_config,
                    multi_stage_pool=pool,
                    seed=int(args.seed),
                )
                env.set_active_stage(stage)
                combo_records = []
                failures = 0
                for episode_index in range(int(args.episodes_per_combo)):
                    rollout = _rollout(
                        env,
                        agent,
                        deterministic=not bool(args.stochastic),
                        horizon=int(config.dwa_horizon_steps),
                    )
                    final_info = rollout["final_info"]
                    image_path = ""
                    if not bool(final_info.get("success", False)):
                        failures += 1
                        status = _status_label(final_info)
                        image_name = "{}_{}_stage{}_ep{:04d}_{}_seed{}.png".format(
                            os.path.splitext(checkpoint_name)[0],
                            str(scene_type),
                            stage,
                            episode_index + 1,
                            status,
                            int(final_info.get("scene_seed", rollout["reset_info"].get("scene_seed", -1))),
                        )
                        image_path = os.path.join(failures_dir, image_name)
                        if failures <= int(args.max_failure_images_per_combo):
                            _plot_failure(
                                image_path,
                                env,
                                rollout,
                                checkpoint_name,
                                stage,
                                episode_index + 1,
                            )
                        else:
                            image_path = ""
                        failure_records.append(
                            {
                                "checkpoint": checkpoint_name,
                                "scene_type": str(scene_type),
                                "stage": int(stage),
                                "episode_index": int(episode_index + 1),
                                "status": _status_label(final_info),
                                "image_path": image_path,
                                "reset_info": _scalar_dict(rollout["reset_info"]),
                                "scene_meta": _scalar_dict(rollout["scene_meta"]),
                                "final_info": _compact_step_info(final_info),
                                "total_reward": float(rollout["total_reward"]),
                                "states": rollout["states"],
                                "steps": rollout["steps"],
                            }
                        )
                    summary = _episode_summary(
                        checkpoint_path,
                        extra,
                        scene_type,
                        stage,
                        episode_index + 1,
                        rollout,
                        image_path,
                    )
                    episode_records.append(summary)
                    combo_records.append(summary)
                env.close() if hasattr(env, "close") else None
                successes = sum(1 for item in combo_records if item["success"])
                print(
                    "  {} stage {} success {}/{} ({:.1f}%) failures {}".format(
                        scene_type,
                        stage,
                        successes,
                        len(combo_records),
                        100.0 * successes / float(len(combo_records)),
                        failures,
                    ),
                    flush=True,
                )

    combo_records = _combo_aggregate(episode_records)
    aggregate = _aggregate(episode_records)
    aggregate["run_dir"] = run_dir
    aggregate["output_dir"] = output_dir
    aggregate["scene_types"] = list(scene_types)
    aggregate["checkpoint_count"] = int(len(checkpoints))
    aggregate["selected_checkpoints"] = [os.path.basename(path) for path in checkpoints]
    aggregate["checkpoint_extras"] = checkpoint_extras
    aggregate["args"] = _json_safe(vars(args))

    _write_jsonl(os.path.join(output_dir, "episode_summary.jsonl"), episode_records)
    _write_jsonl(os.path.join(output_dir, "combo_summary.jsonl"), combo_records)
    _write_jsonl(os.path.join(output_dir, "failure_trajectories.jsonl"), failure_records)
    with open(os.path.join(output_dir, "aggregate_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(aggregate), handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("wrote {}".format(output_dir), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DWA-assisted checkpoints across all curriculum scene types and stages."
    )
    parser.add_argument("--run-dir", default=os.path.join(REPO_ROOT, "runs", "local_parking_20260627_200116_seed0"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "dwa_failres"))
    parser.add_argument("--checkpoint-mode", choices=("all", "stride", "latest"), default="latest")
    parser.add_argument("--checkpoint-stride", type=int, default=5000)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--scene-types", nargs="*", default=None)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--episodes-per-combo", type=int, default=20)
    parser.add_argument("--max-failure-images-per-combo", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if int(args.episodes_per_combo) <= 0:
        raise ValueError("--episodes-per-combo must be positive")
    if int(args.max_failure_images_per_combo) < 0:
        raise ValueError("--max-failure-images-per-combo must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
