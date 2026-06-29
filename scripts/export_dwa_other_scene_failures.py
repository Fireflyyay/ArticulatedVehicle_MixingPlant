#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
from dataclasses import replace

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_SCENE_CONFIG  # noqa: E402
from env.mixing_plant_scene import DEFAULT_SCENE_TYPES, SUPPORTED_SCENE_TYPES  # noqa: E402
from train.curriculum import MultiStageScenePool  # noqa: E402

from analyze_dwa_checkpoints import (  # noqa: E402
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


def _scene_types_from_args(args):
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
    if bool(args.only_other_types):
        values = tuple(scene_type for scene_type in values if scene_type not in DEFAULT_SCENE_TYPES)
    unknown = [scene_type for scene_type in values if scene_type not in SUPPORTED_SCENE_TYPES]
    if unknown:
        raise ValueError("unsupported scene types: {}".format(",".join(unknown)))
    if not values:
        raise ValueError("no scene types selected")
    return values


def _manifest_record(checkpoint_name, scene_type, stage, episode_index, rollout, image_path):
    final_info = rollout["final_info"]
    return {
        "checkpoint": checkpoint_name,
        "scene_type": str(scene_type),
        "actual_scene_type": str(rollout["scene_meta"].get("scene_type", "")),
        "requested_scene_type": str(rollout["scene_meta"].get("requested_scene_type", "")),
        "stage": int(stage),
        "episode_index": int(episode_index),
        "status": _status_label(final_info),
        "scene_seed": int(final_info.get("scene_seed", rollout["reset_info"].get("scene_seed", -1))),
        "scenario_type": str(rollout["reset_info"].get("scenario_type", "")),
        "front_overlap": float(final_info.get("front_overlap", 0.0)),
        "heading_error_deg": float(final_info.get("heading_error_deg", 0.0)),
        "distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
        "dwa_trigger_steps": int(
            sum(1 for step in rollout["steps"] if bool(step["info"].get("dwa_triggered", False)))
        ),
        "dwa_used_steps": int(
            sum(1 for step in rollout["steps"] if bool(step["info"].get("dwa_used", False)))
        ),
        "image_path": os.path.abspath(image_path),
        "reset_info": _scalar_dict(rollout["reset_info"]),
        "scene_meta": _scalar_dict(rollout["scene_meta"]),
    }


def export_failures(args):
    run_dir = os.path.abspath(args.run_dir)
    failures_dir = os.path.abspath(args.failures_dir)
    os.makedirs(failures_dir, exist_ok=True)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    scene_types = _scene_types_from_args(args)
    checkpoints = _select_checkpoints(
        run_dir,
        args.checkpoint_mode,
        args.checkpoint_stride,
        args.checkpoints,
    )
    if not checkpoints:
        raise ValueError("no checkpoints selected from {}".format(run_dir))

    records = []
    print(
        "selected {} checkpoints, scene_types={}".format(
            len(checkpoints),
            ",".join(scene_types),
        ),
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = os.path.basename(checkpoint_path)
        agent, _ = _load_agent(checkpoint_path, args.device)
        print(
            "[{}/{}] {}".format(checkpoint_index + 1, len(checkpoints), checkpoint_name),
            flush=True,
        )
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
                from env.local_parking_env import LocalParkingEnv  # noqa: E402

                env = LocalParkingEnv(
                    config=config,
                    scene_config=scene_config,
                    multi_stage_pool=pool,
                    seed=int(args.seed),
                )
                env.set_active_stage(stage)
                rendered = 0
                failures = 0
                for episode_index in range(int(args.episodes_per_stage)):
                    rollout = _rollout(
                        env,
                        agent,
                        deterministic=not bool(args.stochastic),
                        horizon=int(config.dwa_horizon_steps),
                    )
                    if bool(rollout["final_info"].get("success", False)):
                        continue
                    failures += 1
                    status = _status_label(rollout["final_info"])
                    scene_seed = int(
                        rollout["final_info"].get(
                            "scene_seed",
                            rollout["reset_info"].get("scene_seed", -1),
                        )
                    )
                    image_name = (
                        "{}_{}_{}_stage{}_ep{:04d}_{}_seed{}.png".format(
                            str(args.filename_prefix),
                            os.path.splitext(checkpoint_name)[0],
                            str(scene_type),
                            stage,
                            episode_index + 1,
                            status,
                            scene_seed,
                        )
                    )
                    image_path = os.path.join(failures_dir, image_name)
                    _plot_failure(
                        image_path,
                        env,
                        rollout,
                        checkpoint_name,
                        stage,
                        episode_index + 1,
                    )
                    records.append(
                        _manifest_record(
                            checkpoint_name,
                            scene_type,
                            stage,
                            episode_index + 1,
                            rollout,
                            image_path,
                        )
                    )
                    rendered += 1
                    if rendered >= int(args.max_failures_per_combo):
                        break
                env.close() if hasattr(env, "close") else None
                print(
                    "  {} stage {} failures_seen={} rendered={}".format(
                        scene_type,
                        stage,
                        failures,
                        rendered,
                    ),
                    flush=True,
                )

    manifest_path = os.path.join(failures_dir, str(args.manifest_name))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    print("wrote {} failure images; manifest {}".format(len(records), manifest_path), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Export DWA failure images for non-default curriculum scene types."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(REPO_ROOT, "runs", "local_parking_20260627_143716_seed0"),
    )
    parser.add_argument(
        "--failures-dir",
        default=os.path.join(REPO_ROOT, "outputs", "dwa_failures", "failures"),
    )
    parser.add_argument("--scene-types", nargs="*", default=None)
    parser.add_argument("--only-other-types", action="store_true", default=True)
    parser.add_argument("--include-default-scene-types", dest="only_other_types", action="store_false")
    parser.add_argument("--checkpoint-mode", choices=("all", "stride", "latest"), default="stride")
    parser.add_argument("--checkpoint-stride", type=int, default=4000)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--episodes-per-stage", type=int, default=80)
    parser.add_argument("--max-failures-per-combo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--filename-prefix", default="other_scene")
    parser.add_argument("--manifest-name", default="other_scene_failures_manifest.jsonl")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    if int(args.episodes_per_stage) <= 0:
        raise ValueError("--episodes-per-stage must be positive")
    if int(args.max_failures_per_combo) <= 0:
        raise ValueError("--max-failures-per-combo must be positive")
    export_failures(args)


if __name__ == "__main__":
    main()
