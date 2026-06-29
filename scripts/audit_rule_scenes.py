#!/usr/bin/env python3
import argparse
from dataclasses import replace
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from env.mixing_plant_scene import RULE_SCENE_TYPES  # noqa: E402


def _safe_mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _audit_scene_type(scene_type, samples, stage, seed):
    records = []
    failures = []
    for sample_index in range(int(samples)):
        sample_seed = int(seed) + sample_index
        scene_config = replace(DEFAULT_SCENE_CONFIG, scene_type=scene_type)
        env_config = replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=int(stage),
            scene_pool_size=1,
            reset_scene_retry_count=1,
        )
        try:
            env = LocalParkingEnv(
                config=env_config,
                scene_config=scene_config,
                seed=sample_seed,
            )
            _, info = env.reset(seed=sample_seed)
        except Exception as exc:
            failures.append(
                {
                    "sample_index": int(sample_index),
                    "seed": int(sample_seed),
                    "reason": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            continue
        records.append(dict(info))

    total = int(samples)
    generation_failures = len(failures)
    feasible_mask = [
        bool(
            item.get("reset_feasible_mask_available", False)
            or float(item.get("reset_initial_mask_max", 0.0))
            > float(item.get("reset_initial_mask_required", 0.0))
        )
        for item in records
    ]
    result = {
        "scene_type": str(scene_type),
        "total_samples": int(total),
        "valid_initial_rate": _safe_mean(
            [not bool(item.get("initial_collision", True)) for item in records]
        ),
        "valid_target_rate": _safe_mean(
            [not bool(item.get("nominal_target_collision", True)) for item in records]
        ),
        "collision_free_rate": _safe_mean(
            [
                not bool(item.get("initial_collision", True))
                and not bool(item.get("nominal_target_collision", True))
                for item in records
            ]
        ),
        "feasible_action_mask_rate": _safe_mean(feasible_mask),
        "generation_failure_rate": float(generation_failures) / max(total, 1),
        "average_generation_attempts": _safe_mean(
            [
                float(
                    item.get(
                        "scene_generation_attempts",
                        item.get("scene_generation_attempt_count", 1),
                    )
                )
                for item in records
            ]
        ),
        "obstacle_count_mean": _safe_mean(
            [
                float(
                    item.get(
                        "obstacle_count",
                        item.get("constructed_obstacle_feature_count", 0),
                    )
                )
                for item in records
            ]
        ),
        "corridor_outer_wall_exists_rate": _safe_mean(
            [
                bool(item.get("corridor_outer_wall_exists", False))
                for item in records
            ]
        ),
        "target_heading_into_bay_rate": _safe_mean(
            [
                bool(item.get("target_heading_into_bay", False))
                for item in records
            ]
        ),
        "truck_in_front_rate": _safe_mean(
            [bool(item.get("truck_in_front", False)) for item in records]
        ),
        "truck_perpendicular_rate": _safe_mean(
            [bool(item.get("truck_perpendicular", False)) for item in records]
        ),
        "obstacle_exclusion_valid_rate": _safe_mean(
            [bool(item.get("obstacle_exclusion_valid", False)) for item in records]
        ),
        "successful_samples": int(len(records)),
        "generation_failures": failures,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch-audit rule-based local parking scene generators."
    )
    parser.add_argument(
        "--scene-types",
        nargs="+",
        default=list(RULE_SCENE_TYPES),
        choices=RULE_SCENE_TYPES,
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    results = [
        _audit_scene_type(scene_type, args.samples, args.stage, args.seed)
        for scene_type in args.scene_types
    ]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
