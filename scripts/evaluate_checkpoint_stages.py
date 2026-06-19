import argparse
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG, DEFAULT_VEHICLE_PARAMS
from env.local_parking_env import LocalParkingEnv
from model.continuous_ppo import ContinuousPPOAgent
from train.curriculum import MultiStageScenePool


TASK_FAMILIES = ("head_in",)


def _pad(val, width):
    return str(val).rjust(width)


def _eval_config_for_mode(
    eval_mode,
    hope_code_dir,
    hope_weight_path,
    hope_cache_dir,
    use_teacher_reward,
):
    enable_teacher = eval_mode == "guided"
    return replace(
        DEFAULT_ENV_CONFIG,
        use_hybrid_astar=False,
        rs_potential_enabled=False,
        enable_hope_teacher=enable_teacher,
        hope_code_dir=hope_code_dir,
        hope_weight_path=hope_weight_path,
        hope_cache_dir=hope_cache_dir,
        use_teacher_reward=bool(use_teacher_reward and enable_teacher),
        enable_offpath_reset=False,
        enable_failure_aggregation=False,
    )


def _task_family(reset_info, scene):
    task_family = str(reset_info.get("task_family", ""))
    if task_family in TASK_FAMILIES:
        return task_family
    if str(reset_info.get("goal_orientation_mode", "")) == "head_in":
        return "head_in"
    raise ValueError(
        "unsupported goal orientation mode: {}".format(
            reset_info.get("goal_orientation_mode", "")
        )
    )


def evaluate_checkpoint(
    checkpoint_path,
    episodes_per_family,
    seed,
    device,
    stages,
    eval_mode="deployment",
    hope_code_dir=DEFAULT_ENV_CONFIG.hope_code_dir,
    hope_weight_path=DEFAULT_ENV_CONFIG.hope_weight_path,
    hope_cache_dir=DEFAULT_ENV_CONFIG.hope_cache_dir,
    use_teacher_reward=False,
):
    episodes_per_family = int(episodes_per_family)
    if episodes_per_family < 20:
        raise ValueError("episodes_per_family must be at least 20")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_stage = payload.get("extra", {}).get("stage", "?")
    checkpoint_episode = payload.get("extra", {}).get("episode", "?")
    print("checkpoint: {}".format(checkpoint_path))
    print("  trained stage: {}".format(checkpoint_stage))
    print("  trained episode: {}".format(checkpoint_episode))

    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()

    if eval_mode == "both":
        eval_modes = ("guided", "no_guide", "deployment")
    else:
        eval_modes = (eval_mode,)

    summaries_by_mode = {}

    for mode_name in eval_modes:
        mode_config = _eval_config_for_mode(
            mode_name,
            hope_code_dir=hope_code_dir,
            hope_weight_path=hope_weight_path,
            hope_cache_dir=hope_cache_dir,
            use_teacher_reward=use_teacher_reward,
        )
        multi_pool = MultiStageScenePool(
            pool_size=mode_config.scene_pool_size,
            base_seed=int(seed),
            scene_config=DEFAULT_SCENE_CONFIG,
            family_schedule=mode_config.scene_family_schedule,
        )
        env = LocalParkingEnv(
            config=mode_config,
            multi_stage_pool=multi_pool,
            seed=int(seed),
        )

        stage_summaries = {}

        for stage in stages:
            env.set_active_stage(stage)

            outcomes = []
            sub_scenarios = defaultdict(list)
            families = dict((family, []) for family in TASK_FAMILIES)

            reset_attempts = 0
            while (
                min(len(entries) for entries in families.values())
            ) < episodes_per_family:
                obs, reset_info = env.reset()
                reset_attempts += 1
                family = _task_family(reset_info, env.scene)
                if len(families[family]) >= episodes_per_family:
                    continue
                done = False
                ep_steps = 0
                while not done:
                    raw_action, _, _ = agent.act(obs, deterministic=True)
                    obs, _reward, terminated, truncated, info = env.step(raw_action)
                    done = terminated or truncated
                    ep_steps += 1
                final_info = info
                success = bool(final_info["success"])
                collision = bool(final_info["collision"])
                timeout = bool(final_info["timeout"])
                articulation = bool(final_info.get("articulation_limit_violation", False))
                out_of_bounds = bool(final_info.get("out_of_bounds", False))
                fallback = bool(reset_info.get("fallback_used", False))
                scenario = str(reset_info.get("scenario_type", ""))
                outcomes.append(
                    {
                        "success": success,
                        "collision": collision,
                        "timeout": timeout,
                        "articulation": articulation,
                        "out_of_bounds": out_of_bounds,
                        "fallback": fallback,
                        "scenario": scenario,
                        "task_family": family,
                        "front_overlap": float(final_info["front_overlap"]),
                        "heading_error_deg": float(final_info["heading_error_deg"]),
                        "distance_to_goal": float(final_info["distance_to_goal"]),
                        "episode_steps": ep_steps,
                        "hope_plan_success": bool(
                            final_info.get("hope_plan_success", False)
                        ),
                        "guide_reward": float(final_info.get("guide_reward", 0.0)),
                    }
                )
                sub_scenarios[scenario].append(outcomes[-1])
                families[family].append(outcomes[-1])

                if len(outcomes) > 0 and len(outcomes) % 50 == 0:
                    print(
                        "  mode {} stage {} accepted {}/{} resets={}".format(
                            mode_name,
                            stage,
                            len(outcomes),
                            episodes_per_family * len(TASK_FAMILIES),
                            reset_attempts,
                        )
                    )

            n = len(outcomes)
            if n == 0:
                stage_summaries[stage] = {"episodes": 0, "success_rate": 0.0, "sub": {}}
                continue

            success_rate = sum(1 for o in outcomes if o["success"]) / n
            collision_rate = sum(1 for o in outcomes if o["collision"]) / n
            timeout_rate = sum(1 for o in outcomes if o["timeout"]) / n
            articulation_rate = sum(1 for o in outcomes if o["articulation"]) / n
            oob_rate = sum(1 for o in outcomes if o["out_of_bounds"]) / n
            fallback_rate = sum(1 for o in outcomes if o["fallback"]) / n
            avg_overlap = float(np.mean([o["front_overlap"] for o in outcomes]))
            avg_heading = float(np.mean([o["heading_error_deg"] for o in outcomes]))
            avg_distance = float(np.mean([o["distance_to_goal"] for o in outcomes]))
            avg_steps = float(np.mean([o["episode_steps"] for o in outcomes]))

            sub_tables = {}
            for scenario_name, entries in sorted(sub_scenarios.items()):
                sn = len(entries)
                sub_tables[scenario_name] = {
                    "count": sn,
                    "success_rate": sum(1 for e in entries if e["success"]) / sn if sn else 0.0,
                    "collision_rate": sum(1 for e in entries if e["collision"]) / sn if sn else 0.0,
                }
            family_tables = {}
            for family_name, entries in sorted(families.items()):
                fn = len(entries)
                family_tables[family_name] = {
                    "count": fn,
                    "success_rate": sum(1 for e in entries if e["success"]) / fn if fn else 0.0,
                    "collision_rate": sum(1 for e in entries if e["collision"]) / fn if fn else 0.0,
                }

            stage_summaries[stage] = {
                "episodes": n,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
                "articulation_rate": articulation_rate,
                "out_of_bounds_rate": oob_rate,
                "fallback_rate": fallback_rate,
                "avg_front_overlap": avg_overlap,
                "avg_heading_error_deg": avg_heading,
                "avg_distance": avg_distance,
                "avg_steps": avg_steps,
                "sub_scenarios": sub_tables,
                "families": family_tables,
            }
        env.close() if hasattr(env, "close") else None
        summaries_by_mode[mode_name] = stage_summaries

    return summaries_by_mode


def print_summary_table(stage_summaries):
    if stage_summaries and not all(isinstance(key, int) for key in stage_summaries):
        for mode_name, summaries in stage_summaries.items():
            print()
            print("Mode: {}".format(mode_name))
            print_summary_table(summaries)
        return
    header = (
        "{:<2}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
            "S",
            "eps",
            "succ%",
            "coll%",
            "tout%",
            "art%",
            "oob%",
            "fb%",
            "ovlp",
            "head",
        )
    )
    print()
    print(header)
    print("-" * len(header))
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        print(
            "{:<2}  {:>8}  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.3f}  {:>7.1f}".format(
                stage,
                s["episodes"],
                s["success_rate"] * 100,
                s["collision_rate"] * 100,
                s["timeout_rate"] * 100,
                s["articulation_rate"] * 100,
                s["out_of_bounds_rate"] * 100,
                s["fallback_rate"] * 100,
                s["avg_front_overlap"],
                s["avg_heading_error_deg"],
            )
        )

    for stage in sorted(stage_summaries.keys()):
        sub = stage_summaries[stage].get("sub_scenarios", {})
        if len(sub) > 1:
            print()
            print("Stage {} sub-scenario breakdown:".format(stage))
            sub_header = "  {:<35}  {:>6}  {:>8}  {:>8}".format(
                "scenario", "count", "succ%", "coll%"
            )
            print(sub_header)
            print("  " + "-" * (len(sub_header) - 2))
            for name, entry in sorted(sub.items()):
                print(
                    "  {:<35}  {:>6}  {:>7.1f}%  {:>7.1f}%".format(
                        name,
                        entry["count"],
                        entry["success_rate"] * 100,
                        entry["collision_rate"] * 100,
                    )
                )
        fam = stage_summaries[stage].get("families", {})
        if fam:
            print()
            print("Stage {} family breakdown:".format(stage))
            fam_header = "  {:<14}  {:>6}  {:>8}  {:>8}".format(
                "family", "count", "succ%", "coll%"
            )
            print(fam_header)
            print("  " + "-" * (len(fam_header) - 2))
            for name, entry in sorted(fam.items()):
                print(
                    "  {:<14}  {:>6}  {:>7.1f}%  {:>7.1f}%".format(
                        name,
                        entry["count"],
                        entry["success_rate"] * 100,
                        entry["collision_rate"] * 100,
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint across curriculum stages."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory containing checkpoint_final.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_final.pt",
        help="Checkpoint filename (default: checkpoint_final.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Deterministic episodes per family per stage (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Stages to evaluate (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["guided", "no_guide", "deployment", "both"],
        default="deployment",
        help="guided may call HOPE; no_guide/deployment never call HOPE",
    )
    parser.add_argument("--hope-code-dir", default=DEFAULT_ENV_CONFIG.hope_code_dir)
    parser.add_argument("--hope-weight-path", default=DEFAULT_ENV_CONFIG.hope_weight_path)
    parser.add_argument("--hope-cache-dir", default=DEFAULT_ENV_CONFIG.hope_cache_dir)
    parser.add_argument(
        "--use-teacher-reward",
        action="store_true",
        help="In guided eval, compute teacher reward diagnostics",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print("ERROR: checkpoint not found: {}".format(checkpoint_path))
        sys.exit(1)

    summaries = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        episodes_per_family=args.episodes,
        seed=args.seed,
        device=args.device,
        stages=args.stages,
        eval_mode=args.eval_mode,
        hope_code_dir=args.hope_code_dir,
        hope_weight_path=args.hope_weight_path,
        hope_cache_dir=args.hope_cache_dir,
        use_teacher_reward=args.use_teacher_reward,
    )
    print_summary_table(summaries)


if __name__ == "__main__":
    main()
