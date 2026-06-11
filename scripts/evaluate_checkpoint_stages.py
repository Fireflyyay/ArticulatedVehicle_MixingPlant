import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG, DEFAULT_VEHICLE_PARAMS
from env.local_parking_env import LocalParkingEnv
from model.continuous_ppo import ContinuousPPOAgent
from train.curriculum import MultiStageScenePool


def _pad(val, width):
    return str(val).rjust(width)


def evaluate_checkpoint(checkpoint_path, episodes_per_stage, seed, device, stages):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_stage = payload.get("extra", {}).get("stage", "?")
    checkpoint_episode = payload.get("extra", {}).get("episode", "?")
    print("checkpoint: {}".format(checkpoint_path))
    print("  trained stage: {}".format(checkpoint_stage))
    print("  trained episode: {}".format(checkpoint_episode))

    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()

    multi_pool = MultiStageScenePool(
        pool_size=DEFAULT_ENV_CONFIG.scene_pool_size,
        base_seed=int(seed),
        scene_config=DEFAULT_SCENE_CONFIG,
    )
    env = LocalParkingEnv(
        config=DEFAULT_ENV_CONFIG,
        multi_stage_pool=multi_pool,
        seed=int(seed),
    )

    stage_summaries = {}

    for stage in stages:
        env.set_active_stage(stage)

        outcomes = []
        sub_scenarios = defaultdict(list)

        for ep in range(episodes_per_stage):
            obs, reset_info = env.reset()
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
                    "front_overlap": float(final_info["front_overlap"]),
                    "heading_error_deg": float(final_info["heading_error_deg"]),
                    "distance_to_goal": float(final_info["distance_to_goal"]),
                    "episode_steps": ep_steps,
                }
            )
            sub_scenarios[scenario].append(outcomes[-1])

            if (ep + 1) % 50 == 0:
                elapsed = time.perf_counter()
                print(
                    "  stage {} episode {}/{}".format(stage, ep + 1, episodes_per_stage)
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
        }

    return stage_summaries


def print_summary_table(stage_summaries):
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
        if len(sub) <= 1:
            continue
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
        default=200,
        help="Number of episodes per stage (default: 200)",
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
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print("ERROR: checkpoint not found: {}".format(checkpoint_path))
        sys.exit(1)

    summaries = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        episodes_per_stage=args.episodes,
        seed=args.seed,
        device=args.device,
        stages=args.stages,
    )
    print_summary_table(summaries)


if __name__ == "__main__":
    main()
