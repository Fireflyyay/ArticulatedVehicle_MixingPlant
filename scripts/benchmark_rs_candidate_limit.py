"""
Benchmark RS planner candidate_limit on local head-in parking scenes.
Runs the same episodes with candidate_limit=2 and candidate_limit=10
to measure improvement in RS path validation success rate and time cost.
"""
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
from env.rs_potential import RSPotentialPlanner, RSPotentialOracle
from model.continuous_ppo import ContinuousPPOAgent
from planning.passenger_hybrid_astar import PassengerHybridAStar
from train.curriculum import MultiStageScenePool


def build_rs_planner(candidate_limit):
    collision_checker = PassengerHybridAStar(
        goal_pos_tol=DEFAULT_ENV_CONFIG.planner_position_tolerance,
        goal_heading_tol_deg=DEFAULT_ENV_CONFIG.planner_heading_tolerance_deg,
        front_half_length=DEFAULT_VEHICLE_PARAMS.front_body_length * 0.5,
        front_half_width=DEFAULT_VEHICLE_PARAMS.front_body_width * 0.5,
    )
    planner = RSPotentialPlanner(
        collision_checker=collision_checker,
        turning_radius=DEFAULT_VEHICLE_PARAMS.minimum_turning_radius,
        candidate_limit=candidate_limit,
        sample_step=(
            float(getattr(collision_checker, "step_length", 1.0))
            / float(getattr(collision_checker, "intermediate_checks", 2) + 1)
        ),
    )
    return planner, collision_checker


def evaluate_rs(checkpoint_path, device, episodes_per_stage, candidate_limit):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()

    rs_planner, cchecker = build_rs_planner(candidate_limit)
    oracle = RSPotentialOracle(
        planner=rs_planner,
        enabled=True,
        d_rs=DEFAULT_ENV_CONFIG.rs_potential_d_rs,
        gamma=0.98,
        cost_scale=DEFAULT_ENV_CONFIG.planner_cost_scale,
        potential_coef=DEFAULT_ENV_CONFIG.rs_potential_coef,
        potential_clip=DEFAULT_ENV_CONFIG.rs_potential_clip,
        max_cost=DEFAULT_ENV_CONFIG.planner_max_cost,
        lateral_weight=DEFAULT_ENV_CONFIG.planner_lateral_residual_weight,
        heading_weight=DEFAULT_ENV_CONFIG.planner_goal_heading_weight,
        lateral_clip=DEFAULT_ENV_CONFIG.planner_lateral_clip,
    )

    multi_pool = MultiStageScenePool(
        pool_size=DEFAULT_ENV_CONFIG.scene_pool_size,
        base_seed=0,
        scene_config=DEFAULT_SCENE_CONFIG,
    )
    env = LocalParkingEnv(
        config=DEFAULT_ENV_CONFIG,
        multi_stage_pool=multi_pool,
        rs_planner=rs_planner,
        seed=0,
    )
    env.rs_potential = oracle

    all_stats = {}

    for stage in [1, 2, 3, 4]:
        env.set_active_stage(stage)
        stage_attempts = []
        stage_latched = 0

        for ep in range(episodes_per_stage):
            obs, reset_info = env.reset()
            goal_mode = reset_info.get("goal_orientation_mode", "?")
            done = False
            while not done:
                raw_action, _, _ = agent.act(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(raw_action)
                done = terminated or truncated

            rs_info = oracle.diagnostics()
            rs_latched = bool(rs_info.get("rs_latched", False))
            rs_fail = str(rs_info.get("rs_fail_reason", ""))
            rs_attempt = int(rs_info.get("rs_attempt_count", 0))
            total_time_ms = float(rs_info.get("rs_plan_time_ms_max", 0.0))
            candidate_cnt = int(rs_info.get("rs_candidate_count", 0))
            checked_cnt = int(rs_info.get("rs_checked_candidates", 0))
            collision_checks = int(rs_info.get("rs_collision_checks", 0))
            sample_cnt = int(rs_info.get("rs_sample_count", 0))

            if rs_attempt > 0:
                stage_attempts.append({
                    "mode": goal_mode,
                    "latched": rs_latched,
                    "fail_reason": rs_fail,
                    "candidate_count": candidate_cnt,
                    "checked_candidates": checked_cnt,
                    "collision_checks": collision_checks,
                    "sample_count": sample_cnt,
                    "total_time_ms": total_time_ms,
                })
            if rs_latched:
                stage_latched += 1

            oracle.reset()

        headin_attempted = [a for a in stage_attempts if a["mode"] == "head_in"]

        def summarize(entries):
            if not entries:
                return {"n": 0}
            latched = sum(1 for e in entries if e["latched"])
            reasons = defaultdict(int)
            for e in entries:
                reasons[e["fail_reason"]] += 1
            candidate_counts = [e["candidate_count"] for e in entries if e["candidate_count"] > 0]
            checked = [e["checked_candidates"] for e in entries]
            times = [e["total_time_ms"] for e in entries]
            coll_checks = [e["collision_checks"] for e in entries]
            return {
                "n": len(entries),
                "latched": latched,
                "latch_rate": latched / max(len(entries), 1),
                "reasons": dict(reasons),
                "avg_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0,
                "max_candidates": int(max(candidate_counts)) if candidate_counts else 0,
                "avg_checked": float(np.mean(checked)) if checked else 0,
                "max_checked": int(max(checked)) if checked else 0,
                "avg_time_ms": float(np.mean(times)) if times else 0,
                "max_time_ms": float(max(times)) if times else 0,
                "avg_collision_checks": float(np.mean(coll_checks)) if coll_checks else 0,
            }

        all_stats[stage] = {
            "head_in": summarize(headin_attempted),
            "total_episodes": episodes_per_stage,
            "attempt_count": len(stage_attempts),
        }

    return all_stats


def print_banner(name):
    print()
    print("=" * 80)
    print(f"  {name}")
    print("=" * 80)


def print_compare(a, b, stages):
    for stage in stages:
        print()
        print(f"--- Stage {stage} ---")
        for mode in ("head_in",):
            sa = a[stage][mode]
            sb = b[stage][mode]
            label2 = f"limit=2"
            label10 = f"limit=10"
            if sa["n"] == 0 and sb["n"] == 0:
                continue
            print(f"  [{mode}]")
            print(f"    {'':>15}  {'limit=2':>12}  {'limit=10':>12}  {'chg%':>10}")
            print(f"    {'attempts':>15}  {sa['n']:>12}  {sb['n']:>12}  {'':>10}")
            lr2 = sa["latch_rate"] * 100
            lr10 = sb["latch_rate"] * 100
            lr_chg = (lr10 - lr2)
            print(f"    {'latch_rate':>15}  {lr2:>11.1f}%  {lr10:>11.1f}%  {lr_chg:>+9.1f}%")
            for reason in sorted(set(list(sa["reasons"].keys()) + list(sb["reasons"].keys()))):
                r2 = sa["reasons"].get(reason, 0)
                r10 = sb["reasons"].get(reason, 0)
                r2p = r2 / max(sa["n"], 1) * 100
                r10p = r10 / max(sb["n"], 1) * 100
                chg = r10p - r2p
                print(f"    {'  ' + reason:>20}  {r2:>5d} ({r2p:>5.1f}%)  {r10:>5d} ({r10p:>5.1f}%)  {chg:>+9.1f}%")
            t2 = sa["avg_time_ms"]
            t10 = sb["avg_time_ms"]
            t_chg = (t10 - t2) / max(t2, 0.001) * 100
            print(f"    {'avg_time_ms':>15}  {t2:>12.3f}  {t10:>12.3f}  {t_chg:>+9.1f}%")
            c2 = sa["avg_collision_checks"]
            c10 = sb["avg_collision_checks"]
            c_chg = (c10 - c2) / max(c2, 0.001) * 100
            print(f"    {'avg_coll_checks':>15}  {c2:>12.1f}  {c10:>12.1f}  {c_chg:>+9.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_final.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidates", type=int, nargs="+", default=[2, 10],
                        help="Candidate limits to test (default: 2 10)")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, args.checkpoint)

    results = {}
    for cl in args.candidates:
        print_banner(f"Testing candidate_limit = {cl}")
        stats = evaluate_rs(
            checkpoint_path=ckpt_path,
            device=args.device,
            episodes_per_stage=args.episodes,
            candidate_limit=cl,
        )
        for stage in sorted(stats):
            s = stats[stage]
            h = s["head_in"]
            print(f"  Stage {stage}:")
            print(f"    head_in:  n={h['n']:3d} latched={h['latched']:3d} ({h['latch_rate']*100:5.1f}%) "
                  f"avg_time={h['avg_time_ms']:.3f}ms")
        results[cl] = stats

    if len(results) >= 2:
        print_banner("COMPARISON TABLE")
        print_compare(results[args.candidates[0]], results[args.candidates[1]], sorted(results[args.candidates[0]]))


if __name__ == "__main__":
    main()
