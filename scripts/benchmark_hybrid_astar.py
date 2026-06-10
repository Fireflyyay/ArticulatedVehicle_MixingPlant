#!/usr/bin/env python3
"""Benchmark Hybrid A* planning time across stages and seeds.

Usage:
    PYTHONPATH=src conda run -n HOPE python scripts/benchmark_hybrid_astar.py
"""

import math
import time
import statistics
from collections import Counter

import numpy as np

from config import DEFAULT_SCENE_CONFIG
from env.geometry import wrap_to_pi
from env.mixing_plant_scene import generate_cached_mixing_plant_scene
from env.vehicle import ArticulatedState
from planning.passenger_hybrid_astar import PassengerHybridAStar


def _sample_initial_state(scene, rng, stage):
    """Random feasible initial pose near target bay (replicates env logic)."""
    slot = scene.slot
    goal_mode = scene.metadata.get("goal_orientation_mode", "head_in")

    if goal_mode == "parallel":
        ch = float(scene.metadata["corridor_heading"])
        corg = np.asarray(scene.metadata["corridor_origin"])
        cw = float(scene.metadata["corridor_width"])
        axis = np.asarray([math.cos(ch), math.sin(ch)])
        normal = np.asarray([-axis[1], axis[0]])
        delta = np.asarray(slot.center) - corg
        along_proj = float(np.dot(delta, axis))
        ref_center = corg + axis * along_proj
        half_w = 1.0
        max_lat = max(0.3, cw / 2.0 - half_w - 0.3)
        effective_lat = min(4.0, max_lat)
    else:
        axis = np.asarray([math.cos(slot.theta_goal), math.sin(slot.theta_goal)])
        normal = np.asarray([-axis[1], axis[0]])
        ref_center = np.asarray(slot.center)
        effective_lat = 4.0

    dr = (8.0, 15.0)

    for _ in range(128):
        distance = rng.uniform(*dr)
        lateral = rng.uniform(-effective_lat, effective_lat)
        heading_error = rng.uniform(-math.radians(45), math.radians(45))
        phi = rng.uniform(-math.radians(12), math.radians(12))

        if stage == 3:
            mode = int(rng.integers(0, 3))
            if mode == 0:
                heading_error = math.copysign(
                    rng.uniform(math.radians(35), math.radians(45)),
                    rng.choice((-1.0, 1.0)),
                )
            elif mode == 1:
                lateral = math.copysign(
                    rng.uniform(1.5, effective_lat),
                    rng.choice((-1.0, 1.0)),
                )
            else:
                phi = math.copysign(
                    rng.uniform(math.radians(15), math.radians(30)),
                    rng.choice((-1.0, 1.0)),
                )

        if stage == 4:
            phi = math.copysign(
                rng.uniform(math.radians(18), math.radians(30)),
                rng.choice((-1.0, 1.0)),
            )
            lateral = math.copysign(
                rng.uniform(min(1.8, effective_lat), effective_lat),
                rng.choice((-1.0, 1.0)),
            )

        center = ref_center - distance * axis + lateral * normal
        theta_front = wrap_to_pi(slot.theta_goal + heading_error)
        state = ArticulatedState(
            x_front=float(center[0]),
            y_front=float(center[1]),
            theta_front=float(theta_front),
            theta_rear=float(wrap_to_pi(theta_front - phi)),
        )

        if not scene.is_occupied_world(state.x_front, state.y_front):
            return state

    fallback_dist = 6.0
    center = ref_center - fallback_dist * axis
    return ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(slot.theta_goal),
        theta_rear=float(slot.theta_goal),
    )


def main():
    planner = PassengerHybridAStar(max_expansions=4000)

    STAGES = [1, 2, 3, 4]
    SEEDS_PER_STAGE = 5
    TRIALS_PER_SCENE = 5

    all_by_stage = {}

    print("=" * 90)
    print("Hybrid A* Planning Benchmark  —  PassengerHybridAStar(max_expansions=4000)")
    print("=" * 90)
    print(f"{'Stage':>6}  {'Seed':>5}  {'Trials':>7}  {'Valid':>6}  "
          f"{'Mean(s)':>8}  {'Std(s)':>8}  {'Min(s)':>8}  {'Max(s)':>8}  "
          f"{'AvgExpand':>9}  {'Top Reason(s)'}")
    print("-" * 90)

    all_times = []
    all_valid = 0
    all_total = 0
    all_expanded = []

    for stage in STAGES:
        rng = np.random.default_rng(42 + stage)
        stage_times = []
        stage_valid = 0
        stage_expanded = []
        stage_reasons = Counter()

        for seed_idx in range(SEEDS_PER_STAGE):
            seed_id = seed_idx + (stage - 1) * SEEDS_PER_STAGE
            scene = generate_cached_mixing_plant_scene(
                stage=stage, seed=seed_id,
            )
            slot = scene.slot
            seed_times = []

            seed_valid = 0
            for _ in range(TRIALS_PER_SCENE):
                state = _sample_initial_state(scene, rng, stage)

                t0 = time.perf_counter()
                result = planner.plan_with_cost(scene, state, slot)
                elapsed = time.perf_counter() - t0

                seed_times.append(elapsed)
                stage_times.append(elapsed)
                all_times.append(elapsed)
                if result.valid:
                    stage_valid += 1
                    all_valid += 1
                    seed_valid += 1
                all_total += 1
                stage_expanded.append(result.expanded_nodes)
                all_expanded.append(result.expanded_nodes)
                stage_reasons[result.reason] += 1

            seed_mean = statistics.mean(seed_times)
            print(f"{stage:>6}  {seed_id:>5}  {TRIALS_PER_SCENE:>7}  "
                  f"{seed_valid:>6}  "
                  f"{seed_mean:>8.4f}  {'':>8}  "
                  f"{min(seed_times):>8.4f}  {max(seed_times):>8.4f}  "
                  f"{'':>9}  {'':>14}")

        n = len(stage_times)
        mean_t = statistics.mean(stage_times)
        std_t = statistics.stdev(stage_times) if n > 1 else 0.0
        min_t = min(stage_times)
        max_t = max(stage_times)
        mean_exp = statistics.mean(stage_expanded)
        top_reason = stage_reasons.most_common(1)[0]
        reason_str = f"{top_reason[0]}({top_reason[1]}/{n})"

        print(f"{'─' * 6}  {'─' * 5}  {'─' * 7}  {'─' * 6}  "
              f"{'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  "
              f"{'─' * 9}  {'─' * 14}")
        print(f"{'TOTAL':>6}  {SEEDS_PER_STAGE:>5}  {n:>7}  "
              f"{stage_valid:>6}  "
              f"{mean_t:>8.4f}  {std_t:>8.4f}  {min_t:>8.4f}  {max_t:>8.4f}  "
              f"{mean_exp:>9.0f}  {reason_str}")
        print()

        all_by_stage[stage] = {
            "n": n,
            "valid": stage_valid,
            "mean_time": mean_t,
            "std_time": std_t,
            "min_time": min_t,
            "max_time": max_t,
            "mean_expanded": mean_exp,
            "reasons": dict(stage_reasons),
        }

    print("=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)
    all_mean = statistics.mean(all_times)
    all_std = statistics.stdev(all_times) if len(all_times) > 1 else 0.0
    all_min = min(all_times)
    all_max = max(all_times)
    all_mean_exp = statistics.mean(all_expanded)
    valid_rate = all_valid / all_total * 100

    print(f"  Total trials:      {all_total}")
    print(f"  Valid plans:       {all_valid}/{all_total} ({valid_rate:.1f}%)")
    print(f"  Mean time:         {all_mean:.4f} s")
    print(f"  Std time:          {all_std:.4f} s")
    print(f"  Min time:          {all_min:.4f} s")
    print(f"  Max time:          {all_max:.4f} s")
    print(f"  Mean expansions:   {all_mean_exp:.0f}")
    print()

    # Stage-by-stage table
    print(f"{'Stage':>6}  {'Trials':>7}  {'Valid%':>7}  "
          f"{'Mean(s)':>8}  {'Std(s)':>8}  {'Min(s)':>8}  {'Max(s)':>8}  "
          f"{'AvgExpand':>9}")
    print("-" * 71)
    for stage in STAGES:
        r = all_by_stage[stage]
        vr = r["valid"] / r["n"] * 100
        print(f"{stage:>6}  {r['n']:>7}  {vr:>6.1f}%  "
              f"{r['mean_time']:>8.4f}  {r['std_time']:>8.4f}  "
              f"{r['min_time']:>8.4f}  {r['max_time']:>8.4f}  "
              f"{r['mean_expanded']:>9.0f}")
    print("-" * 71)
    print(f"{'ALL':>6}  {all_total:>7}  {valid_rate:>6.1f}%  "
          f"{all_mean:>8.4f}  {all_std:>8.4f}  "
          f"{all_min:>8.4f}  {all_max:>8.4f}  "
          f"{all_mean_exp:>9.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
