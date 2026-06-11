#!/usr/bin/env python3
"""Benchmark near-goal Reeds-Shepp candidate generation and validation.

Usage:
    PYTHONPATH=src conda run -n HOPE python scripts/benchmark_rs_planner.py
"""

import argparse
from collections import Counter
import math
import statistics

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS
from env.geometry import wrap_to_pi
from env.mixing_plant_scene import generate_cached_mixing_plant_scene
from env.rs_potential import RSPotentialPlanner
from env.vehicle import ArticulatedState
from planning.passenger_hybrid_astar import PassengerHybridAStar


def _percentile(values, percentile):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _latency_summary(records, field):
    values = [float(record[field]) for record in records]
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 95.0),
        "max": max(values),
    }


def _sample_state(scene, checker, rng):
    slot = scene.slot
    axis = np.asarray(
        [math.cos(slot.theta_goal), math.sin(slot.theta_goal)],
        dtype=np.float64,
    )
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    for _ in range(256):
        distance = rng.uniform(0.25, 9.95)
        lateral = rng.uniform(-2.0, 2.0)
        heading_error = rng.uniform(-math.radians(90.0), math.radians(90.0))
        center = np.asarray(slot.center) - distance * axis + lateral * normal
        theta = wrap_to_pi(slot.theta_goal + heading_error)
        state = ArticulatedState(
            x_front=float(center[0]),
            y_front=float(center[1]),
            theta_front=float(theta),
            theta_rear=float(theta),
        )
        if not checker._is_rectangle_occupied(
            scene, state.x_front, state.y_front, state.theta_front
        ):
            return state
    return ArticulatedState(
        x_front=float(slot.x_goal),
        y_front=float(slot.y_goal),
        theta_front=float(slot.theta_goal),
        theta_rear=float(slot.theta_goal),
    )


def _print_latency(label, records):
    print(label)
    for field, name in (
        ("generation_time_ms", "candidate generation"),
        ("collision_time_ms", "collision checking"),
        ("total_time_ms", "total planning"),
    ):
        stats = _latency_summary(records, field)
        print(
            "  {:20s} mean={:8.3f} median={:8.3f} p95={:8.3f} max={:8.3f} ms".format(
                name,
                stats["mean"],
                stats["median"],
                stats["p95"],
                stats["max"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--candidate-limit", type=int, default=2)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    vehicle = DEFAULT_VEHICLE_PARAMS
    checker = PassengerHybridAStar(
        front_half_length=0.5 * vehicle.front_body_length,
        front_half_width=0.5 * vehicle.front_body_width,
    )
    planner = RSPotentialPlanner(
        collision_checker=checker,
        turning_radius=vehicle.minimum_turning_radius,
        candidate_limit=args.candidate_limit,
        sample_step=checker.step_length / float(checker.intermediate_checks + 1),
    )
    rng = np.random.default_rng(args.seed)
    records = []
    reasons = Counter()

    for index in range(args.samples):
        scene = generate_cached_mixing_plant_scene(
            stage=args.stage,
            seed=int(args.seed + index),
        )
        state = _sample_state(scene, checker, rng)
        result = planner.plan(scene, state, scene.slot)
        record = {
            "mode": str(scene.metadata.get("goal_orientation_mode", "")),
            "success": bool(result.valid),
            "reason": str(result.reason),
            "generation_time_ms": float(result.generation_time_ms),
            "collision_time_ms": float(result.collision_time_ms),
            "total_time_ms": float(result.total_time_ms),
            "candidate_count": int(result.candidate_count),
            "checked_candidates": int(result.checked_candidates),
            "collision_checks": int(result.collision_checks),
            "sample_count": int(result.sample_count),
        }
        records.append(record)
        reasons[result.reason] += 1

    successes = [record for record in records if record["success"]]
    failures = [record for record in records if not record["success"]]
    print("RS planner benchmark")
    print(
        "  samples={} stage={} K={} success={}/{} ({:.1f}%)".format(
            args.samples,
            args.stage,
            args.candidate_limit,
            len(successes),
            len(records),
            100.0 * len(successes) / max(len(records), 1),
        )
    )
    print("  reasons={}".format(dict(reasons)))
    _print_latency("all cases", records)
    _print_latency("success cases", successes)
    _print_latency("failure cases", failures)

    for mode in ("head_in", "parallel"):
        selected = [record for record in records if record["mode"] == mode]
        if selected:
            _print_latency("{} cases (n={})".format(mode, len(selected)), selected)

    print("successful-path work")
    print(
        "  mean_sample_points={:.2f} mean_generated_candidates={:.2f} "
        "mean_checked_candidates={:.2f} mean_collision_checks={:.2f}".format(
            statistics.mean(
                [record["sample_count"] for record in successes] or [0.0]
            ),
            statistics.mean(
                [record["candidate_count"] for record in successes] or [0.0]
            ),
            statistics.mean(
                [record["checked_candidates"] for record in successes] or [0.0]
            ),
            statistics.mean(
                [record["collision_checks"] for record in successes] or [0.0]
            ),
        )
    )


if __name__ == "__main__":
    main()
