"""
Analyze the distribution of minimum distances between the vehicle (at sampled
initial states) and scene obstacles, across all 4 curriculum stages.

Reports:
  - Per-stage distance statistics (min, p5, p25, median, p75, p95, max)
  - Per-stage collision/rejection rates
  - Per-stage histogram data (easy to plot)
  - Clearance bucket breakdown per stage
"""
import argparse
import json
import math
import sys
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from config import (
    DEFAULT_SCENE_CONFIG,
    DEFAULT_VEHICLE_PARAMS,
    DEFAULT_ENV_CONFIG,
)
from env.geometry import wrap_to_pi
from env.articulated_action_mask import ArticulatedActionMask
from env.local_parking_env import LocalParkingEnv
from env.mixing_plant_scene import generate_cached_mixing_plant_scene, _bucket_clearance
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


def _sample_state(stage, goal, vehicle_model, scene, rng, config):
    """Mirror _sample_initial_state sampling logic without env dependency."""
    index = stage - 1
    distance_range = config.stage_distance_ranges[index]
    lateral_range = float(config.stage_lateral_ranges[index])
    heading_range = math.radians(config.stage_heading_ranges_deg[index])
    phi_range = math.radians(config.stage_phi_ranges_deg[index])

    axis = np.asarray(
        [math.cos(goal.theta_goal), math.sin(goal.theta_goal)],
        dtype=np.float64,
    )
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    ref_center = np.asarray(goal.center)

    distance = rng.uniform(*distance_range)
    lateral = rng.uniform(-lateral_range, lateral_range)
    heading_error = rng.uniform(-heading_range, heading_range)
    phi = rng.uniform(-phi_range, phi_range)

    scenario = {
        1: "near_goal",
        2: "near_goal_obstacles",
        3: "poor_terminal_pose",
        4: "recovery",
    }[stage]

    if stage == 3:
        pose_mode = int(rng.integers(0, 3))
        if pose_mode == 0:
            min_heading = math.radians(config.poor_pose_min_heading_deg)
            heading_error = math.copysign(
                rng.uniform(min_heading, heading_range),
                rng.choice([-1.0, 1.0]),
            )
            scenario = "poor_terminal_heading"
        elif pose_mode == 1:
            min_lateral = min(
                float(config.poor_pose_min_lateral),
                lateral_range,
            )
            lateral = math.copysign(
                rng.uniform(min_lateral, lateral_range),
                rng.choice([-1.0, 1.0]),
            )
            scenario = "poor_terminal_lateral"
        else:
            min_phi = math.radians(config.poor_pose_min_abs_phi_deg)
            phi = math.copysign(
                rng.uniform(min_phi, phi_range),
                rng.choice([-1.0, 1.0]),
            )
            scenario = "poor_terminal_articulation"

    if stage == 4:
        min_phi = math.radians(config.recovery_min_abs_phi_deg)
        lateral = math.copysign(
            rng.uniform(min(1.8, lateral_range), lateral_range),
            rng.choice([-1.0, 1.0]),
        )
        phi = math.copysign(
            rng.uniform(min_phi, phi_range),
            rng.choice([-1.0, 1.0]),
        )

    center = ref_center - distance * axis + lateral * normal
    theta_front = wrap_to_pi(goal.theta_goal + heading_error)
    state = ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(theta_front),
        theta_rear=float(wrap_to_pi(theta_front - phi)),
    )

    # Collision check
    front_box, rear_box = vehicle_model.body_boxes(state)
    collision = bool(
        scene.prepared_obstacles.intersects(front_box)
        or scene.prepared_obstacles.intersects(rear_box)
    )
    if collision:
        return None, "collision", scenario

    # Body clearance
    clearance = float(
        min(
            front_box.distance(scene.obstacle_union),
            rear_box.distance(scene.obstacle_union),
        )
    )
    return {"clearance": clearance, "scenario": scenario}, None, scenario


def _percentile_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "p5": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _stage4_recovery_bucket(clearance):
    clearance = float(clearance)
    if clearance < 0.30:
        return "tight_recover"
    if clearance < 0.50:
        return "narrow_recover"
    if clearance <= DEFAULT_ENV_CONFIG.recovery_max_body_clearance:
        return "moderate_recover"
    return "open"


def audit_env_resets(
    seeds_per_stage=50,
    resets_per_scene=4,
    config=None,
    reset_retries=None,
):
    if config is None:
        config = DEFAULT_ENV_CONFIG
    action_mask = ArticulatedActionMask.load()
    result = {}
    print("\n" + "=" * 70)
    print("ENV.RESET CANDIDATE-BANK AUDIT")
    print("=" * 70)
    for stage in [1, 2, 3, 4]:
        stage_config = replace(
            config,
            curriculum_stage=stage,
            scene_pool_size=1,
            reset_scene_retry_count=(
                int(reset_retries)
                if reset_retries is not None
                else int(config.reset_scene_retry_count)
            ),
        )
        attempts = 0
        successes = 0
        failures = 0
        collisions = 0
        mask_all_zero = 0
        bank_valid_counts = []
        bank_sizes = []
        clearances = []
        selected_clearance_buckets = defaultdict(int)
        selected_pose_buckets = defaultdict(int)
        corridor_widths = []
        obstacle_feature_counts = []
        scene_retry_counts = []
        failure_examples = []
        for seed in range(int(seeds_per_stage)):
            try:
                env = LocalParkingEnv(
                    config=stage_config,
                    action_mask=action_mask,
                    seed=seed,
                )
            except Exception as exc:
                failures += int(resets_per_scene)
                failure_examples.append("env_init seed {}: {}".format(seed, exc))
                continue
            for _ in range(int(resets_per_scene)):
                attempts += 1
                try:
                    _, info = env.reset()
                except Exception as exc:
                    failures += 1
                    if len(failure_examples) < 5:
                        failure_examples.append("reset seed {}: {}".format(seed, exc))
                    continue
                successes += 1
                collisions += int(bool(info.get("initial_collision", False)))
                mask_all_zero += int(
                    bool(info.get("initial_mask_all_zero_before_floor", False))
                )
                bank_valid_counts.append(
                    int(info.get("reset_candidate_bank_valid_count", 0))
                )
                bank_sizes.append(int(info.get("reset_candidate_bank_size", 0)))
                clearances.append(float(info.get("reset_initial_body_clearance_m", 0.0)))
                selected_clearance_buckets[
                    str(info.get("reset_candidate_selected_clearance_bucket", ""))
                ] += 1
                selected_pose_buckets[
                    str(info.get("reset_candidate_selected_pose_bucket", ""))
                ] += 1
                corridor_widths.append(float(info.get("corridor_width", 0.0)))
                obstacle_feature_counts.append(
                    int(info.get("constructed_obstacle_feature_count", 0))
                )
                scene_retry_counts.append(int(info.get("reset_scene_retry_count", 0)))

        success_rate = successes / float(max(1, attempts))
        collision_rate = collisions / float(max(1, successes))
        mask_zero_rate = mask_all_zero / float(max(1, successes))
        stage_result = {
            "attempts": int(attempts),
            "successes": int(successes),
            "failures": int(failures),
            "success_rate": float(success_rate),
            "collision_rate": float(collision_rate),
            "mask_all_zero_before_floor_rate": float(mask_zero_rate),
            "reset_candidate_bank_valid_count": _percentile_summary(bank_valid_counts),
            "reset_candidate_bank_size": _percentile_summary(bank_sizes),
            "body_clearance_m": _percentile_summary(clearances),
            "selected_clearance_buckets": dict(selected_clearance_buckets),
            "selected_pose_buckets": dict(selected_pose_buckets),
            "corridor_widths": sorted(set(round(value, 3) for value in corridor_widths)),
            "constructed_obstacle_feature_counts": sorted(
                set(obstacle_feature_counts)
            ),
            "scene_retry_count": _percentile_summary(scene_retry_counts),
            "failure_examples": failure_examples,
        }
        if stage == 4:
            recovery_buckets = defaultdict(int)
            for clearance in clearances:
                recovery_buckets[_stage4_recovery_bucket(clearance)] += 1
            stage_result["stage4_clearance_bucket_distribution"] = dict(
                recovery_buckets
            )
        result[str(stage)] = stage_result

        print("\n--- Stage {} env.reset ---".format(stage))
        print(
            "  success_rate={:.3f} failures={} collision_rate={:.3f} mask_all_zero={:.3f}".format(
                success_rate,
                failures,
                collision_rate,
                mask_zero_rate,
            )
        )
        print(
            "  bank_valid median={} p5={} p95={} bank_size median={}".format(
                stage_result["reset_candidate_bank_valid_count"]["median"],
                stage_result["reset_candidate_bank_valid_count"]["p5"],
                stage_result["reset_candidate_bank_valid_count"]["p95"],
                stage_result["reset_candidate_bank_size"]["median"],
            )
        )
        print("  body_clearance:", stage_result["body_clearance_m"])
        print("  selected_clearance_buckets:", dict(selected_clearance_buckets))
        print("  selected_pose_buckets:", dict(selected_pose_buckets))
        print("  corridor_widths:", stage_result["corridor_widths"])
        print("  constructed_obstacle_feature_counts:", stage_result["constructed_obstacle_feature_counts"])
        if failure_examples:
            print("  failures:", failure_examples[:3])
    return result


def analyze(seeds_per_stage=50, samples_per_scene=200, config=None):
    if config is None:
        config = DEFAULT_ENV_CONFIG

    vehicle_model = ArticulatedVehicleModel(DEFAULT_VEHICLE_PARAMS)
    rng = np.random.default_rng(42)

    all_results = {}  # stage -> list of clearance values
    all_rejections = {}  # stage -> dict of reject reason counts
    all_details = {}  # stage -> list of per-scene stats

    for stage in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Stage {stage}")
        print(f"{'='*60}")

        clearances = []
        per_scene_stats = []
        total_attempts = 0
        total_collision = 0
        total_valid = 0

        for seed in range(seeds_per_stage):
            scene = generate_cached_mixing_plant_scene(
                stage=stage,
                seed=seed,
                scene_config=DEFAULT_SCENE_CONFIG,
                task_family="head_in",
            )

            slot = scene.slot
            scene_clearances = []
            scene_attempts = 0
            scene_collision = 0

            while scene_attempts < samples_per_scene:
                result, reject, scenario = _sample_state(
                    stage, slot, vehicle_model, scene, rng, config
                )
                scene_attempts += 1
                if reject == "collision":
                    scene_collision += 1
                    continue
                scene_clearances.append(result["clearance"])

            total_attempts += scene_attempts
            total_collision += scene_collision
            total_valid += len(scene_clearances)

            clearances.extend(scene_clearances)

            if scene_clearances:
                arr = np.array(scene_clearances)
                per_scene_stats.append({
                    "seed": seed,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "attempts": scene_attempts,
                    "collision": scene_collision,
                    "valid": len(scene_clearances),
                })

            if (seed + 1) % 10 == 0:
                print(f"  seeds 0..{seed}: {total_valid} valid / {total_attempts} "
                      f"attempts ({100*total_collision/total_attempts:.1f}% coll)")

        all_results[stage] = np.array(clearances)
        all_rejections[stage] = {
            "attempts": total_attempts,
            "collision": total_collision,
            "valid": total_valid,
            "collision_rate": total_collision / total_attempts if total_attempts else 0,
        }
        all_details[stage] = per_scene_stats

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("DISTANCE DISTRIBUTION SUMMARY")
    print("=" * 70)

    for stage in [1, 2, 3, 4]:
        arr = all_results[stage]
        reject = all_rejections[stage]
        if len(arr) == 0:
            print(f"\nStage {stage}: NO VALID SAMPLES")
            continue

        print(f"\n--- Stage {stage} ---")
        print(f"  Scenes:          {seeds_per_stage}")
        print(f"  Samples/scene:   {samples_per_scene}")
        print(f"  Total attempts:  {reject['attempts']}")
        print(f"  Collision rate:  {100*reject['collision_rate']:.2f}%")
        print(f"  Valid samples:   {reject['valid']}")
        print(f"")
        print(f"  Distance percentiles (m):")
        for pct, label in [(0, "min"), (5, "p5"), (10, "p10"), (25, "p25"),
                           (50, "median"), (75, "p75"), (90, "p90"),
                           (95, "p95"), (100, "max")]:
            print(f"    {label:>6s}: {np.percentile(arr, pct):.4f}")
        print(f"  mean:   {np.mean(arr):.4f}")
        print(f"  std:    {np.std(arr):.4f}")

        # Clearance bucket breakdown
        buckets = defaultdict(int)
        for v in arr:
            buckets[_bucket_clearance(float(v))] += 1
        print(f"\n  Clearance bucket distribution:")
        for b in ["tight", "narrow", "normal", "open"]:
            cnt = buckets.get(b, 0)
            pct = 100 * cnt / len(arr) if len(arr) else 0
            print(f"    {b:>7s}: {cnt:6d} ({pct:5.1f}%)")

        # Per-scene worst case
        scene_mins = [s["min"] for s in all_details[stage]]
        print(f"\n  Per-scene body clearance (min across seeds):")
        print(f"    min:  {np.min(scene_mins):.4f} m")
        print(f"    p5:   {np.percentile(scene_mins, 5):.4f} m")
        print(f"    p25:  {np.percentile(scene_mins, 25):.4f} m")
        print(f"    median: {np.median(scene_mins):.4f} m")

    # --- Export JSON for plotting ---
    output = {
        "config": {
            "seeds_per_stage": seeds_per_stage,
            "samples_per_scene": samples_per_scene,
        },
        "per_stage": {},
    }
    for stage in [1, 2, 3, 4]:
        arr = all_results.get(stage, np.array([]))
        output["per_stage"][str(stage)] = {
            "summary": {
                "n": len(arr),
                "min": float(np.min(arr)) if len(arr) else None,
                "max": float(np.max(arr)) if len(arr) else None,
                "mean": float(np.mean(arr)) if len(arr) else None,
                "std": float(np.std(arr)) if len(arr) else None,
                "p5": float(np.percentile(arr, 5)) if len(arr) else None,
                "p25": float(np.percentile(arr, 25)) if len(arr) else None,
                "p50": float(np.percentile(arr, 50)) if len(arr) else None,
                "p75": float(np.percentile(arr, 75)) if len(arr) else None,
                "p95": float(np.percentile(arr, 95)) if len(arr) else None,
            },
            "rejections": all_rejections.get(stage, {}),
            "clearance_percentiles": {
                str(p): float(np.percentile(arr, p)) for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            } if len(arr) else {},
            # Histogram bins (0 to 10m in 0.1m steps)
            "histogram": {
                "edges": list(np.arange(0, 10.1, 0.1)),
                "counts": list(np.histogram(arr, bins=np.arange(0, 10.1, 0.1))[0].astype(int)),
            } if len(arr) else {},
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Analyze vehicle-obstacle distance distribution across scene stages."
    )
    parser.add_argument("--seeds", type=int, default=50,
                        help="Number of scene seeds per stage (default: 50)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Initial state samples per scene (default: 200)")
    parser.add_argument("--env-reset-samples", type=int, default=4,
                        help="env.reset samples per scene seed; set 0 to skip (default: 4)")
    parser.add_argument("--reset-retries", type=int, default=None,
                        help="Override LocalParkingEnv reset scene retries for audit")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    result = analyze(
        seeds_per_stage=args.seeds,
        samples_per_scene=args.samples,
    )
    if args.env_reset_samples > 0:
        result["env_reset_audit"] = audit_env_resets(
            seeds_per_stage=args.seeds,
            resets_per_scene=args.env_reset_samples,
            reset_retries=args.reset_retries,
        )

    if args.output:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, cls=NpEncoder)
        print(f"\nJSON output written to {args.output}")


if __name__ == "__main__":
    main()
