"""
Evaluate checkpoint across controlled initial vehicle positions, distances,
and scenes. Reports steps-to-success for each configuration.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG, DEFAULT_VEHICLE_PARAMS
from env.geometry import wrap_to_pi
from env.local_parking_env import LocalParkingEnv
from env.vehicle import ArticulatedState
from model.continuous_ppo import ContinuousPPOAgent
from train.curriculum import MultiStageScenePool

TASK_FAMILIES = ("head_in", "parallel_fwd", "parallel_rev")
STAGES = (1, 2, 3, 4)

MAX_STEPS = 400


def _eval_config():
    from dataclasses import replace

    return replace(
        DEFAULT_ENV_CONFIG,
        use_hybrid_astar=False,
        rs_potential_enabled=False,
        enable_hope_teacher=False,
        use_teacher_reward=False,
        enable_offpath_reset=False,
        enable_failure_aggregation=False,
    )


def _task_family_from_scene(env):
    task_family = str(env.scene.metadata.get("task_family", ""))
    if task_family in TASK_FAMILIES:
        return task_family
    goal_mode = env.scene.metadata.get("goal_orientation_mode", "head_in")
    if goal_mode == "head_in":
        return "head_in"
    if bool(env.scene.metadata.get("parallel_reverse", False)):
        return "parallel_rev"
    return "parallel_fwd"


def _setup_env_with_state(env, scene, state):
    """Set env scene/slot/state and re-initialize all dependent components."""
    env.scene = scene
    env.slot = scene.slot
    env.state = state
    env.step_count = 0
    env.scenario_type = "custom_eval"
    env.prev_motion_gear = None
    env.prev_gear_in_obs = 0.0
    env._reset_hope_teacher()
    metrics = env._boxes_and_metrics()
    env.reward_model.reset(
        initial_distance=metrics["distance_to_goal"],
        initial_overlap=metrics["front_overlap"],
        initial_heading_error=metrics["heading_error"],
    )
    env.hybrid_reward.reset(env.scene, env.state, env.slot)
    env.rs_potential.reset()
    env._update_sensors_and_mask()
    obs = env._observation(metrics)
    info = env._base_info(metrics)
    info["scenario_type"] = "custom_eval"
    info["scene_seed"] = int(env.scene.metadata["seed"])
    info["task_family"] = _task_family_from_scene(env)
    info["goal_orientation_mode"] = str(
        env.scene.metadata.get("goal_orientation_mode", "")
    )
    info["fallback_used"] = False
    return obs, info


def _make_initial_state(env, distance, lateral_offset, heading_error_deg, phi_deg):
    """Create an ArticulatedState offset from the goal slot."""
    slot = env.slot
    goal_mode = str(env.scene.metadata.get("goal_orientation_mode", "head_in"))
    heading_error = math.radians(heading_error_deg)
    phi = math.radians(phi_deg)

    if goal_mode == "parallel":
        corridor_heading = float(env.scene.metadata["corridor_heading"])
        corridor_origin = np.asarray(env.scene.metadata["corridor_origin"])
        axis = np.asarray([math.cos(corridor_heading), math.sin(corridor_heading)])
        normal = np.asarray([-axis[1], axis[0]])
        delta = np.asarray(slot.center) - corridor_origin
        along_proj = float(np.dot(delta, axis))
        ref_center = corridor_origin + axis * along_proj
    else:
        axis = np.asarray([math.cos(slot.theta_goal), math.sin(slot.theta_goal)])
        normal = np.asarray([-axis[1], axis[0]])
        ref_center = np.asarray(slot.center)

    center = ref_center - distance * axis + lateral_offset * normal
    theta_front = wrap_to_pi(slot.theta_goal + heading_error)

    return ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(theta_front),
        theta_rear=float(wrap_to_pi(theta_front - phi)),
    )


def _state_is_valid_initial(env, state):
    """Check state is collision-free and within world bounds."""
    if env._state_collides(state):
        return False
    front_box, rear_box = env.vehicle_model.body_boxes(state)
    xmin, ymin, xmax, ymax = env.scene.world_bounds
    for polygon in (front_box, rear_box):
        bx0, by0, bx1, by1 = polygon.bounds
        if bx0 < xmin or by0 < ymin or bx1 > xmax or by1 > ymax:
            return False
    if abs(state.phi) > env.vehicle_params.phi_max + env.config.articulation_tolerance:
        return False
    return True


def run_episode(env, agent, state, device):
    """Run one deterministic episode from the given state. Returns result dict."""
    obs, _info = _setup_env_with_state(env, env.scene, state)

    done = False
    ep_steps = 0
    while not done:
        raw_action, _, _ = agent.act(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(raw_action)
        done = terminated or truncated
        ep_steps += 1

    return {
        "success": bool(info["success"]),
        "collision": bool(info["collision"]),
        "timeout": bool(info["timeout"]),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "articulation": bool(info.get("articulation_limit_violation", False)),
        "front_overlap": float(info["front_overlap"]),
        "heading_error_deg": float(info["heading_error_deg"]),
        "distance_to_goal": float(info["distance_to_goal"]),
        "episode_steps": ep_steps,
        "scene_seed": int(info["scene_seed"]),
        "task_family": str(info["task_family"]),
        "goal_orientation_mode": str(info.get("goal_orientation_mode", "")),
    }


def _get_scene_for_family(pool, family_name, scene_offset):
    """Get a scene of the requested family from the CachedScenePool.

    Since CachedScenePool.get(i) returns scenes[i % N] and scenes are
    built with families in a round-robin schedule, we find the first
    scene matching the target family, then offset from there.
    """
    n_scenes = len(pool._scenes)
    n_families = len(pool.family_schedule)
    # Find all indices matching this family
    matching_indices = []
    for i in range(n_scenes):
        scene = pool._scenes[i]
        fam = str(scene.metadata.get("task_family", ""))
        if fam == family_name:
            matching_indices.append(i)
    if not matching_indices:
        raise RuntimeError(
            f"No scenes of family '{family_name}' in pool (pool_size={n_scenes})"
        )
    idx = matching_indices[scene_offset % len(matching_indices)]
    return pool._scenes[idx]


def evaluate(checkpoint_path, scenes_per_family, seed, quick):
    device = "cpu"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()

    eval_config = _eval_config()
    multi_pool = MultiStageScenePool(
        pool_size=max(scenes_per_family * len(TASK_FAMILIES), 6),
        base_seed=int(seed),
        scene_config=DEFAULT_SCENE_CONFIG,
        family_schedule=eval_config.scene_family_schedule,
    )

    env = LocalParkingEnv(
        config=eval_config,
        multi_stage_pool=multi_pool,
        seed=int(seed),
    )

    if quick:
        distances = [6.0, 10.0, 15.0]
        laterals = [-1.5, 0.0, 1.5]
        headings_deg = [-15.0, 0.0, 15.0]
        phis_deg = [-10.0, 0.0, 10.0]
    else:
        distances = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
        laterals = [-3.0, -1.5, 0.0, 1.5, 3.0]
        headings_deg = [-30.0, -15.0, 0.0, 15.0, 30.0]
        phis_deg = [-20.0, -10.0, 0.0, 10.0, 20.0]

    all_results = []
    total_combos = (
        len(STAGES)
        * len(TASK_FAMILIES)
        * scenes_per_family
        * len(distances)
        * len(laterals)
        * len(headings_deg)
        * len(phis_deg)
    )
    print(f"Total state combinations to evaluate: {total_combos}")

    for stage in STAGES:
        env.set_active_stage(stage)
        pool = multi_pool.pool_for(stage)
        print(f"\n{'='*80}")
        print(f"Stage {stage}")
        print(f"{'='*80}")

        for family_name in TASK_FAMILIES:
            n_valid = 0
            n_invalid = 0
            n_success = 0
            for scene_idx in range(scenes_per_family):
                scene = _get_scene_for_family(pool, family_name, scene_idx)
                env.scene = scene
                env.slot = scene.slot

                for d_axis in distances:
                    for lat in laterals:
                        for hdg in headings_deg:
                            for phi in phis_deg:
                                state = _make_initial_state(env, d_axis, lat, hdg, phi)
                                if not _state_is_valid_initial(env, state):
                                    n_invalid += 1
                                    continue

                                result = run_episode(env, agent, state, device)
                                result["stage"] = stage
                                result["target_family"] = family_name
                                result["init_distance"] = d_axis
                                result["init_lateral"] = lat
                                result["init_heading_deg"] = hdg
                                result["init_phi_deg"] = phi
                                all_results.append(result)
                                n_valid += 1
                                if result["success"]:
                                    n_success += 1

                                    print(
                                        f"  [OK] S{stage} {family_name:>14} "
                                        f"d={d_axis:4.0f} lat={lat:+5.1f} "
                                        f"hdg={hdg:+5.0f} phi={phi:+5.0f} "
                                        f"-> steps={result['episode_steps']:3d}"
                                    )

            print(
                f"  Family {family_name}: valid={n_valid} invalid={n_invalid} "
                f"success={n_success} rate={n_success/max(n_valid,1)*100:.1f}%"
            )

    if hasattr(env, "close"):
        env.close()
    return all_results


def summarize(results):
    if not results:
        print("No results.")
        return

    print("\n" + "=" * 80)
    print("SUMMARY: Success rate by stage and initial distance")
    print("=" * 80)

    for stage in sorted(set(r["stage"] for r in results)):
        stage_results = [r for r in results if r["stage"] == stage]
        print(f"\n--- Stage {stage} ---")
        header = (
            f"{'Dist':>6}  {'Succ%':>8}  {'AvgSteps':>9}  "
            f"{'Min':>5}  {'Max':>5}  {'Count':>6}"
        )
        print(header)
        for dist in sorted(set(r["init_distance"] for r in stage_results)):
            dist_results = [r for r in stage_results if r["init_distance"] == dist]
            n = len(dist_results)
            succ = [r for r in dist_results if r["success"]]
            steps = [r["episode_steps"] for r in succ]
            succ_rate = len(succ) / n * 100 if n > 0 else 0
            if steps:
                mn, mx, avg = min(steps), max(steps), np.mean(steps)
            else:
                mn, mx, avg = 0, 0, float("nan")
            print(
                f"{dist:6.1f}  {succ_rate:7.1f}%  {avg:9.1f}  "
                f"{mn:5d}  {mx:5d}  {n:6d}"
            )

    print("\n" + "=" * 80)
    print("SUMMARY: Steps-to-success by distance (all stages)")
    print("=" * 80)
    succ_all = [r for r in results if r["success"]]
    for dist in sorted(set(r["init_distance"] for r in succ_all)):
        dist_results = [r for r in succ_all if r["init_distance"] == dist]
        steps = [r["episode_steps"] for r in dist_results]
        if steps:
            print(
                f"  d={dist:5.1f}m  n={len(steps):5d}  "
                f"mean={np.mean(steps):6.1f}  med={np.median(steps):6.1f}  "
                f"min={min(steps)}  max={max(steps)}"
            )

    print("\n" + "=" * 80)
    print("SUMMARY: By task family")
    print("=" * 80)
    for fam in TASK_FAMILIES:
        fam_results = [r for r in results if r["target_family"] == fam]
        n = len(fam_results)
        succ = [r for r in fam_results if r["success"]]
        steps = [r["episode_steps"] for r in succ]
        succ_rate = len(succ) / n * 100 if n > 0 else 0
        avg_steps = np.mean(steps) if steps else float("nan")
        col_rate = sum(1 for r in fam_results if r["collision"]) / n * 100 if n > 0 else 0
        print(
            f"  {fam:>14}  succ={succ_rate:5.1f}%  "
            f"coll={col_rate:5.1f}%  avg_steps={avg_steps:6.1f}  n={n}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint at controlled initial positions."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory containing checkpoint_final.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_final.pt",
    )
    parser.add_argument(
        "--scenes-per-family",
        type=int,
        default=1,
        help="Number of scenes per family per stage (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output filename (written to --run-dir)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer grid points",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scenes per family: {args.scenes_per_family}")
    print(f"Quick mode: {args.quick}")
    t0 = time.time()

    results = evaluate(
        checkpoint_path=checkpoint_path,
        scenes_per_family=args.scenes_per_family,
        seed=args.seed,
        quick=args.quick,
    )

    elapsed = time.time() - t0
    print(f"\nEvaluation done in {elapsed:.1f}s ({len(results)} episodes)")

    summarize(results)

    if args.output:
        output_path = os.path.join(args.run_dir, args.output)
        serializable = []
        for r in results:
            d = dict(r)
            for k, v in list(d.items()):
                if isinstance(v, (np.integer,)):
                    d[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    d[k] = float(v)
                elif isinstance(v, np.ndarray):
                    d[k] = v.tolist()
            serializable.append(d)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
