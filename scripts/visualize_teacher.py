#!/usr/bin/env python3
"""Visualize teacher planned trajectories for articulated vehicle parking.

Usage:
    PYTHONPATH=src conda run -n HOPE python scripts/visualize_teacher.py \
        --teacher lattice \
        --family parallel_rev \
        --stage 1 \
        --seeds 0 1 2 3 4
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG, DEFAULT_VEHICLE_PARAMS
from env.local_parking_env import LocalParkingEnv
from env.vehicle import ArticulatedVehicleModel
from env.geometry import wrap_to_pi
from teachers import ArticulatedLatticeTeacher, MultiAnchorTeacher


def plot_vehicle(ax, state, vehicle_model, color_front="#3a86ff", color_rear="#f4a261", alpha=1.0):
    front, rear = vehicle_model.body_boxes(state)
    fx, fy = front.exterior.xy
    rx, ry = rear.exterior.xy
    ax.fill(fx, fy, alpha=alpha * 0.6, color=color_front, edgecolor=color_front, linewidth=0.5)
    ax.fill(rx, ry, alpha=alpha * 0.6, color=color_rear, edgecolor=color_rear, linewidth=0.5)
    hinge_x = state.x_front - vehicle_model.params.front_center_to_hinge * math.cos(state.theta_front)
    hinge_y = state.y_front - vehicle_model.params.front_center_to_hinge * math.sin(state.theta_front)
    ax.plot(hinge_x, hinge_y, 'ko', markersize=3)


def main():
    p = argparse.ArgumentParser(description="Visualize teacher trajectories")
    p.add_argument("--teacher", default="lattice", choices=["lattice", "multi_anchor"])
    p.add_argument("--family", default="parallel_rev",
                   choices=["head_in", "parallel_fwd", "parallel_rev"])
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output-dir", default="outputs/teacher_viz")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vehicle_model = ArticulatedVehicleModel()

    if args.teacher == "multi_anchor":
        teacher = MultiAnchorTeacher()
    else:
        teacher = ArticulatedLatticeTeacher()

    for seed_val in args.seeds:
        env_config = DEFAULT_ENV_CONFIG
        from dataclasses import replace
        env_config = replace(env_config, curriculum_stage=args.stage, scene_pool_size=1,
                           scene_family_schedule=(args.family,), use_hybrid_astar=False,
                           rs_potential_enabled=False)
        env = LocalParkingEnv(config=env_config, seed=seed_val)
        env.reset()

        initial_state = env.state
        scene = env.scene
        slot = env.slot

        print(f"Planning seed={seed_val} family={args.family}...")
        result = teacher.plan_from_state(initial_state, scene, slot, env.vehicle_model)
        print(f"  Success: {result.success}, steps: {result.num_steps}, time: {result.planning_time_ms:.0f}ms")
        if result.success:
            print(f"  Final: overlap={result.final_overlap:.3f}, hdg_err={math.degrees(result.final_heading_error):.1f}°, "
                  f"phi={math.degrees(result.final_phi):.1f}°")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        xmin, ymin, xmax, ymax = scene.world_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for poly in scene.obstacle_polygons:
            ax.fill(*poly.exterior.xy, color="#777777", edgecolor="#555555", linewidth=0.5, alpha=0.7)

        target_front = slot.front_box()
        ax.fill(*target_front.exterior.xy, color="green", alpha=0.4, edgecolor="darkgreen", linewidth=1.0)

        if scene.target_bay:
            ax.fill(*scene.target_bay.polygon.exterior.xy, color="#f6bd60", alpha=0.2, 
                    edgecolor="#f6bd60", linewidth=0.5, linestyle="--")

        ax.annotate(
            "", xy=(slot.x_goal + 2.0 * math.cos(slot.theta_goal),
                    slot.y_goal + 2.0 * math.sin(slot.theta_goal)),
            xytext=(slot.x_goal, slot.y_goal),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        )

        plot_vehicle(ax, initial_state, vehicle_model, color_front="#ff006e", color_rear="#8338ec", alpha=0.8)

        if result.success and result.states:
            xs = [s.x_front for s in result.states]
            ys = [s.y_front for s in result.states]
            ax.plot(xs, ys, "b-", linewidth=1.0, alpha=0.7, label="Front body path")

            for i, s in enumerate(result.states):
                if i % max(1, len(result.states) // 20) == 0:
                    plot_vehicle(ax, s, vehicle_model, alpha=0.15)

            final_state = result.states[-1]
            plot_vehicle(ax, final_state, vehicle_model, color_front="#06d6a0", color_rear="#118ab2", alpha=0.9)

        ax.set_title(f"{args.teacher} | {args.family} | seed={seed_val} | "
                     f"success={result.success} | {result.num_steps} steps | "
                     f"{result.planning_time_ms:.0f}ms")

        fname = os.path.join(args.output_dir, f"{args.teacher}_{args.family}_stage{args.stage}_seed{seed_val}.png")
        fig.savefig(fname, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
