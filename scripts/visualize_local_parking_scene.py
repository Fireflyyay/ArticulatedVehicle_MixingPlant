#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG  # noqa: E402
from dataclasses import replace  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from env.mixing_plant_scene import SUPPORTED_SCENE_TYPES  # noqa: E402


def _plot_polygon(ax, polygon, facecolor, edgecolor, alpha=1.0, linewidth=1.0):
    coords = np.asarray(polygon.exterior.coords)
    ax.fill(
        coords[:, 0],
        coords[:, 1],
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
    )


def _plot_bay(ax, bay):
    is_target = bool(bay.is_target)
    _plot_polygon(
        ax,
        bay.polygon,
        "#f6bd60" if is_target else "#9ecae1",
        "#bc6c25" if is_target else "#3182bd",
        alpha=0.35 if is_target else 0.22,
        linewidth=2.2 if is_target else 1.4,
    )
    mouth = np.asarray(bay.mouth_segment, dtype=np.float64)
    ax.plot(
        mouth[:, 0],
        mouth[:, 1],
        color="#d62828" if is_target else "#3182bd",
        linewidth=3.0 if is_target else 1.5,
        linestyle="--",
        zorder=5,
    )
    center = np.asarray(bay.polygon.centroid.coords[0], dtype=np.float64)
    direction = np.asarray(
        [np.cos(bay.goal_heading), np.sin(bay.goal_heading)],
        dtype=np.float64,
    )
    ax.arrow(
        center[0],
        center[1],
        3.0 * direction[0],
        3.0 * direction[1],
        width=0.12,
        head_width=0.8,
        color="#9b2226" if is_target else "#22577a",
        length_includes_head=True,
        zorder=6,
    )


def _plot_obstacles(ax, scene):
    scene_type = str(scene.metadata.get("scene_type", ""))
    if scene_type == "loading_truck_rectangle_space":
        labels = tuple(scene.metadata.get("constructed_obstacle_labels", ()))
        for index, obstacle in enumerate(scene.obstacle_polygons):
            label = str(labels[index]) if index < len(labels) else ""
            if label == "truck_obstacle":
                _plot_polygon(ax, obstacle, "#d95f02", "#8c2d04", alpha=0.85)
            else:
                _plot_polygon(ax, obstacle, "#7a7a7a", "#555555", alpha=0.90)
        return
    for obstacle in scene.obstacle_polygons:
        _plot_polygon(ax, obstacle, "#777777", "#555555")


def _plot_scene_regions(ax, scene):
    if str(scene.metadata.get("scene_type", "")) != "mixing_station_bay_corridor":
        return
    bounds = scene.metadata.get("corridor_region_bounds")
    if not bounds:
        return
    x0, y0, x1, y1 = tuple(float(value) for value in bounds)
    _plot_polygon(
        ax,
        box(x0, y0, x1, y1),
        "#b7e4c7",
        "#2d6a4f",
        alpha=0.20,
        linewidth=1.4,
    )


def main():
    parser = argparse.ArgumentParser(description="Render a cached local parking scene.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        choices=SUPPORTED_SCENE_TYPES,
        default=DEFAULT_SCENE_CONFIG.scene_type,
    )
    parser.add_argument(
        "--task-family",
        choices=["head_in"],
        default="head_in",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-lidar", action="store_true")
    args = parser.parse_args()

    output = args.output or os.path.join(
        REPO_ROOT,
        "outputs",
        "scenes",
        "local_parking_{}_stage{}_{}_seed{}.png".format(
            args.scene_type,
            args.stage,
            args.task_family,
            args.seed,
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=args.stage,
            scene_pool_size=1,
            scene_family_schedule=(args.task_family,),
        ),
        scene_config=replace(DEFAULT_SCENE_CONFIG, scene_type=args.scene_type),
        seed=args.seed,
    )
    _, info = env.reset(seed=args.seed)
    front_box, rear_box = env.vehicle_model.body_boxes(env.state)
    target = env.slot.front_box()

    fig, ax = plt.subplots(figsize=(9, 9))
    _plot_scene_regions(ax, env.scene)
    _plot_obstacles(ax, env.scene)
    for bay in env.scene.parking_bays:
        _plot_bay(ax, bay)
    _plot_polygon(ax, target, "#8fd175", "#207020", alpha=0.55, linewidth=2.0)
    _plot_polygon(ax, rear_box, "#f4a261", "#9c4f15", alpha=0.85)
    _plot_polygon(ax, front_box, "#3a86ff", "#164a91", alpha=0.85)

    if args.show_lidar:
        rear_center = env.vehicle_model.rear_center(env.state)
        for center, heading, distances, color in (
            (
                (env.state.x_front, env.state.y_front),
                env.state.theta_front,
                env.last_front_lidar_m,
                "#2b6cb0",
            ),
            (rear_center, env.state.theta_rear, env.last_rear_lidar_m, "#b45309"),
        ):
            for beam_angle, distance in zip(env.lidar.beam_angles, distances):
                angle = heading + beam_angle
                end = (
                    center[0] + distance * np.cos(angle),
                    center[1] + distance * np.sin(angle),
                )
                ax.plot([center[0], end[0]], [center[1], end[1]], color=color, alpha=0.12)

    xmin, ymin, xmax, ymax = env.scene.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "{} | stage={} | {} | overlap={:.3f}".format(
            env.scene.metadata.get("scene_type", ""),
            args.stage,
            info["scenario_type"],
            info["front_overlap"],
        ),
        fontsize=10,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print("saved {}".format(output))


if __name__ == "__main__":
    main()
