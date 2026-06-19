#!/usr/bin/env python3
import argparse
import math
import os
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from env.mixing_plant_scene import generate_cached_mixing_plant_scene  # noqa: E402


def _heading_bucket(theta):
    theta = float(theta) % math.pi
    if abs(theta) < 1e-6 or abs(theta - math.pi) < 1e-6:
        return "east_west"
    return "north_south"


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


def _render_scene(output_path, stage, seed):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=stage,
            scene_pool_size=1,
            scene_family_schedule=("head_in",),
        ),
        seed=seed,
    )
    _, info = env.reset(seed=seed)
    front_box, rear_box = env.vehicle_model.body_boxes(env.state)
    target = env.slot.front_box()
    metadata = env.scene.metadata

    fig, ax = plt.subplots(figsize=(9, 9))
    for obstacle in env.scene.obstacle_polygons:
        _plot_polygon(ax, obstacle, "#777777", "#555555")
    for bay in env.scene.parking_bays:
        _plot_bay(ax, bay)
    _plot_polygon(ax, target, "#8fd175", "#207020", alpha=0.55, linewidth=2.0)
    _plot_polygon(ax, rear_box, "#f4a261", "#9c4f15", alpha=0.85)
    _plot_polygon(ax, front_box, "#3a86ff", "#164a91", alpha=0.85)

    xmin, ymin, xmax, ymax = env.scene.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "stage={} | head_in | {} | {} | variant={} | features={}".format(
            stage,
            _heading_bucket(metadata["corridor_heading"]),
            metadata["approach_side_bucket"],
            metadata["obstacle_layout_variant"],
            metadata["constructed_obstacle_feature_count"],
        )
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return metadata, info


def _select_representative_seeds(scan_limit):
    targets = set()
    for stage in range(1, 5):
        for orientation in ("east_west", "north_south"):
            for side in ("left_bay", "right_bay"):
                for variant in range(8):
                    targets.add((stage, orientation, side, variant))

    selected = {}
    for seed in range(int(scan_limit)):
        if len(selected) == len(targets):
            break
        scene = generate_cached_mixing_plant_scene(
            stage=1,
            seed=seed,
            task_family="head_in",
        )
        orientation = _heading_bucket(scene.metadata["corridor_heading"])
        side = scene.metadata["approach_side_bucket"]
        variant = int(scene.metadata["obstacle_layout_variant"])
        for stage in range(1, 5):
            key = (stage, orientation, side, variant)
            if key not in selected:
                selected[key] = seed

    missing = sorted(targets.difference(selected))
    if missing:
        raise RuntimeError(
            "missing {} scene type keys after {} seeds: {}".format(
                len(missing),
                scan_limit,
                missing[:10],
            )
        )
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Render representative images for every discrete scene type."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "scenes"),
    )
    parser.add_argument("--scan-limit", type=int, default=20000)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    selected = _select_representative_seeds(args.scan_limit)
    manifest_path = os.path.join(output_dir, "scene_type_catalog_manifest.csv")
    with open(manifest_path, "w") as manifest:
        manifest.write(
            "file,stage,task_family,corridor_orientation,approach_side,"
            "obstacle_layout_variant,seed,complexity,constructed_obstacles,"
            "constructed_walls,parking_bays\n"
        )
        for key in sorted(selected):
            stage, orientation, side, variant = key
            seed = selected[key]
            filename = "type_stage{}_head_in_{}_{}_variant{}_seed{}.png".format(
                stage,
                orientation,
                side,
                variant,
                seed,
            )
            output_path = os.path.join(output_dir, filename)
            metadata, _ = _render_scene(output_path, stage, seed)
            manifest.write(
                "{},{},head_in,{},{},{},{},{},{},{},{}\n".format(
                    filename,
                    stage,
                    orientation,
                    side,
                    variant,
                    seed,
                    metadata["scene_complexity_bucket"],
                    metadata["constructed_obstacle_feature_count"],
                    metadata["constructed_wall_feature_count"],
                    metadata["parking_bay_count"],
                )
            )

    print("rendered {} scene type images".format(len(selected)))
    print("manifest {}".format(manifest_path))
    print("output_dir {}".format(output_dir))


if __name__ == "__main__":
    main()
