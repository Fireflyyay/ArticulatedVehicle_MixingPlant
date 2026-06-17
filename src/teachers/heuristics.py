import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS
from env.geometry import wrap_to_pi


class TeacherHeuristic:
    def __init__(
        self,
        vehicle_params=None,
        weights: Optional[Dict[str, float]] = None,
        grid_resolution: float = 1.0,
    ):
        self.p = vehicle_params or DEFAULT_VEHICLE_PARAMS
        self._default_weights = {
            "w_grid": 2.0,
            "w_pos": 1.0,
            "w_theta": 0.3,
            "w_phi": 0.15,
            "w_entry": 0.5,
            "w_anchor": 0.0,
        }
        self.weights = dict(self._default_weights)
        if weights:
            self.weights.update(weights)
        self.grid_resolution = float(grid_resolution)
        self._grid_cache: Dict[int, np.ndarray] = {}

    def configure_for_family(self, task_family: str):
        if task_family == "parallel_rev":
            self.weights.update({
                "w_grid": 2.5,
                "w_pos": 0.3,
                "w_theta": 0.3,
                "w_phi": 0.15,
                "w_entry": 0.8,
                "w_anchor": 2.0,
            })
        elif task_family == "parallel_fwd":
            self.weights.update({
                "w_grid": 2.0,
                "w_pos": 1.0,
                "w_theta": 0.5,
                "w_phi": 0.15,
                "w_entry": 0.5,
                "w_anchor": 0.5,
            })
        else:
            self.weights.update({
                "w_grid": 2.0,
                "w_pos": 1.0,
                "w_theta": 0.3,
                "w_phi": 0.15,
                "w_entry": 0.3,
                "w_anchor": 0.0,
            })

    def compute(
        self,
        x: float,
        y: float,
        theta_f: float,
        phi: float,
        slot,
        scene,
        anchors_xy: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        h_pos = math.hypot(x - slot.x_goal, y - slot.y_goal)
        h_theta = abs(wrap_to_pi(theta_f - slot.theta_goal))
        h_phi = abs(phi)
        h_grid = self._grid_distance(x, y, slot, scene)
        h_entry = self._entry_alignment(x, y, theta_f, phi, scene)
        h_anchor = 0.0
        if anchors_xy and self.weights.get("w_anchor", 0.0) > 0.0:
            goal_xy = (float(slot.x_goal), float(slot.y_goal))
            h_anchor = self._anchor_soft_min(
                (x, y), anchors_xy, goal_xy, scene
            )
        return (
            self.weights["w_grid"] * h_grid
            + self.weights["w_pos"] * h_pos
            + self.weights["w_theta"] * h_theta
            + self.weights["w_phi"] * h_phi
            + self.weights["w_entry"] * h_entry
            + self.weights["w_anchor"] * h_anchor
        )

    def _grid_distance(self, x, y, slot, scene):
        key = id(scene)
        if key not in self._grid_cache:
            self._grid_cache[key] = self._build_grid_bfs(slot, scene)
        bfs = self._grid_cache[key]
        xmin, ymin, xmax, ymax = scene.world_bounds
        col = int((float(x) - xmin) / self.grid_resolution)
        row = int((float(y) - ymin) / self.grid_resolution)
        w = bfs.shape[1]
        h = bfs.shape[0]
        if 0 <= col < w and 0 <= row < h:
            dist = bfs[row, col]
            if dist < float("inf"):
                return float(dist) * self.grid_resolution
        return math.hypot(x - slot.x_goal, y - slot.y_goal)

    def _build_grid_bfs(self, slot, scene):
        xmin, ymin, xmax, ymax = scene.world_bounds
        w = int((xmax - xmin) / self.grid_resolution)
        h = int((ymax - ymin) / self.grid_resolution)
        bfs = np.full((h, w), float("inf"), dtype=np.float32)
        goal_col = int((slot.x_goal - xmin) / self.grid_resolution)
        goal_row = int((slot.y_goal - ymin) / self.grid_resolution)
        if not (0 <= goal_col < w and 0 <= goal_row < h):
            return bfs
        q = deque()
        bfs[goal_row, goal_col] = 0.0
        q.append((goal_row, goal_col))
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while q:
            r, c = q.popleft()
            cur_dist = bfs[r, c]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bfs[nr, nc] > cur_dist + 1:
                    wx = xmin + nc * self.grid_resolution + self.grid_resolution * 0.5
                    wy = ymin + nr * self.grid_resolution + self.grid_resolution * 0.5
                    if not scene.is_occupied_world(wx, wy):
                        bfs[nr, nc] = cur_dist + 1.0
                        q.append((nr, nc))
        return bfs

    def _entry_alignment(self, x, y, theta_f, phi, scene):
        if not hasattr(scene, "target_bay") or scene.target_bay is None:
            return 0.0
        bay = scene.target_bay
        mx, my = bay.mouth_center
        d_mouth = math.hypot(x - mx, y - my)
        if d_mouth > 12.0:
            return 0.0
        heading_to_entry = math.atan2(my - y, mx - x)
        heading_error = abs(wrap_to_pi(theta_f - heading_to_entry))
        decay = max(0.0, 1.0 - d_mouth / 12.0)
        return decay * min(heading_error, math.pi)

    def _anchor_soft_min(
        self,
        current_xy: Tuple[float, float],
        anchors_xy: List[Tuple[float, float]],
        goal_xy: Tuple[float, float],
        scene,
    ):
        if not anchors_xy:
            return 0.0
        cx, cy = current_xy
        anchor_costs = []
        for ax, ay in anchors_xy:
            d_to_anchor = math.hypot(cx - ax, cy - ay)
            d_anchor_goal = math.hypot(ax - goal_xy[0], ay - goal_xy[1])
            anchor_costs.append(d_to_anchor + d_anchor_goal)
        if not anchor_costs:
            return 0.0
        best = min(anchor_costs)
        softmin_sum = 0.0
        weight_sum = 0.0
        temp = 1.0
        for c in anchor_costs:
            w = math.exp(-c / temp)
            softmin_sum += w * c
            weight_sum += w
        if weight_sum > 0:
            return softmin_sum / weight_sum
        return best

    def grid_cost_map(self, scene) -> np.ndarray:
        key = id(scene)
        if key not in self._grid_cache:
            return np.full(
                (
                    int((scene.world_bounds[3] - scene.world_bounds[1]) / self.grid_resolution),
                    int((scene.world_bounds[2] - scene.world_bounds[0]) / self.grid_resolution),
                ),
                float("inf"),
                dtype=np.float32,
            )
        return self._grid_cache[key] * self.grid_resolution


def compute_bay_entry_distance(x, y, scene) -> float:
    if not hasattr(scene, "target_bay"):
        return float("inf")
    bay = scene.target_bay
    mx, my = bay.mouth_center
    return math.hypot(x - mx, y - my)


def count_gear_switches(gears: List[int]) -> int:
    switches = 0
    for i in range(1, len(gears)):
        if gears[i] != gears[i - 1]:
            switches += 1
    return switches
