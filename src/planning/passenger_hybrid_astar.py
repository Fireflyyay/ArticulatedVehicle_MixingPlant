import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.geometry import wrap_to_pi


@dataclass
class PlannerResult:
    """Result from a Hybrid A* planning attempt, used for potential-based shaping."""
    valid: bool
    path: Optional[np.ndarray]        # (N, 3) float32 debug / viz path, or None
    cost: float                        # total search cost from start to goal
    goal_cost: float                   # terminal residual portion of cost
    reason: str                        # "success" / "no_path" / "start_occupied" etc.
    expanded_nodes: int


class PassengerHybridAStar:
    """Small optional passenger-car guide; never used as a hard teacher."""

    def __init__(
        self,
        wheelbase=2.8,
        step_length=1.0,
        steering_angles=(-0.5, 0.0, 0.5),
        heading_bins=36,
        max_expansions=4000,
        goal_pos_tol=0.8,
        goal_heading_tol_deg=15.0,
        front_half_length=2.225,
        front_half_width=1.508,
        footprint_samples=3,
        intermediate_checks=2,
        rear_half_length=0.0,
        rear_half_width=0.0,
        front_center_to_hinge=0.0,
        rear_center_to_hinge=0.0,
    ):
        self.wheelbase = float(wheelbase)
        self.step_length = float(step_length)
        self.steering_angles = tuple(float(v) for v in steering_angles)
        self.heading_bins = int(heading_bins)
        self.max_expansions = int(max_expansions)
        self.goal_pos_tol = float(goal_pos_tol)
        self.goal_heading_tol_rad = math.radians(float(goal_heading_tol_deg))
        self.front_half_length = float(front_half_length)
        self.front_half_width = float(front_half_width)
        self.footprint_samples = max(1, int(footprint_samples))
        self.intermediate_checks = max(0, int(intermediate_checks))
        self.rear_half_length = float(rear_half_length)
        self.rear_half_width = float(rear_half_width)
        self.front_center_to_hinge = float(front_center_to_hinge)
        self.rear_center_to_hinge = float(rear_center_to_hinge)

    @staticmethod
    def _wrap_angle(angle, bins):
        return int(round((wrap_to_pi(angle) + math.pi) / (2.0 * math.pi) * bins)) % bins

    def _key(self, scene, state):
        x, y, theta = state
        xmin, ymin, _, _ = scene.world_bounds
        col = int((x - xmin) / scene.resolution)
        row = int((y - ymin) / scene.resolution)
        heading = self._wrap_angle(theta, self.heading_bins)
        return row, col, heading

    @staticmethod
    def _heuristic(state, slot):
        return math.hypot(state[0] - slot.x_goal, state[1] - slot.y_goal)

    @staticmethod
    def _heading_error(state, slot):
        return abs(wrap_to_pi(state[2] - slot.theta_goal))

    def _rectangle_sample_points(self, x, y, theta, half_length=None, half_width=None):
        c = math.cos(theta)
        s = math.sin(theta)
        hl = half_length if half_length is not None else self.front_half_length
        hw = half_width if half_width is not None else self.front_half_width
        points = []
        n = self.footprint_samples
        for i in range(n):
            for j in range(n):
                lx = -hl + (2.0 * hl) * i / max(1, n - 1)
                ly = -hw + (2.0 * hw) * j / max(1, n - 1)
                wx = x + c * lx - s * ly
                wy = y + s * lx + c * ly
                points.append((wx, wy))
        return points

    def _is_rectangle_occupied(self, scene, x, y, theta):
        for px, py in self._rectangle_sample_points(x, y, theta):
            if scene.is_occupied_world(px, py):
                return True
        return False

    def _check_two_bodies(self, scene, x_f, y_f, theta_f, x_r, y_r, theta_r):
        for px, py in self._rectangle_sample_points(x_f, y_f, theta_f):
            if scene.is_occupied_world(px, py):
                return True
        if self.rear_half_length > 0:
            for px, py in self._rectangle_sample_points(
                x_r, y_r, theta_r,
                half_length=self.rear_half_length,
                half_width=self.rear_half_width,
            ):
                if scene.is_occupied_world(px, py):
                    return True
        return False

    def _motion_primitive_valid(self, scene, x0, y0, theta0, x1, y1, theta1):
        if self._is_rectangle_occupied(scene, x1, y1, theta1):
            return False
        if self.intermediate_checks == 0:
            return True
        for k in range(1, self.intermediate_checks + 1):
            alpha = k / (self.intermediate_checks + 1)
            xm = x0 + alpha * (x1 - x0)
            ym = y0 + alpha * (y1 - y0)
            thetam = wrap_to_pi(theta0 + alpha * wrap_to_pi(theta1 - theta0))
            if self._is_rectangle_occupied(scene, xm, ym, thetam):
                return False
        return True

    def _goal_reached(self, state, slot):
        return (
            self._heuristic(state, slot) <= self.goal_pos_tol
            and self._heading_error(state, slot) <= self.goal_heading_tol_rad
        )

    def plan_with_cost(self, scene, state, slot):
        if self._is_rectangle_occupied(
            scene, state.x_front, state.y_front, state.theta_front
        ):
            return PlannerResult(
                valid=False,
                path=None,
                cost=0.0,
                goal_cost=0.0,
                reason="start_occupied",
                expanded_nodes=0,
            )

        start = (float(state.x_front), float(state.y_front), float(state.theta_front))
        start_key = self._key(scene, start)
        queue = [(self._heuristic(start, slot), 0.0, start_key, start)]
        costs = {start_key: 0.0}
        parents = {start_key: None}
        states = {start_key: start}
        expanded = 0

        for _ in range(self.max_expansions):
            if not queue:
                return PlannerResult(
                    valid=False,
                    path=None,
                    cost=0.0,
                    goal_cost=0.0,
                    reason="no_path",
                    expanded_nodes=expanded,
                )
            _, cost, key, current = heapq.heappop(queue)
            if cost > costs.get(key, float("inf")) + 1e-9:
                continue
            expanded += 1

            if self._goal_reached(current, slot):
                path = []
                cur = key
                while cur is not None:
                    path.append(states[cur])
                    cur = parents[cur]
                path_arr = np.asarray(path[::-1], dtype=np.float32)
                # Compute terminal residual cost (heading error at goal region)
                terminal_residual_cost = self._heading_error(path[-1], slot)
                total_cost = costs[key] + terminal_residual_cost
                return PlannerResult(
                    valid=True,
                    path=path_arr,
                    cost=float(total_cost),
                    goal_cost=float(terminal_residual_cost),
                    reason="success",
                    expanded_nodes=expanded,
                )

            for direction in (1.0, -1.0):
                for steering in self.steering_angles:
                    travel = direction * self.step_length
                    theta_next = wrap_to_pi(
                        current[2] + travel * math.tan(steering) / self.wheelbase
                    )
                    x_next = current[0] + travel * math.cos(current[2])
                    y_next = current[1] + travel * math.sin(current[2])
                    if not self._motion_primitive_valid(
                        scene,
                        current[0], current[1], current[2],
                        x_next, y_next, theta_next,
                    ):
                        continue
                    successor = (x_next, y_next, theta_next)
                    successor_key = self._key(scene, successor)
                    new_cost = cost + self.step_length + 0.05 * abs(steering)
                    if new_cost >= costs.get(successor_key, float("inf")):
                        continue
                    costs[successor_key] = new_cost
                    parents[successor_key] = key
                    states[successor_key] = successor
                    priority = new_cost + self._heuristic(successor, slot)
                    heapq.heappush(
                        queue,
                        (priority, new_cost, successor_key, successor),
                    )
        return PlannerResult(
            valid=False,
            path=None,
            cost=0.0,
            goal_cost=0.0,
            reason="max_expansions",
            expanded_nodes=expanded,
        )

    def plan(self, scene, state, slot):
        result = self.plan_with_cost(scene, state, slot)
        if result.valid:
            return result.path
        return None
