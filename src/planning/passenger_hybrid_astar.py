import heapq
import math

import numpy as np

from env.geometry import wrap_to_pi


class PassengerHybridAStar:
    """Small optional passenger-car guide; never used as a hard teacher."""

    def __init__(
        self,
        wheelbase=2.8,
        step_length=1.0,
        steering_angles=(-0.5, 0.0, 0.5),
        heading_bins=36,
        max_expansions=4000,
    ):
        self.wheelbase = float(wheelbase)
        self.step_length = float(step_length)
        self.steering_angles = tuple(float(v) for v in steering_angles)
        self.heading_bins = int(heading_bins)
        self.max_expansions = int(max_expansions)

    def _key(self, scene, state):
        x, y, theta = state
        xmin, ymin, _, _ = scene.world_bounds
        col = int((x - xmin) / scene.resolution)
        row = int((y - ymin) / scene.resolution)
        heading = int(
            round((wrap_to_pi(theta) + math.pi) / (2.0 * math.pi) * self.heading_bins)
        ) % self.heading_bins
        return row, col, heading

    @staticmethod
    def _heuristic(state, slot):
        return math.hypot(state[0] - slot.x_goal, state[1] - slot.y_goal)

    def plan(self, scene, state, slot):
        start = (float(state.x_front), float(state.y_front), float(state.theta_front))
        start_key = self._key(scene, start)
        queue = [(self._heuristic(start, slot), 0.0, start_key, start)]
        costs = {start_key: 0.0}
        parents = {start_key: None}
        states = {start_key: start}

        for _ in range(self.max_expansions):
            if not queue:
                return None
            _, cost, key, current = heapq.heappop(queue)
            if cost > costs.get(key, float("inf")) + 1e-9:
                continue
            if (
                self._heuristic(current, slot) <= 1.0
                and abs(wrap_to_pi(current[2] - slot.theta_goal)) <= math.radians(20.0)
            ):
                path = []
                cursor = key
                while cursor is not None:
                    path.append(states[cursor])
                    cursor = parents[cursor]
                return np.asarray(path[::-1], dtype=np.float32)

            for direction in (1.0, -1.0):
                for steering in self.steering_angles:
                    travel = direction * self.step_length
                    theta_next = wrap_to_pi(
                        current[2] + travel * math.tan(steering) / self.wheelbase
                    )
                    x_next = current[0] + travel * math.cos(current[2])
                    y_next = current[1] + travel * math.sin(current[2])
                    if scene.is_occupied_world(x_next, y_next):
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
        return None
