import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from shapely.geometry import Polygon


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def oriented_box(center, heading, length, width):
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    local = np.asarray(
        [
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c = math.cos(float(heading))
    s = math.sin(float(heading))
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    world = local.dot(rotation.T) + np.asarray(center, dtype=np.float64)
    return Polygon(world)


def polygon_corners_in_frame(polygon, origin, heading):
    corners = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    delta = corners - np.asarray(origin, dtype=np.float64)
    c = math.cos(float(heading))
    s = math.sin(float(heading))
    world_to_local = np.asarray([[c, s], [-s, c]], dtype=np.float64)
    return delta.dot(world_to_local.T)


def overlap_ratio(current, target):
    target_area = float(target.area)
    if target_area <= 0.0:
        return 0.0
    return float(current.intersection(target).area / target_area)


@dataclass(frozen=True)
class DirectedParkingSlot:
    x_goal: float
    y_goal: float
    theta_goal: float
    front_body_length: float
    front_body_width: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_goal, self.y_goal)

    def front_box(self):
        return oriented_box(
            self.center,
            self.theta_goal,
            self.front_body_length,
            self.front_body_width,
        )

    def position_error_in_slot_frame(self, x_front, y_front):
        dx = float(x_front) - self.x_goal
        dy = float(y_front) - self.y_goal
        c = math.cos(self.theta_goal)
        s = math.sin(self.theta_goal)
        return np.asarray([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)

    def target_corners_in_ego_frame(self, x_front, y_front, theta_front):
        return polygon_corners_in_frame(
            self.front_box(),
            (x_front, y_front),
            theta_front,
        ).astype(np.float32)
