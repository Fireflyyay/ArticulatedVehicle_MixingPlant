"""Reeds-Shepp candidate generation for local parking guidance.

The analytical path families and interpolation structure are adapted from
PythonRobotics' MIT-licensed Reeds-Shepp implementation:
https://github.com/AtsushiSakai/PythonRobotics

Copyright (c) 2016 - now Atsushi Sakai and other contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from env.geometry import wrap_to_pi


@dataclass
class ReedsSheppPath:
    lengths: List[float]
    segment_types: List[str]
    total_length: float
    poses: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    directions: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int8)
    )


def _mod2pi(angle):
    value = math.fmod(float(angle), math.copysign(2.0 * math.pi, float(angle)))
    if value < -math.pi:
        value += 2.0 * math.pi
    elif value > math.pi:
        value -= 2.0 * math.pi
    return value


def _polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def _left_straight_left(x, y, phi):
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = _mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]
    return False, [], []


def _left_straight_right(x, y, phi):
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1_sq = u1 * u1
    if u1_sq >= 4.0:
        u = math.sqrt(u1_sq - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, v], ["L", "S", "R"]
    return False, [], []


def _straight_left_straight(x, y, phi):
    """Extended S(|)C(|)S family used by HOPE-style candidate search."""
    phi = _mod2pi(phi)
    if not (0.0 < phi < math.pi * 0.99) or abs(math.tan(phi)) < 1e-12:
        return False, [], []
    x_intersection = -y / math.tan(phi) + x
    first = x_intersection - math.tan(phi / 2.0)
    diagonal = math.hypot(x - x_intersection, y)
    last = diagonal - math.tan(phi / 2.0)
    if y < 0.0:
        last = -diagonal - math.tan(phi / 2.0)
    return True, [first, phi, last], ["S", "L", "S"]


def _left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + math.pi / 2.0)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]
    return False, [], []


def _left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + math.pi / 2.0)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]
    return False, [], []


def _left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if 1e-12 < u1 <= 4.0:
        u = math.acos(1.0 - u1 * u1 * 0.125)
        ratio = 2.0 * math.sin(u) / u1
        ratio = max(-1.0, min(1.0, ratio))
        a = math.asin(ratio)
        t = _mod2pi(-a + theta + math.pi / 2.0)
        v = _mod2pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]
    return False, [], []


def _left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 <= 2.0:
        a = math.acos((u1 + 2.0) * 0.25)
        t = _mod2pi(theta + a + math.pi / 2.0)
        u = _mod2pi(a)
        v = _mod2pi(phi - t + 2.0 * u)
        if t >= 0.0 and u >= 0.0 and v >= 0.0:
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]
    return False, [], []


def _left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    u2 = (20.0 - u1 * u1) / 16.0
    if 1e-12 < u1 and 0.0 <= u2 <= 1.0:
        u = math.acos(u2)
        ratio = 2.0 * math.sin(u) / u1
        ratio = max(-1.0, min(1.0, ratio))
        a = math.asin(ratio)
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]
    return False, [], []


def _left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        root = math.sqrt(u1 * u1 - 4.0)
        u = root - 2.0
        a = math.atan2(2.0, root)
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi + math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -v], ["L", "R", "S", "L"]
    return False, [], []


def _left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        root = math.sqrt(u1 * u1 - 4.0)
        u = root - 2.0
        a = math.atan2(root, 2.0)
        t = _mod2pi(theta - a + math.pi / 2.0)
        v = _mod2pi(t - phi - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, math.pi / 2.0, -v], ["L", "S", "R", "L"]
    return False, [], []


def _left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        t = _mod2pi(theta + math.pi / 2.0)
        u = u1 - 2.0
        v = _mod2pi(phi - t - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -v], ["L", "R", "S", "R"]
    return False, [], []


def _left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 2.0:
        t = _mod2pi(theta)
        u = u1 - 2.0
        v = _mod2pi(phi - t - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, math.pi / 2.0, -v], ["L", "S", "L", "R"]
    return False, [], []


def _left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(zeta, eta)
    if u1 >= 4.0:
        root = math.sqrt(u1 * u1 - 4.0)
        u = root - 4.0
        a = math.atan2(2.0, root)
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return (
                True,
                [t, -math.pi / 2.0, -u, -math.pi / 2.0, v],
                ["L", "R", "S", "L", "R"],
            )
    return False, [], []


_PATH_FUNCTIONS = (
    _straight_left_straight,
    _left_straight_left,
    _left_straight_right,
    _left_x_right_x_left,
    _left_x_right_left,
    _left_right_x_left,
    _left_right_x_left_right,
    _left_x_right_left_x_right,
    _left_x_right90_straight_left,
    _left_x_right90_straight_right,
    _left_straight_right90_x_left,
    _left_straight_left90_x_right,
    _left_x_right90_straight_left90_x_right,
)


def _reflect(segment_types):
    return [
        "R" if segment == "L" else "L" if segment == "R" else "S"
        for segment in segment_types
    ]


def _append_path(paths, lengths, segment_types, min_length):
    total = float(sum(abs(value) for value in lengths))
    if total <= min_length:
        return
    for existing in paths:
        if existing.segment_types != segment_types:
            continue
        if abs(existing.total_length - total) <= min_length:
            return
    paths.append(
        ReedsSheppPath(
            lengths=[float(value) for value in lengths],
            segment_types=list(segment_types),
            total_length=total,
        )
    )


def _normalized_candidates(start, goal, max_curvature, sample_step):
    dx = float(goal[0]) - float(start[0])
    dy = float(goal[1]) - float(start[1])
    dtheta = float(goal[2]) - float(start[2])
    c = math.cos(float(start[2]))
    s = math.sin(float(start[2]))
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    min_length = max(float(sample_step) * max_curvature * 0.1, 1e-6)
    paths = []

    for path_function in _PATH_FUNCTIONS:
        transforms = (
            (x, y, dtheta, False, False),
            (-x, y, -dtheta, True, False),
            (x, -y, -dtheta, False, True),
            (-x, -y, dtheta, True, True),
        )
        for tx, ty, ttheta, time_flip, reflect in transforms:
            valid, lengths, segment_types = path_function(tx, ty, ttheta)
            if not valid:
                continue
            if time_flip:
                lengths = [-value for value in lengths]
            if reflect:
                segment_types = _reflect(segment_types)
            _append_path(paths, lengths, segment_types, min_length)
    return paths


def _interpolate(distance, segment_length, segment_type, max_curvature, origin):
    origin_x, origin_y, origin_yaw = origin
    if segment_type == "S":
        x = origin_x + distance / max_curvature * math.cos(origin_yaw)
        y = origin_y + distance / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:
        local_dx = math.sin(distance) / max_curvature
        if segment_type == "L":
            local_dy = (1.0 - math.cos(distance)) / max_curvature
            yaw = origin_yaw + distance
        else:
            local_dy = (1.0 - math.cos(distance)) / -max_curvature
            yaw = origin_yaw - distance
        x = (
            origin_x
            + math.cos(origin_yaw) * local_dx
            - math.sin(origin_yaw) * local_dy
        )
        y = (
            origin_y
            + math.sin(origin_yaw) * local_dx
            + math.cos(origin_yaw) * local_dy
        )
    direction = 1 if segment_length >= 0.0 else -1
    return float(x), float(y), float(yaw), direction


def _sample_local_path(lengths, segment_types, max_curvature, sample_step):
    normalized_step = float(sample_step) * max_curvature
    origin = (0.0, 0.0, 0.0)
    poses = []
    directions = []
    for segment_length, segment_type in zip(lengths, segment_types):
        step = normalized_step if segment_length >= 0.0 else -normalized_step
        distances = np.arange(0.0, segment_length, step, dtype=np.float64)
        distances = np.append(distances, segment_length)
        for distance in distances:
            x, y, yaw, direction = _interpolate(
                float(distance),
                float(segment_length),
                segment_type,
                max_curvature,
                origin,
            )
            poses.append((x, y, yaw))
            directions.append(direction)
        origin = poses[-1]
    return np.asarray(poses, dtype=np.float64), np.asarray(directions, dtype=np.int8)


def _to_world(local_poses, start):
    c = math.cos(float(start[2]))
    s = math.sin(float(start[2]))
    world = np.empty_like(local_poses)
    world[:, 0] = c * local_poses[:, 0] - s * local_poses[:, 1] + float(start[0])
    world[:, 1] = s * local_poses[:, 0] + c * local_poses[:, 1] + float(start[1])
    world[:, 2] = [
        wrap_to_pi(float(yaw) + float(start[2])) for yaw in local_poses[:, 2]
    ]
    return world


def generate_reeds_shepp_paths(
    start: Sequence[float],
    goal: Sequence[float],
    turning_radius: float,
    sample_step: float,
):
    """Return all analytical candidates, sampled and sorted by path length."""
    radius = float(turning_radius)
    if radius <= 0.0:
        raise ValueError("turning_radius must be positive")
    if float(sample_step) <= 0.0:
        raise ValueError("sample_step must be positive")
    max_curvature = 1.0 / radius
    paths = _normalized_candidates(start, goal, max_curvature, sample_step)
    for path in paths:
        local_poses, directions = _sample_local_path(
            path.lengths,
            path.segment_types,
            max_curvature,
            sample_step,
        )
        path.poses = _to_world(local_poses, start)
        path.directions = directions
        path.lengths = [float(value) / max_curvature for value in path.lengths]
        path.total_length = float(path.total_length) / max_curvature
    return sorted(paths, key=lambda item: item.total_length)
