import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.geometry import overlap_ratio, wrap_to_pi
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class HopeTeacherTrajectory:
    available: bool = False
    plan_success: bool = False
    fail_reason: str = "disabled"
    cache_hit: bool = False
    cache_key: str = ""
    planning_time_ms: float = 0.0
    code_loaded: bool = False
    weight_loaded: bool = False
    weight_keys: List[str] = field(default_factory=list)
    path_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=np.float32))
    path_length: float = 0.0
    num_gear_switches: int = 0
    anchor_xy: Optional[Tuple[float, float]] = None
    anchor_s: float = 0.0
    path_valid_after_articulated_check: bool = False
    weak_topology_only: bool = False
    reward_mode: str = "disabled"
    min_clearance: float = 0.0
    terminal_heading_error: float = 0.0
    terminal_overlap: float = 0.0
    terminal_position_error: float = 0.0
    terminal_heading_consistent: bool = False
    terminal_success_consistent: bool = False
    collision: bool = False
    out_of_bounds: bool = False
    reference_point: str = "front_body_center"
    coordinate_frame: str = "mixing_plant_world_xy_yaw"

    @property
    def reward_available(self):
        return self.plan_success and self.reward_mode in ("strong", "weak_topology")

    def to_json_dict(self):
        return {
            "available": bool(self.available),
            "plan_success": bool(self.plan_success),
            "fail_reason": str(self.fail_reason),
            "cache_key": str(self.cache_key),
            "planning_time_ms": float(self.planning_time_ms),
            "code_loaded": bool(self.code_loaded),
            "weight_loaded": bool(self.weight_loaded),
            "weight_keys": list(self.weight_keys),
            "path_points": self.path_points.astype(float).tolist(),
            "path_length": float(self.path_length),
            "num_gear_switches": int(self.num_gear_switches),
            "anchor_xy": list(self.anchor_xy) if self.anchor_xy is not None else None,
            "anchor_s": float(self.anchor_s),
            "path_valid_after_articulated_check": bool(
                self.path_valid_after_articulated_check
            ),
            "weak_topology_only": bool(self.weak_topology_only),
            "reward_mode": str(self.reward_mode),
            "min_clearance": float(self.min_clearance),
            "terminal_heading_error": float(self.terminal_heading_error),
            "terminal_overlap": float(self.terminal_overlap),
            "terminal_position_error": float(self.terminal_position_error),
            "terminal_heading_consistent": bool(self.terminal_heading_consistent),
            "terminal_success_consistent": bool(self.terminal_success_consistent),
            "collision": bool(self.collision),
            "out_of_bounds": bool(self.out_of_bounds),
            "reference_point": str(self.reference_point),
            "coordinate_frame": str(self.coordinate_frame),
        }

    @classmethod
    def from_json_dict(cls, payload):
        result = cls(
            available=bool(payload.get("available", False)),
            plan_success=bool(payload.get("plan_success", False)),
            fail_reason=str(payload.get("fail_reason", "")),
            cache_hit=True,
            cache_key=str(payload.get("cache_key", "")),
            planning_time_ms=float(payload.get("planning_time_ms", 0.0)),
            code_loaded=bool(payload.get("code_loaded", False)),
            weight_loaded=bool(payload.get("weight_loaded", False)),
            weight_keys=list(payload.get("weight_keys", [])),
            path_points=np.asarray(payload.get("path_points", []), dtype=np.float32).reshape(-1, 5),
            path_length=float(payload.get("path_length", 0.0)),
            num_gear_switches=int(payload.get("num_gear_switches", 0)),
            anchor_xy=(
                tuple(float(v) for v in payload["anchor_xy"])
                if payload.get("anchor_xy") is not None
                else None
            ),
            anchor_s=float(payload.get("anchor_s", 0.0)),
            path_valid_after_articulated_check=bool(
                payload.get("path_valid_after_articulated_check", False)
            ),
            weak_topology_only=bool(payload.get("weak_topology_only", False)),
            reward_mode=str(payload.get("reward_mode", "disabled")),
            min_clearance=float(payload.get("min_clearance", 0.0)),
            terminal_heading_error=float(payload.get("terminal_heading_error", 0.0)),
            terminal_overlap=float(payload.get("terminal_overlap", 0.0)),
            terminal_position_error=float(payload.get("terminal_position_error", 0.0)),
            terminal_heading_consistent=bool(
                payload.get("terminal_heading_consistent", False)
            ),
            terminal_success_consistent=bool(
                payload.get("terminal_success_consistent", False)
            ),
            collision=bool(payload.get("collision", False)),
            out_of_bounds=bool(payload.get("out_of_bounds", False)),
            reference_point=str(payload.get("reference_point", "front_body_center")),
            coordinate_frame=str(
                payload.get("coordinate_frame", "mixing_plant_world_xy_yaw")
            ),
        )
        return result


@contextmanager
def _temporarily_prepend_path(path):
    path = os.path.abspath(path)
    inserted = False
    if path not in sys.path:
        sys.path.insert(0, path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


@contextmanager
def _temporarily_hide_modules(prefixes):
    saved = {}
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            saved[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


class HopeTeacherAdapter:
    """Training-only adapter for HOPE coarse topology guidance."""

    def __init__(self, config, vehicle_params, rng=None):
        self.config = config
        self.vehicle_params = vehicle_params
        self.rng = np.random.default_rng() if rng is None else rng
        self.hope_code_dir = self._resolve_path(config.hope_code_dir)
        self.hope_src_dir = os.path.join(self.hope_code_dir, "src")
        self.hope_weight_path = self._resolve_path(config.hope_weight_path)
        self.cache_dir = self._resolve_path(config.hope_cache_dir)
        self._rs_module = None
        self._code_load_error = ""
        self._weight_load_error = ""
        self._weight_keys = []
        self._weight_probe_mode = "none"
        self._code_loaded = False
        self._weight_loaded = False
        self._memory_cache: Dict[str, HopeTeacherTrajectory] = {}
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_hope_code()
        self._probe_weight_file()

    @staticmethod
    def _resolve_path(path):
        if path is None or str(path).strip() == "":
            return ""
        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(REPO_ROOT, path))

    @property
    def available(self):
        return bool(self._code_loaded and self._weight_loaded)

    def _load_hope_code(self):
        rs_path = os.path.join(self.hope_src_dir, "env", "reeds_shepp.py")
        if not os.path.isfile(rs_path):
            self._code_load_error = "hope_reeds_shepp_missing"
            return
        try:
            module_name = "_hope_reeds_shepp_{}".format(
                hashlib.sha1(rs_path.encode("utf-8")).hexdigest()[:10]
            )
            spec = importlib.util.spec_from_file_location(module_name, rs_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "calc_all_paths"):
                self._code_load_error = "hope_reeds_shepp_no_calc_all_paths"
                return
            self._rs_module = module
            self._code_loaded = True
        except Exception as exc:
            self._code_load_error = "hope_code_import_error:{}:{}".format(
                type(exc).__name__,
                str(exc),
            )

    def _probe_weight_file(self):
        if not self.hope_weight_path:
            self._weight_load_error = "hope_weight_path_empty"
            return
        if not os.path.isfile(self.hope_weight_path):
            self._weight_load_error = "hope_weight_missing"
            return
        try:
            import torch

            with _temporarily_hide_modules(("model",)):
                with _temporarily_prepend_path(self.hope_src_dir):
                    try:
                        checkpoint = torch.load(
                            self.hope_weight_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                    except TypeError:
                        checkpoint = torch.load(
                            self.hope_weight_path,
                            map_location="cpu",
                        )
            if hasattr(checkpoint, "keys"):
                self._weight_keys = [str(key) for key in list(checkpoint.keys())[:16]]
            else:
                self._weight_keys = [type(checkpoint).__name__]
            self._weight_loaded = True
            self._weight_probe_mode = "torch_load"
            del checkpoint
        except Exception as exc:
            self._weight_load_error = "hope_weight_load_error:{}:{}".format(
                type(exc).__name__,
                str(exc),
            )
            if zipfile.is_zipfile(self.hope_weight_path):
                try:
                    with zipfile.ZipFile(self.hope_weight_path) as archive:
                        names = archive.namelist()
                    if names:
                        self._weight_keys = ["zip:{}".format(names[0].split("/")[0])]
                        self._weight_loaded = True
                        self._weight_probe_mode = "file_probe_only"
                except Exception:
                    pass

    def _cache_file(self, cache_key):
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(cache_key))
        return os.path.join(self.cache_dir, safe + ".json")

    def _scene_obstacle_hash(self, scene):
        hasher = hashlib.sha1()
        try:
            hasher.update(np.asarray(scene.occupancy_grid, dtype=np.uint8).tobytes())
        except Exception:
            for polygon in getattr(scene, "obstacle_polygons", []):
                hasher.update(np.asarray(polygon.bounds, dtype=np.float64).tobytes())
        return hasher.hexdigest()[:16]

    def cache_key(self, scene, state, slot):
        task_family = str(scene.metadata.get("task_family", ""))
        payload = {
            "scene_seed": int(scene.metadata.get("seed", -1)),
            "stage": int(scene.metadata.get("stage", -1)),
            "task_family": task_family,
            "goal_orientation_mode": str(scene.metadata.get("goal_orientation_mode", "")),
            "initial": [
                round(float(state.x_front), 2),
                round(float(state.y_front), 2),
                round(float(wrap_to_pi(state.theta_front)), 3),
                round(float(wrap_to_pi(state.theta_rear)), 3),
            ],
            "goal": [
                round(float(slot.x_goal), 2),
                round(float(slot.y_goal), 2),
                round(float(wrap_to_pi(slot.theta_goal)), 3),
            ],
            "obstacles": self._scene_obstacle_hash(scene),
            "vehicle": [
                round(float(self.vehicle_params.front_body_length), 3),
                round(float(self.vehicle_params.rear_body_length), 3),
                round(float(self.vehicle_params.front_body_width), 3),
                round(float(self.vehicle_params.minimum_turning_radius), 3),
            ],
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def plan_episode(self, scene, state, slot, vehicle_model):
        start_time = time.perf_counter()
        cache_key = self.cache_key(scene, state, slot)
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            cached.cache_hit = True
            return cached
        cache_file = self._cache_file(cache_key)
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as handle:
                    cached = HopeTeacherTrajectory.from_json_dict(json.load(handle))
                cached.cache_hit = True
                cached.cache_key = cache_key
                self._memory_cache[cache_key] = cached
                return cached
            except Exception:
                pass

        if not self._code_loaded:
            return self._store_failure(
                cache_key,
                "hope_code_unavailable:{}".format(self._code_load_error),
                start_time,
            )
        if not self._weight_loaded:
            return self._store_failure(
                cache_key,
                "hope_weight_unavailable:{}".format(self._weight_load_error),
                start_time,
            )

        max_curvature = 1.0 / max(
            float(self.vehicle_params.minimum_turning_radius),
            1e-6,
        )
        try:
            paths = self._rs_module.calc_all_paths(
                float(state.x_front),
                float(state.y_front),
                float(state.theta_front),
                float(slot.x_goal),
                float(slot.y_goal),
                float(slot.theta_goal),
                max_curvature,
                step_size=0.5,
            )
        except Exception as exc:
            return self._store_failure(
                cache_key,
                "hope_rs_exception:{}".format(type(exc).__name__),
                start_time,
            )
        if not paths:
            return self._store_failure(cache_key, "hope_rs_no_path", start_time)

        candidates = sorted(paths, key=lambda item: float(getattr(item, "L", float("inf"))))
        weak_candidate = None
        best_failure = "hope_rs_no_valid_articulated_path"
        for path in candidates[:64]:
            result = self._path_to_result(
                path=path,
                cache_key=cache_key,
                scene=scene,
                slot=slot,
                vehicle_model=vehicle_model,
                planning_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            if result.path_valid_after_articulated_check:
                return self._store_result(result)
            if result.weak_topology_only and weak_candidate is None:
                weak_candidate = result
            if result.fail_reason:
                best_failure = result.fail_reason

        if weak_candidate is not None:
            weak_candidate.planning_time_ms = (time.perf_counter() - start_time) * 1000.0
            return self._store_result(weak_candidate)
        return self._store_failure(cache_key, best_failure, start_time)

    def _store_failure(self, cache_key, reason, start_time):
        result = HopeTeacherTrajectory(
            available=self.available,
            plan_success=False,
            fail_reason=str(reason),
            cache_hit=False,
            cache_key=str(cache_key),
            planning_time_ms=(time.perf_counter() - start_time) * 1000.0,
            code_loaded=self._code_loaded,
            weight_loaded=self._weight_loaded,
            weight_keys=list(self._weight_keys),
        )
        return self._store_result(result)

    def _store_result(self, result):
        result.code_loaded = self._code_loaded
        result.weight_loaded = self._weight_loaded
        result.weight_keys = list(self._weight_keys)
        self._memory_cache[result.cache_key] = result
        try:
            with open(self._cache_file(result.cache_key), "w", encoding="utf-8") as handle:
                json.dump(result.to_json_dict(), handle, sort_keys=True)
        except Exception:
            pass
        return result

    def _path_to_result(self, path, cache_key, scene, slot, vehicle_model, planning_time_ms):
        xs = np.asarray(getattr(path, "x", []), dtype=np.float64)
        ys = np.asarray(getattr(path, "y", []), dtype=np.float64)
        yaws = np.asarray(getattr(path, "yaw", []), dtype=np.float64)
        dirs = np.asarray(getattr(path, "directions", []), dtype=np.float64)
        if len(xs) < 2 or len(xs) != len(ys) or len(xs) != len(yaws):
            return HopeTeacherTrajectory(
                available=True,
                plan_success=False,
                fail_reason="hope_rs_malformed_path",
                cache_key=cache_key,
            )
        if len(dirs) != len(xs):
            dirs = np.ones_like(xs)
        dirs = np.where(dirs >= 0.0, 1.0, -1.0)
        seg = np.hypot(np.diff(xs), np.diff(ys))
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        path_length = float(cumulative[-1])
        points = np.stack(
            [
                xs.astype(np.float32),
                ys.astype(np.float32),
                np.asarray([wrap_to_pi(v) for v in yaws], dtype=np.float32),
                dirs.astype(np.float32),
                cumulative.astype(np.float32),
            ],
            axis=1,
        )
        audit = self._articulated_audit(points, scene, slot, vehicle_model)
        anchor_xy, anchor_s = self._select_anchor(points, scene, slot)
        gear_switches = int(np.sum(np.diff(np.sign(points[:, 3])) != 0))
        reward_mode = "disabled"
        plan_success = False
        weak_topology_only = False
        fail_reason = ""
        if audit["path_valid_after_articulated_check"]:
            reward_mode = "strong"
            plan_success = True
        elif (
            not audit["collision"]
            and not audit["out_of_bounds"]
            and audit["terminal_heading_consistent"]
        ):
            reward_mode = "weak_topology"
            plan_success = True
            weak_topology_only = True
        else:
            fail_reason = audit["fail_reason"]

        return HopeTeacherTrajectory(
            available=True,
            plan_success=plan_success,
            fail_reason=fail_reason,
            cache_hit=False,
            cache_key=cache_key,
            planning_time_ms=float(planning_time_ms),
            path_points=points,
            path_length=path_length,
            num_gear_switches=gear_switches,
            anchor_xy=anchor_xy,
            anchor_s=anchor_s,
            path_valid_after_articulated_check=bool(
                audit["path_valid_after_articulated_check"]
            ),
            weak_topology_only=weak_topology_only,
            reward_mode=reward_mode,
            min_clearance=float(audit["min_clearance"]),
            terminal_heading_error=float(audit["terminal_heading_error"]),
            terminal_overlap=float(audit["terminal_overlap"]),
            terminal_position_error=float(audit["terminal_position_error"]),
            terminal_heading_consistent=bool(audit["terminal_heading_consistent"]),
            terminal_success_consistent=bool(audit["terminal_success_consistent"]),
            collision=bool(audit["collision"]),
            out_of_bounds=bool(audit["out_of_bounds"]),
        )

    def _articulated_audit(self, points, scene, slot, vehicle_model):
        collision = False
        out_of_bounds = False
        min_clearance = float("inf")
        xmin, ymin, xmax, ymax = scene.world_bounds
        for point in points:
            state = ArticulatedState(
                x_front=float(point[0]),
                y_front=float(point[1]),
                theta_front=float(point[2]),
                theta_rear=float(point[2]),
            )
            front_box, rear_box = vehicle_model.body_boxes(state)
            if (
                front_box.bounds[0] < xmin
                or front_box.bounds[1] < ymin
                or front_box.bounds[2] > xmax
                or front_box.bounds[3] > ymax
                or rear_box.bounds[0] < xmin
                or rear_box.bounds[1] < ymin
                or rear_box.bounds[2] > xmax
                or rear_box.bounds[3] > ymax
            ):
                out_of_bounds = True
            if (
                scene.prepared_obstacles.intersects(front_box)
                or scene.prepared_obstacles.intersects(rear_box)
            ):
                collision = True
                min_clearance = 0.0
                break
            try:
                min_clearance = min(
                    min_clearance,
                    float(front_box.distance(scene.obstacle_union)),
                    float(rear_box.distance(scene.obstacle_union)),
                )
            except Exception:
                min_clearance = min(min_clearance, 0.0)

        final = points[-1]
        final_state = ArticulatedState(
            x_front=float(final[0]),
            y_front=float(final[1]),
            theta_front=float(final[2]),
            theta_rear=float(final[2]),
        )
        final_front, _ = vehicle_model.body_boxes(final_state)
        terminal_overlap = float(overlap_ratio(final_front, slot.front_box()))
        terminal_heading_error = abs(float(wrap_to_pi(final_state.theta_front - slot.theta_goal)))
        terminal_position_error = float(
            math.hypot(final_state.x_front - slot.x_goal, final_state.y_front - slot.y_goal)
        )
        terminal_heading_consistent = terminal_heading_error <= float(
            self.config.success_heading_error
        )
        terminal_success_consistent = (
            terminal_overlap >= float(self.config.success_overlap)
            and terminal_heading_consistent
        )
        valid = (
            not collision
            and not out_of_bounds
            and terminal_success_consistent
        )
        fail_reason = ""
        if collision:
            fail_reason = "articulated_collision"
        elif out_of_bounds:
            fail_reason = "articulated_out_of_bounds"
        elif not terminal_heading_consistent:
            fail_reason = "terminal_heading_mismatch"
        elif not terminal_success_consistent:
            fail_reason = "terminal_success_mismatch"
        return {
            "path_valid_after_articulated_check": bool(valid),
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "min_clearance": float(min_clearance if math.isfinite(min_clearance) else 0.0),
            "terminal_overlap": terminal_overlap,
            "terminal_heading_error": terminal_heading_error,
            "terminal_position_error": terminal_position_error,
            "terminal_heading_consistent": bool(terminal_heading_consistent),
            "terminal_success_consistent": bool(terminal_success_consistent),
            "fail_reason": fail_reason,
        }

    def _select_anchor(self, points, scene, slot):
        if len(points) == 0:
            return None, 0.0
        xy = points[:, :2]
        reverse = np.where(points[:, 3] < 0.0)[0]
        if getattr(scene, "target_bay", None) is not None:
            ref = np.asarray(scene.target_bay.mouth_center, dtype=np.float64)
        else:
            ref = np.asarray([slot.x_goal, slot.y_goal], dtype=np.float64)
        if len(reverse) > 0:
            distances = np.linalg.norm(xy[reverse] - ref.reshape(1, 2), axis=1)
            index = int(reverse[int(np.argmin(distances))])
        else:
            distances = np.linalg.norm(xy - ref.reshape(1, 2), axis=1)
            index = int(np.argmin(distances))
        return (float(points[index, 0]), float(points[index, 1])), float(points[index, 4])

    def diagnostics(self, trajectory, guide_weight, dropout_rate, dropped):
        if trajectory is None:
            trajectory = HopeTeacherTrajectory()
        return {
            "hope_teacher_enabled": True,
            "hope_teacher_available": bool(trajectory.available),
            "hope_code_loaded": bool(trajectory.code_loaded),
            "hope_weight_loaded": bool(trajectory.weight_loaded),
            "hope_code_load_error": str(self._code_load_error),
            "hope_weight_load_error": str(self._weight_load_error),
            "hope_weight_probe_mode": str(self._weight_probe_mode),
            "hope_plan_success": bool(trajectory.plan_success),
            "hope_plan_fail_reason": str(trajectory.fail_reason),
            "hope_cache_hit": bool(trajectory.cache_hit),
            "hope_path_length": float(trajectory.path_length),
            "hope_path_valid_after_articulated_check": bool(
                trajectory.path_valid_after_articulated_check
            ),
            "hope_reward_mode": str(trajectory.reward_mode),
            "hope_collision_margin": float(trajectory.min_clearance),
            "hope_terminal_heading_error": float(trajectory.terminal_heading_error),
            "hope_terminal_overlap": float(trajectory.terminal_overlap),
            "hope_terminal_position_error": float(trajectory.terminal_position_error),
            "hope_num_gear_switches": int(trajectory.num_gear_switches),
            "hope_anchor_s": float(trajectory.anchor_s),
            "hope_planning_time_ms": float(trajectory.planning_time_ms),
            "guide_weight_current": float(guide_weight),
            "guide_dropout_rate": float(dropout_rate),
            "guide_dropped": bool(dropped),
        }

    @staticmethod
    def disabled_diagnostics():
        return {
            "hope_teacher_enabled": False,
            "hope_teacher_available": False,
            "hope_code_loaded": False,
            "hope_weight_loaded": False,
            "hope_code_load_error": "",
            "hope_weight_load_error": "",
            "hope_weight_probe_mode": "none",
            "hope_plan_success": False,
            "hope_plan_fail_reason": "disabled",
            "hope_cache_hit": False,
            "hope_path_length": 0.0,
            "hope_path_valid_after_articulated_check": False,
            "hope_reward_mode": "disabled",
            "hope_collision_margin": 0.0,
            "hope_terminal_heading_error": 0.0,
            "hope_terminal_overlap": 0.0,
            "hope_terminal_position_error": 0.0,
            "hope_num_gear_switches": 0,
            "hope_anchor_s": 0.0,
            "hope_planning_time_ms": 0.0,
            "guide_weight_current": 0.0,
            "guide_dropout_rate": 1.0,
            "guide_dropped": True,
        }

    def compute_guidance(self, trajectory, previous_state, current_state, action_gear, guide_weight):
        zero = {
            "guide_reward": 0.0,
            "guide_progress_reward": 0.0,
            "guide_lateral_error": 0.0,
            "guide_heading_error": 0.0,
            "guide_anchor_error": 0.0,
            "guide_gear_agreement": 0.0,
            "guide_corridor_penalty": 0.0,
            "guide_tangent_reward": 0.0,
            "guide_anchor_reward": 0.0,
            "guide_gear_reward": 0.0,
        }
        if (
            trajectory is None
            or not trajectory.reward_available
            or float(guide_weight) <= 0.0
            or len(trajectory.path_points) < 2
        ):
            return 0.0, zero

        prev_proj = self._project_state(previous_state, trajectory.path_points)
        curr_proj = self._project_state(current_state, trajectory.path_points)
        corridor_width = max(float(self.config.teacher_corridor_width), 1e-6)
        progress_delta = curr_proj["s"] - prev_proj["s"]
        progress_reward = float(np.clip(progress_delta / corridor_width, -1.0, 1.0))
        lateral_error = abs(float(curr_proj["lateral_error"]))
        corridor_penalty = -float(
            np.clip(max(0.0, lateral_error - corridor_width) / corridor_width, 0.0, 2.0) ** 2
        )
        heading_error = abs(
            float(wrap_to_pi(current_state.theta_front - curr_proj["tangent_heading"]))
        )
        tangent_reward = math.cos(heading_error)
        anchor_error = 0.0
        anchor_reward = 0.0
        if trajectory.anchor_xy is not None:
            ax, ay = trajectory.anchor_xy
            prev_anchor = math.hypot(previous_state.x_front - ax, previous_state.y_front - ay)
            curr_anchor = math.hypot(current_state.x_front - ax, current_state.y_front - ay)
            anchor_error = curr_anchor
            anchor_reward = float(np.clip((prev_anchor - curr_anchor) / corridor_width, -1.0, 1.0))

        prior_direction = 1 if curr_proj["direction"] >= 0.0 else -1
        executed_direction = 0
        if int(action_gear) == 0:
            executed_direction = 1
        elif int(action_gear) == 1:
            executed_direction = -1
        if executed_direction == 0:
            gear_agreement = 0.0
            gear_reward = 0.0
        elif executed_direction == prior_direction:
            gear_agreement = 1.0
            gear_reward = 1.0
        else:
            gear_agreement = 0.0
            gear_reward = -0.2

        if trajectory.reward_mode == "weak_topology":
            raw = 0.25 * float(self.config.teacher_progress_weight) * progress_reward
            corridor_penalty = 0.0
            tangent_reward = 0.0
            anchor_reward = 0.0
            gear_reward = 0.0
        else:
            raw = (
                float(self.config.teacher_progress_weight) * progress_reward
                + corridor_penalty
                + float(self.config.teacher_heading_weight) * tangent_reward
                + float(self.config.teacher_anchor_weight) * anchor_reward
                + float(self.config.teacher_gear_weight) * gear_reward
            )
        clipped = float(
            np.clip(
                raw,
                -float(self.config.teacher_reward_clip),
                float(self.config.teacher_reward_clip),
            )
        )
        weighted = float(guide_weight) * clipped
        diagnostics = {
            "guide_reward": weighted,
            "guide_progress_reward": progress_reward,
            "guide_lateral_error": lateral_error,
            "guide_heading_error": heading_error,
            "guide_anchor_error": anchor_error,
            "guide_gear_agreement": gear_agreement,
            "guide_corridor_penalty": corridor_penalty,
            "guide_tangent_reward": tangent_reward,
            "guide_anchor_reward": anchor_reward,
            "guide_gear_reward": gear_reward,
        }
        return weighted, diagnostics

    def _project_state(self, state, points):
        xy = points[:, :2].astype(np.float64)
        pos = np.asarray([state.x_front, state.y_front], dtype=np.float64)
        best = {
            "s": 0.0,
            "lateral_error": float("inf"),
            "tangent_heading": float(points[0, 2]),
            "direction": float(points[0, 3]),
        }
        for index in range(len(xy) - 1):
            a = xy[index]
            b = xy[index + 1]
            ab = b - a
            length_sq = float(np.dot(ab, ab))
            if length_sq <= 1e-12:
                continue
            t = float(np.clip(np.dot(pos - a, ab) / length_sq, 0.0, 1.0))
            proj = a + t * ab
            error_vec = pos - proj
            lateral = float(np.linalg.norm(error_vec))
            if lateral < best["lateral_error"]:
                seg_len = math.sqrt(length_sq)
                s = float(points[index, 4]) + t * seg_len
                tangent = math.atan2(float(ab[1]), float(ab[0]))
                best = {
                    "s": s,
                    "lateral_error": lateral,
                    "tangent_heading": tangent,
                    "direction": float(points[index, 3]),
                }
        return best

    def sample_offpath_state(self, trajectory, rng, episode_progress, scene, vehicle_model):
        if trajectory is None or not trajectory.reward_available or len(trajectory.path_points) < 3:
            return None, "no_teacher_path"
        points = trajectory.path_points
        idx = int(rng.integers(1, max(2, len(points) - 1)))
        base = points[idx]
        prev_pt = points[max(0, idx - 1)]
        next_pt = points[min(len(points) - 1, idx + 1)]
        tangent = math.atan2(float(next_pt[1] - prev_pt[1]), float(next_pt[0] - prev_pt[0]))
        normal = tangent + 0.5 * math.pi
        scale = float(np.clip(episode_progress, 0.0, 1.0))
        lateral_amp = float(self.config.teacher_corridor_width) * (0.2 + 0.8 * scale)
        along_amp = 0.5 * lateral_amp
        heading_amp = math.radians(10.0 + 25.0 * scale)
        phi_amp = min(float(self.vehicle_params.phi_max) * 0.75, math.radians(8.0 + 18.0 * scale))
        for _ in range(16):
            lateral = float(rng.uniform(-lateral_amp, lateral_amp))
            along = float(rng.uniform(-along_amp, along_amp))
            heading = wrap_to_pi(float(base[2]) + float(rng.uniform(-heading_amp, heading_amp)))
            phi = float(rng.uniform(-phi_amp, phi_amp))
            x = float(base[0]) + along * math.cos(tangent) + lateral * math.cos(normal)
            y = float(base[1]) + along * math.sin(tangent) + lateral * math.sin(normal)
            state = ArticulatedState(
                x_front=x,
                y_front=y,
                theta_front=heading,
                theta_rear=wrap_to_pi(heading - phi),
            )
            if abs(state.phi) > float(self.vehicle_params.phi_max):
                continue
            front_box, rear_box = vehicle_model.body_boxes(state)
            xmin, ymin, xmax, ymax = scene.world_bounds
            if (
                front_box.bounds[0] < xmin
                or front_box.bounds[1] < ymin
                or front_box.bounds[2] > xmax
                or front_box.bounds[3] > ymax
                or rear_box.bounds[0] < xmin
                or rear_box.bounds[1] < ymin
                or rear_box.bounds[2] > xmax
                or rear_box.bounds[3] > ymax
            ):
                continue
            if (
                scene.prepared_obstacles.intersects(front_box)
                or scene.prepared_obstacles.intersects(rear_box)
            ):
                continue
            return state, "sampled"
        return None, "no_collision_free_sample"

    def record_failure(self, scene, state, slot, info):
        path = os.path.join(self.cache_dir, "failure_aggregation.jsonl")
        record = {
            "scene_seed": int(scene.metadata.get("seed", -1)),
            "task_family": str(scene.metadata.get("task_family", "")),
            "goal_orientation_mode": str(scene.metadata.get("goal_orientation_mode", "")),
            "state": [
                float(state.x_front),
                float(state.y_front),
                float(state.theta_front),
                float(state.theta_rear),
                float(state.v),
                float(state.phi_dot),
            ],
            "goal": [float(slot.x_goal), float(slot.y_goal), float(slot.theta_goal)],
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "timeout": bool(info.get("timeout", False)),
            "out_of_bounds": bool(info.get("out_of_bounds", False)),
            "articulation_limit_violation": bool(
                info.get("articulation_limit_violation", False)
            ),
        }
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return True
        except Exception:
            return False
