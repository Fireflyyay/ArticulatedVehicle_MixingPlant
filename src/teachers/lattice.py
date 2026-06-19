import heapq
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS
from env.vehicle import ArticulatedState, ArticulatedVehicleModel
from env.geometry import DirectedParkingSlot, overlap_ratio, wrap_to_pi
from teachers.base import PlanResult, TeacherBase
from teachers.primitive_set import PrimitiveSet
from teachers.heuristics import (
    TeacherHeuristic,
    compute_bay_entry_distance,
    count_gear_switches,
)


GRID_XY = 0.25
N_THETA = 36
N_PHI = 15

BEAM_WIDTH_COARSE = 64
BEAM_WIDTH_EXACT = 16
BEAM_PRE_FILTER_MULT = 3


class ArticulatedLatticeTeacher(TeacherBase):
    def __init__(
        self,
        vehicle_params=None,
        heuristic_weights: Optional[Dict[str, float]] = None,
        max_expansions: int = 8000,
        max_time_ms: float = 2000.0,
        max_open_size: int = 30000,
        w_heuristic: float = 1.5,
        w_path: float = 1.0,
        w_time: float = 0.5,
        w_gear: float = 3.0,
        w_phi_cost: float = 0.3,
        w_phidot_cost: float = 0.1,
        w_clearance: float = 5.0,
        w_stall: float = 8.0,
    ):
        super().__init__(name="lattice")
        self.p = vehicle_params or DEFAULT_VEHICLE_PARAMS
        self.primitive_set = PrimitiveSet(vehicle_params=self.p)
        self.heuristic = TeacherHeuristic(
            vehicle_params=self.p,
            weights=heuristic_weights,
        )
        self.max_expansions = max_expansions
        self.max_time_ms = max_time_ms
        self.max_open_size = max_open_size
        self.w_heuristic = w_heuristic
        self.w_path = w_path
        self.w_time = w_time
        self.w_gear = w_gear
        self.w_phi_cost = w_phi_cost
        self.w_phidot_cost = w_phidot_cost
        self.w_clearance = w_clearance
        self.w_stall = w_stall

    def configure_for_family(self, task_family: str):
        self.heuristic.configure_for_family(task_family)

    def _state_key(self, state: ArticulatedState, gear: int):
        x_bin = int(round(state.x_front / GRID_XY))
        y_bin = int(round(state.y_front / GRID_XY))
        theta_norm = (wrap_to_pi(state.theta_front) + math.pi) / (2.0 * math.pi)
        t_bin = int(round(theta_norm * N_THETA)) % N_THETA
        p_bin = int(round((state.phi + self.p.phi_max) / (2.0 * self.p.phi_max) * (N_PHI - 1)))
        p_bin = max(0, min(N_PHI - 1, p_bin))
        return (x_bin, y_bin, t_bin, p_bin, gear)

    def _edge_cost(
        self,
        duration: float,
        path_length: float,
        prev_gear: int,
        new_gear: int,
        phi: float,
        phi_dot_abs: float,
        clearance: float,
        is_stall: bool,
    ) -> float:
        cost = 0.0
        cost += self.w_path * path_length
        cost += self.w_time * duration
        if prev_gear is not None and new_gear != prev_gear:
            cost += self.w_gear
        cost += self.w_phi_cost * abs(phi)
        cost += self.w_phidot_cost * phi_dot_abs
        if clearance < 0.3:
            cost += self.w_clearance * (0.3 - clearance) / 0.3
        if is_stall:
            cost += self.w_stall
        return cost

    def _is_stall(self, speed_ratio: float) -> bool:
        return speed_ratio < 1e-6

    def plan_from_state(
        self,
        state: ArticulatedState,
        scene,
        slot: DirectedParkingSlot,
        vehicle_model: ArticulatedVehicleModel,
    ) -> PlanResult:
        start_time = time.perf_counter()
        task_family = self._scenario_family_from_scene(scene)
        seed = int(scene.metadata.get("seed", 0))
        self.configure_for_family(task_family)

        metrics = self._compute_metrics(state, slot, vehicle_model)
        if self._check_success(state, slot, vehicle_model):
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return PlanResult(
                success=True,
                states=[state],
                total_cost=0.0,
                path_length=0.0,
                num_steps=0,
                planning_time_ms=elapsed,
                teacher_name=self.name,
                scenario_family=task_family,
                seed=seed,
                final_position_error=metrics["distance"],
                final_heading_error=metrics["heading_error"],
                final_phi=metrics["phi"],
                final_overlap=metrics["front_overlap"],
                min_clearance=self._compute_clearance(state, scene, vehicle_model),
            )

        if self._check_collision(state, scene, vehicle_model):
            return PlanResult(
                success=False,
                fail_reason="start_collision",
                planning_time_ms=(time.perf_counter() - start_time) * 1000.0,
                teacher_name=self.name,
                scenario_family=task_family,
                seed=seed,
            )

        start_gear = 0
        start_key = self._state_key(state, start_gear)
        g_cost: Dict[Tuple, float] = {start_key: 0.0}
        parent: Dict[Tuple, Optional[Tuple]] = {start_key: None}
        parent_prim: Dict[Tuple, Optional[Tuple]] = {start_key: None}
        stored_states: Dict[Tuple, ArticulatedState] = {start_key: state}
        stored_gears: Dict[Tuple, int] = {start_key: start_gear}
        stored_consecutive_stalls: Dict[Tuple, int] = {start_key: 0}

        anchors_xy = None
        anchors_xy = [(float(slot.x_goal), float(slot.y_goal))]

        h_start = self.heuristic.compute(
            state.x_front, state.y_front, state.theta_front, state.phi,
            slot, scene, anchors_xy,
        )
        f_start = h_start * self.w_heuristic

        open_list = []
        tie_breaker = 0
        heapq.heappush(open_list, (f_start, tie_breaker, 0.0, start_key))
        tie_breaker += 1

        best_result = None
        best_f = float("inf")
        expanded = 0

        while open_list and expanded < self.max_expansions:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > self.max_time_ms:
                break
            if len(open_list) > self.max_open_size:
                open_list.sort()
                open_list = open_list[: self.max_open_size]
                heapq.heapify(open_list)

            f_val, _, current_g, current_key = heapq.heappop(open_list)
            if current_g > g_cost.get(current_key, float("inf")) + 1e-6:
                continue
            expanded += 1

            current_state = stored_states.get(current_key)
            if current_state is None:
                continue
            current_gear = stored_gears.get(current_key, 0)
            consecutive_stalls = stored_consecutive_stalls.get(current_key, 0)

            if self._check_success(current_state, slot, vehicle_model):
                path_states, path_gears, path_prims = self._reconstruct_path(
                    current_key, parent, parent_prim, stored_states, stored_gears
                )
                exact_ok = self._validate_path_exact(
                    path_states, path_gears, path_prims, scene, vehicle_model,
                )
                if exact_ok:
                    actions_norm, actions_phys, v_cmds, phi_dot_cmds = self._build_action_sequence(
                        path_states, path_gears, path_prims, vehicle_model
                    )
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    metrics = self._compute_metrics(current_state, slot, vehicle_model)
                    return self._make_result(
                        True, path_states, actions_norm, actions_phys,
                        current_g, task_family, seed, elapsed, vehicle_model,
                        scene, metrics, expanded,
                    )
                else:
                    g_cost[current_key] = float("inf")

            prim_set_name = self.primitive_set.select_primitive_set(
                d_goal=math.hypot(
                    current_state.x_front - slot.x_goal,
                    current_state.y_front - slot.y_goal,
                ),
                d_entry=compute_bay_entry_distance(
                    current_state.x_front, current_state.y_front, scene
                ),
                clearance=self._compute_clearance(current_state, scene, vehicle_model),
                phi_abs=abs(current_state.phi),
            )
            is_fine = prim_set_name == "fine"
            primitives = self.primitive_set.get_primitives(prim_set_name)
            table = self.primitive_set.get_table(prim_set_name)
            n_prims = len(primitives)
            phi_bin = self.primitive_set._nearest_phi_bin(current_state.phi)

            g_only_candidates = []
            for pi in range(n_prims):
                entry = table[phi_bin][pi]
                if entry is None:
                    continue
                prim = primitives[pi]
                if prim.speed_ratio < 1e-6 and abs(current_state.phi) < math.radians(5.0):
                    continue
                if prim.speed_ratio < 1e-6 and consecutive_stalls >= 2:
                    continue

                new_gear = entry["prim_gear"]
                if prim.speed_ratio < 1e-6:
                    new_gear = current_gear

                edge_cost = self._edge_cost(
                    duration=entry["prim_duration"],
                    path_length=abs(entry["prim_v_cmd"]) * entry["prim_duration"],
                    prev_gear=current_gear,
                    new_gear=new_gear,
                    phi=entry["final_phi"],
                    phi_dot_abs=abs(entry["prim_phi_dot_cmd"]),
                    clearance=1.0,
                    is_stall=self._is_stall(prim.speed_ratio),
                )
                g_only_candidates.append({
                    "pi": pi,
                    "entry": entry,
                    "prim": prim,
                    "new_gear": new_gear,
                    "edge_cost": edge_cost,
                    "new_g": current_g + edge_cost,
                })

            g_only_candidates.sort(key=lambda c: c["new_g"])
            pre_filter_count = min(len(g_only_candidates), BEAM_WIDTH_COARSE * BEAM_PRE_FILTER_MULT)
            pre_candidates = g_only_candidates[:pre_filter_count]

            candidates = []
            for pc in pre_candidates:
                entry = pc["entry"]
                pi = pc["pi"]
                prim = pc["prim"]

                dx = entry["final_dx"]
                dy = entry["final_dy"]
                c = math.cos(current_state.theta_front)
                s = math.sin(current_state.theta_front)
                nx = current_state.x_front + c * dx - s * dy
                ny = current_state.y_front + s * dx + c * dy
                ntheta_f = wrap_to_pi(current_state.theta_front + entry["final_dtheta_f"])
                nphi = entry["final_phi"]

                h_val = self.heuristic.compute(
                    nx, ny, ntheta_f, nphi, slot, scene, anchors_xy,
                )
                new_f = pc["new_g"] + self.w_heuristic * h_val

                candidates.append({
                    "prim_idx": pi,
                    "entry": entry,
                    "prim": prim,
                    "new_g": pc["new_g"],
                    "new_f": new_f,
                    "new_gear": pc["new_gear"],
                    "edge_cost": pc["edge_cost"],
                    "grid_checked": False,
                    "grid_occupied": False,
                })

            candidates.sort(key=lambda c: c["new_f"])
            candidates = candidates[:BEAM_WIDTH_COARSE]

            for cand in candidates:
                cand["grid_occupied"] = self.primitive_set.quick_occupancy_check(
                    cand["entry"], current_state.theta_front,
                    current_state.x_front, current_state.y_front, scene,
                )
                cand["grid_checked"] = True

            grid_free = [c for c in candidates if c["grid_checked"] and not c["grid_occupied"]]
            grid_hit = [c for c in candidates if c["grid_checked"] and c["grid_occupied"]]

            grid_free.sort(key=lambda c: c["new_f"])
            grid_hit.sort(key=lambda c: c["new_f"])

            exact_candidates = grid_free[:BEAM_WIDTH_EXACT] + grid_hit[:BEAM_WIDTH_EXACT]

            for cand in exact_candidates:
                if cand.get("grid_occupied", False):
                    rollout = self.primitive_set.exact_rollout(
                        current_state, cand["prim_idx"], is_fine,
                        vehicle_model, scene,
                    )
                    if rollout["collision"]:
                        continue
                    final_state = rollout["final_state"]
                else:
                    entry = cand["entry"]
                    c = math.cos(current_state.theta_front)
                    s = math.sin(current_state.theta_front)
                    nx = current_state.x_front + c * entry["final_dx"] - s * entry["final_dy"]
                    ny = current_state.y_front + s * entry["final_dx"] + c * entry["final_dy"]
                    ntheta_f = wrap_to_pi(current_state.theta_front + entry["final_dtheta_f"])
                    nphi = entry["final_phi"]
                    ntheta_r = wrap_to_pi(ntheta_f - nphi)
                    final_state = ArticulatedState(
                        x_front=nx, y_front=ny,
                        theta_front=ntheta_f, theta_rear=ntheta_r,
                        v=abs(entry["prim_v_cmd"]) if entry["prim_gear"] == 0 else -abs(entry["prim_v_cmd"]),
                        phi_dot=entry["prim_phi_dot_cmd"],
                    )
                next_key = self._state_key(final_state, cand["new_gear"])
                if cand["new_g"] >= g_cost.get(next_key, float("inf")):
                    continue

                g_cost[next_key] = cand["new_g"]
                parent[next_key] = current_key
                parent_prim[next_key] = (cand["prim_idx"], is_fine, cand["new_gear"])
                stored_states[next_key] = final_state
                stored_gears[next_key] = cand["new_gear"]
                new_consecutive = 1 if self._is_stall(cand["prim"].speed_ratio) else 0
                if self._is_stall(cand["prim"].speed_ratio) and consecutive_stalls > 0:
                    new_consecutive = consecutive_stalls + 1
                stored_consecutive_stalls[next_key] = new_consecutive

                tie_breaker += 1
                heapq.heappush(open_list, (cand["new_f"], tie_breaker, cand["new_g"], next_key))

        elapsed = (time.perf_counter() - start_time) * 1000.0
        is_timeout = elapsed > self.max_time_ms
        fail_reason = "timeout" if is_timeout else "max_expansions"

        if best_result is not None:
            best_result.planning_time_ms = elapsed
            return best_result

        return PlanResult(
            success=False,
            fail_reason=fail_reason,
            timeout=is_timeout,
            planning_time_ms=elapsed,
            teacher_name=self.name,
            scenario_family=task_family,
            seed=seed,
        )

    def _validate_path_exact(self, path_states, path_gears, path_prims, scene, vehicle_model):
        current = path_states[0]
        for i in range(1, len(path_states)):
            prim_info = path_prims[i]
            if prim_info is None:
                continue
            prim_idx, is_fine, gear = prim_info
            primitives = (
                self.primitive_set.fine_primitives
                if is_fine
                else self.primitive_set.coarse_primitives
            )
            prim = primitives[prim_idx]
            n_steps = max(1, int(round(prim.duration / self.p.dt)))
            for _ in range(n_steps):
                current = vehicle_model.step(current, (prim.v_cmd, prim.phi_dot_cmd), dt=self.p.dt)
                front_box, rear_box = vehicle_model.body_boxes(current)
                if (
                    scene.prepared_obstacles.intersects(front_box)
                    or scene.prepared_obstacles.intersects(rear_box)
                ):
                    return False
        return True

    def _reconstruct_path(self, goal_key, parent, parent_prim, stored_states, stored_gears):
        path_states = []
        path_gears = []
        path_prims = []
        cur = goal_key
        while cur is not None:
            path_states.append(stored_states[cur])
            path_gears.append(stored_gears.get(cur, 0))
            path_prims.append(parent_prim.get(cur))
            cur = parent.get(cur)
        path_states.reverse()
        path_gears.reverse()
        path_prims.reverse()
        return path_states, path_gears, path_prims

    def _build_action_sequence(self, path_states, path_gears, path_prims, vehicle_model):
        actions_norm = []
        actions_phys = []
        v_cmds_list = []
        phi_dot_cmds_list = []

        for i in range(1, len(path_states)):
            prim_info = path_prims[i]
            if prim_info is None:
                continue
            prim_idx, is_fine, gear = prim_info
            primitives = (
                self.primitive_set.fine_primitives
                if is_fine
                else self.primitive_set.coarse_primitives
            )
            prim = primitives[prim_idx]
            n_steps = max(1, int(round(prim.duration / self.p.dt)))

            state_before = path_states[i - 1]
            current = state_before
            for s_idx in range(n_steps):
                current = vehicle_model.step(current, (prim.v_cmd, prim.phi_dot_cmd), dt=self.p.dt)
                v_cmds_list.append(prim.v_cmd)
                phi_dot_cmds_list.append(prim.phi_dot_cmd)
                norm_action = self._compute_normalized_action(
                    state_before if s_idx == 0 else prev_tmp,
                    prim.v_cmd, prim.phi_dot_cmd,
                )
                phys_action = np.array([prim.v_cmd, prim.phi_dot_cmd], dtype=np.float32)
                actions_norm.append(norm_action)
                actions_phys.append(phys_action)
                prev_tmp = current
                state_before = current

        return actions_norm, actions_phys, v_cmds_list, phi_dot_cmds_list

    def _compute_normalized_action(self, state, v_cmd, phi_dot_cmd):
        phi = state.phi
        phi_dot_max = self.p.phi_dot_max
        phi_max = self.p.phi_max
        dt = self.p.dt
        phi_dot_lower = max(-phi_dot_max, (-phi_max - phi) / dt)
        phi_dot_upper = min(phi_dot_max, (phi_max - phi) / dt)
        if phi_dot_upper <= phi_dot_lower:
            phi_dot_norm = 0.0
        else:
            alpha = (phi_dot_cmd - phi_dot_lower) / (phi_dot_upper - phi_dot_lower)
            phi_dot_norm = 2.0 * alpha - 1.0
        phi_dot_norm = float(np.clip(phi_dot_norm, -1.0, 1.0))
        if v_cmd >= 0:
            v_norm = v_cmd / self.p.parking_v_forward_max
        else:
            v_norm = v_cmd / self.p.parking_v_reverse_max
        v_norm = float(np.clip(v_norm, -1.0, 1.0))
        return np.array([v_norm, phi_dot_norm], dtype=np.float32)

    def _make_result(
        self, success, path_states, actions_norm, actions_phys,
        total_cost, task_family, seed, elapsed, vehicle_model, scene,
        final_metrics, expanded,
    ):
        path_length = 0.0
        for i in range(1, len(path_states)):
            s0 = path_states[i - 1]
            s1 = path_states[i]
            path_length += math.hypot(s1.x_front - s0.x_front, s1.y_front - s0.y_front)

        gears = []
        for a in actions_phys:
            gears.append(0 if a[0] >= 0 else 1)
        n_switches = count_gear_switches(gears)
        n_zero = sum(1 for a in actions_phys if abs(a[0]) < 1e-6)

        final_state = path_states[-1] if path_states else path_states[0]
        min_clear = self._compute_clearance(final_state, scene, vehicle_model)
        for s in path_states:
            cl = self._compute_clearance(s, scene, vehicle_model)
            if cl < min_clear:
                min_clear = cl

        return PlanResult(
            success=success,
            states=path_states,
            actions_normalized=actions_norm,
            actions_physical=actions_phys,
            total_cost=total_cost,
            path_length=path_length,
            num_steps=len(actions_norm),
            num_gear_switches=n_switches,
            num_zero_speed_steps=n_zero,
            final_position_error=final_metrics["distance"],
            final_heading_error=final_metrics["heading_error"],
            final_phi=final_metrics["phi"],
            final_overlap=final_metrics["front_overlap"],
            min_clearance=min_clear,
            collision=False,
            timeout=False,
            planning_time_ms=elapsed,
            teacher_name=self.name,
            scenario_family=task_family,
            seed=seed,
        )

    def label_first_action(
        self,
        state: ArticulatedState,
        scene,
        slot: DirectedParkingSlot,
        vehicle_model: ArticulatedVehicleModel,
    ) -> Tuple[Optional[np.ndarray], Optional[PlanResult]]:
        result = self.plan_from_state(state, scene, slot, vehicle_model)
        if result.success and result.actions_normalized:
            return result.actions_normalized[0], result
        return None, result
