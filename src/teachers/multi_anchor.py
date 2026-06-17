import math
import time
from typing import List, Optional, Tuple

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS
from env.vehicle import ArticulatedState, ArticulatedVehicleModel
from env.geometry import DirectedParkingSlot, wrap_to_pi
from teachers.base import PlanResult, TeacherBase
from teachers.lattice import ArticulatedLatticeTeacher
from teachers.heuristics import TeacherHeuristic


class MultiAnchorTeacher(TeacherBase):
    def __init__(
        self,
        vehicle_params=None,
        lattice_teacher=None,
        max_anchors: int = 20,
        top_anchors: int = 5,
        segment_budget_ratio: float = 0.33,
        max_total_time_ms: float = 3000.0,
        w_reverse_bonus: float = 1.5,
        **lattice_kwargs,
    ):
        super().__init__(name="multi_anchor")
        self.p = vehicle_params or DEFAULT_VEHICLE_PARAMS
        self.top_anchors = min(top_anchors, max_anchors)
        self.segment_budget_ratio = segment_budget_ratio
        self.max_total_time_ms = max_total_time_ms
        self.w_reverse_bonus = w_reverse_bonus

        if lattice_teacher is not None:
            self.lattice = lattice_teacher
        else:
            self.lattice = ArticulatedLatticeTeacher(
                vehicle_params=self.p, **lattice_kwargs
            )

    def _generate_anchors(
        self,
        slot: DirectedParkingSlot,
        scene,
        vehicle_model: ArticulatedVehicleModel,
    ) -> List[dict]:
        anchors = []
        bay = getattr(scene, "target_bay", None)
        if bay is None:
            gx, gy = slot.x_goal, slot.y_goal
            for offset in [1.0, 2.0, 3.0, 4.0]:
                anchors.append({
                    "x": gx,
                    "y": gy,
                    "heading": slot.theta_goal,
                    "label": f"goal_offset_{offset}",
                })
            return anchors

        inward = bay.inward_heading
        goal_heading = slot.theta_goal
        corridor_heading = getattr(bay, "corridor_heading", inward + math.pi * 0.5)
        mx, my = bay.mouth_center

        inward_dx = math.cos(inward)
        inward_dy = math.sin(inward)
        corridor_dx = math.cos(corridor_heading)
        corridor_dy = math.sin(corridor_heading)
        perp_dx = inward_dx
        perp_dy = inward_dy

        for dist in [-3.0, 0.0, 3.0, 6.0]:
            ax = mx + corridor_dx * dist
            ay = my + corridor_dy * dist
            anchors.append({
                "x": ax,
                "y": ay,
                "heading": goal_heading,
                "label": f"mouth_center_offset_{dist:.0f}m",
            })

        for depth in [2.0, 4.0, 6.0]:
            ax = mx + inward_dx * depth
            ay = my + inward_dy * depth
            anchors.append({
                "x": ax,
                "y": ay,
                "heading": goal_heading,
                "label": f"bay_depth_{depth:.0f}m",
            })

        for lateral in [-3.0, 3.0]:
            for dist in [-4.0, 0.0, 4.0]:
                ax = mx + corridor_dx * dist + perp_dx * lateral
                ay = my + corridor_dy * dist + perp_dy * lateral
                anchors.append({
                    "x": ax,
                    "y": ay,
                    "heading": goal_heading,
                    "label": f"flank_lat{int(lateral)}_off{int(dist)}",
                })

        for dist in [-10.0, -6.0, 6.0, 10.0]:
            ax = slot.x_goal + corridor_dx * dist
            ay = slot.y_goal + corridor_dy * dist
            anchors.append({
                "x": ax,
                "y": ay,
                "heading": goal_heading,
                "label": f"corridor_along_{dist:.0f}m",
            })

        half_len = 0.5 * self.p.front_body_length
        for dist in [2.0, 4.0]:
            ax = slot.x_goal - math.cos(goal_heading) * (half_len + dist)
            ay = slot.y_goal - math.sin(goal_heading) * (half_len + dist)
            anchors.append({
                "x": ax,
                "y": ay,
                "heading": goal_heading,
                "label": f"approach_reverse_{dist:.0f}m",
            })

        ax, ay = slot.x_goal, slot.y_goal
        anchors.append({
            "x": ax,
            "y": ay,
            "heading": goal_heading,
            "label": "goal_exact",
        })

        seen = set()
        filtered = []
        for a in anchors:
            key = (round(a["x"], 2), round(a["y"], 2))
            if key in seen:
                continue
            seen.add(key)
            state = ArticulatedState(
                x_front=a["x"],
                y_front=a["y"],
                theta_front=a["heading"],
                theta_rear=a["heading"],
            )
            if not self._check_collision(state, scene, vehicle_model):
                xmin, ymin, xmax, ymax = scene.world_bounds
                if xmin < a["x"] < xmax and ymin < a["y"] < ymax:
                    a["phi"] = 0.0
                    a["state"] = state
                    filtered.append(a)

        return filtered[:self.top_anchors * 4]

    def _score_anchors(
        self,
        start_state: ArticulatedState,
        anchors: List[dict],
        slot: DirectedParkingSlot,
        scene,
        task_family: str,
    ) -> List[Tuple[float, dict]]:
        scored = []
        sx, sy = start_state.x_front, start_state.y_front
        gx, gy = slot.x_goal, slot.y_goal
        start_to_goal = math.hypot(sx - gx, sy - gy)

        is_parallel_rev = (task_family == "parallel_rev")
        goal_vec_x = gx - sx
        goal_vec_y = gy - sy

        for a in anchors:
            ax, ay = a["x"], a["y"]
            d_start_anchor = math.hypot(sx - ax, sy - ay)
            d_anchor_goal = math.hypot(ax - gx, ay - gy)
            score = d_start_anchor + d_anchor_goal

            heading_align = 0.0
            anchor_to_goal_dir = math.atan2(gy - ay, gx - ax)
            heading_error = abs(wrap_to_pi(a["heading"] - anchor_to_goal_dir))
            heading_align = 1.0 - math.cos(heading_error)
            score += 1.0 * heading_align

            if is_parallel_rev and start_to_goal > 0.5:
                anchor_to_goal_dx = gx - ax
                anchor_to_goal_dy = gy - ay
                reverse_bonus = 0.0
                dot = (
                    (anchor_to_goal_dx * goal_vec_x + anchor_to_goal_dy * goal_vec_y)
                    / (start_to_goal * max(d_anchor_goal, 0.01))
                )
                dot = max(-1.0, min(1.0, dot))
                if dot < -0.3:
                    reverse_bonus = min(1.0, (-dot - 0.3) / 0.7)
                score -= self.w_reverse_bonus * reverse_bonus

            scored.append((score, a))

        scored.sort(key=lambda item: item[0])
        return scored

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
        self.lattice.configure_for_family(task_family)

        if self._check_collision(state, scene, vehicle_model):
            return PlanResult(
                success=False,
                fail_reason="start_collision",
                planning_time_ms=(time.perf_counter() - start_time) * 1000.0,
                teacher_name=self.name,
                scenario_family=task_family,
                seed=seed,
            )

        if self._check_success(state, slot, vehicle_model):
            return PlanResult(
                success=True,
                states=[state],
                total_cost=0.0,
                planning_time_ms=(time.perf_counter() - start_time) * 1000.0,
                teacher_name=self.name,
                scenario_family=task_family,
                seed=seed,
            )

        remaining_ms = self.max_total_time_ms - (time.perf_counter() - start_time) * 1000.0
        if remaining_ms <= 0:
            return self._fail_result(start_time, "timeout", task_family, seed)

        is_parallel_rev = (task_family == "parallel_rev")
        direct_budget_ratio = 0.3 if is_parallel_rev else 1.0
        direct_budget_ms = min(remaining_ms * direct_budget_ratio * 0.5, 1500.0)

        orig_expansions = self.lattice.max_expansions
        orig_time = self.lattice.max_time_ms
        try:
            self.lattice.max_expansions = max(500, int(orig_expansions * direct_budget_ratio * 0.5))
            self.lattice.max_time_ms = direct_budget_ms
            direct_result = self.lattice.plan_from_state(state, scene, slot, vehicle_model)
        finally:
            self.lattice.max_expansions = orig_expansions
            self.lattice.max_time_ms = orig_time

        if direct_result.success:
            direct_result.teacher_name = self.name
            return direct_result

        anchors = self._generate_anchors(slot, scene, vehicle_model)
        if not anchors:
            return self._fail_result(start_time, "no_anchors", task_family, seed)

        scored = self._score_anchors(state, anchors, slot, scene, task_family)
        top_anchors = [a for _, a in scored[: self.top_anchors]]

        remaining_ms = self.max_total_time_ms - (time.perf_counter() - start_time) * 1000.0
        if remaining_ms <= 0:
            return self._fail_result(start_time, "timeout", task_family, seed)

        n_anchors = len(top_anchors)
        segment_share = min(self.segment_budget_ratio, 1.0 / max(1, n_anchors * 2))
        seg_time = min(remaining_ms * segment_share, 1000.0)
        seg_expansions = max(300, int(orig_expansions * segment_share))

        valid_results = []
        valid_scores = []

        try:
            self.lattice.max_expansions = seg_expansions
            self.lattice.max_time_ms = seg_time

            for ai, anchor in enumerate(top_anchors):
                elapsed = (time.perf_counter() - start_time) * 1000.0
                if elapsed > self.max_total_time_ms:
                    break

                seg1 = self.lattice.plan_from_state(
                    state, scene,
                    DirectedParkingSlot(
                        x_goal=anchor["x"],
                        y_goal=anchor["y"],
                        theta_goal=anchor["heading"],
                        front_body_length=slot.front_body_length,
                        front_body_width=slot.front_body_width,
                    ),
                    vehicle_model,
                )

                if not seg1.success:
                    continue

                seg1_end = seg1.states[-1] if seg1.states else state
                seg2 = self.lattice.plan_from_state(
                    seg1_end, scene, slot, vehicle_model,
                )

                if not seg2.success:
                    continue

                combined = self._combine_results(
                    seg1, seg2, task_family, seed, scene, vehicle_model,
                )
                valid_results.append((ai, anchor, seg1, seg2, combined))
                valid_scores.append((scored[ai][0], combined))

        finally:
            self.lattice.max_expansions = orig_expansions
            self.lattice.max_time_ms = orig_time

        if valid_results:
            valid_scores.sort(key=lambda x: x[1].total_cost)
            best = valid_scores[0][1]
            elapsed = (time.perf_counter() - start_time) * 1000.0
            best.planning_time_ms = elapsed
            best.teacher_name = self.name
            _, anchor, _, _, _ = valid_results[0]
            best.selected_anchor_sequence = [anchor.get("label", "?")]
            return best

        return self._fail_result(start_time, "no_anchor_path", task_family, seed)

    def _combine_results(self, seg1, seg2, task_family, seed, scene, vehicle_model):
        all_states = list(seg1.states) + list(seg2.states[1:])
        all_actions_norm = list(seg1.actions_normalized) + list(seg2.actions_normalized)
        all_actions_phys = list(seg1.actions_physical) + list(seg2.actions_physical)
        total_cost = seg1.total_cost + seg2.total_cost
        path_length = seg1.path_length + seg2.path_length
        num_steps = len(all_actions_norm)
        n_switches = seg1.num_gear_switches + seg2.num_gear_switches
        n_zero = seg1.num_zero_speed_steps + seg2.num_zero_speed_steps

        final_state = all_states[-1]
        min_clear = min(seg1.min_clearance, seg2.min_clearance)

        return PlanResult(
            success=True,
            states=all_states,
            actions_normalized=all_actions_norm,
            actions_physical=all_actions_phys,
            total_cost=total_cost,
            path_length=path_length,
            num_steps=num_steps,
            num_gear_switches=n_switches,
            num_zero_speed_steps=n_zero,
            final_position_error=seg2.final_position_error,
            final_heading_error=seg2.final_heading_error,
            final_phi=seg2.final_phi,
            final_overlap=seg2.final_overlap,
            min_clearance=min_clear,
            planning_time_ms=seg1.planning_time_ms + seg2.planning_time_ms,
            teacher_name=self.name,
            scenario_family=task_family,
            seed=seed,
        )

    def _fail_result(self, start_time, reason, task_family, seed):
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return PlanResult(
            success=False,
            fail_reason=reason,
            timeout="timeout" in reason,
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
