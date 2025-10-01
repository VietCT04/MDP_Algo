"""
Fixed Dubins-aware controller that follows curve geometry without bugs.
(Updated: CarMissionManager now prefers shortest-time Hamiltonian planning.)
"""

import math
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from algorithm.lib.car import CarState, CarAction, CarCommand, normalize_angle, angle_difference
from algorithm.config import config
from algorithm.lib.path import DubinsPath, DubinsPathType
from algorithm.lib.pathfinding import CarPathPlanner

class DubinsSegmentType(Enum):
    LEFT_TURN = "L"
    RIGHT_TURN = "R"
    STRAIGHT = "S"

@dataclass
class DubinsSegmentInfo:
    segment_type: 'DubinsSegmentType'
    segment_index: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    start_heading: float
    end_heading: float
    center_point: Optional[Tuple[float, float]] = None

@dataclass
class CarStatus:
    estimated_state: CarState
    confidence_radius: float
    last_command: Optional[CarCommand] = None
    commands_executed: int = 0


class DubinsAwareController:
    """Controller that follows Dubins path geometry with proper arc movement."""

    
    def _angle_on_circle(self, x: float, y: float, cx: float, cy: float) -> float:
        return math.atan2(y - cy, x - cx)

    def _remaining_arc(self, cur_ang: float, end_ang: float, left: bool) -> float:
        """Unsigned remaining sweep from cur_ang to end_ang going CCW (left=True) or CW (left=False)."""
        if left:
            return (end_ang - cur_ang) % (2*math.pi)   # CCW
        else:
            return (cur_ang - end_ang) % (2*math.pi)   # CW

    def __init__(self):
        self.waypoint_tolerance = config.pathfinding.waypoint_tolerance
        self.angle_tolerance = config.pathfinding.angle_tolerance
        self.max_forward_step = config.pathfinding.max_forward_step
        self.turning_radius = config.car.turning_radius

        self.current_path: Optional[List[DubinsPath]] = None
        self.path_index: int = 0
        self.current_segment: Optional[DubinsSegmentInfo] = None
        self.image_recognition_time: int = 0
        self.at_target: bool = False

    def set_path(self, path_segments: List[DubinsPath]):
        self.current_path = path_segments
        self.path_index = 0
        self.current_segment = None
        self.image_recognition_time = 0
        self.at_target = False

        print(f"Controller: Set path with {len(path_segments)} segments")
        for i, segment in enumerate(path_segments):
            label = segment.path_type.value.upper() if hasattr(segment.path_type, "value") else str(segment.path_type)
            print(f"  Path {i}: {label} ({segment.length:.1f}cm)")

    def get_next_command(self, car_status: CarStatus) -> Optional[CarCommand]:
        if not self.current_path:
            return CarCommand(CarAction.STOP, {"reason": "no_path"}, car_status.estimated_state)
        if self.path_index >= len(self.current_path):
            return CarCommand(CarAction.STOP, {"reason": "path_complete"}, car_status.estimated_state)

        cur_path = self.current_path[self.path_index]
        current_state = car_status.estimated_state

        # Initialize first segment of current Dubins path
        if self.current_segment is None:
            self.current_segment = self._initialize_first_segment(cur_path)
            if self.current_segment is None:
                self.path_index += 1
                return self.get_next_command(car_status)

        # If we've reached end of this Dubins path, do image recognition pause
        if self._is_at_dubins_path_end(current_state, cur_path):
            if not self.at_target:
                self.at_target = True
                self.image_recognition_time = 20
            if self.image_recognition_time > 0:
                self.image_recognition_time -= 1
                return CarCommand(CarAction.STOP, {"reason": "image_recognition", "frames_remaining": self.image_recognition_time}, current_state)
            # move on
            self.path_index += 1
            self.current_segment = None
            self.at_target = False
            return self.get_next_command(car_status)

        # Segment transition
        if self._is_current_segment_complete(current_state):
            next_seg = self._get_next_segment(cur_path)
            if next_seg is not None:
                self.current_segment = next_seg

        return self._generate_dubins_command(current_state)

    # ---- segment helpers (same as your working version) ----
    def _initialize_first_segment(self, dubins_path: DubinsPath) -> Optional[DubinsSegmentInfo]:
        path_type = dubins_path.path_type.value.lower()
        waypoints = dubins_path.waypoints
        if len(waypoints) < 2 or len(path_type) < 1:
            return None
        segment_char = path_type[0].upper()
        segment_type = DubinsSegmentType(segment_char)
        start_point = waypoints[0]
        end_point = waypoints[1]
        start_heading = dubins_path.start_state.theta

        if segment_type == DubinsSegmentType.STRAIGHT:
            end_heading = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        else:
            center = self._calculate_turn_center(start_point, start_heading, segment_type)
            end_heading = self._calculate_heading_at_point(end_point, center, segment_type)

        seg = DubinsSegmentInfo(segment_type, 0, start_point, end_point, start_heading, end_heading)
        if segment_type != DubinsSegmentType.STRAIGHT:
            seg.center_point = self._calculate_turn_center(start_point, start_heading, segment_type)
        return seg

    def _calculate_turn_center(self, point: Tuple[float, float], heading: float, turn_type: DubinsSegmentType) -> Tuple[float, float]:
        x, y = point
        r = self.turning_radius
        if turn_type == DubinsSegmentType.LEFT_TURN:
            return (x - r * math.sin(heading), y + r * math.cos(heading))
        else:
            return (x + r * math.sin(heading), y - r * math.cos(heading))

    def _calculate_heading_at_point(self, point: Tuple[float, float], center: Tuple[float, float], turn_type: DubinsSegmentType) -> float:
        px, py = point
        cx, cy = center
        dx, dy = px - cx, py - cy
        if turn_type == DubinsSegmentType.LEFT_TURN:
            heading = math.atan2(dx, -dy)
        else:
            heading = math.atan2(-dx, dy)
        return normalize_angle(heading)

    def _is_current_segment_complete(self, current_state: CarState) -> bool:
        if not self.current_segment:
            return True
        end_x, end_y = self.current_segment.end_point
        distance = math.hypot(current_state.x - end_x, current_state.y - end_y)
        if self.current_segment.segment_type != DubinsSegmentType.STRAIGHT:
            angle_diff = abs(angle_difference(current_state.theta, self.current_segment.end_heading))
            return distance <= self.waypoint_tolerance and angle_diff <= self.angle_tolerance
        return distance <= self.waypoint_tolerance

    def _is_at_dubins_path_end(self, current_state: CarState, dubins_path: DubinsPath) -> bool:
        end_state = dubins_path.end_state
        distance = math.hypot(current_state.x - end_state.x, current_state.y - end_state.y)
        angle_diff_v = abs(angle_difference(current_state.theta, end_state.theta))
        return distance <= (self.waypoint_tolerance * 2) and angle_diff_v <= (self.angle_tolerance * 3)

    def _get_next_segment(self, dubins_path: DubinsPath) -> Optional[DubinsSegmentInfo]:
        if not self.current_segment or self.current_segment.segment_index >= 2:
            return None
        path_type = dubins_path.path_type.value.lower()
        next_index = self.current_segment.segment_index + 1
        waypoints = dubins_path.waypoints
        if next_index >= len(path_type) or next_index + 1 >= len(waypoints):
            return None
        segment_char = path_type[next_index].upper()
        segment_type = DubinsSegmentType(segment_char)
        start_point = waypoints[next_index]
        end_point = waypoints[next_index + 1]
        start_heading = self.current_segment.end_heading
        if segment_type == DubinsSegmentType.STRAIGHT:
            end_heading = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        else:
            center = self._calculate_turn_center(start_point, start_heading, segment_type)
            end_heading = self._calculate_heading_at_point(end_point, center, segment_type)
        seg = DubinsSegmentInfo(segment_type, next_index, start_point, end_point, start_heading, end_heading)
        if segment_type != DubinsSegmentType.STRAIGHT:
            seg.center_point = self._calculate_turn_center(start_point, start_heading, segment_type)
        return seg

    def _generate_dubins_command(self, current_state: CarState) -> CarCommand:
        if not self.current_segment:
            return CarCommand(CarAction.STOP, {"reason": "no_segment"}, current_state)
        if self.current_segment.segment_type == DubinsSegmentType.STRAIGHT:
            return self._cmd_straight(current_state)
        return self._cmd_arc(current_state)

    def _cmd_straight(self, current_state: CarState) -> CarCommand:
        end_x, end_y = self.current_segment.end_point
        target_heading = math.atan2(end_y - current_state.y, end_x - current_state.x)
        heading_error = angle_difference(current_state.theta, target_heading)

        if abs(heading_error) > self.angle_tolerance * 2:
            turn_amount = min(abs(heading_error), 0.2)
            if heading_error > 0:
                new_theta = normalize_angle(current_state.theta + turn_amount)
                return CarCommand(CarAction.TURN_LEFT, {"angle": turn_amount}, CarState(current_state.x, current_state.y, new_theta))
            else:
                new_theta = normalize_angle(current_state.theta - turn_amount)
                return CarCommand(CarAction.TURN_RIGHT, {"angle": turn_amount}, CarState(current_state.x, current_state.y, new_theta))

        distance_to_end = math.hypot(end_x - current_state.x, end_y - current_state.y)
        move_distance = min(distance_to_end, self.max_forward_step * 1.5)
        expected_x = current_state.x + move_distance * math.cos(current_state.theta)
        expected_y = current_state.y + move_distance * math.sin(current_state.theta)
        return CarCommand(CarAction.FORWARD, {"distance": move_distance}, CarState(expected_x, expected_y, current_state.theta))

    # def _cmd_arc(self, current_state: CarState) -> CarCommand:
    #     cx, cy = self.current_segment.center_point
    #     cur_ang = math.atan2(current_state.y - cy, current_state.x - cx)
    #     angular_step = self.max_forward_step / self.turning_radius
    #     direction = +1 if self.current_segment.segment_type == DubinsSegmentType.LEFT_TURN else -1
    #     target_angle = cur_ang + direction * angular_step
    #     new_x = cx + self.turning_radius * math.cos(target_angle)
    #     new_y = cy + self.turning_radius * math.sin(target_angle)
    #     if self.current_segment.segment_type == DubinsSegmentType.LEFT_TURN:
    #         new_theta = normalize_angle(target_angle + math.pi / 2)
    #         action = CarAction.TURN_LEFT
    #     else:
    #         new_theta = normalize_angle(target_angle - math.pi / 2)
    #         action = CarAction.TURN_RIGHT
    #     turn_angle = abs(angle_difference(current_state.theta, new_theta))
    #     return CarCommand(action, {"angle": turn_angle, "arc_movement": True}, CarState(new_x, new_y, new_theta))
    
    def _cmd_arc(self, current_state: CarState) -> CarCommand:
        cx, cy = self.current_segment.center_point

        cur_ang = self._angle_on_circle(current_state.x, current_state.y, cx, cy)

        angular_step = self.max_forward_step / self.turning_radius  # rad

        left = (self.current_segment.segment_type == DubinsSegmentType.LEFT_TURN)
        direction = +1 if left else -1

        ex, ey = self.current_segment.end_point
        end_ang = self._angle_on_circle(ex, ey, cx, cy)

        remaining = self._remaining_arc(cur_ang, end_ang, left)
        step = min(angular_step, remaining)

        target_angle = cur_ang + direction * step
        new_x = cx + self.turning_radius * math.cos(target_angle)
        new_y = cy + self.turning_radius * math.sin(target_angle)

        if left:
            new_theta = normalize_angle(target_angle + math.pi / 2)
            action = CarAction.TURN_LEFT
        else:
            new_theta = normalize_angle(target_angle - math.pi / 2)
            action = CarAction.TURN_RIGHT

        turn_angle = abs(angle_difference(current_state.theta, new_theta))

        return CarCommand(
            action,
            {"angle": turn_angle, "arc_movement": True},
            CarState(new_x, new_y, new_theta)
    )

    # ---- public status/update ----
    def update_car_position(self, car_status: CarStatus, executed_command: CarCommand, actual_result: Dict[str, Any] = None) -> CarStatus:
        if actual_result and 'measured_position' in actual_result:
            measured_pos = actual_result['measured_position']
            new_state = CarState(measured_pos['x'], measured_pos['y'], measured_pos['theta'])
            new_confidence = config.car.position_drift
        else:
            new_state = executed_command.expected_end_state
            new_confidence = car_status.confidence_radius + config.car.position_drift
        return CarStatus(new_state, new_confidence, executed_command, car_status.commands_executed + 1)

    def get_path_progress(self) -> Dict[str, Any]:
        if not self.current_path:
            return {"status": "no_path"}
        total = len(self.current_path)
        cur_idx = self.path_index
        dubins_progress = 0.0
        if self.current_segment and cur_idx < total:
            dubins_progress = (self.current_segment.segment_index + 1) / 3.0
        overall = (cur_idx + dubins_progress) / max(1, total)
        return {
            "status": "active" if cur_idx < total else "complete",
            "total_segments": total,
            "current_segment": cur_idx,
            "dubins_segment_index": self.current_segment.segment_index if self.current_segment else -1,
            "dubins_segment_type": self.current_segment.segment_type.value if self.current_segment else "None",
            "overall_progress": overall,
            "at_target": self.at_target,
            "image_recognition_time": self.image_recognition_time
        }


class CarMissionManager:
    """Mission manager using the Dubins-aware controller."""

    def __init__(self):
        self.path_planner = CarPathPlanner()
        self.controller = DubinsAwareController()
        self.car_status: Optional[CarStatus] = None
        self.mission_targets: List[int] = []
        self.visited_targets: List[int] = []

    def initialize_car(self, x: float, y: float, theta: float = 0.0):
        self.car_status = CarStatus(CarState(x, y, theta), 1.0)

    def add_obstacle(self, x: int, y: int, image_side: str) -> int:
        self.path_planner.add_obstacle(x, y, image_side)
        return len(self.path_planner.obstacles) - 1

    def plan_mission(self, target_obstacle_indices: List[int]) -> bool:
        if not self.car_status:
            return False

        print(f"Planning mission to visit obstacles: {target_obstacle_indices}")

        # Prefer exact shortest-time Hamiltonian (B.3)
        segments = self.path_planner.plan_visiting_path(
            self.car_status.estimated_state, target_obstacle_indices
        )
        if not segments:
            print("Mission planning failed - no valid path found")
            return False

        self.controller.set_path(segments)
        self.mission_targets = target_obstacle_indices.copy()
        self.visited_targets = []
        print(f"Mission planned: {len(segments)} segments to visit {len(target_obstacle_indices)} targets")
        return True

    def get_next_action(self) -> Optional[CarCommand]:
        if not self.car_status:
            return None
        return self.controller.get_next_command(self.car_status)

    def execute_command(self, command: CarCommand, actual_result: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.car_status:
            return {"status": "error", "message": "Car not initialized"}

        self.car_status = self.controller.update_car_position(self.car_status, command, actual_result)

        progress = self.controller.get_path_progress()
        current_segment = progress.get("current_segment", 0)
        if (current_segment > len(self.visited_targets) and
                len(self.visited_targets) < len(self.mission_targets)):
            target_idx = len(self.visited_targets)
            self.visited_targets.append(self.mission_targets[target_idx])

        return {
            "status": "success",
            "command_executed": command.action.value,
            "estimated_position": {
                "x": self.car_status.estimated_state.x,
                "y": self.car_status.estimated_state.y,
                "theta": self.car_status.estimated_state.theta
            },
            "position_confidence_radius": self.car_status.confidence_radius,
            "progress": progress
        }

    def get_status(self) -> Dict[str, Any]:
        if not self.car_status:
            return {"status": "not_initialized"}
        progress = self.controller.get_path_progress()
        return {
            "car_status": {
                "position": {
                    "x": self.car_status.estimated_state.x,
                    "y": self.car_status.estimated_state.y,
                    "theta": self.car_status.estimated_state.theta,
                    "theta_degrees": math.degrees(self.car_status.estimated_state.theta)
                },
                "confidence_radius": self.car_status.confidence_radius,
                "commands_executed": self.car_status.commands_executed
            },
            "mission": {
                "targets": self.mission_targets,
                "visited": self.visited_targets,
                "progress": progress
            },
            "obstacles_count": len(self.path_planner.obstacles)
        }