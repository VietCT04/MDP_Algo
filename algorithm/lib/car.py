import math
from enum import Enum
from dataclasses import dataclass

class CarAction(Enum):
    """Actions the car can perform."""
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"


@dataclass
class CarState:
    x: float  # cm from bottom-left corner
    y: float  # cm from bottom-left corner
    theta: float  # orientation in radians (0 = facing east)

    def distance_to(self, other: 'CarState') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: 'CarState') -> float:
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass
class CarCommand:
    action: CarAction
    parameters: dict
    expected_end_state: CarState

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π] range."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_difference(angle1: float, angle2: float) -> float:
    """Shortest angular difference between two angles."""
    diff = angle2 - angle1
    return normalize_angle(diff)