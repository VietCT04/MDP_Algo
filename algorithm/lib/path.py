from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

from algorithm.lib.car import CarState


class DubinsPathType(Enum):
    """Types of Dubins paths for car-like motion."""
    RSR = "rsr"  # Right-Straight-Right
    RSL = "rsl"  # Right-Straight-Left
    LSR = "lsr"  # Left-Straight-Right
    LSL = "lsl"  # Left-Straight-Left
    RLR = "rlr"  # Right-Left-Right
    LRL = "lrl"  # Left-Right-Left


@dataclass
class Obstacle:
    """Obstacle in the arena with an image on one side."""
    x: int  # bottom-left corner x coordinate (cm)
    y: int  # bottom-left corner y coordinate (cm)
    image_side: str  # 'E', 'N', 'W', 'S' - which side has the image


@dataclass
class DubinsPath:
    """
    A Dubins path between two car configurations.

    NOTE: We extend the basic structure with:
      - segment_types: ['L'|'R'|'S', 'L'|'R'|'S', 'L'|'R'|'S']
      - segment_lengths: [arc1_cm, straight_cm, arc2_cm]  (straight_cm may be 0 for RLR/LRL)
      - metadata: e.g., {"reverse_on_straight": True}
    """
    path_type: DubinsPathType
    length: float  # total path length in cm
    start_state: CarState
    end_state: CarState
    waypoints: List[Tuple[float, float]]  # sequence of (x,y) points: [start, t1, t2, end]
    turn_points: List[Tuple[float, float]]  # [t1, t2]
    segment_types: List[str] = field(default_factory=list)
    segment_lengths: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)