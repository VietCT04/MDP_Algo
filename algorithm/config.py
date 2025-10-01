from dataclasses import dataclass

@dataclass
class RobotConfig:
    width: float = 20
    length: float = 21
    turning_radius: float = 15.0
    camera_distance: float = 25.0
    linear_speed_cm_s: float = 10.0
    angular_speed_rad_s: float = 1.2
    reverse_linear_speed_cm_s: float = 8.0
    image_recognition_time_s: float = 0.0
    forward_motion_error: float = 0.0
    turn_angle_error: float = 0.0
    position_drift: float = 0.0
    scan_retreat_cm: float = 12.0

@dataclass
class vision:
    scan_seconds = 5.0

@dataclass
class Arena:
    size: int = 200
    grid_cell_size: int = 10
    obstacle_size: int = 10
    collision_buffer: int = 3

@dataclass
class PathfindingParams:
    waypoint_tolerance: float = 10.0
    angle_tolerance: float = 0.3
    max_forward_step: float = 8.0
    max_turn_step: float = 0.2

@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True

class Config:
    def __init__(self, config_file: str = "car_config.json"):
        self.config_file = config_file
        self.car = RobotConfig()
        self.arena = Arena()
        self.pathfinding = PathfindingParams()
        self.server = ServerConfig()
        self.vision = vision()

    def get_grid_size(self) -> int:
        return self.arena.size // self.arena.grid_cell_size

# Global configuration
config = Config()
