"""
Flask server for robot car pathfinding API.
Provides HTTP endpoints for car control, path planning, and status monitoring.
"""

from algorithm.lib.path import DubinsPath
from algorithm.lib.pathfinding import CarPathPlanner
from flask import Flask, request, jsonify
import logging
from typing import Dict, Any, List
import math
import json

from algorithm.lib.controller import CarMissionManager
from algorithm.lib.car import CarState, CarCommand, CarAction
from algorithm.config import config
from algorithm.lib.orders_export import export_compact_orders_from_paths


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global mission manager instance
from threading import Lock

mission_manager: CarMissionManager | None = None
_initialized = False
_init_lock = Lock()

def init_app_once():
    """Initialize mission manager exactly once per process."""
    global mission_manager, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        mission_manager = CarMissionManager()
        logger.info("Car pathfinding server initialized")
        _initialized = True

# Ensure initialization before handling any request
@app.before_request
def _ensure_initialized():
    init_app_once()

# @app.before_first_request
# def initialize_server():
#     """Initialize the mission manager on first request."""
#     global mission_manager
#     mission_manager = CarMissionManager()
#     logger.info("Car pathfinding server initialized")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "robot_car_pathfinding",
        "config": {
            "turning_radius": config.car.turning_radius,
            "arena_size": f"{config.arena.size}x{config.arena.size}cm"
        }
    })


@app.route('/car/initialize', methods=['POST'])
def initialize_car():
    """
    Initialize car at starting position.

    Expected JSON:
    {
        "x": 20.0,
        "y": 20.0,
        "theta": 0.0  // optional, defaults to 0
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        theta = float(data.get('theta', 0))

        # Validate position is within arena
        if not (0 <= x <= config.arena.size and 0 <= y <= config.arena.size):
            return jsonify({"error": "Position outside arena bounds"}), 400

        mission_manager.initialize_car(x, y, theta)

        logger.info(f"Car initialized at ({x}, {y}) facing {theta:.3f} rad")

        return jsonify({
            "status": "success",
            "message": "Car initialized",
            "position": {"x": x, "y": y, "theta": theta}
        })

    except Exception as e:
        logger.error(f"Car initialization failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

import math
from algorithm.lib.car import CarAction  # ensure this import exists

import math
from algorithm.lib.car import CarAction  # ensure this import exists

import math
from algorithm.lib.car import CarAction

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi  # (-pi, pi]

def _signed_delta(a: float, b: float) -> float:
    # shortest signed angle taking you from a -> b
    return _wrap_pi(b - a)

def _emit_token_orders(
    mgr,
    *,
    num_targets: int | None = None,
    max_steps: int = 100000,
    include_pose: bool = True,
):
    tokens = []
    stops_seen = 0
    steps = 0

    while True:
        cmd = mgr.get_next_action()
        if not cmd:
            break

        start_theta = mgr.car_status.estimated_state.theta
        end = cmd.expected_end_state
        act = cmd.action

        if act == CarAction.FORWARD:
            dist = float(cmd.parameters.get("distance", 0.0))
            token = {"op": "F", "value_cm": round(dist, 2)}

        elif act in (CarAction.TURN_LEFT, CarAction.TURN_RIGHT):
            # Decide direction from the *actual* heading change
            dtheta = _signed_delta(start_theta, end.theta)
            op = "L" if dtheta >= 0.0 else "R"
            token = {"op": op, "value_deg": round(abs(math.degrees(dtheta)), 1)}

        elif act == CarAction.STOP:
            token = {"op": "S"}

        else:
            token = {"op": act.name}

        if include_pose and end is not None:
            token["pose"] = {
                "x": round(end.x, 2),
                "y": round(end.y, 2),
                "theta": round(end.theta, 4),
                "theta_deg": round(math.degrees(end.theta), 1),
            }

        tokens.append(token)

        # Advance planner (simulate perfect execution)
        if end is not None:
            mgr.execute_command(cmd, {
                "measured_position": {"x": end.x, "y": end.y, "theta": end.theta}
            })
        else:
            s = mgr.car_status.estimated_state
            mgr.execute_command(cmd, {
                "measured_position": {"x": s.x, "y": s.y, "theta": s.theta}
            })

        if act == CarAction.STOP:
            stops_seen += 1
            if num_targets is not None and stops_seen >= num_targets:
                break

        steps += 1
        if steps >= max_steps:
            raise RuntimeError("Exceeded max_steps while compiling orders (planner did not finish).")

    return tokens




def _pretty_strings(tokens):
    """
    Optional: human-friendly strings, e.g. ["F 10cm", "L 90°", "R 45°", "S"]
    """
    out = []
    for t in tokens:
        if t["op"] == "F":
            out.append(f"F {t['value_cm']}cm")
        elif t["op"] in ("L", "R"):
            out.append(f"{t['op']} {t['value_deg']}°")
        else:
            out.append("S")
    return out

@app.post("/mission/compile_orders")
def compile_orders():
    """
    Plan a full mission from a single request and return ALL movement orders at once.

    Request JSON shape (now supports obstacle_id):
    {
      "start": {"x": 20.0, "y": 20.0, "theta": 0.0},
      "obstacles": [
        {"obstacle_id": 0, "x": 60, "y": 120, "image_side": "N"},
        {"obstacle_id": 1, "x": 140, "y": 80, "image_side": "E"},
        {"obstacle_id": 2, "x": 180, "y": 180, "image_side": "S"}
      ],
      "targets": [0, 1, 2]   // optional; defaults to all obstacles in order
    }
    """
    try:
        # ---- permissive JSON object parsing ----
        data = request.get_json(silent=True)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return jsonify({"error": "Body was a JSON string, but not valid JSON."}), 400
        if data is None:
            raw = (request.get_data(as_text=True) or "").strip()
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return jsonify({"error": "Body must be valid JSON."}), 400
            else:
                data = {}
        if not isinstance(data, dict):
            return jsonify({"error": "Body must be a JSON object (or a JSON string containing an object)."}), 400
        # ---- end parsing block ----
        mode = "discrete"

        start = data.get("start") or {}
        sx = float(start.get("x", 20.0))
        sy = float(start.get("y", 20.0))
        st = float(start.get("theta", 0.0))

        obstacles = data.get("obstacles", [])
        if not isinstance(obstacles, list) or not obstacles:
            return jsonify({"error": "obstacles must be a non-empty list"}), 400

        # Build a mapping from internal obstacle index -> user obstacle_id (or fallback to index)
        index_to_obstacle_id: list[int] = []
        for i, o in enumerate(obstacles):
            if not all(k in o for k in ("x", "y", "image_side")):
                return jsonify({"error": f"obstacle #{i} missing x/y/image_side"}), 400
            if str(o["image_side"]).upper() not in ("N", "E", "S", "W"):
                return jsonify({"error": f"obstacle #{i} image_side must be one of N,E,S,W"}), 400
            try:
                oid = int(o.get("obstacle_id", i))   # <--- NEW: accept obstacle_id, fallback to position
            except (TypeError, ValueError):
                return jsonify({"error": f"obstacle #{i} obstacle_id must be integer if provided"}), 400
            index_to_obstacle_id.append(oid)

        if "targets" in data:
            if not isinstance(data["targets"], list):
                return jsonify({"error": "targets must be a list of indices"}), 400
            targets = list(map(int, data["targets"]))
        else:
            targets = list(range(len(obstacles)))

        planner = CarPathPlanner()
        for o in obstacles:
            x = int(o["x"]); y = int(o["y"]); side = str(o["image_side"]).upper()
            if not (0 <= x <= config.arena.size - config.arena.obstacle_size and
                    0 <= y <= config.arena.size - config.arena.obstacle_size):
                return jsonify({"error": f"obstacle ({x},{y}) outside valid area"}), 400
            planner.add_obstacle(x, y, side)

        start_state = CarState(sx, sy, st)

        if mode == "discrete":  # <--- NEW: grid A* with only 45°/90° turns + F/B
            orders = planner.plan_visiting_orders_discrete(start_state, targets)
            if not orders:
                return jsonify({"status":"failed","message":"Discrete planner failed"}), 400
            payload = {"orders": orders}
        else:
            # ---- existing Dubins mission path via CarMissionManager ----
            tmp = CarMissionManager()
            tmp.initialize_car(sx, sy, st)
            for o in obstacles:
                tmp.add_obstacle(int(o["x"]), int(o["y"]), str(o["image_side"]).upper())

            if not tmp.plan_mission(targets):
                return jsonify({"status":"failed","message":"Could not plan mission"}), 400

            planned_paths = tmp.controller.current_path
            for p in planned_paths:
                if not isinstance(p, DubinsPath):
                    raise RuntimeError("Expected DubinsPath in planned paths")
            payload = export_compact_orders_from_paths(planned_paths)

        # --- Annotate each 'S' with obstacle_id in targets order (same as before) ---
        target_id_queue = [index_to_obstacle_id[t] for t in targets]
        next_idx = 0
        if isinstance(payload, dict) and isinstance(payload.get("orders"), list):
            for cmd in payload["orders"]:
                if cmd.get("op") == "S":
                    if next_idx < len(target_id_queue):
                        cmd["obstacle_id"] = target_id_queue[next_idx]
                        next_idx += 1

        return jsonify(payload)

    except Exception as e:
        logger.exception("compile_orders failed")
        return jsonify({"error": str(e)}), 500


@app.route('/obstacles/add', methods=['POST'])
def add_obstacle():
    """
    Add an obstacle to the arena.

    Expected JSON:
    {
        "x": 50,
        "y": 50,
        "image_side": "S"  // 'E', 'N', 'W', 'S'
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        x = int(data.get('x'))
        y = int(data.get('y'))
        image_side = data.get('image_side', '').upper()

        if image_side not in ['E', 'N', 'W', 'S']:
            return jsonify({"error": "image_side must be E, N, W, or S"}), 400

        # Validate obstacle position
        if not (0 <= x <= config.arena.size - config.arena.obstacle_size and
                0 <= y <= config.arena.size - config.arena.obstacle_size):
            return jsonify({"error": "Obstacle position outside valid area"}), 400

        obstacle_id = mission_manager.add_obstacle(x, y, image_side)

        logger.info(f"Obstacle {obstacle_id} added at ({x}, {y}) with image on {image_side} side")

        return jsonify({
            "status": "success",
            "obstacle_id": obstacle_id,
            "position": {"x": x, "y": y},
            "image_side": image_side
        })

    except Exception as e:
        logger.error(f"Add obstacle failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/mission/plan', methods=['POST'])
def plan_mission():
    """
    Plan a mission to visit specified obstacles.

    Expected JSON:
    {
        "targets": [0, 1, 2]  // obstacle indices to visit
    }
    """
    global mission_manager

    try:
        data = request.get_json() or {}
        targets = data.get('targets', [])

        if not isinstance(targets, list):
            return jsonify({"error": "targets must be a list of obstacle indices"}), 400

        success = mission_manager.plan_mission(targets)

        if not success:
            return jsonify({
                "status": "failed",
                "message": "Could not plan path to visit specified obstacles"
            }), 400

        logger.info(f"Mission planned to visit obstacles: {targets}")

        return jsonify({
            "status": "success",
            "message": f"Mission planned to visit {len(targets)} obstacles",
            "targets": targets
        })

    except Exception as e:
        logger.error(f"Mission planning failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/car/next_action', methods=['GET'])
def get_next_action():
    """Get the next action for the car to perform."""
    global mission_manager

    try:
        command = mission_manager.get_next_action()

        if not command:
            return jsonify({
                "status": "no_action",
                "message": "No action available (mission not planned or complete)"
            })

        response = {
            "status": "success",
            "action": command.action.value,
            "parameters": command.parameters,
            "expected_end_state": {
                "x": command.expected_end_state.x,
                "y": command.expected_end_state.y,
                "theta": command.expected_end_state.theta
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Get next action failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/car/execute', methods=['POST'])
def execute_command():
    """
    Execute a car command and update position.

    Expected JSON:
    {
        "action": "forward",
        "parameters": {"distance": 10.0},
        "actual_result": {  // optional - actual sensor readings
            "measured_position": {"x": 25.2, "y": 20.1, "theta": 0.05}
        }
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        action_str = data.get('action', '')
        parameters = data.get('parameters', {})
        actual_result = data.get('actual_result', {})

        # Parse action
        try:
            action = CarAction(action_str)
        except ValueError:
            return jsonify({"error": f"Invalid action: {action_str}"}), 400

        # Create command (we need expected end state, but server doesn't know it)
        # This is a limitation - ideally the command comes from get_next_action
        current_status = mission_manager.car_status
        if not current_status:
            return jsonify({"error": "Car not initialized"}), 400

        # Create a dummy command for execution tracking
        command = CarCommand(action, parameters, current_status.estimated_state)

        result = mission_manager.execute_command(command, actual_result)

        logger.info(f"Executed {action_str} with result: {result['status']}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/car/position', methods=['POST'])
def update_car_position():
    """
    Manually update car position (for position corrections).

    Expected JSON:
    {
        "x": 25.5,
        "y": 30.2,
        "theta": 0.52
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        x = float(data.get('x'))
        y = float(data.get('y'))
        theta = float(data.get('theta'))

        if not mission_manager.car_status:
            return jsonify({"error": "Car not initialized"}), 400

        # Update position directly
        mission_manager.car_status.estimated_state = CarState(x, y, theta)
        mission_manager.car_status.confidence_radius = 1.0  # Reset confidence

        logger.info(f"Car position updated to ({x:.2f}, {y:.2f}, {theta:.3f})")

        return jsonify({
            "status": "success",
            "message": "Position updated",
            "position": {"x": x, "y": y, "theta": theta}
        })

    except Exception as e:
        logger.error(f"Position update failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get comprehensive system status."""
    global mission_manager

    try:
        status = mission_manager.get_status()
        return jsonify({
            "status": "success",
            "data": status
        })

    except Exception as e:
        logger.error(f"Get status failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_system():
    """Reset the entire system."""
    global mission_manager

    try:
        mission_manager = CarMissionManager()
        logger.info("System reset complete")

        return jsonify({
            "status": "success",
            "message": "System reset complete"
        })

    except Exception as e:
        logger.error(f"System reset failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Robot Car Pathfinding Server")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /car/initialize      - Initialize car position")
    print("  POST /obstacles/add       - Add obstacle to arena")
    print("  POST /mission/plan        - Plan mission to visit obstacles")
    print("  GET  /car/next_action     - Get next action for car")
    print("  POST /car/execute         - Execute car command")
    print("  POST /car/position        - Update car position")
    print("  GET  /status              - Get system status")
    print("  POST /reset               - Reset system")
    print("\nKey features:")
    print("  • Dubins path planning for car-like motion")
    print("  • Position uncertainty handling")
    print("  • Hamiltonian path optimization")
    print("  • Collision avoidance")
    print(f"\nStarting server on {config.server.host}:{config.server.port}")
    print("=" * 60)

    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug
    )