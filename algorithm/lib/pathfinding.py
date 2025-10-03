"""
Enhanced pathfinding with multiple algorithms for robot car navigation.
Now includes a shortest-time Hamiltonian (TSP) solver using Dubins time as edge cost.
"""
from heapq import heappush, heappop
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Iterable
from itertools import permutations, combinations
from dataclasses import dataclass

from algorithm.lib.car import CarState
from algorithm.lib.path import Obstacle, DubinsPath, DubinsPathType
from algorithm.config import config


@dataclass
class PathfindingResult:
    """Result of pathfinding with debug info."""
    paths: List[DubinsPath]
    total_length: float
    algorithm_used: str
    debug_info: Dict


# ------------ Dubins planner (unchanged geometry) -----------------
class DubinsPlanner:
    """Plans Dubins paths for car-like robot motion."""

    def __init__(self, turning_radius: float = None):
        self.turning_radius = turning_radius or config.car.turning_radius

    def plan_path(self, start: CarState, goal: CarState) -> Optional[DubinsPath]:
        """Plan shortest Dubins path between two car states. Returns None if no valid path exists."""
        best_path = None
        min_length = float('inf')
        for path_type in DubinsPathType:
            try:
                path = self._compute_path(start, goal, path_type)
                if path and path.length > 0 and path.length < min_length: # compare lengths of the valid paths
                    min_length = path.length
                    best_path = path
            except Exception:
                continue
        return best_path

    def _compute_path(self, start: CarState, goal: CarState, path_type: DubinsPathType) -> Optional[DubinsPath]:
        r = self.turning_radius
        if path_type == DubinsPathType.RSR:
            return self._rsr_path(start, goal, r)
        elif path_type == DubinsPathType.RSL:
            return self._rsl_path(start, goal, r)
        elif path_type == DubinsPathType.LSR:
            return self._lsr_path(start, goal, r)
        elif path_type == DubinsPathType.LSL:
            return self._lsl_path(start, goal, r)
        elif path_type == DubinsPathType.RLR:
            return self._rlr_path(start, goal, r)
        else:
            return self._lrl_path(start, goal, r)

    # ---- individual families (same as your working version) ----
    def _rsr_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x + r * math.sin(start.theta)
        c1y = start.y - r * math.cos(start.theta)
        c2x = goal.x + r * math.sin(goal.theta)
        c2y = goal.y - r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        ux, uy = -dy / d, dx / d  # external tangent
        t1x, t1y = c1x + r * ux, c1y + r * uy
        t2x, t2y = c2x + r * ux, c2y + r * uy
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, True)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, True)
        length = abs(alpha * r) + d + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.RSR, length, start, goal, waypoints, turn_points)

    def _rsl_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x + r * math.sin(start.theta)
        c1y = start.y - r * math.cos(start.theta)
        c2x = goal.x - r * math.sin(goal.theta)
        c2y = goal.y + r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 2 * r:
            return None
        try:
            phi = math.acos(2 * r / d)
        except ValueError:
            return None
        theta_t = math.atan2(dy, dx) + phi
        tx, ty = math.cos(theta_t), math.sin(theta_t)
        t1x, t1y = c1x + r * tx, c1y + r * ty
        t2x, t2y = c2x - r * tx, c2y - r * ty
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, True)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, False)
        straight = math.hypot(t2x - t1x, t2y - t1y)
        length = abs(alpha * r) + straight + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.RSL, length, start, goal, waypoints, turn_points)

    def _lsr_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x - r * math.sin(start.theta)
        c1y = start.y + r * math.cos(start.theta)
        c2x = goal.x + r * math.sin(goal.theta)
        c2y = goal.y - r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 2 * r:
            return None
        try:
            phi = math.acos(2 * r / d)
        except ValueError:
            return None
        theta_t = math.atan2(dy, dx) - phi
        tx, ty = math.cos(theta_t), math.sin(theta_t)
        t1x, t1y = c1x + r * tx, c1y + r * ty
        t2x, t2y = c2x - r * tx, c2y - r * ty
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, False)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, True)
        straight = math.hypot(t2x - t1x, t2y - t1y)
        length = abs(alpha * r) + straight + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.LSR, length, start, goal, waypoints, turn_points)

    def _lsl_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x - r * math.sin(start.theta)
        c1y = start.y + r * math.cos(start.theta)
        c2x = goal.x - r * math.sin(goal.theta)
        c2y = goal.y + r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        ux, uy = dy / d, -dx / d  # external tangent (opposite sign of RSR)
        t1x, t1y = c1x + r * ux, c1y + r * uy
        t2x, t2y = c2x + r * ux, c2y + r * uy
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, False)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, False)
        length = abs(alpha * r) + d + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.LSL, length, start, goal, waypoints, turn_points)

    def _rlr_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x + r * math.sin(start.theta)
        c1y = start.y - r * math.cos(start.theta)
        c2x = goal.x + r * math.sin(goal.theta)
        c2y = goal.y - r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d > 4 * r or d < 1e-3:
            return None
        mx, my = (c1x + c2x) / 2, (c1y + c2y) / 2
        h_sq = 4 * r * r - d * d / 4
        if h_sq < 0:
            return None
        h = math.sqrt(h_sq)
        if d <= 0:
            return None
        px, py = -dy / d, dx / d
        c3x, c3y = mx + h * px, my + h * py
        t1x, t1y = (c1x + c3x) / 2, (c1y + c3y) / 2
        t2x, t2y = (c2x + c3x) / 2, (c2y + c3y) / 2
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, True)
        beta = self._arc_angle(t1x, t1y, c3x, c3y, t2x, t2y, False)
        gamma = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, True)
        length = abs(alpha * r) + abs(beta * r) + abs(gamma * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.RLR, length, start, goal, waypoints, turn_points)

    def _lrl_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x - r * math.sin(start.theta)
        c1y = start.y + r * math.cos(start.theta)
        c2x = goal.x - r * math.sin(goal.theta)
        c2y = goal.y + r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d > 4 * r or d < 1e-3:
            return None
        mx, my = (c1x + c2x) / 2, (c1y + c2y) / 2
        h_sq = 4 * r * r - d * d / 4
        if h_sq < 0:
            return None
        h = math.sqrt(h_sq)
        if d <= 0:
            return None
        px, py = dy / d, -dx / d
        c3x, c3y = mx + h * px, my + h * py
        t1x, t1y = (c1x + c3x) / 2, (c1y + c3y) / 2
        t2x, t2y = (c2x + c3x) / 2, (c2y + c3y) / 2
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, False)
        beta = self._arc_angle(t1x, t1y, c3x, c3y, t2x, t2y, True)
        gamma = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, False)
        length = abs(alpha * r) + abs(beta * r) + abs(gamma * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.LRL, length, start, goal, waypoints, turn_points)

    def _arc_angle(self, px: float, py: float, cx: float, cy: float,
                   qx: float, qy: float, clockwise: bool) -> float:
        v1x, v1y = px - cx, py - cy
        v2x, v2y = qx - cx, qy - cy
        angle = math.atan2(v2y, v2x) - math.atan2(v1y, v1x)
        if clockwise:
            if angle > 0:
                angle -= 2 * math.pi
        else:
            if angle < 0:
                angle += 2 * math.pi
        return angle


# ------------------- High-level planner with TSP -------------------
class CarPathPlanner:
    """Path planner with several strategies. Prefers shortest-time Hamiltonian (Held–Karp)."""

    def __init__(self):
        self.dubins_planner = DubinsPlanner()
        self.arena_size = config.arena.size
        self.grid_size = config.get_grid_size()
        self.obstacles: List[Obstacle] = []

        # Collision grid (1 = blocked, 0 = free)
        self.collision_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

    # ---- geometry helpers ----

    def _ensure_meta(self, seg: DubinsPath) -> Dict:
        if getattr(seg, "metadata", None) is None:
            seg.metadata = {}
        return seg.metadata

    @staticmethod
    def _rect_inflated(o: Obstacle, inflate: float) -> Tuple[float, float, float, float]:
        s = config.arena.obstacle_size
        return (o.x - inflate, o.y - inflate, o.x + s + inflate, o.y + s + inflate)

    @staticmethod
    def _point_in_rect(px: float, py: float, r: Tuple[float,float,float,float]) -> bool:
        xmin, ymin, xmax, ymax = r
        return (xmin <= px <= xmax) and (ymin <= py <= ymax)

    @staticmethod
    def _circle_center(point: Tuple[float,float], heading: float, turn: str, R: float) -> Tuple[float,float]:
        x, y = point
        if turn.upper() == 'L':
            return (x - R * math.sin(heading), y + R * math.cos(heading))
        else:
            return (x + R * math.sin(heading), y - R * math.cos(heading))

    @staticmethod
    def _norm_angle(a: float) -> float:
        # 0..2π
        while a < 0: a += 2*math.pi
        while a >= 2*math.pi: a -= 2*math.pi
        return a

    def _sample_dubins(self, path: DubinsPath, step_cm: float = 3.0) -> Iterable[Tuple[float,float]]:
        """Yield centerline sample points (x,y) every ~step_cm along the Dubins path."""
        wps = path.waypoints

        if len(wps) <= 2:  # retreat line uses exactly 2 waypoints
            a, b = wps[0], wps[-1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            n = max(1, int(L / step_cm))
            for k in range(n + 1):
                t = k / n
                yield (a[0] + t * dx, a[1] + t * dy)
            return
    

        typ = (path.path_type.value if hasattr(path.path_type, "value") else str(path.path_type)).lower()
        R = float(config.car.turning_radius)

        for i in range(3):
            seg = typ[i]
            a = wps[i]; b = wps[i+1]
            if seg == 's':
                dx, dy = b[0]-a[0], b[1]-a[1]
                L = math.hypot(dx, dy)
                n = max(1, int(L / step_cm))
                for k in range(n+1):
                    t = k / n
                    yield (a[0] + t*dx, a[1] + t*dy)
            else:
                if i == 0:
                    heading = path.start_state.theta
                    c = self._circle_center(a, heading, seg, R)
                    ang0 = math.atan2(a[1]-c[1], a[0]-c[0])
                    ang1 = math.atan2(b[1]-c[1], b[0]-c[0])
                elif i == 2:
                    heading = path.end_state.theta
                    c = self._circle_center(b, heading, seg, R)  # center determined by goal config
                    ang0 = math.atan2(a[1]-c[1], a[0]-c[0])
                    ang1 = math.atan2(b[1]-c[1], b[0]-c[0])
                else:
                    # rarely arc in the middle, but keep safe
                    c = self._circle_center(a, path.start_state.theta, seg, R)
                    ang0 = math.atan2(a[1]-c[1], a[0]-c[0])
                    ang1 = math.atan2(b[1]-c[1], b[0]-c[0])

                A0, A1 = self._norm_angle(ang0), self._norm_angle(ang1)

                if seg == 'l':  # CCW
                    if A1 <= A0: A1 += 2*math.pi
                    length = (A1 - A0) * R
                    n = max(1, int(length / step_cm))
                    for k in range(n+1):
                        a_ = A0 + (A1 - A0) * (k / n)
                        yield (c[0] + R*math.cos(a_), c[1] + R*math.sin(a_))
                else:  # 'r' CW
                    if A0 <= A1: A0 += 2*math.pi
                    length = (A0 - A1) * R
                    n = max(1, int(length / step_cm))
                    for k in range(n+1):
                        a_ = A0 - (A0 - A1) * (k / n)
                        yield (c[0] + R*math.cos(a_), c[1] + R*math.sin(a_))

                        

    def _build_forward_retreat(self, at: CarState, retreat_cm: float) -> Optional[DubinsPath]:
        """
        Create a small Dubins (forward-only) path that ends 'retreat_cm' behind the scan pose
        (same heading). This is not a pure straight-backward move (Dubins can't go backward),
        but it gives you a valid path segment that *ends* at the desired retreat pose.
        """
        d = float(retreat_cm)
        if d <= 0:
            return None

        # desired retreat pose: go 'd' behind the scan pose along its heading
        bx = at.x - d * math.cos(at.theta)
        by = at.y - d * math.sin(at.theta)
        end = CarState(bx, by, at.theta)

        # keep within arena
        margin = float(config.car.width) / 2.0 + float(config.arena.collision_buffer)
        if not (margin <= bx <= self.arena_size - margin and margin <= by <= self.arena_size - margin):
            return None

        # plan a (forward) Dubins from scan pose -> retreat pose
        seg = self.dubins_planner.plan_path(at, end)
        # require the retreat segment to be collision-free
        if seg and not self._path_intersects_obstacles_strict(seg):
            return seg
        return None


    def _append_scan_retreats(self, segments: List[DubinsPath]) -> List[DubinsPath]:
        """
        After each scan stop (i.e., after each segment that ends at a target), append a short
        retreat segment if enabled by config.car.scan_retreat_cm.
        """
        retreat = float(getattr(config.car, "scan_retreat_cm", 0.0))
        if retreat <= 0 or not segments:
            return segments

        out: List[DubinsPath] = []
        for i, seg in enumerate(segments):
            out.append(seg)
            end_pose = seg.end_state
            rseg = self._build_forward_retreat(end_pose, retreat)
            if rseg:
                out.append(rseg)
                print(f"[retreat] Added {rseg.path_type.value.upper()} {rseg.length:.1f}cm after segment {i}")
            else:
                print(f"[retreat] Skipped after segment {i} (blocked or infeasible)")
        return out
    
    # ---------- HYBRID A* helpers ----------

    def _bin_state(self, s: CarState, xy_res: float, th_bins: int) -> Tuple[int,int,int]:
        ix = int(round(s.x / xy_res))
        iy = int(round(s.y / xy_res))
        th = (s.theta % (2*math.pi))
        ih = int(round(th / (2*math.pi) * th_bins)) % th_bins
        return ix, iy, ih

    def _goal_reached(self, a: CarState, b: CarState,
                      pos_tol_cm: float, ang_tol_rad: float) -> bool:
        if math.hypot(a.x - b.x, a.y - b.y) > pos_tol_cm:
            return False
        dth = (a.theta - b.theta + math.pi) % (2*math.pi) - math.pi
        return abs(dth) <= ang_tol_rad

    def _primitive_time(self, curvature: float, ds: float) -> float:
        v_lin = max(1e-6, config.car.linear_speed_cm_s)
        omega = max(1e-6, config.car.angular_speed_rad_s)
        if abs(curvature) < 1e-9:
            return ds / v_lin
        else:
            return abs(curvature * ds) / omega  # time = |Δθ| / ω, Δθ = k*ds

    def _simulate_step(self, s: CarState, curvature: float, ds: float) -> CarState:
        """Bicycle kinematics integration for short arc length ds."""
        if abs(curvature) < 1e-9:
            nx = s.x + ds * math.cos(s.theta)
            ny = s.y + ds * math.sin(s.theta)
            nt = s.theta
        else:
            k = curvature
            dth = k * ds
            nx = s.x + (math.sin(s.theta + dth) - math.sin(s.theta)) / k
            ny = s.y + (-math.cos(s.theta + dth) + math.cos(s.theta)) / k
            nt = s.theta + dth
        return CarState(nx, ny, nt)

    def _primitive_collision_free(self, s: CarState, curvature: float, ds: float) -> bool:
        """Check the tiny motion primitive against walls & inflated obstacles."""
        inflation = float(config.car.width) / 2.0 + float(config.arena.collision_buffer)
        rects = [self._rect_inflated(o, inflation) for o in self.obstacles]
        margin = inflation
        size = float(self.arena_size)

        # sample along the primitive every ~2.5 cm
        samples = max(2, int(ds / 2.5))
        for i in range(samples + 1):
            lam = i / samples
            if abs(curvature) < 1e-9:
                x = s.x + lam * ds * math.cos(s.theta)
                y = s.y + lam * ds * math.sin(s.theta)
            else:
                k = curvature
                dth = k * ds * lam
                x = s.x + (math.sin(s.theta + dth) - math.sin(s.theta)) / k
                y = s.y + (-math.cos(s.theta + dth) + math.cos(s.theta)) / k

            if x < margin or x > (size - margin) or y < margin or y > (size - margin):
                return False
            for r in rects:
                if self._point_in_rect(x, y, r):
                    return False
        return True

    def _dubins_lower_bound_time(self, s: CarState, g: CarState) -> float:
        """Admissible heuristic: Dubins *length* / v_lin (ignores obstacles)."""
        v_lin = max(1e-6, config.car.linear_speed_cm_s)
        p = self.dubins_planner.plan_path(s, g)
        if p:
            return p.length / v_lin
        # fallback to straight-line lower bound if Dubins fails
        return (math.hypot(g.x - s.x, g.y - s.y)) / v_lin

    def _shortcut_and_dubinize(self, states: List[CarState]) -> Optional[List[DubinsPath]]:
        """Compress a continuous waypoint chain into a few collision-free Dubins segments."""
        if len(states) < 2:
            return []
        segs: List[DubinsPath] = []
        i = 0
        while i < len(states) - 1:
            best_j = i + 1
            best_path = None
            # try farthest feasible connection first
            for j in range(len(states) - 1, i, -1):
                p = self.dubins_planner.plan_path(states[i], states[j])
                if p and self._path_collision_free(p):
                    best_j = j
                    best_path = p
                    break
            if best_path is None:
                # fallback to immediate neighbor (should almost never fail)
                p = self.dubins_planner.plan_path(states[i], states[i + 1])
                if not p or not self._path_collision_free(p):
                    return None
                best_path = p
                best_j = i + 1
            segs.append(best_path)
            i = best_j
        return segs

    def plan_visiting_orders_discrete(self, start_state: CarState, obstacle_indices: List[int]) -> List[Dict]:
        """
        Discrete planner with car-like constraints:
        - Turns are *arcs* at fixed radius R, by 45° or 90° (no in-place spins).
        - Straight moves are forward/back by one grid cell (coalesced).
        Emits legacy orders: L/R (angle_deg ∈ {45,90}), F/B (value_cm integer), S (scan), optional B retreat.
        """
        cell = float(config.arena.grid_cell_size)
        R = float(config.car.turning_radius)
        v_lin = max(1e-6, float(config.car.linear_speed_cm_s))
        # linear speed along an arc; prefer configured value if present, else R*omega
        v_arc = float(getattr(config.car, "turn_linear_cm_s", 0.0)) or (R * max(1e-6, float(getattr(config.car, "angular_speed_rad_s", 0.0))) or v_lin)
        scan_sec = float(getattr(config.vision, "scan_seconds", 0.0))
        retreat_cm = float(getattr(config.car, "scan_retreat_cm", 0.0))

        def _qpos(x: float, y: float) -> Tuple[float, float]:
            return (round(x / cell) * cell, round(y / cell) * cell)

        def _qhead(theta: float) -> int:
            # heading index in {0..7} for multiples of 45°
            step = int(round((theta % (2*math.pi)) / (math.pi/4))) % 8
            return step

        def _idx_theta(hidx: int) -> float:
            return hidx * (math.pi/4)

        def _inside_arena(x: float, y: float) -> bool:
            inflation = float(config.car.width)/2.0 + float(config.arena.collision_buffer)
            return (inflation <= x <= self.arena_size - inflation) and (inflation <= y <= self.arena_size - inflation)

        def _rects_inflated():
            inflation = float(config.car.width)/2.0 + float(config.arena.collision_buffer)
            return [self._rect_inflated(o, inflation) for o in self.obstacles]

        def _seg_collision_free(ax: float, ay: float, bx: float, by: float) -> bool:
            rects = _rects_inflated()
            size = float(self.arena_size)
            inflation = float(config.car.width)/2.0 + float(config.arena.collision_buffer)
            margin = inflation
            L = max(1e-6, math.hypot(bx-ax, by-ay))
            samples = max(2, int(L / 2.5))
            for i in range(samples+1):
                t = i / samples
                x = ax + t*(bx-ax); y = ay + t*(by-ay)
                if x < margin or x > size - margin or y < margin or y > size - margin:
                    return False
                for r in rects:
                    if self._point_in_rect(x, y, r): return False
            return True

        def _arc_end_pose(x: float, y: float, th: float, turn_dir: str, steps_45: int) -> Tuple[float,float,float]:
            """turn_dir in {'L','R'}, steps_45 in {1,2} (1=45°, 2=90°). Returns (nx,ny,nθ)."""
            dth = steps_45 * (math.pi/4) * (1 if turn_dir == 'L' else -1)
            # circle center using current pose and turn_dir
            if turn_dir == 'L':
                cx = x - R * math.sin(th)
                cy = y + R * math.cos(th)
            else:
                cx = x + R * math.sin(th)
                cy = y - R * math.cos(th)
            # end pose on that circle after dth
            th2 = (th + dth) % (2*math.pi)
            nx = cx + (R * math.sin(th2)) if turn_dir == 'L' else cx + (R * math.sin(th2))
            ny = cy - (R * math.cos(th2)) if turn_dir == 'L' else cy - (R * math.cos(th2))
            return nx, ny, th2

        def _arc_collision_free(x: float, y: float, th: float, turn_dir: str, steps_45: int) -> bool:
            rects = _rects_inflated()
            size = float(self.arena_size)
            inflation = float(config.car.width)/2.0 + float(config.arena.collision_buffer)
            margin = inflation

            if turn_dir == 'L':
                cx = x - R * math.sin(th); cy = y + R * math.cos(th)
                a0 = math.atan2(y - cy, x - cx); a1 = a0 + steps_45 * (math.pi/4)
                # sample CCW
                if a1 <= a0: a1 += 2*math.pi
                length = R * (a1 - a0)
                n = max(3, int(length / 2.5))
                for k in range(n+1):
                    a = a0 + (a1 - a0) * (k / n)
                    sx = cx + R * math.cos(a); sy = cy + R * math.sin(a)
                    if sx < margin or sx > size - margin or sy < margin or sy > size - margin:
                        return False
                    for r in rects:
                        if self._point_in_rect(sx, sy, r): return False
            else:
                cx = x + R * math.sin(th); cy = y - R * math.cos(th)
                a0 = math.atan2(y - cy, x - cx); a1 = a0 - steps_45 * (math.pi/4)
                # sample CW
                if a0 <= a1: a0 += 2*math.pi
                length = R * (a0 - a1)
                n = max(3, int(length / 2.5))
                for k in range(n+1):
                    a = a0 - (a0 - a1) * (k / n)
                    sx = cx + R * math.cos(a); sy = cy + R * math.sin(a)
                    if sx < margin or sx > size - margin or sy < margin or sy > size - margin:
                        return False
                    for r in rects:
                        if self._point_in_rect(sx, sy, r): return False
            return True

        def _neighbors(x: float, y: float, hidx: int):
            """
            Neighbor generator:
            ('arc','L',1) -> left 45° arc;  ('arc','L',2) -> left 90° arc
            ('arc','R',1) -> right 45° arc; ('arc','R',2) -> right 90° arc
            ('move',+1) forward one cell;   ('move',-1) backward one cell
            """
            th = _idx_theta(hidx)

            # arcs (require radius)
            for turn_dir in ('L', 'R'):
                for steps_45 in (1, 2):
                    if _arc_collision_free(x, y, th, turn_dir, steps_45):
                        nx, ny, nth = _arc_end_pose(x, y, th, turn_dir, steps_45)
                        if _inside_arena(nx, ny):
                            nh = _qhead(nth)
                            # quantize xy for graph key
                            qx, qy = _qpos(nx, ny)
                            yield (qx, qy, nh, ('arc', turn_dir, steps_45, (nx, ny, nth)))

            # forward / backward straight
            dx, dy = cell * math.cos(th), cell * math.sin(th)
            fx, fy = x + dx, y + dy
            if _inside_arena(fx, fy) and _seg_collision_free(x, y, fx, fy):
                yield (fx, fy, hidx, ('move', +1, (fx, fy, th)))
            bx, by = x - dx, y - dy
            if _inside_arena(bx, by) and _seg_collision_free(x, y, bx, by):
                yield (bx, by, hidx, ('move', -1, (bx, by, th)))

        def _cost(action) -> float:
            kind = action[0]
            if kind == 'arc':
                _, _, steps_45, _ = action
                dth = steps_45 * (math.pi/4)
                s = R * dth
                return s / v_arc
            else:
                # one cell
                return cell / v_lin

        def _heur(x: float, y: float, gx: float, gy: float) -> float:
            # Straight-line time lower bound (ignores needed turns)
            return math.hypot(gx - x, gy - y) / v_lin

        def _astar(start: CarState, goal: CarState):
            sx, sy = _qpos(start.x, start.y)
            sh = _qhead(start.theta)
            gx, gy = _qpos(goal.x, goal.y)

            import heapq
            start_key = (sx, sy, sh)
            gscore = {start_key: 0.0}
            came = {}  # (x,y,h) -> ((px,py,ph), action)
            openh = []
            heapq.heappush(openh, ( _heur(sx, sy, gx, gy), 0.0, start_key, (sx, sy, _idx_theta(sh)) ))

            visited_pose_for_key = {start_key: (sx, sy, _idx_theta(sh))}

            while openh:
                _, g, key, pose = heapq.heappop(openh)
                x, y, th = pose
                if math.hypot(x - gx, y - gy) <= cell * 0.5:
                    # reconstruct primitive actions
                    actions = []
                    cur_key = key
                    while cur_key in came:
                        prev_key, act = came[cur_key]
                        actions.append(act)
                        cur_key = prev_key
                    actions.reverse()
                    return actions, pose  # final continuous pose

                kx, ky, kh = key
                for nx, ny, nh, act in _neighbors(kx, ky, kh):
                    nk = (nx, ny, nh)
                    ng = g + _cost(act)
                    if ng + 1e-9 < gscore.get(nk, float('inf')):
                        gscore[nk] = ng
                        came[nk] = (key, act)
                        # store best pose seen for this key (for final pose output)
                        if act[0] == 'arc':
                            _, _, _, cont = act
                            visited_pose_for_key[nk] = cont
                        else:
                            _, _, cont = act
                            visited_pose_for_key[nk] = cont
                        f = ng + _heur(nx, ny, gx, gy)
                        heapq.heappush(openh, (f, ng, nk, visited_pose_for_key[nk]))
            return None, None

        def _actions_to_orders(actions, start_pose: CarState):
            """
            Convert primitives to orders. Coalesce straight runs; arcs become L/R with angle 45 or 90.
            All outputs are integers.
            """
            orders: List[Dict] = []
            x, y, th = start_pose.x, start_pose.y, start_pose.theta

            def flush_run(run_kind: Optional[str], steps: int):
                if not run_kind or steps == 0: return
                dist = int(round(steps * cell))
                pose = {"x": round(x,2), "y": round(y,2), "theta": round(th,4), "theta_deg": round(math.degrees(th),1)}
                if run_kind == 'F':
                    orders.append({"op": "F", "value_cm": dist, "pose": pose})
                else:
                    orders.append({"op": "B", "value_cm": dist, "pose": pose})

            run_kind, steps = None, 0

            for act in actions:
                if act[0] == 'move':
                    _, sign, cont = act
                    nk = 'F' if sign > 0 else 'B'
                    if run_kind is None or run_kind == nk:
                        run_kind = nk; steps += 1
                    else:
                        flush_run(run_kind, steps); run_kind, steps = nk, 1
                    x, y, th = cont  # already updated pose from neighbor
                else:
                    # arc turn
                    flush_run(run_kind, steps); run_kind, steps = None, 0
                    _, turn_dir, steps_45, cont = act
                    angle = int(45 * steps_45)
                    # after arc, pose is 'cont'
                    x2, y2, th2 = cont
                    pose = {"x": round(x2,2), "y": round(y2,2), "theta": round(th2,4), "theta_deg": round(math.degrees(th2),1)}
                    if turn_dir == 'L':
                        orders.append({"op": "L", "angle_deg": angle, "pose": pose})
                    else:
                        orders.append({"op": "R", "angle_deg": angle, "pose": pose})
                    x, y, th = x2, y2, th2

            flush_run(run_kind, steps)
            return orders, CarState(x, y, th)

        # -------- plan sequentially for the requested targets --------
        cur = start_state
        orders_out: List[Dict] = []
        for idx in obstacle_indices:
            if idx < 0 or idx >= len(self.obstacles):
                continue
            goal = self.get_image_target_position(self.obstacles[idx])

            actions, final_pose = _astar(cur, goal)
            if actions is None:
                return []

            seg_orders, cur = _actions_to_orders(actions, cur)
            orders_out.extend(seg_orders)

            # Scan at the target
            s = {"op": "S", "pose": {
                    "x": round(cur.x,2), "y": round(cur.y,2),
                    "theta": round(cur.theta,4), "theta_deg": round(math.degrees(cur.theta),1)
                }}
            if scan_sec > 0:
                s["scan_seconds"] = round(scan_sec, 2)
            orders_out.append(s)

            # Optional straight retreat
            if retreat_cm > 0:
                bx = cur.x - retreat_cm * math.cos(cur.theta)
                by = cur.y - retreat_cm * math.sin(cur.theta)
                if _seg_collision_free(cur.x, cur.y, bx, by):
                    orders_out.append({
                        "op": "B",
                        "value_cm": int(round(retreat_cm)),
                        "pose": {"x": round(bx,2), "y": round(by,2),
                                "theta": round(cur.theta,4), "theta_deg": round(math.degrees(cur.theta),1)}
                    })
                    cur = CarState(bx, by, cur.theta)

        return orders_out



        
    
    def plan_hybrid_astar(self, start: CarState, goal: CarState) -> Optional[List[DubinsPath]]:
        """
        Hybrid A*: A* over (x,y,theta) with short motion primitives,
        Dubins lower-bound heuristic, and an analytic Dubins connect near the goal.
        Returns a list of DubinsPath segments if successful.
        """
        # --- parameters (tune if needed) ---
        ds = 8.0  # cm per primitive step
        R = max(1e-6, float(config.car.turning_radius))
        # a small set of curvatures (left/straight/right); add mid values for agility
        curvatures = [-1.0/R, -0.5/R, 0.0, 0.5/R, 1.0/R]
        xy_res = max(2.5, config.arena.grid_cell_size / 2)  # binning resolution for closed-set
        th_bins = 32
        pos_tol = 15.0  # cm (goal snapping distance)
        ang_tol = math.radians(30.0)
        max_iters = 20000

        start_key = self._bin_state(start, xy_res, th_bins)
        h0 = self._dubins_lower_bound_time(start, goal)

        # open-set items: (f=g+h, g, unique_id, key, state)
        uid = 0
        open_heap = []
        heappush(open_heap, (h0, 0.0, uid, start_key, start))
        uid += 1

        best_g: Dict[Tuple[int,int,int], float] = {start_key: 0.0}
        parent: Dict[Tuple[int,int,int], Tuple[Tuple[int,int,int], CarState]] = {}

        iters = 0
        while open_heap and iters < max_iters:
            iters += 1
            f, g, _, key, s = heappop(open_heap)

            # Analytic expansion if we're close enough: try to snap with a final Dubins
            if self._goal_reached(s, goal, pos_tol, ang_tol):
                final = self.dubins_planner.plan_path(s, goal)
                if final and self._path_collision_free(final):
                    # reconstruct states -> smooth -> prepend final segment at the end
                    chain: List[CarState] = [s]
                    k = key
                    while k in parent:
                        pk, ps = parent[k]
                        chain.append(ps)
                        k = pk
                    chain.append(start)
                    chain.reverse()
                    segs = self._shortcut_and_dubinize(chain)
                    if segs is None:
                        return None
                    segs.append(final)
                    return segs

            # Expand successors
            for kappa in curvatures:
                if not self._primitive_collision_free(s, kappa, ds):
                    continue
                ns = self._simulate_step(s, kappa, ds)
                nkey = self._bin_state(ns, xy_res, th_bins)
                step_cost = self._primitive_time(kappa, ds)
                ng = g + step_cost

                if nkey in best_g and ng >= best_g[nkey] - 1e-9:
                    continue
                best_g[nkey] = ng
                parent[nkey] = (key, s)
                nh = self._dubins_lower_bound_time(ns, goal)
                heappush(open_heap, (ng + nh, ng, uid, nkey, ns))
                uid += 1

        return None  # give up

    def _path_collision_free(self, path: DubinsPath) -> bool:
        inflation = float(config.car.width) / 2.0 + float(config.arena.collision_buffer)
        rects = [self._rect_inflated(o, inflation) for o in self.obstacles]

        for x, y in self._sample_dubins(path, step_cm=3.0):
            for r in rects:
                if self._point_in_rect(x, y, r):
                    return False
        return True

    def _best_dubins_between(self, start, goal) -> Optional[DubinsPath]:
        candidates = self._all_dubins_candidates(start, goal)  # your existing generator for 6 types
        # keep only collision-free ones
        ok = [p for p in candidates if self._path_collision_free(p)]
        if not ok:
            return None
        # pick shortest-TIME (you already have this cost)
        return min(ok, key=self._time_cost)

    # ---- world / obstacle management ----
    def add_obstacle(self, x: int | float, y: int | float, image_side: str):
        # snap to grid & cast (optional but robust)
        cell = config.arena.grid_cell_size
        x = int(round(x / cell) * cell)
        y = int(round(y / cell) * cell)

        obstacle = Obstacle(x, y, image_side)
        self.obstacles.append(obstacle)
        self._update_collision_grid()
        print(f"Added obstacle {len(self.obstacles)-1} at ({x}, {y}) with image on {image_side} side")


    def _update_collision_grid(self):
        self.collision_grid.fill(0)
        buffer = config.arena.collision_buffer
        cell = config.arena.grid_cell_size
        obs_size = config.arena.obstacle_size

        for obs in self.obstacles:
            # use floor/ceil then clamp, then cast to int
            min_x = max(0, math.floor((obs.x - buffer) / cell))
            max_x = min(self.grid_size, math.ceil((obs.x + obs_size + buffer) / cell))
            min_y = max(0, math.floor((obs.y - buffer) / cell))
            max_y = min(self.grid_size, math.ceil((obs.y + obs_size + buffer) / cell))

            # guard against empty or inverted slices
            if min_x < max_x and min_y < max_y:
                self.collision_grid[int(min_y):int(max_y), int(min_x):int(max_x)] = 1


    def get_image_target_position(self, obstacle: Obstacle) -> CarState:
        """Pose where the robot should be to scan the image."""
        d = config.car.camera_distance * 0.8
        size = config.arena.obstacle_size
        if obstacle.image_side == 'S':
            x = obstacle.x + size / 2
            y = obstacle.y - d
            theta = math.pi / 2
        elif obstacle.image_side == 'N':
            x = obstacle.x + size / 2
            y = obstacle.y + size + d
            theta = 3 * math.pi / 2
        elif obstacle.image_side == 'E':
            x = obstacle.x + size + d
            y = obstacle.y + size / 2
            theta = math.pi
        else:  # 'W'
            x = obstacle.x - d
            y = obstacle.y + size / 2
            theta = 0.0
        return CarState(x, y, theta)
    
    def _build_retreat_line(self, at: CarState, retreat_cm: float) -> Optional[DubinsPath]:
        d = float(retreat_cm)
        if d <= 0:
            return None

        # end point d cm behind the scan pose along –heading
        bx = at.x - d * math.cos(at.theta)
        by = at.y - d * math.sin(at.theta)
        end = CarState(bx, by, at.theta)

        # fabricate a simple “line” path (2 waypoints)
        waypoints = [(at.x, at.y), (bx, by)]
        length = math.hypot(bx - at.x, by - at.y)

        # You can keep any enum value; sampler ignores it for 2-point paths
        p = DubinsPath(DubinsPathType.LSL, length, at, end, waypoints, turn_points=[])

        # respect walls/obstacles; skip if unsafe
        if self._path_intersects_obstacles_strict(p):
            return None
        return p

    # ------------------ Public planning API ------------------
    def plan_visiting_path(self, start_state: CarState, obstacle_indices: List[int]) -> List[DubinsPath]:
        print(f"Planning path from ({start_state.x:.1f}, {start_state.y:.1f}) to visit obstacles: {obstacle_indices}")
        print(f"Available obstacles: {len(self.obstacles)}")

        if not obstacle_indices:
            print("No obstacles to visit")
            return []

        cur = start_state
        result_segments: List[DubinsPath] = []
        retreat_cm = float(getattr(config.car, "scan_retreat_cm", 0.0))

        print(f"Scan retreat after each target: {retreat_cm:.1f}cm")

        for idx in obstacle_indices:
            if idx < 0 or idx >= len(self.obstacles):
                print(f"Skip invalid obstacle index {idx}")
                continue

            goal = self.get_image_target_position(self.obstacles[idx])

            # 1) Try Hybrid A*
            segs = self.plan_hybrid_astar(cur, goal)
            if segs:
                # keep all segments, but only the final one is a scan-stop
                result_segments.extend(segs)
                last = segs[-1]
                self._ensure_meta(last)["scan_stop"] = True
                self._ensure_meta(last)["target_index"] = idx
                cur = goal
            else:
                # 2) Fallback: single-shot Dubins
                direct = self.dubins_planner.plan_path(cur, goal)
                if direct and not self._path_intersects_obstacles_strict(direct):
                    result_segments.append(direct)
                    self._ensure_meta(direct)["scan_stop"] = True
                    self._ensure_meta(direct)["target_index"] = idx
                    cur = goal
                else:
                    print(f"[HybridA*] No valid path to obstacle {idx}")
                    return []

            # --- RETREAT: straight line backwards from the scan pose (no scan here) ---
            if retreat_cm > 0:
                rseg = self._build_retreat_line(cur, retreat_cm)
                if rseg:
                    m = self._ensure_meta(rseg)
                    m["retreat"] = True
                    m["retreat_cm"] = retreat_cm
                    #print(f"[retreat] line {rseg.length:.1f}cm from obstacle {idx}")
                    result_segments.append(rseg)
                    cur = rseg.end_state
                else:
                    print(f"[retreat] skipped (unsafe) after obstacle {idx}")

        # for i, seg in enumerate(result_segments):
        #     m = self._ensure_meta(seg)
        #     print(f"  Segment {i}: {seg.path_type.value.upper()} {seg.length:.1f}cm, "
        #           f"from ({seg.start_state.x:.1f}, {seg.start_state.y:.1f}, {math.degrees(seg.start_state.theta):.1f}°) "
        #           f"to ({seg.end_state.x:.1f}, {seg.end_state.y:.1f}, {math.degrees(seg.end_state.theta):.1f}°) "
        #           f"{'(scan)' if m.get('scan_stop', False) else ''}{'(retreat)' if m.get('retreat', False) else ''}")

        for i in result_segments:
            m = self._ensure_meta(i)
            if m.get("scan_stop", False):
                print(f"  Segment to obstacle {m.get('target_index', -1)}: {i.path_type.value.upper()} {i.length:.1f}cm, "
                      f"from ({i.start_state.x:.1f}, {i.start_state.y:.1f}, {math.degrees(i.start_state.theta):.1f}°) "
                      f"to ({i.end_state.x:.1f}, {i.end_state.y:.1f}, {math.degrees(i.end_state.theta):.1f}°) (scan)")
            elif m.get("retreat", False):
                print(f"  Retreat segment: {i.path_type.value.upper()} {i.length:.1f}cm, "
                      f"from ({i.start_state.x:.1f}, {i.start_state.y:.1f}, {math.degrees(i.start_state.theta):.1f}°) "
                      f"to ({i.end_state.x:.1f}, {i.end_state.y:.1f}, {math.degrees(i.end_state.theta):.1f}°) (retreat)")
            else:
                print(f"  Transit segment: {i.path_type.value.upper()} {i.length:.1f}cm, "
                      f"from ({i.start_state.x:.1f}, {i.start_state.y:.1f}, {math.degrees(i.start_state.theta):.1f}°) "
                      f"to ({i.end_state.x:.1f}, {i.end_state.y:.1f}, {math.degrees(i.end_state.theta):.1f}°)")

        total = sum(p.length for p in result_segments)
        print(f"Mission planned: {len(result_segments)} segments, total {total:.1f}cm")
        return result_segments
        
        # # 1) Try optimal Held–Karp (shortest-time Hamiltonian)
        # hk_paths = self.plan_shortest_time_hamiltonian(start_state, obstacle_indices)
        # if hk_paths:
        #     total = sum(p.length for p in hk_paths)
        #     print(f"Success with shortest_time_hamiltonian: {len(hk_paths)} segments, {total:.1f}cm")
        #     return hk_paths

        # # 2) Fall back to previous strategies if HK fails for any reason
        # for algo in (self._greedy_nearest_neighbor, self._exhaustive_search, self._fallback_simple_path):
        #     try:
        #         result = algo(start_state, obstacle_indices)
        #         if result.paths:
        #             print(f"Success with {result.algorithm_used}: {len(result.paths)} segments, {result.total_length:.1f}cm")
        #             return result.paths
        #         else:
        #             print(f"Failed with {result.algorithm_used}: {result.debug_info}")
        #     except Exception as e:
        #         print(f"Error with algorithm: {e}")

    # ------------------ B.3: Held–Karp TSP on time ------------------
    def plan_shortest_time_hamiltonian(self, start_state: CarState, obstacle_indices: List[int]) -> List[DubinsPath]:
        """Compute the true shortest-time visiting order using Held–Karp DP."""
        # Build target states
        targets: List[Tuple[int, CarState]] = []
        # Thang: At this stage, we can add additional points between the obstacles to avoid the collisions
        # on all the Dubin paths
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))

        n = len(targets)
        if n == 0:
            return []

        # Precompute costs and keep the actual best Dubins path for each directed edge
        INF = 1e12
        edge_time_start = [INF] * n
        edge_path_start: List[Optional[DubinsPath]] = [None] * n

        edge_time = [[INF] * n for _ in range(n)]
        edge_path: List[List[Optional[DubinsPath]]] = [[None] * n for _ in range(n)]

        # From start -> each target
        for j in range(n):
            dest = targets[j][1]
            path = self.dubins_planner.plan_path(start_state, dest)
            if path and not self._path_intersects_obstacles_strict(path):
                edge_time_start[j] = self._estimate_time_for_path(path)
                edge_path_start[j] = path

        # Between targets
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = targets[i][1]
                b = targets[j][1]
                path = self.dubins_planner.plan_path(a, b)
                if path and not self._path_intersects_obstacles_strict(path):
                    edge_time[i][j] = self._estimate_time_for_path(path)
                    edge_path[i][j] = path

        # If any target unreachable from start, abort
        if any(t >= INF for t in edge_time_start):
            return []

        # Held–Karp DP: dp[(mask, j)] = (time, prev_j)
        # mask over n targets (0..n-1). Ending at j.
        dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}

        # Base cases: start -> j
        for j in range(n):
            dp[(1 << j, j)] = (edge_time_start[j], None)

        # Iterate over subset sizes
        for size in range(2, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for k in subset:
                    mask |= (1 << k)
                for j in subset:
                    best = (INF, None)
                    prev_mask = mask ^ (1 << j)
                    for i in subset:
                        if i == j:
                            continue
                        if (prev_mask, i) in dp and edge_time[i][j] < INF:
                            cand = dp[(prev_mask, i)][0] + edge_time[i][j]
                            if cand < best[0]:
                                best = (cand, i)
                    if best[0] < INF:
                        dp[(mask, j)] = best

        # Pick best final end node (no return to start needed)
        full_mask = (1 << n) - 1
        best_final = (INF, None)
        best_end = None
        for j in range(n):
            if (full_mask, j) in dp and dp[(full_mask, j)][0] < best_final[0]:
                best_final = dp[(full_mask, j)]
                best_end = j

        if best_end is None:
            return []

        # Reconstruct order
        order: List[int] = []
        mask = full_mask
        j = best_end
        while j is not None:
            order.append(j)
            time_val, prev = dp[(mask, j)]
            if prev is None:
                break
            mask ^= (1 << j)
            j = prev
        order.reverse()

        # Build actual path sequence: start -> first -> ... -> last
        segments: List[DubinsPath] = []
        # Start to first
        first = order[0]
        segments.append(edge_path_start[first])
        # Between targets
        for a, b in zip(order[:-1], order[1:]):
            segments.append(edge_path[a][b])

        return segments

    # ---- helpers: time estimate & collision checks ----
    def _estimate_time_for_path(self, path: DubinsPath) -> float:
        """
        Convert a Dubins path into time using straight vs arc speeds:
          t = L_straight / v_lin + L_arc / (r * omega)
        """
        v_lin = max(1e-6, config.car.linear_speed_cm_s)
        omega = max(1e-6, config.car.angular_speed_rad_s)
        r = max(1e-6, config.car.turning_radius)

        # Straight length: only if the middle segment is 'S'
        straight_len = 0.0
        if 's' in path.path_type.value:
            # waypoints[1] and [2] are tangent points (t1, t2) across the straight
            if len(path.waypoints) >= 3:
                x1, y1 = path.waypoints[1]
                x2, y2 = path.waypoints[2]
                straight_len = math.hypot(x2 - x1, y2 - y1)

        arc_len = max(0.0, path.length - straight_len)
        t_straight = straight_len / v_lin
        t_arc = arc_len / (r * omega)

        # Optional constant per-stop recognition time — same for every target, so it
        # doesn't affect ordering; keep 0 to avoid bias.
        return t_straight + t_arc

    def _path_intersects_obstacles(self, path: DubinsPath) -> bool:
        return self._path_intersects_obstacles_strict(path, buffer_reduction=0)

    def _path_intersects_obstacles_strict(self, path: DubinsPath, buffer_reduction: float = 0) -> bool:
        """
        True if the Dubins *arc* path intersects any inflated obstacle or violates arena margins.
        Uses continuous geometry so collision_buffer affects results directly.
        """
        # centerline clearance = car half width + (possibly reduced) extra buffer
        inflation = float(config.car.width) / 2.0 + float(config.arena.collision_buffer)
        #print(f"Collision check with inflation: {inflation:.1f}cm")
        rects = [self._rect_inflated(o, inflation) for o in self.obstacles]
        #print(f"Checking against {len(rects)} obstacles")
        #print(rects)

        # keep some margin from arena walls as well
        margin = inflation
        size = float(self.arena_size)

        for x, y in self._sample_dubins(path, step_cm=3.0):   # follows the true arcs
            # walls
            # print(f"  Sample point: ({x:.1f}, {y:.1f})")
            if x < margin or x > (size - margin) or y < margin or y > (size - margin):
                print("    Out of bounds!")
                return True
            # obstacles
            for r in rects:
                if self._point_in_rect(x, y, r):
                    print(f"    Collides with rect {r}")
                    return True
        print("    No collision detected")
        return False

    # -------- legacy strategies kept as fallbacks --------
    def _greedy_nearest_neighbor(self, start_state: CarState, obstacle_indices: List[int]) -> PathfindingResult:
        targets = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))
        if not targets:
            return PathfindingResult([], 0, "greedy_nearest_neighbor", {"error": "no_valid_targets"})

        path_segments = []
        total_length = 0.0
        current_state = start_state
        remaining = targets.copy()

        while remaining:
            best = None
            best_seg = None
            best_len = float('inf')
            for cand_idx, cand_state in remaining:
                seg = self.dubins_planner.plan_path(current_state, cand_state)
                if seg and not self._path_intersects_obstacles(seg) and seg.length < best_len:
                    best_len = seg.length
                    best = (cand_idx, cand_state)
                    best_seg = seg
            if not best_seg:
                return PathfindingResult([], 0, "greedy_nearest_neighbor", {"error": "no_valid_path", "remaining": len(remaining)})
            path_segments.append(best_seg)
            total_length += best_seg.length
            current_state = best_seg.end_state
            remaining.remove(best)

        return PathfindingResult(path_segments, total_length, "greedy_nearest_neighbor", {"segments": len(path_segments)})

    def _exhaustive_search(self, start_state: CarState, obstacle_indices: List[int]) -> PathfindingResult:
        if len(obstacle_indices) > 7:
            return PathfindingResult([], 0, "exhaustive_search", {"error": "too_many_targets"})
        targets = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))
        if not targets:
            return PathfindingResult([], 0, "exhaustive_search", {"error": "no_valid_targets"})

        best_path = None
        min_len = float('inf')
        attempts = 0
        for perm in permutations(targets):
            attempts += 1
            path_segments = []
            total = 0.0
            cur = start_state
            valid = True
            for _, tgt in perm:
                seg = self.dubins_planner.plan_path(cur, tgt)
                if not seg or self._path_intersects_obstacles_strict(seg):
                    valid = False
                    break
                path_segments.append(seg)
                total += seg.length
                cur = tgt
            if valid and total < min_len:
                min_len = total
                best_path = path_segments

        if best_path:
            return PathfindingResult(best_path, min_len, "exhaustive_search", {"attempts": attempts, "segments": len(best_path)})
        return PathfindingResult([], 0, "exhaustive_search", {"error": "no_valid_permutation", "attempts": attempts})

    def _fallback_simple_path(self, start_state: CarState, obstacle_indices: List[int]) -> PathfindingResult:
        targets = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))
        if not targets:
            return PathfindingResult([], 0, "fallback_simple", {"error": "no_valid_targets"})

        path_segments = []
        total = 0.0
        cur = start_state
        for obs_idx, tgt in targets:
            seg = self.dubins_planner.plan_path(cur, tgt)
            if seg and not self._path_intersects_obstacles_strict(seg):
                path_segments.append(seg)
                total += seg.length
                cur = tgt
            else:
                print(f"Warning: Could not plan path to obstacle {obs_idx}")
        return PathfindingResult(path_segments, total, "fallback_simple", {"segments": len(path_segments), "warnings": True})


if __name__ == "__main__":
    # Simple smoke test
    planner = CarPathPlanner()
    planner.add_obstacle(50, 50, 'S')
    planner.add_obstacle(100, 100, 'E')
    planner.add_obstacle(150, 50, 'N')
    start = CarState(20, 20, 0)
    paths = planner.plan_visiting_path(start, [0, 1, 2])
    if paths:
        total = sum(p.length for p in paths)
        print(f"Final result: {len(paths)} segments, total length: {total:.1f}cm")
        for i, seg in enumerate(paths):
            print(f"  Segment {i}: {seg.path_type.value.upper()} ({seg.length:.1f}cm)")
    else:
        print("No valid path found")