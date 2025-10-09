"""
Lattice A* path planner with real turning-radius arcs, long forward primitives,
and configurable forward slip before L/R turns.

Key points
----------
- State space: (i, j, h) on a fine grid, h∈{0..7} (every 45°).
- Motion primitives (generated on-the-fly):
  * F d, B d with d ∈ {10, 20, ..., 200} cm along current heading
  * L/R arc with angle in {45°, 90°} at radius R = config.car.turning_radius
    (now modeled as: pre-slip straight of s(cm) → arc)
- Costs:
  * Straight: time = distance / v_lin
  * Arc: time = (slip / v_lin) + (arc_length / v_arc) + small per-45° bias
  * Reverse multiplies time by BACKWARD_MULT (>1)
- Safety:
  * Robot inflation = car.width/2 + arena.collision_buffer
  * Segment/arc collision sampled ~every 2.5 cm (slip is also checked)
  * Wall margin = inflation
- Output orders:
  * [{'op':'F'|'B','value_cm':...,'pose':{x,y,theta,theta_deg}},
     {'op':'L'|'R','angle_deg':..., 'radius_cm':..., 'pre_slip_cm': s,
      'pose':{x,y,theta,theta_deg}}, ...]
    Straights (F/B) are merged automatically.
- Goal tolerance:
  * From config.planner.{goal_pos_epsilon_cm, goal_heading_epsilon_deg}
    (fallbacks: 7.5 cm and 10°)

Assumes:
- algorithm.lib.car.CarState
- algorithm.lib.path.Obstacle
- algorithm.config.config
"""

import math
import heapq
from typing import List, Dict, Tuple, Optional

from algorithm.lib.car import CarState
from algorithm.lib.path import Obstacle
from algorithm.config import config

deg = math.degrees
rad = math.radians


class CarPathPlanner:
    def __init__(self, grid_N: int = 2000):
        """
        grid_N: number of grid cells per side (e.g., 2000 for 1 cm if arena.size=2000,
                or 0.1 cm if arena.size=200).
        """
        self.field_cm = float(config.arena.size)
        self.grid_N   = int(grid_N)
        self.res_cm   = self.field_cm / self.grid_N  # grid resolution in cm/cell

        self.obstacles: List[Obstacle] = []

        # Debugging
        self.debug = True
        self._expand_log_every = 5000

        # Goal tolerances (safe defaults)
        planner_cfg = getattr(config, "planner", None)
        self.goal_pos_eps_cm: float = float(getattr(planner_cfg, "goal_pos_epsilon_cm", 7.5))
        self.goal_head_eps_deg: float = float(getattr(planner_cfg, "goal_heading_epsilon_deg", 10.0))

    # --------------------- Logging ---------------------
    def _log(self, msg: str):
        if self.debug:
            print(msg)

    # --------------------- Turn-slip model ---------------------
    def _turn_slip(self, angle_deg: float) -> float:
        """
        Return forward slip distance (cm) executed immediately BEFORE an L/R turn.
        Config options (in order of precedence):
          - config.planner.turn_forward_slip_cm_per_deg
          - config.planner.turn_forward_slip_cm_45 / _90
        """
        pcfg = getattr(config, "planner", None)
        if pcfg is None:
            return 0.0

        per_deg = getattr(pcfg, "turn_forward_slip_cm_per_deg", None)
        if per_deg is not None:
            try:
                return float(per_deg) * float(angle_deg)
            except Exception:
                pass

        # Discrete values
        s45 = float(getattr(pcfg, "turn_forward_slip_cm_45", 0.0))
        s90 = float(getattr(pcfg, "turn_forward_slip_cm_90", s45 * 2.0 if s45 > 0 else 0.0))

        # If we get a non 45/90 angle (future), interpolate
        if abs(angle_deg - 45.0) < 1e-6:
            return s45
        if abs(angle_deg - 90.0) < 1e-6:
            return s90
        # linear interpolation by degrees
        if s45 > 0.0 or s90 > 0.0:
            # prefer s90 slope if set
            base = s90 / 90.0 if s90 > 0.0 else s45 / 45.0
            return base * float(angle_deg)

        return 0.0

    # --------------------- Obstacle mgmt ---------------------
    def add_obstacle(self, x: float, y: float, image_side: str):
        """Obstacles are axis-aligned squares, pass top-left (x,y) in cm."""
        gx = round(x / self.res_cm) * self.res_cm
        gy = round(y / self.res_cm) * self.res_cm
        self.obstacles.append(Obstacle(gx, gy, image_side))
        self._log(f"[add_obstacle] ({gx:.2f},{gy:.2f}) side={image_side}")

    # --------------------- Geometry / Safety ---------------------
    @staticmethod
    def _inflation() -> float:
        return float(config.car.width)/2.0 + float(config.arena.collision_buffer)

    def _inside_walls(self, x: float, y: float) -> bool:
        m = self._inflation()
        return (m <= x <= self.field_cm - m) and (m <= y <= self.field_cm - m)

    def _rect_infl(self, o: Obstacle, infl: float) -> Tuple[float,float,float,float]:
        s = float(config.arena.obstacle_size)
        return (o.x - infl, o.y - infl, o.x + s + infl, o.y + s + infl)

    @staticmethod
    def _pt_in_rect(px: float, py: float, r: Tuple[float,float,float,float]) -> bool:
        x0, y0, x1, y1 = r
        return (x0 <= px <= x1) and (y0 <= py <= y1)

    def _segment_clear(self, ax: float, ay: float, bx: float, by: float, allow_start_outside=False) -> bool:
        infl = self._inflation()
        rects = [self._rect_infl(o, infl) for o in self.obstacles]
        L = max(1e-6, math.hypot(bx-ax, by-ay))
        n = max(2, int(L / 2.5))
        for i in range(n+1):
            t = i/n
            x = ax + t*(bx-ax)
            y = ay + t*(by-ay)
            if i == 0 and allow_start_outside:
                continue
            if not self._inside_walls(x, y):
                self._log(f"[seg_clear] OOB at ({x:.2f},{y:.2f})")
                return False
            for r in rects:
                if self._pt_in_rect(x, y, r):
                    self._log(f"[seg_clear] obstacle hit at ({x:.2f},{y:.2f}) in rect {r}")
                    return False
        return True

    def _arc_end_pose(self, x: float, y: float, th: float, turn_dir: str, angle_deg: float) -> Tuple[float,float,float]:
        """
        End pose after executing: slip straight s(cm) along heading, then arc turn.
        """
        s = self._turn_slip(angle_deg)
        # slip first
        x1 = x + s * math.cos(th)
        y1 = y + s * math.sin(th)

        R = float(config.car.turning_radius)
        dpsi = rad(angle_deg) * (1 if turn_dir.upper() == 'L' else -1)
        if turn_dir.upper() == 'L':
            cx = x1 - R * math.sin(th); cy = y1 + R * math.cos(th)
        else:
            cx = x1 + R * math.sin(th); cy = y1 - R * math.cos(th)
        a0 = math.atan2(y1 - cy, x1 - cx)
        a1 = a0 + dpsi
        nx = cx + R * math.cos(a1); ny = cy + R * math.sin(a1)
        nth = (th + dpsi) % (2*math.pi)
        return nx, ny, nth

    def _arc_clear(self, x: float, y: float, th: float, turn_dir: str, angle_deg: float, allow_start_outside=False) -> bool:
        """
        Check slip segment and the arc path for collisions and wall violations.
        """
        s = self._turn_slip(angle_deg)
        # Check the slip straight
        sx = x + s * math.cos(th)
        sy = y + s * math.sin(th)
        if s > 0.0:
            if not self._segment_clear(x, y, sx, sy, allow_start_outside=allow_start_outside):
                self._log(f"[arc_clear] slip blocked (s={s:.2f} cm)")
                return False

        # Now check the arc starting from (sx,sy)
        R = float(config.car.turning_radius)
        infl = self._inflation()
        rects = [self._rect_infl(o, infl) for o in self.obstacles]
        dpsi = rad(angle_deg) * (1 if turn_dir.upper() == 'L' else -1)

        if turn_dir.upper() == 'L':
            cx = sx - R * math.sin(th); cy = sy + R * math.cos(th)
            a0 = math.atan2(sy - cy, sx - cx); a1 = a0 + dpsi
            if a1 <= a0: a1 += 2*math.pi
            length = R * (a1 - a0)
            n = max(3, int(length/2.5))
            for k in range(n+1):
                a = a0 + (a1 - a0) * (k/n)
                px = cx + R * math.cos(a); py = cy + R * math.sin(a)
                if k == 0 and allow_start_outside:
                    continue
                if not self._inside_walls(px, py):
                    self._log(f"[arc_clear L] OOB at ({px:.2f},{py:.2f})")
                    return False
                for r in rects:
                    if self._pt_in_rect(px, py, r):
                        self._log(f"[arc_clear L] obstacle hit at ({px:.2f},{py:.2f}) in rect {r}")
                        return False
        else:
            cx = sx + R * math.sin(th); cy = sy - R * math.cos(th)
            a0 = math.atan2(sy - cy, sx - cx); a1 = a0 + dpsi
            if a0 <= a1: a0 += 2*math.pi
            length = R * (a0 - a1)
            n = max(3, int(length/2.5))
            for k in range(n+1):
                a = a0 - (a0 - a1) * (k/n)
                px = cx + R * math.cos(a); py = cy + R * math.sin(a)
                if k == 0 and allow_start_outside:
                    continue
                if not self._inside_walls(px, py):
                    self._log(f"[arc_clear R] OOB at ({px:.2f},{py:.2f})")
                    return False
                for r in rects:
                    if self._pt_in_rect(px, py, r):
                        self._log(f"[arc_clear R] obstacle hit at ({px:.2f},{py:.2f}) in rect {r}")
                        return False
        return True

    # --------------------- Quantization ---------------------
    def _xy_to_idx(self, x: float, y: float) -> Tuple[int,int]:
        return int(round(x / self.res_cm)), int(round(y / self.res_cm))

    def _idx_to_xy(self, i: int, j: int) -> Tuple[float,float]:
        return i * self.res_cm, j * self.res_cm

    @staticmethod
    def _th_to_bin(theta: float) -> int:
        return int(round((theta % (2*math.pi)) / (math.pi/4))) % 8

    @staticmethod
    def _bin_to_th(h: int) -> float:
        return h * (math.pi / 4)

    # --------------------- Motion primitives ---------------------
    def _successors(self, i: int, j: int, h: int, allow_start_outside=False):
        """
        Generate neighbor states lazily. Order is important:
        1) Forward (prefer straight)
        2) Backward
        3) Arcs (L/R 45°, then 90°)
        """
        x, y = self._idx_to_xy(i, j)
        th   = self._bin_to_th(h)

        STEPS = list(range(10, 201, 10))  # 10..200 cm
        c, s = math.cos(th), math.sin(th)

        # FORWARD
        for d in STEPS:
            nx, ny = x + d*c, y + d*s
            if self._segment_clear(x, y, nx, ny, allow_start_outside=allow_start_outside):
                ii, jj = self._xy_to_idx(nx, ny)
                yield (ii, jj, h, ("move", +1, d, (nx, ny, th)))
            else:
                break  # longer along same ray will also be blocked

        # BACKWARD (discourage but allow)
        for d in STEPS:
            nx, ny = x - d*c, y - d*s
            if self._segment_clear(x, y, nx, ny, allow_start_outside=allow_start_outside):
                ii, jj = self._xy_to_idx(nx, ny)
                yield (ii, jj, h, ("move", -1, d, (nx, ny, th)))
            else:
                break

        # ARCS (with slip)
        for turn_dir in ("L","R"):
            for ang in (45, 90):  # smaller overshoot first
                if self._arc_clear(x, y, th, turn_dir, ang, allow_start_outside=allow_start_outside):
                    nx, ny, nth = self._arc_end_pose(x, y, th, turn_dir, ang)
                    ii, jj = self._xy_to_idx(nx, ny)
                    nh = self._th_to_bin(nth)
                    yield (ii, jj, nh, ("arc", turn_dir, ang, (nx, ny, nth)))

    # --------------------- Costs & Heuristic ---------------------
    def _cost(self, action) -> float:
        v_lin = max(1e-6, float(config.car.linear_speed_cm_s))
        v_arc = min(
            v_lin,
            float(getattr(config.car, "turn_linear_cm_s", 0.0)) or
            float(config.car.turning_radius) * float(getattr(config.car, "angular_speed_rad_s", 0.0)) or
            v_lin
        )
        TURN_EPS = float(getattr(config.car, "turn_bias_seconds", 0.05))  # small per 45°
        BACKWARD_MULT = float(getattr(config.car, "backward_cost_mult", 1.25))

        kind = action[0]
        if kind == "move":
            _, sign, d_cm, _ = action
            t = d_cm / v_lin
            if sign < 0:
                t *= BACKWARD_MULT
            return t
        else:
            _, _turn_dir, ang, _ = action
            s_slip = self._turn_slip(float(ang))
            s_arc = float(config.car.turning_radius) * rad(ang)
            base = (s_slip / v_lin) + (s_arc / v_arc)
            per45 = int(round(ang / 45))
            return base + TURN_EPS * per45

    def _heur(self, i: int, j: int, h: int, gi: int, gj: int, gh: int) -> float:
        """Admissible: straight-line time + tiny heading-bias."""
        v_lin = max(1e-6, float(config.car.linear_speed_cm_s))
        dx = (gi - i) * self.res_cm
        dy = (gj - j) * self.res_cm
        dist = math.hypot(dx, dy)
        t_lin = dist / v_lin

        delta = abs(gh - h)
        ticks = min(delta, 8 - delta)
        TURN_EPS = float(getattr(config.car, "turn_bias_seconds", 0.05))
        return t_lin + TURN_EPS * ticks

    # --------------------- Helpers ---------------------
    @staticmethod
    def _angdiff(a: float, b: float) -> float:
        """Smallest absolute difference between two angles (radians)."""
        return abs((a - b + math.pi) % (2*math.pi) - math.pi)

    def _reconstruct(self, came, goal_key):
        path_actions = []
        k = goal_key
        while k in came:
            pk, action = came[k]
            path_actions.append(action)
            k = pk
        path_actions.reverse()
        return path_actions

    # --------------------- A* search ---------------------
    def _a_star(
        self,
        start: CarState,
        goal: CarState,
        pos_tol_cm: Optional[float] = None,
        heading_tol_deg: Optional[float] = None,
        max_expansions: int = 400000,
    ):
        # Select tolerances (use config defaults if None)
        pos_tol = float(pos_tol_cm if pos_tol_cm is not None else self.goal_pos_eps_cm)
        head_tol_rad = rad(heading_tol_deg if heading_tol_deg is not None else self.goal_head_eps_deg)

        si, sj = self._xy_to_idx(start.x, start.y)
        sh = self._th_to_bin(start.theta)
        gi, gj = self._xy_to_idx(goal.x, goal.y)
        gh = self._th_to_bin(goal.theta)

        start_key = (si, sj, sh)
        gscore = { start_key: 0.0 }
        came: Dict[Tuple[int,int,int], Tuple[Tuple[int,int,int], tuple]] = {}

        openh = []
        heapq.heappush(
            openh,
            (self._heur(si, sj, sh, gi, gj, gh), 0.0, start_key, (start.x, start.y, start.theta))
        )

        expansions = 0

        while openh:
            f, g, key, pose = heapq.heappop(openh)
            i, j, h = key
            x, y, th = pose

            expansions += 1
            if expansions % self._expand_log_every == 0:
                self._log(f"[A*] expansions={expansions}, open={len(openh)}, g={g:.3f}, f={f:.3f}")

            # tolerant goal test
            if (math.hypot(x - goal.x, y - goal.y) <= pos_tol) and (self._angdiff(th, goal.theta) <= head_tol_rad):
                actions = self._reconstruct(came, key)
                return actions, (x, y, th)

            # neighbors
            for ni, nj, nh, action in self._successors(i, j, h, allow_start_outside=True if expansions == 1 else False):
                nk = (ni, nj, nh)
                ng = g + self._cost(action)
                if ng + 1e-9 < gscore.get(nk, float('inf')):
                    gscore[nk] = ng
                    came[nk] = (key, action)
                    _, _, _, cont = action
                    nx, ny, nth = cont
                    f_new = ng + self._heur(ni, nj, nh, gi, gj, gh)
                    heapq.heappush(openh, (f_new, ng, nk, (nx, ny, nth)))

            if expansions > max_expansions:
                self._log("[A*] GAVE UP: expansion limit")
                break

        return None, None

    # --------------------- Orders ---------------------
    def _compress_orders(self, actions: List[tuple], start_pose: CarState) -> Tuple[List[Dict], CarState]:
        """Merge consecutive F or B with same heading; emit arc orders as-is (with slip value)."""
        orders: List[Dict] = []
        x, y, th = start_pose.x, start_pose.y, start_pose.theta

        run_kind = None  # 'F' or 'B'
        run_dist = 0

        def flush():
            nonlocal run_kind, run_dist, x, y, th
            if run_kind and run_dist > 0:
                nx = x + (run_dist if run_kind == 'F' else -run_dist) * math.cos(th)
                ny = y + (run_dist if run_kind == 'F' else -run_dist) * math.sin(th)
                orders.append({
                    "op": run_kind,
                    "value_cm": int(round(run_dist)),
                    "pose": {"x": round(nx,2), "y": round(ny,2),
                             "theta": round(th,4), "theta_deg": round(deg(th),1)}
                })
                x, y = nx, ny
            run_kind = None
            run_dist = 0

        for a in actions:
            if a[0] == "move":
                _, sign, d_cm, _cont = a
                want = 'F' if sign > 0 else 'B'
                if run_kind in (None, want):
                    run_kind = want
                    run_dist += d_cm
                else:
                    flush()
                    run_kind = want
                    run_dist = d_cm
            else:
                # arc (with slip embedded in the end pose)
                flush()
                _, turn_dir, ang, cont = a
                nx, ny, nth = cont
                slip = self._turn_slip(float(ang))
                orders.append({
                    "op": "L" if turn_dir == "L" else "R",
                    "angle_deg": int(ang),
                    "radius_cm": float(config.car.turning_radius),
                    "pre_slip_cm": round(slip, 2),
                    "pose": {"x": round(nx,2), "y": round(ny,2),
                             "theta": round(nth,4), "theta_deg": round(deg(nth),1)}
                })
                x, y, th = nx, ny, nth

        flush()
        return orders, CarState(x, y, th)

    # --------------------- Public API ---------------------
    def plan_orders_between(self, start: CarState, goal: CarState) -> List[Dict]:
        self._log(
            f"[plan_between] start=({start.x:.2f},{start.y:.2f},{deg(start.theta):.1f}°) "
            f"goal=({goal.x:.2f},{goal.y:.2f},{deg(goal.theta):.1f}°) "
            f"grid={self.grid_N}x{self.grid_N} res={self.res_cm:.3f}cm "
            f"tol=±{self.goal_pos_eps_cm:.1f}cm, ±{self.goal_head_eps_deg:.1f}°"
        )

        actions, _final_pose = self._a_star(
            start, goal,
            pos_tol_cm=self.goal_pos_eps_cm,
            heading_tol_deg=self.goal_head_eps_deg
        )
        if actions is None:
            self._log("[plan_between] FAILED: no path")
            return []

        orders, _ = self._compress_orders(actions, start)
        self._log(f"[plan_between] SUCCESS: {len(orders)} orders")
        return orders

    # ---------- robust scan pose ----------
    def get_image_target_position(self, obstacle: Obstacle) -> CarState:
        size = float(config.arena.obstacle_size)
        cell = max(self.res_cm, float(getattr(config.arena, "grid_cell_size", self.res_cm)))
        infl = self._inflation()

        # camera distance (car or vision), scaled if desired
        cam_dist = (
            getattr(getattr(config, "car", object()), "camera_distance", None)
            or getattr(getattr(config, "vision", object()), "camera_distance", None)
            or getattr(getattr(config, "vision", object()), "camera_distance_cm", None)
            or 30.0
        )
        scale = float(getattr(getattr(config, "planner", object()), "scan_distance_scale", 1.0))
        d_nominal = float(cam_dist) * scale

        keepout = infl + 0.51 * cell
        d = max(d_nominal, keepout)

        side = obstacle.image_side.upper()
        if side == 'S':
            x = obstacle.x + size/2.0; y = obstacle.y - d; th = math.pi/2
        elif side == 'N':
            x = obstacle.x + size/2.0; y = obstacle.y + size + d; th = 3*math.pi/2
        elif side == 'E':
            x = obstacle.x + size + d; y = obstacle.y + size/2.0; th = math.pi
        else:  # 'W'
            x = obstacle.x - d; y = obstacle.y + size/2.0; th = 0.0

        # Nudge outward (≤ 20 cm) if still illegal
        step = 1.0
        tries = 0
        while (not self._inside_walls(x, y)) or any(self._pt_in_rect(x, y, self._rect_infl(o, infl)) for o in self.obstacles):
            if tries > 20:
                return None
            if side == 'S':      y -= step
            elif side == 'N':    y += step
            elif side == 'E':    x += step
            else:                x -= step
            tries += 1

        self._log(f"[scan_pose] side={side} cam={cam_dist}cm scale={scale} -> d={d:.1f}cm keepout={keepout:.1f}cm")
        return CarState(x, y, th)

    def plan_visiting_orders_discrete(self, start_state: CarState, obstacle_indices: List[int]) -> List[Dict]:
        """Visit obstacles in the given order; emits S (scan) between legs."""
        orders_out: List[Dict] = []
        scan_sec = float(getattr(config.vision, "scan_seconds", 0.0))
        cur = start_state

        for idx in obstacle_indices:
            if not (0 <= idx < len(self.obstacles)):
                continue
            goal = self.get_image_target_position(self.obstacles[idx])
            if goal is None:
                print("Cannot")
                continue
            self._log(f"[visit] target#{idx} scan_pose=({goal.x:.2f},{goal.y:.2f},{deg(goal.theta):.1f}°)")
            segment_orders = self.plan_orders_between(cur, goal)
            if not segment_orders:
                self._log(f"[visit] FAIL to reach obstacle {idx}")
                return []
            orders_out.extend(segment_orders)

            last_pose = segment_orders[-1]["pose"]
            s = {"op": "S", "obstacle_id": idx, "pose": last_pose}
            if scan_sec > 0:
                s["scan_seconds"] = round(scan_sec, 2)
            orders_out.append(s)

            cur = CarState(last_pose["x"], last_pose["y"], last_pose["theta"])

        return orders_out


# --------------------- Example quick test ---------------------
if __name__ == "__main__":
    planner = CarPathPlanner(grid_N=2000)
    planner.add_obstacle(50, 130, 'S')
    planner.add_obstacle(20, 170, 'E')

    start = CarState(20, 20, math.pi/2)   # facing up
    goal  = CarState(55, 106, math.pi/2)  # e.g., scan pose

    orders = planner.plan_orders_between(start, goal)
    if orders:
        print({"orders_count": len(orders), "last_pose": orders[-1]["pose"]})
    else:
        print({"status": "failed", "message": "Planner failed"})
