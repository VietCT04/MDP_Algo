"""
Simplest discrete planner with ARC turns (no in-place spin).

Key behavior
------------
- L/R are *arc segments* at radius R = config.car.turning_radius (default 24.5 cm).
  They update x,y,theta along a circular path (no in-place rotation).
- Prefer long FORWARD legs; only arc when you must change heading.
- Snap scan pose onto the current forward ray (within ~10 cm lateral) to avoid tiny sidesteps.
- Axis-first routing (Y-then-X, else X-then-Y). If blocked, fall back to a permissive "bug" walker.
- Rich debug logs via print() so you can trace every choice & collision check.

Assumes:
- algorithm.lib.car.CarState
- algorithm.lib.path.Obstacle
- algorithm.config.config
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from algorithm.lib.car import CarState
from algorithm.lib.path import Obstacle
from algorithm.config import config


deg = math.degrees

class CarPathPlanner:
    def __init__(self):
        self.arena_size = float(config.arena.size)
        self.grid_size  = int(config.get_grid_size())
        self.obstacles: List[Obstacle] = []
        self.collision_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.debug = True  # flip to False to silence logs

    # ---------- logging ----------
    def _log(self, msg: str):
        if self.debug:
            print(msg)

    # ---------- world / obstacles ----------
    def add_obstacle(self, x: int | float, y: int | float, image_side: str):
        cell = float(config.arena.grid_cell_size)
        x = int(round(x / cell) * cell)
        y = int(round(y / cell) * cell)
        obstacle = Obstacle(x, y, image_side)
        self.obstacles.append(obstacle)
        self._update_collision_grid()
        self._log(f"[add_obstacle] ({x:.2f},{y:.2f}) side={image_side}")

    def _update_collision_grid(self):
        self.collision_grid.fill(0)
        buffer  = float(config.arena.collision_buffer)
        cell    = float(config.arena.grid_cell_size)
        obs_sz  = float(config.arena.obstacle_size)
        for obs in self.obstacles:
            min_x = max(0, math.floor((obs.x - buffer) / cell))
            max_x = min(self.grid_size, math.ceil((obs.x + obs_sz + buffer) / cell))
            min_y = max(0, math.floor((obs.y - buffer) / cell))
            max_y = min(self.grid_size, math.ceil((obs.y + obs_sz + buffer) / cell))
            if min_x < max_x and min_y < max_y:
                self.collision_grid[int(min_y):int(max_y), int(min_x):int(max_x)] = 1

    # ---------- geometry & safety ----------
    @staticmethod
    def _inflation() -> float:
        return float(config.car.width) / 2.0 + float(config.arena.collision_buffer)

    def _rect_infl(self, o: Obstacle, infl: float) -> Tuple[float,float,float,float]:
        s = float(config.arena.obstacle_size)
        return (o.x - infl, o.y - infl, o.x + s + infl, o.y + s + infl)

    @staticmethod
    def _pt_in_rect(px: float, py: float, r: Tuple[float,float,float,float]) -> bool:
        x0, y0, x1, y1 = r
        return (x0 <= px <= x1) and (y0 <= py <= y1)

    def _inside_walls(self, x: float, y: float) -> bool:
        m = self._inflation()
        return (m <= x <= self.arena_size - m) and (m <= y <= self.arena_size - m)

    def _pose_clear(self, x: float, y: float) -> bool:
        if not self._inside_walls(x, y):
            return False
        infl = self._inflation()
        for o in self.obstacles:
            if self._pt_in_rect(x, y, self._rect_infl(o, infl)):
                return False
        return True

    def _segment_clear(self, ax: float, ay: float, bx: float, by: float) -> bool:
        """Collision check for a straight segment."""
        infl = self._inflation()
        rects = [self._rect_infl(o, infl) for o in self.obstacles]
        m = infl
        size = self.arena_size
        L = max(1e-6, math.hypot(bx-ax, by-ay))
        samples = max(2, int(L / 2.5))
        for i in range(samples + 1):
            t = i / samples
            x = ax + t*(bx-ax)
            y = ay + t*(by-ay)
            if x < m or x > size - m or y < m or y > size - m:
                self._log(f"[seg_clear] out-of-bounds sample ({x:.2f},{y:.2f})")
                return False
            for r in rects:
                if self._pt_in_rect(x, y, r):
                    self._log(f"[seg_clear] obstacle hit at ({x:.2f},{y:.2f}) in rect {r}")
                    return False
        return True
    
    # --- NEW: exact arc delta for a single turn (used for compensation) ---
    def _arc_delta(self, th: float, turn_dir: str, angle_deg: float) -> tuple[float, float]:
        """
        Returns (dx, dy) displacement caused by an arc of 'angle_deg' at radius R
        starting at heading 'th'. Positive 'angle_deg' always; turn_dir in {'L','R'}.
        Formula:
        dx = R [sin(th + dψ) - sin(th)]
        dy = R [cos(th) - cos(th + dψ)]
        where dψ = +angle for L, -angle for R (in radians).
        """
        R = float(config.car.turning_radius)
        sgn = 1.0 if turn_dir.upper() == 'L' else -1.0
        dpsi = math.radians(angle_deg) * sgn
        dx = R * (math.sin(th + dpsi) - math.sin(th))
        dy = R * (math.cos(th) - math.cos(th + dpsi))
        return dx, dy


    # --- PATCH: safer heading change with controllable step size & better rounding ---
    def _face_heading(self, target_th: float, x: float, y: float, th: float,
                    orders: list[dict], max_step_deg: int = 90) -> tuple[float,float,float]:
        """
        Rotate toward target_th using arcs (no in-place spin).
        max_step_deg lets us enforce a 45° final approach if desired.
        """
        d = (target_th - th + math.pi) % (2*math.pi) - math.pi
        if abs(d) < math.radians(1.0):
            return x, y, th

        turn_dir = 'L' if d > 0 else 'R'
        remaining = abs(d)

        # Use only 90s and 45s; clamp largest step by max_step_deg.
        steps = []
        if max_step_deg >= 90:
            n90 = int(remaining // math.radians(90))
            remaining -= n90 * math.radians(90)
            steps += [90] * n90

        # Whatever remains ~(<90°) -> round to nearest 45°
        n45 = int(round(remaining / math.radians(45)))
        # If max_step_deg=45, split any 90s we had queued (shouldn't happen if we respect max_step_deg)
        if max_step_deg == 45 and steps:
            # turn queued 90s into two 45s each
            steps = [45 for _ in range(2*len(steps))]

        steps += [45] * n45

        for ang in steps:
            x, y, th = self._emit_arc(x, y, th, turn_dir, ang, orders)
        return x, y, th


    # --- PATCH: axis-first plan with FINAL-ARC COMPENSATION (keeps scan x/y tight) ---
    def _axis_plan(self, start: CarState, goal: CarState) -> Optional[List[Dict]]:
        """
        Try Y-then-X (preferred), else X-then-Y.
        For the final approach leg we force a single 45° arc and PRE-COMPENSATE the prior straight
        so that after the arc we land exactly on goal.x (for NS scans) or goal.y (for EW scans).
        """
        SCAN_LATERAL_TOL_CM = 10.0  # your ±10 band
        orders: List[Dict] = []
        x, y, th = start.x, start.y, start.theta
        gx, gy, gth = goal.x, goal.y, goal.theta

        def fwd_to(nx: float, ny: float):
            nonlocal x, y, th, orders
            x, y = self._emit_forward(x, y, nx, ny, th, orders)

        # -------------------- Y then X --------------------
        try:
            # 1) Go to target Y first
            dy = gy - y
            if abs(dy) > 1e-6:
                want_y = math.pi/2 if dy > 0 else 3*math.pi/2
                # normal rotate (90/45s ok)
                x, y, th = self._face_heading(want_y, x, y, th, orders, max_step_deg=90)
                fwd_to(x, gy)

            # 2) Now approach X with a compensated final 45° arc toward gth
            dx = gx - x
            if abs(dx) > 1e-6:
                # We want the *last* heading change before scanning to be a single 45° arc.
                # First, pre-rotate to be “almost” along +X or -X (at ±45° from gth), then forward,
                # then a 45° arc into gth, then we’re done with X (and at gth already).
                # Decide turn direction for the final 45° based on (gth - current th).
                # Step A: aim to an intermediate heading so that the remaining delta to gth is exactly 45°.
                # Compute which side (L or R) is the shorter 45° into gth:
                cand = []
                for final_dir in ('L','R'):
                    # if we did a final 45° of this direction, what would the pre-arc heading be?
                    pre_th = (gth - math.radians(45) if final_dir=='L' else gth + math.radians(45)) % (2*math.pi)
                    # how far (and which way) from current th to that pre_th?
                    dpre = (pre_th - th + math.pi) % (2*math.pi) - math.pi
                    cand.append((abs(dpre), final_dir, pre_th))
                cand.sort(key=lambda t: t[0])  # pick smaller absolute rotate to pre_th
                final_dir, pre_th = cand[0][1], cand[0][2]

                # Step B: rotate to pre_th (allow 90s/45s)
                x, y, th = self._face_heading(pre_th, x, y, th, orders, max_step_deg=90)

                # Step C: compute the displacement of the FINAL 45° arc and pre-compensate X
                dxf, dyf = self._arc_delta(th, final_dir, 45)
                x_wp = gx - dxf  # after the final arc, x should be gx exactly
                self._log(f"[compensate] final {final_dir}45 causes Δx={dxf:+.2f}, Δy={dyf:+.2f} -> "
                        f"pre-adjust X to {x_wp:.2f} so post-arc x≈{gx:.2f}")

                # forward along current heading to x_wp (heading is pre_th which is ±45° from gth)
                # We need to move purely in X here; pre_th may not be 0/π, so do a short correction:
                # Easiest: force a brief rotate to 0/π, forward to x_wp, then rotate back to pre_th.
                # (Keeps it simple & predictable.)
                want_x = 0.0 if (x_wp - x) >= 0 else math.pi
                x, y, th = self._face_heading(want_x, x, y, th, orders, max_step_deg=90)
                fwd_to(x_wp, y)
                x, y, th = self._face_heading(pre_th, x, y, th, orders, max_step_deg=90)

                # Step D: do the final 45° arc into gth, no extra drift surprises now
                x, y, th = self._emit_arc(x, y, th, final_dir, 45, orders)

            # 3) We are already at gth after the compensated arc; tiny cleanup rotate (should be 0)
            x, y, th = self._face_heading(gth, x, y, th, orders, max_step_deg=45)

            # Validate lateral band at scan:
            if abs(x - gx) > SCAN_LATERAL_TOL_CM:
                self._log(f"[warn] lateral miss after compensation: |x-gx|={abs(x-gx):.2f}cm > {SCAN_LATERAL_TOL_CM}")
            return orders

        except RuntimeError as e:
            self._log(f"[axis_plan YX] blocked: {e}")
            orders.clear()
            x, y, th = start.x, start.y, start.theta

        # -------------------- X then Y --------------------
        try:
            # 1) Go to target X first
            dx = gx - x
            if abs(dx) > 1e-6:
                want_x = 0.0 if dx > 0 else math.pi
                x, y, th = self._face_heading(want_x, x, y, th, orders, max_step_deg=90)
                fwd_to(gx, y)

            # 2) Compensate final 45° into gth before moving along Y
            dy = gy - y
            if abs(dy) > 1e-6:
                cand = []
                for final_dir in ('L','R'):
                    pre_th = (gth - math.radians(45) if final_dir=='L' else gth + math.radians(45)) % (2*math.pi)
                    dpre = (pre_th - th + math.pi) % (2*math.pi) - math.pi
                    cand.append((abs(dpre), final_dir, pre_th))
                cand.sort(key=lambda t: t[0])
                final_dir, pre_th = cand[0][1], cand[0][2]

                x, y, th = self._face_heading(pre_th, x, y, th, orders, max_step_deg=90)

                dxf, dyf = self._arc_delta(th, final_dir, 45)
                y_wp = gy - dyf
                self._log(f"[compensate] final {final_dir}45 causes Δx={dxf:+.2f}, Δy={dyf:+.2f} -> "
                        f"pre-adjust Y to {y_wp:.2f} so post-arc y≈{gy:.2f}")

                want_y = math.pi/2 if (y_wp - y) >= 0 else 3*math.pi/2
                x, y, th = self._face_heading(want_y, x, y, th, orders, max_step_deg=90)
                fwd_to(x, y_wp)
                x, y, th = self._face_heading(pre_th, x, y, th, orders, max_step_deg=90)

                x, y, th = self._emit_arc(x, y, th, final_dir, 45, orders)

            x, y, th = self._face_heading(gth, x, y, th, orders, max_step_deg=45)

            if abs(y - gy) > SCAN_LATERAL_TOL_CM:
                self._log(f"[warn] lateral miss after compensation: |y-gy|={abs(y-gy):.2f}cm > {SCAN_LATERAL_TOL_CM}")
            return orders

        except RuntimeError as e:
            self._log(f"[axis_plan XY] blocked: {e}")
            return None


    # ---------- arc geometry ----------
    def _arc_end_pose(self, x: float, y: float, th: float, turn_dir: str, angle_deg: float) -> Tuple[float,float,float]:
        """End pose after an arc of 'angle_deg' (positive) at radius R about current heading."""
        R = float(config.car.turning_radius)
        dpsi = math.radians(angle_deg) * (1 if turn_dir.upper() == 'L' else -1)
        if turn_dir.upper() == 'L':
            cx = x - R * math.sin(th)
            cy = y + R * math.cos(th)
        else:
            cx = x + R * math.sin(th)
            cy = y - R * math.cos(th)
        a0 = math.atan2(y - cy, x - cx)
        a1 = a0 + dpsi
        nx = cx + R * math.cos(a1)
        ny = cy + R * math.sin(a1)
        nth = (th + dpsi) % (2*math.pi)
        return nx, ny, nth

    def _arc_clear(self, x: float, y: float, th: float, turn_dir: str, angle_deg: float) -> bool:
        """Collision check along the arc."""
        R = float(config.car.turning_radius)
        infl = self._inflation()
        rects = [self._rect_infl(o, infl) for o in self.obstacles]
        m = infl
        size = self.arena_size
        dpsi = math.radians(angle_deg) * (1 if turn_dir.upper() == 'L' else -1)

        if turn_dir.upper() == 'L':
            cx = x - R * math.sin(th); cy = y + R * math.cos(th)
            a0 = math.atan2(y - cy, x - cx); a1 = a0 + dpsi
            if a1 <= a0: a1 += 2*math.pi
            length = R * (a1 - a0)
            n = max(3, int(length / 2.5))
            for k in range(n+1):
                a = a0 + (a1 - a0) * (k / n)
                sx = cx + R * math.cos(a); sy = cy + R * math.sin(a)
                if sx < m or sx > size - m or sy < m or sy > size - m:
                    self._log(f"[arc_clear L] OOB at ({sx:.2f},{sy:.2f})")
                    return False
                for r in rects:
                    if self._pt_in_rect(sx, sy, r):
                        self._log(f"[arc_clear L] obstacle hit at ({sx:.2f},{sy:.2f}) in rect {r}")
                        return False
        else:
            cx = x + R * math.sin(th); cy = y - R * math.cos(th)
            a0 = math.atan2(y - cy, x - cx); a1 = a0 + dpsi
            if a0 <= a1: a0 += 2*math.pi
            length = R * (a0 - a1)
            n = max(3, int(length / 2.5))
            for k in range(n+1):
                a = a0 - (a0 - a1) * (k / n)
                sx = cx + R * math.cos(a); sy = cy + R * math.sin(a)
                if sx < m or sx > size - m or sy < m or sy > size - m:
                    self._log(f"[arc_clear R] OOB at ({sx:.2f},{sy:.2f})")
                    return False
                for r in rects:
                    if self._pt_in_rect(sx, sy, r):
                        self._log(f"[arc_clear R] obstacle hit at ({sx:.2f},{sy:.2f}) in rect {r}")
                        return False
        return True

    def _emit_arc(self, x: float, y: float, th: float, turn_dir: str, angle_deg: int, orders: List[Dict]) -> Tuple[float,float,float]:
        """Emit an arc (L/R) of angle_deg at radius R; update pose; log."""
        if angle_deg == 0:
            return x, y, th
        if not self._arc_clear(x, y, th, turn_dir, angle_deg):
            raise RuntimeError(f"arc blocked: {turn_dir}{angle_deg}")
        nx, ny, nth = self._arc_end_pose(x, y, th, turn_dir, angle_deg)
        orders.append({
            "op": "L" if turn_dir.upper() == "L" else "R",
            "angle_deg": int(angle_deg),
            "radius_cm": float(config.car.turning_radius),
            "pose": {
                "x": round(nx, 2), "y": round(ny, 2),
                "theta": round(nth, 4), "theta_deg": round(deg(nth), 1)
            }
        })
        self._log(f"[emit_arc] {turn_dir}{angle_deg}° @R={config.car.turning_radius:.2f} "
                  f"from ({x:.2f},{y:.2f},{deg(th):.1f}°) -> ({nx:.2f},{ny:.2f},{deg(nth):.1f}°)")
        return nx, ny, nth

    def _emit_forward(self, sx: float, sy: float, ex: float, ey: float, th: float, orders: List[Dict]) -> Tuple[float,float]:
        if not self._segment_clear(sx, sy, ex, ey):
            raise RuntimeError("forward leg blocked")
        dist = int(round(math.hypot(ex - sx, ey - sy)))
        orders.append({
            "op": "F",
            "value_cm": dist,
            "pose": {"x": round(ex,2), "y": round(ey,2), "theta": round(th,4), "theta_deg": round(deg(th),1)}
        })
        self._log(f"[emit_forward] {dist}cm ({sx:.2f},{sy:.2f}) -> ({ex:.2f},{ey:.2f}) heading {deg(th):.1f}°")
        return ex, ey

    # ---------- robust scan pose ----------
    def get_image_target_position(self, obstacle: Obstacle) -> CarState:
        size = float(config.arena.obstacle_size)
        cell = float(config.arena.grid_cell_size)
        infl = self._inflation()

        d_nominal = float(config.car.camera_distance) * 0.8
        d = max(d_nominal, infl + 0.51*cell)

        side = obstacle.image_side.upper()
        if side == 'S':
            x = obstacle.x + size/2.0; y = obstacle.y - d; th = math.pi/2
        elif side == 'N':
            x = obstacle.x + size/2.0; y = obstacle.y + size + d; th = 3*math.pi/2
        elif side == 'E':
            x = obstacle.x + size + d; y = obstacle.y + size/2.0; th = math.pi
        else:  # 'W'
            x = obstacle.x - d; y = obstacle.y + size/2.0; th = 0.0

        if not self._pose_clear(x, y):
            # push one more half-cell outward if needed
            if side == 'S':      y -= 0.51*cell
            elif side == 'N':    y += 0.51*cell
            elif side == 'E':    x += 0.51*cell
            else:                x -= 0.51*cell
            if not self._pose_clear(x, y):
                raise ValueError("scan pose not clear")
        self._log(f"[scan_pose] side={side} -> ({x:.2f},{y:.2f}) facing {deg(th):.1f}°")
        return CarState(x, y, th)

    # ---------- goal snap to forward ray ----------
    def _snap_goal_to_forward_ray(self, start: CarState, goal: CarState, tol_cm: float = 10.0) -> CarState:
        """If already aligned, snap laterally so we can just drive straight."""
        sx, sy, sth = start.x, start.y, start.theta
        gx, gy, gth = goal.x, goal.y, goal.theta

        def is_ns(th):
            th = (th + 2*math.pi) % (2*math.pi)
            return min(abs(th - math.pi/2), abs(th - 3*math.pi/2)) < math.radians(10)

        def is_ew(th):
            th = (th + 2*math.pi) % (2*math.pi)
            return min(abs(th - 0.0), abs(th - math.pi)) < math.radians(10)

        snapped = None
        if is_ns(gth) and is_ns(sth) and abs(sx - gx) <= tol_cm:
            ngx = round(sx, 2)
            if self._pose_clear(ngx, gy) and self._segment_clear(sx, sy, ngx, gy):
                snapped = CarState(ngx, gy, gth)
        elif is_ew(gth) and is_ew(sth) and abs(sy - gy) <= tol_cm:
            ngy = round(sy, 2)
            if self._pose_clear(gx, ngy) and self._segment_clear(sx, sy, gx, ngy):
                snapped = CarState(gx, ngy, gth)

        if snapped:
            self._log(f"[snap] goal snapped from ({gx:.2f},{gy:.2f}) to ({snapped.x:.2f},{snapped.y:.2f})")
            return snapped
        return goal

    # ---------- axis-first plan using ARC turns ----------
    def _axis_plan(self, start: CarState, goal: CarState) -> Optional[List[Dict]]:
        """
        Try Y-then-X, else X-then-Y.
        Heading changes are done with ARC turns at radius R (45°/90° steps).
        """
        orders: List[Dict] = []
        x, y, th = start.x, start.y, start.theta
        gx, gy, gth = goal.x, goal.y, goal.theta

        def face_heading(target_th: float):
            """Turn from 'th' to 'target_th' using arcs (prefer 90°, then 45°)."""
            nonlocal x, y, th, orders
            # normalize delta to [-pi,pi]
            d = (target_th - th + math.pi) % (2*math.pi) - math.pi
            if abs(d) < math.radians(1):
                return
            turn_dir = 'L' if d > 0 else 'R'
            rem = abs(d)
            # chunk into 90° then leftover 45° (rounded)
            ninety = int(rem // (math.pi/2))
            rem -= ninety * (math.pi/2)
            forty5 = int(round(rem / (math.pi/4)))
            for _ in range(ninety):
                x, y, th = self._emit_arc(x, y, th, turn_dir, 90, orders)
            if forty5:
                x, y, th = self._emit_arc(x, y, th, turn_dir, 45, orders)

        def fwd_to(nx: float, ny: float):
            nonlocal x, y, th, orders
            x, y = self._emit_forward(x, y, nx, ny, th, orders)

        eps = 1e-6

        # Try Y then X
        try:
            dy = gy - y
            if abs(dy) > eps:
                face_heading(math.pi/2 if dy > 0 else 3*math.pi/2)
                fwd_to(x, gy)
            dx = gx - x
            if abs(dx) > eps:
                face_heading(0.0 if dx > 0 else math.pi)
                fwd_to(gx, y)
            face_heading(gth)
            return orders
        except RuntimeError as e:
            self._log(f"[axis_plan YX] blocked: {e}")
            orders.clear()
            x, y, th = start.x, start.y, start.theta

        # Try X then Y
        try:
            dx = gx - x
            if abs(dx) > eps:
                face_heading(0.0 if dx > 0 else math.pi)
                fwd_to(gx, y)
            dy = gy - y
            if abs(dy) > eps:
                face_heading(math.pi/2 if dy > 0 else 3*math.pi/2)
                fwd_to(x, gy)
            face_heading(gth)
            return orders
        except RuntimeError as e:
            self._log(f"[axis_plan XY] blocked: {e}")
            return None

    # ---------- permissive bug fallback (arc-based) ----------
    def _bug_fallback(self, start: CarState, goal: CarState) -> Optional[List[Dict]]:
        cell = float(config.arena.grid_cell_size)
        x, y, th = start.x, start.y, start.theta
        gx, gy, gth = goal.x, goal.y, goal.theta
        orders: List[Dict] = []
        steps = 0
        self._log("[bug] entering fallback walker")

        while steps < 2000:
            steps += 1
            # reach?
            if math.hypot(gx - x, gy - y) <= 0.5*cell:
                # final heading
                d = (gth - th + math.pi) % (2*math.pi) - math.pi
                if abs(d) > math.radians(1):
                    turn_dir = 'L' if d > 0 else 'R'
                    rem = abs(d)
                    ninety = int(rem // (math.pi/2)); rem -= ninety*(math.pi/2)
                    forty5 = int(round(rem / (math.pi/4)))
                    for _ in range(ninety):
                        x, y, th = self._emit_arc(x, y, th, turn_dir, 90, orders)
                    if forty5:
                        x, y, th = self._emit_arc(x, y, th, turn_dir, 45, orders)
                return orders

            want = math.atan2(gy - y, gx - x)
            d = (want - th + math.pi) % (2*math.pi) - math.pi

            if abs(d) > math.radians(22.5):
                # swing 45° toward the target
                x, y, th = self._emit_arc(x, y, th, 'L' if d > 0 else 'R', 45, orders)
                continue

            # otherwise try a 1-cell forward step
            nx = x + cell * math.cos(th)
            ny = y + cell * math.sin(th)
            try:
                x, y = self._emit_forward(x, y, nx, ny, th, orders)
            except RuntimeError:
                # blocked straight; try a small 45° arc to skirt
                x, y, th = self._emit_arc(x, y, th, 'L', 45, orders)

        self._log("[bug] gave up")
        return None

    # ---------- public API ----------
    def plan_visiting_orders_discrete(self, start_state: CarState, obstacle_indices: List[int]) -> List[Dict]:
        orders_out: List[Dict] = []
        scan_sec   = float(getattr(config.vision, "scan_seconds", 0.0))
        retreat_cm = float(getattr(config.car, "scan_retreat_cm", 0.0))

        cur = start_state
        self._log(f"[plan] Start: ({cur.x:.2f},{cur.y:.2f},{deg(cur.theta):.1f}°); "
                  f"turning_radius={config.car.turning_radius:.2f}, inflation={self._inflation():.2f}")

        for idx in obstacle_indices:
            if not (0 <= idx < len(self.obstacles)):
                self._log(f"[plan] skip invalid obstacle index {idx}")
                continue

            goal_raw = self.get_image_target_position(self.obstacles[idx])
            goal     = self._snap_goal_to_forward_ray(cur, goal_raw, tol_cm=10.0)
            self._log(f"[plan] Target #{idx}: raw=({goal_raw.x:.2f},{goal_raw.y:.2f},{deg(goal_raw.theta):.1f}°) "
                      f"-> used=({goal.x:.2f},{goal.y:.2f},{deg(goal.theta):.1f}°)")

            # Try axis-first plan; else fallback walker
            plan = self._axis_plan(cur, goal) or self._bug_fallback(cur, goal)
            if plan is None:
                self._log(f"[plan] FAILED reaching obstacle {idx}")
                return []

            # Apply plan to accumulate final pose
            x, y, th = cur.x, cur.y, cur.theta
            for cmd in plan:
                if cmd["op"] in ("L","R"):
                    x = cmd["pose"]["x"]; y = cmd["pose"]["y"]; th = cmd["pose"]["theta"]
                else:  # F
                    x = cmd["pose"]["x"]; y = cmd["pose"]["y"]; th = cmd["pose"]["theta"]
                orders_out.append(cmd)

            cur = CarState(x, y, th)

            # Scan
            s = {"op": "S", "obstacle_id": idx, "pose": {
                    "x": round(cur.x, 2), "y": round(cur.y, 2),
                    "theta": round(cur.theta, 4), "theta_deg": round(deg(cur.theta), 1)
                }}
            if scan_sec > 0:
                s["scan_seconds"] = round(scan_sec, 2)
            orders_out.append(s)
            self._log(f"[scan] at ({cur.x:.2f},{cur.y:.2f}) heading {deg(cur.theta):.1f}°")

            # Optional retreat straight back (B)
            if retreat_cm > 0:
                bx = cur.x - retreat_cm * math.cos(cur.theta)
                by = cur.y - retreat_cm * math.sin(cur.theta)
                if self._segment_clear(cur.x, cur.y, bx, by):
                    orders_out.append({
                        "op":"B","value_cm":int(round(retreat_cm)),
                        "pose":{"x":round(bx,2),"y":round(by,2),
                                "theta":round(cur.theta,4),"theta_deg":round(deg(cur.theta),1)}
                    })
                    self._log(f"[retreat] {retreat_cm:.1f}cm to ({bx:.2f},{by:.2f})")
                    cur = CarState(bx, by, cur.theta)
                else:
                    self._log("[retreat] skipped: blocked")

        self._log(f"[plan] DONE; emitted {len(orders_out)} orders")
        return orders_out
