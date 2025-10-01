import pygame
import math
import sys
import os, json
import time 
from typing import List, Tuple

from algorithm.lib.car import CarState, CarAction
from algorithm.lib.path import Obstacle, DubinsPath, DubinsPathType
from algorithm.lib.controller import CarMissionManager
from algorithm.config import config
from datetime import datetime


class Colors:
    BLACK = (0, 0, 0); WHITE = (255, 255, 255); RED = (255, 0, 0)
    GREEN = (0, 255, 0); BLUE = (0, 0, 255); YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0); PURPLE = (128, 0, 128); GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200); DARK_GREEN = (0, 100, 0); LIGHT_GREEN = (144, 238, 144)
    LIGHT_BLUE = (173, 216, 230); DARK_BLUE = (0, 0, 139); MAGENTA = (255, 0, 255); CYAN = (0, 255, 255)


class ImprovedCarVisualizer:
    """Pygame-based car pathfinding visualizer with true Dubins curves and an obstacle editor."""

    def __init__(self, width=1200, height=900):
        pygame.init()
        self.screen_width = width; self.screen_height = height
        self.arena_size = config.arena.size

        self.offset_x = 100
        self.offset_y = 75

        # Sidebar geometry
        self.sidebar_w = 280
        self.sidebar_margin = 28        # a bit larger, reliable gap
        safety = 0.985                  # shrink arena ~1.5% to avoid rounding overlap

        # Arena must fit with sidebar on the right
        usable_w = self.screen_width - (self.offset_x + self.sidebar_w + self.sidebar_margin)
        usable_h = self.screen_height - (self.offset_y + 75)  # bottom margin
        self.scale = safety * min(usable_w / self.arena_size, usable_h / self.arena_size)



        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Enhanced Robot Car Pathfinding - Dubins Paths")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        self.obstacles: List[Obstacle] = []
        self.planned_paths: List[DubinsPath] = []
        self.car_state: CarState | None = None
        self.target_positions: List[CarState] = []
        self.car_trail: List[CarState] = []
        self.current_target_highlight = -1
        self.image_recognition_status = False
        self.status_messages: List[str] = []

        self.run_times: list[float] = []  
        self.current_run_start: float | None = None
        self.current_run_elapsed: float = 0.0
        self._run_recorded: bool = False  

        self.running = True; self.paused = False
        self.show_grid = True; self.show_paths = True; self.show_waypoints = True
        self.show_turning_circles = False
        self.animation_speed = 1

        # --- editor (pre-start) ---
        self.editor_mode = True
        self.current_side = 'N'
        self.editor_preview_pos: Tuple[float, float] | None = None

    def arena_pixel_rect(self):
        w = int(self.arena_size * self.scale)
        h = int(self.arena_size * self.scale)
        x = self.offset_x
        y = self.offset_y
        return x, y, w, h


    # ---------- transforms ----------

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(self.offset_x + x * self.scale), int(self.offset_y + (self.arena_size - y) * self.scale)

    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        return (sx - self.offset_x) / self.scale, self.arena_size - (sy - self.offset_y) / self.scale
    
    # ---------- map save/load ----------

    def _maps_dir(self) -> str:
        # folder next to this script: ./maps
        base = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(base, "maps")
        os.makedirs(d, exist_ok=True)
        return d

    def save_current_map(self) -> str:
        """
        Save current obstacles to a timestamped JSON file.
        Returns the filepath saved.
        """
        data = {
            "meta": {
                "arena_size": self.arena_size,
                "obstacle_size": getattr(config.arena, "obstacle_size", 10),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            },
            "obstacles": [
                {"x": float(o.x), "y": float(o.y), "side": str(o.image_side)}
                for o in self.obstacles
            ]
        }
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self._maps_dir(), f"map-{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.add_status_message(f"Saved map: {os.path.basename(path)}")
        print(f"[MAP] Saved: {path}")
        return path

    def load_latest_map(self) -> bool:
        """
        Load the most recent map JSON from ./maps.
        Returns True if loaded, False otherwise.
        """
        d = self._maps_dir()
        files = sorted(
            (f for f in os.listdir(d) if f.lower().endswith(".json")),
            reverse=True
        )
        if not files:
            self.add_status_message("No saved maps found.")
            return False

        path = os.path.join(d, files[0])
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.add_status_message(f"Failed to load: {files[0]}")
            print(f"[MAP] Load error: {e}")
            return False

        obs = []
        obs_size = getattr(config.arena, "obstacle_size", 10)
        cell = config.arena.grid_cell_size

        for it in data.get("obstacles", []):
            x = float(it["x"]); y = float(it["y"]); side = str(it["side"])
            # clamp to arena
            x = max(0.0, min(self.arena_size - obs_size, x))
            y = max(0.0, min(self.arena_size - obs_size, y))
            # snap to grid & cast to int
            x = int(round(x / cell) * cell)
            y = int(round(y / cell) * cell)
            obs.append(Obstacle(x, y, side))

        self.obstacles = obs
        self.add_status_message(f"Loaded map: {os.path.basename(path)}")
        print(f"[MAP] Loaded: {path}")
        return True


    # ---------- drawing ----------

    def draw_grid(self):
        if not self.show_grid:
            return
        grid = 10
        for i in range(0, self.arena_size + 1, grid):
            sx1, sy1 = self.world_to_screen(i, 0); sx2, sy2 = self.world_to_screen(i, self.arena_size)
            pygame.draw.line(self.screen, Colors.LIGHT_GRAY, (sx1, sy1), (sx2, sy2))
            sx1, sy1 = self.world_to_screen(0, i); sx2, sy2 = self.world_to_screen(self.arena_size, i)
            pygame.draw.line(self.screen, Colors.LIGHT_GRAY, (sx1, sy1), (sx2, sy2))

    def draw_arena(self):
        tl = self.world_to_screen(0, self.arena_size)
        w = int(self.arena_size * self.scale); h = int(self.arena_size * self.scale)
        pygame.draw.rect(self.screen, Colors.BLACK, (tl[0], tl[1], w, h), 3)

        stl = self.world_to_screen(0, 40)
        sw = int(40 * self.scale); sh = int(40 * self.scale)
        pygame.draw.rect(self.screen, Colors.LIGHT_GREEN, (stl[0], stl[1], sw, sh))
        pygame.draw.rect(self.screen, Colors.DARK_GREEN, (stl[0], stl[1], sw, sh), 2)
        text = self.font.render("START", True, Colors.DARK_GREEN)
        text_pos = self.world_to_screen(20, 20)
        self.screen.blit(text, text.get_rect(center=text_pos))

    def draw_obstacle(self, obstacle: Obstacle, obs_id: int):
        size = config.arena.obstacle_size
        top_left = self.world_to_screen(obstacle.x, obstacle.y + size)
        w = int(size * self.scale); h = int(size * self.scale)

        if obs_id == self.current_target_highlight:
            pygame.draw.rect(self.screen, Colors.MAGENTA, (top_left[0] - 3, top_left[1] - 3, w + 6, h + 6))

        pygame.draw.rect(self.screen, Colors.RED, (top_left[0], top_left[1], w, h))
        pygame.draw.rect(self.screen, Colors.BLACK, (top_left[0], top_left[1], w, h), 2)

        b = max(4, int(3 * self.scale))
        if obstacle.image_side == 'S':
            pos = self.world_to_screen(obstacle.x, obstacle.y - 1)
            pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0], pos[1], w, b))
        elif obstacle.image_side == 'N':
            pos = self.world_to_screen(obstacle.x, obstacle.y + size + 1)
            pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0], pos[1], w, b))
        elif obstacle.image_side == 'E':
            pos = self.world_to_screen(obstacle.x + size + 1, obstacle.y + size)
            pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0], pos[1], b, h))
        else:
            pos = self.world_to_screen(obstacle.x - 1, obstacle.y + size)
            pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0] - b, pos[1], b, h))

        center = self.world_to_screen(obstacle.x + size / 2, obstacle.y + size / 2)
        t = self.font.render(str(obs_id), True, Colors.WHITE)
        self.screen.blit(t, t.get_rect(center=center))

    def draw_car(self, s: CarState, color=Colors.BLUE):
        cw = config.car.width * self.scale; cl = config.car.length * self.scale
        corners = [(-cl/2, -cw/2), (cl/2, -cw/2), (cl/2, cw/2), (-cl/2, cw/2)]
        c, si = math.cos(s.theta), math.sin(s.theta)
        sc = []
        for x, y in corners:
            rx, ry = x * c - y * si, x * si + y * c
            wx, wy = s.x + rx / self.scale, s.y + ry / self.scale
            sc.append(self.world_to_screen(wx, wy))

        body_color = Colors.CYAN if self.image_recognition_status and (pygame.time.get_ticks() // 200) % 2 else color
        pygame.draw.polygon(self.screen, body_color, sc)
        pygame.draw.polygon(self.screen, Colors.BLACK, sc, 2)

        L = 20
        sx, sy = self.world_to_screen(s.x, s.y)
        ex, ey = self.world_to_screen(s.x + L * c, s.y + L * si)
        pygame.draw.line(self.screen, Colors.RED, (sx, sy), (ex, ey), 4)
        ah = 8; aa = 0.4
        lx, ly = ex - ah * math.cos(s.theta - aa), ey - ah * math.sin(s.theta - aa)
        rx, ry = ex - ah * math.cos(s.theta + aa), ey - ah * math.sin(s.theta + aa)
        pygame.draw.polygon(self.screen, Colors.RED, [(ex, ey), (lx, ly), (rx, ry)])

    # --------- true Dubins curve drawing ----------

    def _circle_center(self, point: Tuple[float, float], heading: float, turn: str) -> Tuple[float, float]:
        r = config.car.turning_radius
        x, y = point
        if turn.upper() == 'L':
            return (x - r * math.sin(heading), y + r * math.cos(heading))
        else:
            return (x + r * math.sin(heading), y - r * math.cos(heading))

    def _draw_arc(self, cx: float, cy: float, start: Tuple[float, float], end: Tuple[float, float],
                  turn: str, width: int, color):
        """Draw an arc between start and end around center (cx,cy)."""
        r = config.car.turning_radius
        a0 = math.atan2(start[1] - cy, start[0] - cx)
        a1 = math.atan2(end[1] - cy, end[0] - cx)

        def norm(a):
            while a < 0: a += 2 * math.pi
            while a >= 2 * math.pi: a -= 2 * math.pi
            return a

        a0 = norm(a0); a1 = norm(a1)
        points = []
        steps = max(8, int(abs(a1 - a0) * r / 2))  # smoothness based on arc length

        if turn.upper() == 'L':
            if a1 <= a0: a1 += 2 * math.pi
            angles = [a0 + (a1 - a0) * i / steps for i in range(steps + 1)]
        else:
            if a0 <= a1: a0 += 2 * math.pi
            angles = [a0 - (a0 - a1) * i / steps for i in range(steps + 1)]

        for ang in angles:
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            points.append(self.world_to_screen(x, y))
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, width)

    def draw_curved_dubins_path(self, path: DubinsPath, color=Colors.BLUE, width=3):
        if not self.show_paths or not path or not path.waypoints:
            return

        wps = path.waypoints

        # --- 1) Simple polyline (e.g., retreat line with 2 points) ---
        if len(wps) < 4:
            pts = [self.world_to_screen(x, y) for (x, y) in wps]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, width)

            if self.show_waypoints:
                for i, (x, y) in enumerate(wps):
                    pos = self.world_to_screen(x, y)
                    # endpoints a bit bigger
                    if i in (0, len(wps) - 1):
                        pygame.draw.circle(self.screen, color, pos, 5)
                        pygame.draw.circle(self.screen, Colors.BLACK, pos, 5, 2)
                    else:
                        pygame.draw.circle(self.screen, color, pos, 4)

            # label
            mid = wps[len(wps)//2]
            label_txt = getattr(path.path_type, "value", str(path.path_type)).upper()
            self.screen.blit(self.small_font.render(label_txt, True, color),
                            (self.world_to_screen(mid[0] + 8, mid[1] + 8)))
            return

        # --- 2) Classic Dubins with 4 waypoints (L/R, S, L/R) ---
        ptype = (path.path_type.value if isinstance(path.path_type, DubinsPathType)
                else str(path.path_type)).lower()
        if len(ptype) < 3:
            # Safety: fall back to polyline
            pts = [self.world_to_screen(x, y) for (x, y) in wps]
            pygame.draw.lines(self.screen, color, False, pts, width)
            return

        # segment 0: turn or straight
        if ptype[0] in ('l', 'r'):
            start, t1 = wps[0], wps[1]
            cx, cy = self._circle_center(start, path.start_state.theta, ptype[0])
            self._draw_arc(cx, cy, start, t1, ptype[0], width, color)
        else:
            pygame.draw.line(self.screen, color, self.world_to_screen(*wps[0]),
                            self.world_to_screen(*wps[1]), width)

        # segment 1: straight
        pygame.draw.line(self.screen, color, self.world_to_screen(*wps[1]),
                        self.world_to_screen(*wps[2]), width)

        # segment 2: turn or straight
        if ptype[2] in ('l', 'r'):
            t2, end = wps[2], wps[3]
            cx, cy = self._circle_center(end, path.end_state.theta, ptype[2])
            self._draw_arc(cx, cy, t2, end, ptype[2], width, color)
        else:
            pygame.draw.line(self.screen, color, self.world_to_screen(*wps[2]),
                            self.world_to_screen(*wps[3]), width)

        # waypoint dots
        if self.show_waypoints:
            for i, (x, y) in enumerate(wps):
                pos = self.world_to_screen(x, y)
                if i == 0:
                    pygame.draw.circle(self.screen, Colors.GREEN, pos, 6)
                    pygame.draw.circle(self.screen, Colors.BLACK, pos, 6, 2)
                elif i == len(wps) - 1:
                    pygame.draw.circle(self.screen, Colors.RED, pos, 6)
                    pygame.draw.circle(self.screen, Colors.BLACK, pos, 6, 2)
                else:
                    pygame.draw.circle(self.screen, color, pos, 4)

        # label
        mid = wps[len(wps)//2]
        label_pos = self.world_to_screen(mid[0] + 8, mid[1] + 8)
        label = (path.path_type.value if isinstance(path.path_type, DubinsPathType)
                else str(path.path_type)).upper()
        self.screen.blit(self.small_font.render(label, True, color), label_pos)

    # ---------- targets, trail, UI ----------

    def draw_target_position(self, tstate: CarState, tid: int, is_current=False):
        pos = self.world_to_screen(tstate.x, tstate.y)
        size = 12 if is_current else 8
        color = Colors.CYAN if is_current else Colors.PURPLE
        pygame.draw.rect(self.screen, color, (pos[0] - size, pos[1] - size, size * 2, size * 2))
        pygame.draw.rect(self.screen, Colors.BLACK, (pos[0] - size, pos[1] - size, size * 2, size * 2), 2)

        L = 20
        end = self.world_to_screen(tstate.x + L * math.cos(tstate.theta),
                                   tstate.y + L * math.sin(tstate.theta))
        pygame.draw.line(self.screen, color, pos, end, 4)
        ah = 6; aa = 0.4
        lx = end[0] - ah * math.cos(tstate.theta - aa); ly = end[1] - ah * math.sin(tstate.theta - aa)
        rx = end[0] - ah * math.cos(tstate.theta + aa); ry = end[1] - ah * math.sin(tstate.theta + aa)
        pygame.draw.polygon(self.screen, color, [end, (lx, ly), (rx, ry)])

        text = self.small_font.render(f'T{tid}', True, color)
        self.screen.blit(text, (pos[0] + 15, pos[1] - 15))

    def draw_car_trail(self):
        if len(self.car_trail) < 2: return
        pts = [self.world_to_screen(s.x, s.y) for s in self.car_trail]
        for i in range(len(pts) - 1):
            surf = pygame.Surface((self.screen_width, self.screen_height))
            alpha = max(50, int(255 * (i + 1) / len(pts)))
            surf.set_alpha(alpha); surf.fill(Colors.WHITE); surf.set_colorkey(Colors.WHITE)
            pygame.draw.line(surf, Colors.ORANGE, pts[i], pts[i + 1], 2)
            self.screen.blit(surf, (0, 0))

    def draw_ui(self):
        title = self.font.render("Enhanced Robot Car Pathfinding - Dubins Paths", True, Colors.BLACK)
        self.screen.blit(title, (10, 10))

        ctrls = ["SPACE - Play/Pause", "G - Toggle Grid", "P - Toggle Paths",
                 "W - Toggle Waypoints", "T - Toggle Turning Circles", "+/- - Speed", "R - Reset", "ESC - Exit"]
        y = 35
        for c in ctrls:
            self.screen.blit(self.small_font.render(c, True, Colors.BLACK), (10, y)); y += 18

        panel_x = self.screen_width - self.sidebar_w - 10   # 10px outer margin
        panel_y = 10
        pw = self.sidebar_w
        ph = 250
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, (panel_x, panel_y, pw, ph))
        pygame.draw.rect(self.screen, Colors.BLACK, (panel_x, panel_y, pw, ph), 2)


        if self.car_state:
            lines = ["CAR STATUS:",
                     f"Position: ({self.car_state.x:.1f}, {self.car_state.y:.1f})",
                     f"Heading: {math.degrees(self.car_state.theta):.1f}°",
                     f"Speed: {self.animation_speed}x"]
            if self.image_recognition_status: lines.append(">>> SCANNING IMAGE <<<")
            for i, ln in enumerate(lines):
                color = Colors.CYAN if "SCANNING" in ln else Colors.BLACK
                self.screen.blit(self.small_font.render(ln, True, color), (panel_x + 10, panel_y + 15 + i * 18))

        stats_y = panel_y + 10 + 5 * 18  # below CAR STATUS lines
        if self.current_run_start is not None and not self._run_recorded:
            live = f"Timer (live): {self.current_run_elapsed:0.2f}s"
            self.screen.blit(self.small_font.render(live, True, Colors.DARK_BLUE), (panel_x + 10, stats_y))
            stats_y += 18

        if self.run_times:
            last5 = self.run_times[-5:]
            best = min(self.run_times)
            avg = sum(self.run_times) / len(self.run_times)
            self.screen.blit(self.small_font.render(f"Runs: {len(self.run_times)}  Best: {best:0.2f}s  Avg: {avg:0.2f}s", True, Colors.BLACK), (panel_x + 10, stats_y))
            stats_y += 18
            self.screen.blit(self.small_font.render("Last 5:", True, Colors.BLACK), (panel_x + 10, stats_y))
            stats_y += 18
            self.screen.blit(self.small_font.render(", ".join(f"{t:0.2f}s" for t in last5), True, Colors.BLACK), (panel_x + 10, stats_y))


        msg_y = panel_y + 160
        for i, msg in enumerate(self.status_messages[-4:]):
            self.screen.blit(self.small_font.render(msg, True, Colors.DARK_BLUE), (panel_x + 10, msg_y + i * 18))

        if self.paused and not self.editor_mode:
            self.screen.blit(self.font.render("PAUSED", True, Colors.RED),
                             (self.screen_width - 100, self.screen_height - 30))

    def add_status_message(self, m: str):
        self.status_messages.append(m)
        if len(self.status_messages) > 10: self.status_messages.pop(0)

    # ---------- editor ----------

    def draw_editor_overlay(self):
        panel_x = 10; panel_y = self.screen_height - 120; pw = 720; ph = 110
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, (panel_x, panel_y, pw, ph))
        pygame.draw.rect(self.screen, Colors.BLACK, (panel_x, panel_y, pw, ph), 2)
        lines = [
            "EDITOR MODE: Left-click to place obstacle (snapped to grid). Right-click removes last.",
            "Keys: W/A/S/D set image side • BACKSPACE undo • C clear • M save • L load-latest • ENTER start",
            f"Current side: {self.current_side}"
        ]

        for i, ln in enumerate(lines):
            self.screen.blit(self.small_font.render(ln, True, Colors.BLACK), (panel_x + 10, panel_y + 10 + i*22))

        if self.editor_preview_pos is not None:
            size = config.arena.obstacle_size
            px, py = self.editor_preview_pos
            # keep within arena visual
            px = max(0, min(self.arena_size - size, px))
            py = max(0, min(self.arena_size - size, py))

            top_left = self.world_to_screen(px, py + size)
            w = int(size * self.scale); h = int(size * self.scale)
            surf = pygame.Surface((w, h)); surf.set_alpha(100); surf.fill(Colors.RED)
            self.screen.blit(surf, (top_left[0], top_left[1]))
            pygame.draw.rect(self.screen, Colors.BLACK, (top_left[0], top_left[1], w, h), 2)

            b = max(4, int(3 * self.scale))
            if self.current_side == 'S':
                pos = self.world_to_screen(px, py - 1)
                pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0], pos[1], w, b))
            elif self.current_side == 'N':
                pos = self.world_to_screen(px, py + size + 1)
                pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0], pos[1], w, b))
            elif self.current_side == 'E':
                pos = self.world_to_screen(px + size + 1, py + size)
                pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0], pos[1], b, h))
            else:
                pos = self.world_to_screen(px - 1, py + size)
                pygame.draw.rect(self.screen, Colors.YELLOW, (pos[0] - b, pos[1], b, h))

    def handle_events_editor(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False

                elif e.key == pygame.K_RETURN:
                    # commit and plan mission
                    if self.obstacles:
                        self.setup_simulation_from_editor()
                        self.editor_mode = False
                        self.add_status_message("Mission planned successfully!")
                    else:
                        self.add_status_message("Place at least one obstacle before starting.")

                # WASD -> N/S/W/E
                elif e.key in (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d):
                    self.current_side = {
                        pygame.K_w: 'N',  # W = North
                        pygame.K_s: 'S',  # S = South
                        pygame.K_a: 'W',  # A = West
                        pygame.K_d: 'E',  # D = East
                    }[e.key]

                elif e.key == pygame.K_BACKSPACE:
                    if self.obstacles:
                        self.obstacles.pop()

                elif e.key == pygame.K_c:
                    self.obstacles.clear()

                # --- Save / Load maps (must be inside KEYDOWN) ---
                elif e.key == pygame.K_m:
                    self.save_current_map()
                elif e.key == pygame.K_l:
                    self.load_latest_map()

            elif e.type == pygame.MOUSEMOTION:
                wx, wy = self.screen_to_world(*e.pos)
                gx = round(wx / 10) * 10
                gy = round(wy / 10) * 10
                self.editor_preview_pos = (gx, gy)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:  # left click: place obstacle
                    wx, wy = self.screen_to_world(*e.pos)
                    x = round(wx / 10) * 10
                    y = round(wy / 10) * 10
                    size = config.arena.obstacle_size
                    x = max(0, min(self.arena_size - size, x))
                    y = max(0, min(self.arena_size - size, y))
                    self.obstacles.append(Obstacle(x, y, self.current_side))
                elif e.button == 3:  # right click: remove last
                    if self.obstacles:
                        self.obstacles.pop()



    # ---------- input / loop ----------

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: self.running = False
                elif e.key == pygame.K_SPACE: self.paused = not self.paused
                elif e.key == pygame.K_g: self.show_grid = not self.show_grid
                elif e.key == pygame.K_p: self.show_paths = not self.show_paths
                elif e.key == pygame.K_w: self.show_waypoints = not self.show_waypoints
                elif e.key == pygame.K_t: self.show_turning_circles = not self.show_turning_circles
                elif e.key == pygame.K_r: self.reset()
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS): self.animation_speed = min(5, self.animation_speed + 1)
                elif e.key == pygame.K_MINUS: self.animation_speed = max(1, self.animation_speed - 1)

    def reset(self):
        self.car_trail = []; self.status_messages = []
        if self.car_state: self.car_trail.append(CarState(self.car_state.x, self.car_state.y, self.car_state.theta))

    def update(self): pass

    def render(self):
        self.screen.fill(Colors.WHITE)
        self.draw_grid(); self.draw_arena()
        for i, obs in enumerate(self.obstacles): self.draw_obstacle(obs, i)
        for i, ts in enumerate(self.target_positions): self.draw_target_position(ts, i, i == self.current_target_highlight)
        colors = [Colors.BLUE, Colors.DARK_GREEN, Colors.RED, Colors.PURPLE, Colors.ORANGE]
        for i, path in enumerate(self.planned_paths): self.draw_curved_dubins_path(path, colors[i % len(colors)])
        self.draw_car_trail()
        if self.car_state: self.draw_car(self.car_state)
        self.draw_ui()
        pygame.display.flip()

    def run(self):
        # editor phase
        while self.running and self.editor_mode:
            self.handle_events_editor()
            self.screen.fill(Colors.WHITE)
            self.draw_grid()
            self.draw_arena()
            for i, obs in enumerate(self.obstacles):
                self.draw_obstacle(obs, i)
            self.draw_editor_overlay()
            pygame.display.flip()
            self.clock.tick(60)

        # simulation phase
        while self.running:
            self.handle_events()
            if not self.paused:
                for _ in range(self.animation_speed): self.update()
            self.render(); self.clock.tick(60)
        pygame.quit(); sys.exit()


class EnhancedCarSimulation(ImprovedCarVisualizer):
    """Enhanced car simulation with proper Dubins path following + pre-start obstacle editor."""

    def __init__(self):
        super().__init__()
        # Start in editor; mission will be planned after ENTER
        self.simulation_step = 0
        self.max_steps = 10000
        self.step_delay = 0

    def setup_simulation_from_editor(self):
        # Build manager from obstacles placed in editor
        self.manager = CarMissionManager()
        self.manager.initialize_car(20, 20, 0)

        for o in self.obstacles:
            self.manager.add_obstacle(o.x, o.y, o.image_side)

        target_indices = list(range(len(self.obstacles)))
        if self.manager.plan_mission(target_indices):
            self.planned_paths = self.manager.controller.current_path
            self.target_positions = [
                self.manager.path_planner.get_image_target_position(o)
                for o in self.manager.path_planner.obstacles
            ]
            self.current_run_start = time.perf_counter()
            self.current_run_elapsed = 0.0
            self._run_recorded = False
            self.add_status_message("Mission planned successfully!")
        else:
            self.add_status_message("Mission planning failed!")
            self.current_run_start = None
            self._run_recorded = True  # don't record anything

        self.car_state = self.manager.car_status.estimated_state
        self.car_trail = [CarState(self.car_state.x, self.car_state.y, self.car_state.theta)]

    def update(self):
        # --- TIMER: live update while run is active ---
        if self.current_run_start is not None and not self._run_recorded:
            self.current_run_elapsed = time.perf_counter() - self.current_run_start

        if self.simulation_step >= self.max_steps:
            return

        cmd = self.manager.get_next_action()
        if not cmd:
            self.add_status_message("No more commands")
            return

        # --- TIMER: capture completion exactly once when STOP appears ---
        if cmd.action == CarAction.STOP and not self._run_recorded:
            elapsed = time.perf_counter() - self.current_run_start if self.current_run_start else 0.0
            self.run_times.append(elapsed)
            self._run_recorded = True
            self.add_status_message(f"Mission complete! Run time: {elapsed:.2f}s")
        elif cmd.action == CarAction.STOP:
            # already recorded; keep old message style if you want
            self.add_status_message("Mission complete!")

        result = self.manager.execute_command(cmd)
        self.car_state = self.manager.car_status.estimated_state
        self.car_trail.append(CarState(self.car_state.x, self.car_state.y, self.car_state.theta))

        progress = result.get('progress', {})
        self.image_recognition_status = progress.get('at_target', False)
        if progress.get('current_segment', -1) < len(self.target_positions):
            self.current_target_highlight = progress.get('current_segment', -1)

        if cmd.action == CarAction.FORWARD:
            d = cmd.parameters.get('distance', 0)
            if d > 0:
                self.add_status_message(f"Forward {d:.1f}cm")
        elif cmd.action in (CarAction.TURN_LEFT, CarAction.TURN_RIGHT):
            ang = cmd.parameters.get('angle', 0)
            self.add_status_message(
                f"Turn {'left' if cmd.action==CarAction.TURN_LEFT else 'right'} {math.degrees(ang):.1f}°"
            )

        if progress.get('image_recognition_time', 0) > 0:
            self.add_status_message(f"Scanning... ({progress['image_recognition_time']} frames)")

        visited = len(self.manager.visited_targets)
        if hasattr(self, '_last_visited_count') and visited > self._last_visited_count:
            self.add_status_message(f"Target {visited - 1} completed!")
        self._last_visited_count = visited
        self.simulation_step += 1



if __name__ == "__main__":
    print("Enhanced Car Pathfinding Visualization")
    print("=" * 45)
    print("Features:")
    print("- True Dubins curves (rendered)")
    print("- Smooth controller (curvature & P-control)")
    print("- Image recognition stops")
    print("- Real-time status")
    print("- Pre-start obstacle editor (click to place, N/S/E/W to set side, Enter to start)")
    simulation = EnhancedCarSimulation()
    simulation.run()
