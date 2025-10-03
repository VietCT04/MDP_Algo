# viewer_arc_orders.py
import pygame, math, sys, os, json, time
from typing import List, Tuple, Optional

from algorithm.lib.car import CarState
from algorithm.lib.path import Obstacle
from algorithm.lib.pathfinding import CarPathPlanner   # <-- arc-orders planner
from algorithm.config import config


# ---------- colors ----------
class Colors:
    BLACK=(0,0,0); WHITE=(255,255,255); RED=(255,0,0); GREEN=(0,255,0); BLUE=(0,0,255)
    YELLOW=(255,255,0); ORANGE=(255,165,0); PURPLE=(128,0,128); GRAY=(140,140,140)
    LIGHT_GRAY=(210,210,210); DARK=(30,30,30); CYAN=(0,255,255); MAGENTA=(255,0,255)


deg = math.degrees
rad = math.radians


# ---------- visualizer ----------
class ArcOrdersVisualizer:
    def __init__(self, width=1200, height=900):
        pygame.init()
        self.W, self.H = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Arc-Orders Planner • Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 18)

        # world / scale
        self.field_cm = float(config.arena.size)  # 200 or 2000 (cm)
        self.offset_x = 80
        self.offset_y = 60
        # leave room for side panel
        side_panel_w = 260
        usable_w = self.W - (self.offset_x + side_panel_w + 20)
        usable_h = self.H - (self.offset_y + 50)
        self.scale = min(usable_w / self.field_cm, usable_h / self.field_cm)
        self.side_panel = pygame.Rect(self.W - side_panel_w - 10, 10, side_panel_w, 270)

        # data
        self.obstacles: List[Obstacle] = []
        self.orders: List[dict] = []
        self.order_idx = 0
        self.car: Optional[CarState] = None
        self.trail: List[CarState] = []
        self.status: List[str] = []

        # editor
        self.editor_mode = True
        self.preview: Optional[Tuple[float,float]] = None
        self.current_side = 'N'
        self.start_pose = CarState(20.0, 20.0, math.pi/2)   # editor shows start pose; ENTER plans from here

        # anim
        self.running = True
        self.paused = False
        self.anim_speed = 1       # 1..5
        self.step_cm = 4.0        # forward step per frame (before speed multiplier)
        self.arc_step_deg = 3.0   # degrees per frame for arcs (before speed multiplier)
        self.scan_pause_frames = 40  # how long to "flash" at scan
        self.scan_left = 0

        # drawing toggles
        self.show_grid = True
        self.show_path_preview = True
        self.debug = True  # prints planner logs if your planner does it too

    # ----------- utils -----------
    def log(self, msg: str):
        if self.debug:
            print(msg)

    def add_msg(self, m: str):
        self.status.append(m)
        if len(self.status) > 12:
            self.status.pop(0)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        # y-up world -> y-down screen
        return (int(self.offset_x + x * self.scale),
                int(self.offset_y + (self.field_cm - y) * self.scale))

    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        return ((sx - self.offset_x) / self.scale,
                self.field_cm - (sy - self.offset_y) / self.scale)

    # ----------- editor maps -----------
    def _maps_dir(self):
        base = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(base, "maps"); os.makedirs(d, exist_ok=True); return d

    def save_map(self):
        data = {
            "meta":{"arena_size":self.field_cm,"obstacle_size":config.arena.obstacle_size},
            "obstacles":[{"x":o.x,"y":o.y,"side":o.image_side} for o in self.obstacles]
        }
        path = os.path.join(self._maps_dir(), f"map-{int(time.time())}.json")
        with open(path,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)
        self.add_msg(f"Saved {os.path.basename(path)}")

    def load_latest(self):
        files = [f for f in os.listdir(self._maps_dir()) if f.endswith(".json")]
        if not files:
            self.add_msg("No saved maps")
            return
        files.sort(reverse=True)
        path = os.path.join(self._maps_dir(), files[0])
        try:
            data = json.load(open(path,"r",encoding="utf-8"))
        except Exception as e:
            self.add_msg("Load failed"); self.log(f"[MAP] load error: {e}"); return
        self.obstacles.clear()
        cell = config.arena.grid_cell_size
        size = config.arena.obstacle_size
        for it in data.get("obstacles",[]):
            x = max(0, min(self.field_cm - size, round(float(it["x"])/cell)*cell))
            y = max(0, min(self.field_cm - size, round(float(it["y"])/cell)*cell))
            self.obstacles.append(Obstacle(x,y,str(it["side"])))
        self.add_msg(f"Loaded {files[0]}")

    #--- replace draw_dynamic_grid ---
    def draw_dynamic_grid(self):  # <- replace the whole function
        if not self.show_grid: 
            return
        # arena border
        x0, y0 = self.world_to_screen(0, self.field_cm)
        w = int(self.field_cm * self.scale); h = int(self.field_cm * self.scale)
        pygame.draw.rect(self.screen, Colors.DARK, (x0, y0, w, h), 3)

        minor = 10   # cm
        major = 100  # cm

        # verticals
        x_cm = 0
        while x_cm <= self.field_cm:
            sx, _ = self.world_to_screen(x_cm, 0)
            col = Colors.LIGHT_GRAY if (x_cm % major) else Colors.BLACK
            width = 1 if (x_cm % major) else 2
            pygame.draw.line(self.screen, col, (sx, y0), (sx, y0 + h), width)
            x_cm += minor

        # horizontals
        y_cm = 0
        while y_cm <= self.field_cm:
            _, sy = self.world_to_screen(0, y_cm)
            col = Colors.LIGHT_GRAY if (y_cm % major) else Colors.BLACK
            width = 1 if (y_cm % major) else 2
            pygame.draw.line(self.screen, col, (x0, sy), (x0 + w, sy), width)
            y_cm += minor

        # 40×40 start zone
        st = self.world_to_screen(0, 40)
        sw, sh = int(40 * self.scale), int(40 * self.scale)
        pygame.draw.rect(self.screen, (180, 255, 180), (st[0], st[1], sw, sh))
        pygame.draw.rect(self.screen, (40, 120, 40), (st[0], st[1], sw, sh), 2)
        self.screen.blit(
            self.small.render("START", True, (30, 90, 30)),
            self.small.render("START", True, (30, 90, 30)).get_rect(center=self.world_to_screen(20, 20)),
        )



    # --- replace draw_obstacles ---
    def draw_obstacles(self):
        size_cm = config.arena.obstacle_size  # 10 cm
        MIN_PX = 10                           # never draw smaller than 10 px

        band_px = max(3, int(2 * self.scale))
        for i, o in enumerate(self.obstacles):
            # draw centered with min pixel size
            cx, cy = self.world_to_screen(o.x + size_cm/2, o.y + size_cm/2)
            w = h = max(int(size_cm * self.scale), MIN_PX)
            tl = (cx - w // 2, cy - h // 2)

            pygame.draw.rect(self.screen, Colors.RED, (tl[0], tl[1], w, h))
            pygame.draw.rect(self.screen, Colors.BLACK, (tl[0], tl[1], w, h), 2)

            # image-side band (screen-aligned so it stays visible)
            if o.image_side == 'S':
                pygame.draw.rect(self.screen, Colors.YELLOW, (tl[0], tl[1] + h - band_px, w, band_px))
            elif o.image_side == 'N':
                pygame.draw.rect(self.screen, Colors.YELLOW, (tl[0], tl[1], w, band_px))
            elif o.image_side == 'E':
                pygame.draw.rect(self.screen, Colors.YELLOW, (tl[0] + w - band_px, tl[1], band_px, h))
            else:  # 'W'
                pygame.draw.rect(self.screen, Colors.YELLOW, (tl[0], tl[1], band_px, h))

            # id label
            self.screen.blit(
                self.small.render(str(i), True, Colors.WHITE),
                self.small.render(str(i), True, Colors.WHITE).get_rect(center=(cx, cy)),
            )



    def draw_car(self, s: CarState, scanning=False):
        MIN_W, MIN_L = 10, 16  # px
        cw = max(int(config.car.width * self.scale), MIN_W)
        cl = max(int(config.car.length * self.scale), MIN_L)

        corners = [(-cl/2, -cw/2), (cl/2, -cw/2), (cl/2, cw/2), (-cl/2, cw/2)]
        c, si = math.cos(s.theta), math.sin(s.theta)
        poly = []
        for x, y in corners:
            rx, ry = x * c - y * si, x * si + y * c
            wx, wy = s.x + rx / self.scale, s.y + ry / self.scale
            poly.append(self.world_to_screen(wx, wy))

        body = Colors.CYAN if scanning and (pygame.time.get_ticks() // 200) % 2 else (0, 150, 255)
        pygame.draw.polygon(self.screen, body, poly)
        pygame.draw.polygon(self.screen, Colors.BLACK, poly, 2)

        # heading arrow
        L = 20
        sx, sy = self.world_to_screen(s.x, s.y)
        ex, ey = self.world_to_screen(s.x + L * c, s.y + L * si)
        pygame.draw.line(self.screen, Colors.ORANGE, (sx, sy), (ex, ey), 3)


    def draw_orders_preview(self):
        if not self.show_path_preview or not self.orders: return
        prev = None
        pose = CarState(self.start_pose.x, self.start_pose.y, self.start_pose.theta)
        R_default = float(config.car.turning_radius)
        for cmd in self.orders:
            if cmd["op"] in ("F","B"):
                ex, ey = cmd["pose"]["x"], cmd["pose"]["y"]
                pygame.draw.line(self.screen, (0,120,255), self.world_to_screen(pose.x, pose.y),
                                 self.world_to_screen(ex, ey), 3)
                pose = CarState(ex, ey, pose.theta)
            elif cmd["op"] in ("L","R"):
                turn = cmd["op"]; ang = float(cmd["angle_deg"])
                R = float(cmd.get("radius_cm", R_default))
                cx, cy = self.arc_center(pose.x, pose.y, pose.theta, turn, R)
                nx, ny, nth = self.arc_end(pose.x, pose.y, pose.theta, turn, ang, R)
                # sample arc
                steps = max(6, int( (R*rad(ang))/ 3.0 ))
                pts=[]
                for i in range(steps+1):
                    a = ang * (i/steps)
                    tx, ty, _ = self.arc_end(pose.x, pose.y, pose.theta, turn, a, R)
                    pts.append(self.world_to_screen(tx, ty))
                pygame.draw.lines(self.screen, (0,120,255), False, pts, 3)
                pose = CarState(nx, ny, nth)

    def draw_panel(self):
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, self.side_panel)
        pygame.draw.rect(self.screen, Colors.BLACK, self.side_panel, 2)
        x,y=self.side_panel.x+10,self.side_panel.y+10
        lines = [
            "Arc-Orders Visualizer",
            f"Arena: {int(self.field_cm)} cm",
            f"Speed: {self.anim_speed}x",
            f"Orders: {len(self.orders)}",
            f"Mode: {'EDITOR' if self.editor_mode else 'SIM'}",
            "Keys:",
            "  ENTER plan/run   M save map",
            "  L load latest    G grid",
            "  +/- speed        SPACE pause",
            "  W/A/S/D set side",
            "  R reset editor",
        ]
        for s in lines:
            self.screen.blit(self.small.render(s, True, Colors.BLACK), (x,y)); y+=18
        # status tail
        y+=6
        for s in self.status[-6:]:
            self.screen.blit(self.small.render(s, True, (20,20,120)), (x,y)); y+=18

    # ----------- geometry -----------
    def arc_center(self, x: float, y: float, th: float, turn: str, R: float) -> Tuple[float,float]:
        if turn=='L':
            return (x - R*math.sin(th), y + R*math.cos(th))
        else:
            return (x + R*math.sin(th), y - R*math.cos(th))

    def arc_end(self, x: float, y: float, th: float, turn: str, ang_deg: float, R: float) -> Tuple[float,float,float]:
        dpsi = rad(ang_deg) * (1 if turn=='L' else -1)
        if turn=='L':
            cx, cy = x - R*math.sin(th), y + R*math.cos(th)
        else:
            cx, cy = x + R*math.sin(th), y - R*math.cos(th)
        a0 = math.atan2(y - cy, x - cx)
        a1 = a0 + dpsi
        nx = cx + R*math.cos(a1)
        ny = cy + R*math.sin(a1)
        nth = (th + dpsi) % (2*math.pi)
        return nx, ny, nth

    # ----------- planning -----------
    def plan_from_editor(self):
        self.orders.clear()
        planner = CarPathPlanner()
        for o in self.obstacles:
            planner.add_obstacle(o.x, o.y, o.image_side)
        start = CarState(self.start_pose.x, self.start_pose.y, self.start_pose.theta)
        targets = list(range(len(planner.obstacles)))
        self.add_msg(f"Planning from ({start.x:.0f},{start.y:.0f},{deg(start.theta):.0f}°) for {len(targets)} targets")
        orders = planner.plan_visiting_orders_discrete(start, targets)
        if not orders:
            self.add_msg("Planner failed.")
            return False
        self.orders = orders
        self.add_msg(f"Plan OK: {len(self.orders)} orders")
        # init sim pose
        self.car = CarState(start.x, start.y, start.theta)
        self.trail = [CarState(self.car.x, self.car.y, self.car.theta)]
        self.order_idx = 0
        self.scan_left = 0
        return True

    # ----------- animation step -----------
    def advance_one_frame(self):
        if self.paused or self.car is None or self.order_idx >= len(self.orders):
            return

        # scanning pause
        if self.scan_left > 0:
            self.scan_left -= 1
            return

        cmd = self.orders[self.order_idx]
        op = cmd["op"]

        if op in ("F","B"):
            # move along current heading (cmd already contains end pose)
            target_x = float(cmd["pose"]["x"])
            target_y = float(cmd["pose"]["y"])
            # distance to target along the current heading
            dx = target_x - self.car.x
            dy = target_y - self.car.y
            remain = math.hypot(dx, dy)
            if remain < 1e-6:
                # snap & move next
                self.car = CarState(target_x, target_y, self.car.theta)
                self.trail.append(self.car)
                self.order_idx += 1
                return
            step = self.step_cm * self.anim_speed
            if step >= remain:
                nx, ny = target_x, target_y
            else:
                nx = self.car.x + dx * (step/remain)
                ny = self.car.y + dy * (step/remain)
            self.car = CarState(nx, ny, self.car.theta)
            self.trail.append(self.car)

        elif op in ("L","R"):
            # turn along arc to target angle (we trust the stored pose as end)
            target_th = float(cmd["pose"]["theta"])
            end_x = float(cmd["pose"]["x"])
            end_y = float(cmd["pose"]["y"])
            R = float(cmd.get("radius_cm", config.car.turning_radius))
            # how much remaining angle?
            d = (target_th - self.car.theta + math.pi) % (2*math.pi) - math.pi
            if abs(d) < rad(1.0):
                self.car = CarState(end_x, end_y, target_th)
                self.trail.append(self.car)
                self.order_idx += 1
                return
            step_deg = self.arc_step_deg * self.anim_speed
            step_deg = min(step_deg, abs(deg(d)))
            nx, ny, nth = self.arc_end(self.car.x, self.car.y, self.car.theta, op, step_deg, R)
            self.car = CarState(nx, ny, nth)
            self.trail.append(self.car)

        elif op == "S":
            # scanning pause
            self.scan_left = self.scan_pause_frames
            self.order_idx += 1

        elif op == "B":
            # Backward is emitted as a straight to a new pose; treat same as F
            target_x = float(cmd["pose"]["x"])
            target_y = float(cmd["pose"]["y"])
            dx = target_x - self.car.x
            dy = target_y - self.car.y
            remain = math.hypot(dx, dy)
            if remain < 1e-6:
                self.car = CarState(target_x, target_y, self.car.theta)
                self.trail.append(self.car)
                self.order_idx += 1
                return
            step = self.step_cm * self.anim_speed
            if step >= remain:
                nx, ny = target_x, target_y
            else:
                nx = self.car.x + dx * (step/remain)
                ny = self.car.y + dy * (step/remain)
            self.car = CarState(nx, ny, self.car.theta)
            self.trail.append(self.car)
        else:
            # unknown op; skip
            self.order_idx += 1

    # ----------- event handling -----------
    def handle_events_editor(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: self.running=False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: self.running=False
                elif e.key == pygame.K_SPACE: self.paused = not self.paused
                elif e.key == pygame.K_g: self.show_grid = not self.show_grid
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS): self.anim_speed = min(5, self.anim_speed+1)
                elif e.key == pygame.K_MINUS: self.anim_speed = max(1, self.anim_speed-1)

                # place/plan controls
                elif e.key == pygame.K_RETURN:
                    # plan and start sim
                    if self.plan_from_editor():
                        self.editor_mode = False
                elif e.key == pygame.K_m:
                    self.save_map()
                elif e.key == pygame.K_l:
                    self.load_latest()
                elif e.key == pygame.K_r:
                    self.obstacles.clear(); self.orders.clear(); self.trail.clear()

                # set image side
                elif e.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d):
                    self.current_side = {pygame.K_w:'N', pygame.K_a:'W', pygame.K_s:'S', pygame.K_d:'E'}[e.key]

                # rotate start heading 45° steps
                elif e.key == pygame.K_q: self.start_pose = CarState(self.start_pose.x, self.start_pose.y, (self.start_pose.theta + rad(45))%(2*math.pi))
                elif e.key == pygame.K_e: self.start_pose = CarState(self.start_pose.x, self.start_pose.y, (self.start_pose.theta - rad(45))%(2*math.pi))

            elif e.type == pygame.MOUSEMOTION:
                wx, wy = self.screen_to_world(*e.pos)
                cell = config.arena.grid_cell_size
                gx = round(wx / cell) * cell
                gy = round(wy / cell) * cell
                self.preview = (gx, gy)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    if self.preview:
                        x,y = self.preview
                        size = config.arena.obstacle_size
                        x = max(0, min(self.field_cm - size, x))
                        y = max(0, min(self.field_cm - size, y))
                        self.obstacles.append(Obstacle(x,y,self.current_side))
                elif e.button == 3:
                    if self.obstacles: self.obstacles.pop()

    def handle_events_sim(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: self.running=False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: self.running=False
                elif e.key == pygame.K_SPACE: self.paused = not self.paused
                elif e.key == pygame.K_g: self.show_grid = not self.show_grid
                elif e.key == pygame.K_p: self.show_path_preview = not self.show_path_preview
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS): self.anim_speed = min(5, self.anim_speed+1)
                elif e.key == pygame.K_MINUS: self.anim_speed = max(1, self.anim_speed-1)

    # ----------- render -----------
    def draw_editor_overlay(self):
        # draw the start car first (still fine)
        self.draw_car(self.start_pose, scanning=False)

        # move the help panel to the right, under the sidebar
        panel_x = self.side_panel.x
        panel_y = self.side_panel.bottom + 10
        pw = self.side_panel.width
        ph = 100
        # keep inside window
        if panel_y + ph + 10 > self.H:
            ph = max(60, self.H - panel_y - 10)

        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, (panel_x, panel_y, pw, ph))
        pygame.draw.rect(self.screen, Colors.BLACK, (panel_x, panel_y, pw, ph), 2)

        lines = [
            "EDITOR: Left-click place • Right-click undo",
            "W/A/S/D set image side • Q/E rotate start 45°",
            "ENTER plan & run • M save • L load-latest",
            "G toggle grid • +/- speed • R reset • ESC quit",
            f"Current side: {self.current_side} | Start θ={deg(self.start_pose.theta):.0f}°"
        ]
        y = panel_y + 10
        for ln in lines:
            self.screen.blit(self.small.render(ln, True, Colors.BLACK), (panel_x + 10, y))
            y += 20


    def render(self):
        self.screen.fill(Colors.WHITE)
        self.draw_dynamic_grid()
        self.draw_obstacles()
        if self.editor_mode:
            self.draw_editor_overlay()
        else:
            self.draw_orders_preview()
            if self.trail and len(self.trail)>1:
                pts=[self.world_to_screen(s.x,s.y) for s in self.trail]
                pygame.draw.lines(self.screen, Colors.ORANGE, False, pts, 2)
            if self.car:
                self.draw_car(self.car, scanning=(self.scan_left>0))
        self.draw_panel()
        pygame.display.flip()

    # ----------- main loops -----------
    def run(self):
        # editor phase
        while self.running and self.editor_mode:
            self.handle_events_editor()
            self.render()
            self.clock.tick(60)

        # simulation phase
        while self.running:
            self.handle_events_sim()
            for _ in range(self.anim_speed):
                self.advance_one_frame()
            self.render()
            self.clock.tick(60)


if __name__ == "__main__":
    viz = ArcOrdersVisualizer()
    viz.run()
