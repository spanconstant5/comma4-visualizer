#!/usr/bin/env python3
"""
Comma 4 Openpilot Drive Visualizer
Standalone desktop application replicating the comma ui-alt / replay-clipper view.

Usage:
  python comma4_viz.py                       # Demo mode (synthetic data)
  python comma4_viz.py /path/to/route        # Play real openpilot route
  python comma4_viz.py --record              # Record demo to video

Controls:
  SPACE      - Pause / Resume
  LEFT/RIGHT - Scrub ±1 second
  R          - Reset to start
  +/-        - Adjust playback speed
  Q / ESC    - Quit

Install:  pip install pygame pycapnp opencv-python-headless
Package:  pyinstaller --onefile --windowed comma4_viz.py
"""

import pygame
import sys
import os
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Window & Layout ──────────────────────────────────────────────────────────

WINDOW_W = 1440
WINDOW_H = 936
FPS = 20

# Hardware-accelerated video decode (GPU). On by default; disable with the
# env var COMMA4_NO_HWACCEL=1 or the --no-gpu flag if a driver misbehaves.
USE_HW = os.environ.get("COMMA4_NO_HWACCEL", "") not in ("1", "true", "True")

MENUBAR_H = 26
HEADER_H = 85
FOOTER_H = 45
MARGIN = 10

CAMERA_W = 615
TELEM_W = 370
TOPDOWN_W = WINDOW_W - CAMERA_W - TELEM_W - 4 * MARGIN

CONTENT_Y = MENUBAR_H + HEADER_H + 5
CONTENT_H = WINDOW_H - MENUBAR_H - HEADER_H - FOOTER_H - 15
CAMERA_H = (CONTENT_H - MARGIN) // 2

ROAD_RECT = pygame.Rect(MARGIN, CONTENT_Y, CAMERA_W, CAMERA_H)
WIDE_RECT = pygame.Rect(MARGIN, CONTENT_Y + CAMERA_H + MARGIN, CAMERA_W, CAMERA_H)
TELEM_RECT = pygame.Rect(CAMERA_W + 2 * MARGIN, CONTENT_Y, TELEM_W, CONTENT_H)
_RIGHT_X = CAMERA_W + TELEM_W + 3 * MARGIN
TOPDOWN_RECT = pygame.Rect(_RIGHT_X, CONTENT_Y, TOPDOWN_W, CONTENT_H)
# Right column split: radar/confidence on top, OSM map below
_RADAR_H = int(CONTENT_H * 0.60)
RADAR_RECT = pygame.Rect(_RIGHT_X, CONTENT_Y, TOPDOWN_W, _RADAR_H)
MAP_RECT = pygame.Rect(_RIGHT_X, CONTENT_Y + _RADAR_H + MARGIN,
                       TOPDOWN_W, CONTENT_H - _RADAR_H - MARGIN)


def scrub_rect():
    """Clickable timeline rect, drawn just above the footer."""
    y = WINDOW_H - FOOTER_H - 15
    return pygame.Rect(MARGIN + 55, y, WINDOW_W - 2 * MARGIN - 110, 10)

# ── Colors ───────────────────────────────────────────────────────────────────

BG = (13, 18, 28)
PANEL_BG = (18, 24, 36)
CAM_BORDER = (38, 175, 95)
ROAD_COL = (72, 76, 82)
SHOULDER = (50, 72, 42)
SKY_TOP = (50, 75, 110)
SKY_BOT = (100, 130, 160)
LANE_WHITE = (195, 200, 205)
LANE_YELLOW = (195, 170, 45)
PATH_FILL = (255, 200, 40, 90)
CHEVRON = (255, 200, 40)
LEAD_CYAN = (40, 200, 220)
WHITE = (230, 230, 230)
DIM = (110, 120, 135)
ORANGE = (255, 155, 50)
GREEN_T = (55, 200, 115)
CYAN_T = (40, 200, 220)
RED_T = (220, 55, 55)
WHEEL_COL = (210, 215, 220)
WHEEL_BG = (30, 40, 55)
BAR_BG = (45, 55, 65)
BAR_GREEN = (55, 195, 95)
BAR_ORANGE = (215, 135, 40)
CONF_GREEN = (40, 215, 95)
RADAR_DOT = (190, 200, 215)
TRAJ_RED = (210, 55, 55)
EGO_YELLOW = (255, 195, 40)
OTHER_CAR = (180, 185, 190)

# ── Route Data Adapter ───────────────────────────────────────────────────────

class Projector:
    """Projects model points (device/calib frame, metres) to camera-panel pixels.

    Replicates openpilot's road/wide projection: intrinsics @ view_from_calib,
    scaled from the full sensor frame to the panel. Built once per route from the
    log's extrinsicsCalibration.
    """
    VIEW_FROM_DEVICE = None  # set below once numpy is imported

    def __init__(self, calib, panel_w, panel_h, cam_config):
        import numpy as np
        self.np = np
        if Projector.VIEW_FROM_DEVICE is None:
            Projector.VIEW_FROM_DEVICE = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
        self.height = float(calib.get("height", 1.22))
        rpy = calib.get("rpy", [0, 0, 0])
        wide = calib.get("wide", [0, 0, 0])
        dfc = self._euler2rot(rpy)
        vfd = Projector.VIEW_FROM_DEVICE
        view_from_calib = vfd @ dfc
        view_from_wide = vfd @ self._euler2rot(wide) @ dfc

        nW, nH, nfl = cam_config["narrow"]
        wW, wH, wfl = cam_config["wide"]

        def intr(W, H, fl):
            return np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]])
        def scale(W, H):
            return np.array([[panel_w / W, 0, 0], [0, panel_h / H, 0], [0, 0, 1]])
        self.T_road = scale(nW, nH) @ intr(nW, nH, nfl) @ view_from_calib
        self.T_wide = scale(wW, wH) @ intr(wW, wH, wfl) @ view_from_wide

    def _euler2rot(self, e):
        np = self.np
        phi, theta, psi = e
        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        cz, sz = np.cos(psi), np.sin(psi)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def project(self, pts3d, wide):
        """pts3d: Nx3 device points. Returns Nx2 screen pts (NaN where behind)."""
        np = self.np
        T = self.T_wide if wide else self.T_road
        proj = T @ pts3d.T              # 3xN
        z = proj[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            screen = (proj[:2] / z).T   # Nx2
        screen[z <= 0] = np.nan
        return screen


class RouteDataSource:
    """Wraps a loaded Route to serve DrivingState frames to the renderer."""

    def __init__(self, route_dir: str, progress_cb=None, use_hw=True, cache_frames=600):
        from route_loader import Route, VideoReader, ThreadedCamera

        def cb(msg, frac=None):
            print(f"  {msg}")
            if progress_cb:
                progress_cb(msg, frac)

        self.route = Route(route_dir, progress_cb=cb)
        self._car = self.route.car_fingerprint
        self._model = self.route.model_name
        self._runner = self.route.model_runner
        # Build the camera projector from the route's calibration + camera config
        self._projector = Projector(self.route.calib, ROAD_RECT.width, ROAD_RECT.height,
                                    self.route.cam_config)

        if progress_cb:
            progress_cb("Opening camera video...", None)
        self.road_video = VideoReader(self.route.segment_spans, "road",
                                      use_hw=use_hw, cache_frames=cache_frames)
        self.wide_video = VideoReader(self.route.segment_spans, "wide",
                                      use_hw=use_hw, cache_frames=cache_frames)
        # Background decoders for smooth, non-blocking playback
        self._road_thread = ThreadedCamera(self.road_video) if self.road_video.available else None
        self._wide_thread = ThreadedCamera(self.wide_video) if self.wide_video.available else None
        self._cached_surfaces = {}

    @property
    def duration(self):
        return self.route.duration

    def generate(self, t: float) -> 'DrivingState':
        """Convert route frame data to a DrivingState for the renderer."""
        f = self.route.get_frame(t)
        s = DrivingState()
        s.timestamp = f.t
        s.speed_mph = f.speed_mph
        # vCruise is in km/h -> mph for the MAX badge
        s.max_speed = int(round(f.cruise_speed * 0.621371)) if f.cruise_speed > 0 else 0
        s.steer_target = f.desired_curvature * 500 if f.desired_curvature else f.steering_angle * 0.5
        s.steer_applied = s.steer_target
        s.steer_actual = f.steering_angle
        s.steer_delta = s.steer_target - s.steer_actual
        s.hands_on = f.steering_torque != 0 or getattr(f, 'steeringPressed', False)
        s.throttle = 8.0 if f.gas_pressed else max(0, f.a_ego * 5)
        s.brake = 15.0 if f.brake_pressed else max(0, -f.a_ego * 8)
        s.gas_pressed = f.gas_pressed
        s.brake_pressed = f.brake_pressed
        s.a_ego = f.a_ego
        s.a_target = f.a_target
        s.a_cmd = f.a_ego * 0.9
        s.a_out = f.a_ego * 0.9
        s.lead_dist = f.lead_dist if f.lead_present else 0
        s.lead_speed = f.lead_v_lead
        # radar yRel is left-positive; model/device frame is right-positive.
        # Normalize to right-positive so camera + top-down agree.
        s.lead_y = -f.lead_y_rel
        # Nuanced engagement (sunnypilot MADS-aware)
        s.engage_status = getattr(f, "engage_status", "disengaged")
        s.engaged = s.engage_status == "engaged"
        s.op_active = s.engage_status != "disengaged"
        s.alert1 = f.alert1
        s.alert2 = f.alert2
        s.alert_status = f.alert_status
        s.alert_size = f.alert_size
        s.left_blinker = f.left_blinker
        s.right_blinker = f.right_blinker
        s.confidence = f.confidence
        # Real model trajectory for the path overlay
        s.path_x = f.model_path_x
        s.path_y = f.model_path_y
        s.path_z = f.model_path_z
        s.model = f.model
        s.projector = self._projector
        s.road_name = f.road_name
        s.map_speed_limit = f.map_speed_limit
        s.lat = f.lat
        s.lon = f.lon
        s.gps_track = self.route.gps_track
        s.nav_maneuver = f.nav_maneuver
        s.nav_street = f.nav_street
        s.nav_dist = f.nav_dist

        # Build radar points from model path
        pts = []
        if f.model_path_x and f.model_path_y:
            for x, y in zip(f.model_path_x[:20], f.model_path_y[:20]):
                pts.append((y, x))
        # Add lead car cluster
        if f.lead_present and f.lead_dist > 0:
            for _ in range(5):
                pts.append((f.lead_y_rel + random.uniform(-1, 1),
                            f.lead_dist + random.uniform(-1, 1)))
        s.radar_pts = pts
        s._car_fingerprint = self._car
        return s

    def get_road_frame(self, t: float):
        """Playback road frame (non-blocking, from the decode thread)."""
        if self._road_thread is None:
            return None
        self._road_thread.set_target(t)
        return self._to_surface(self._road_thread.frame())

    def get_wide_frame(self, t: float):
        """Playback wide frame (non-blocking, from the decode thread)."""
        if self._wide_thread is None:
            return None
        self._wide_thread.set_target(t)
        return self._to_surface(self._wide_thread.frame())

    def get_road_frame_exact(self, t: float):
        """Exact road frame for export (blocking, precise)."""
        if self._road_thread is None:
            return None
        return self._to_surface(self._road_thread.exact(t))

    def get_wide_frame_exact(self, t: float):
        """Exact wide frame for export (blocking, precise)."""
        if self._wide_thread is None:
            return None
        return self._to_surface(self._wide_thread.exact(t))

    def _to_surface(self, frame):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class DrivingState:
    timestamp: float = 0.0
    speed_mph: float = 42.0
    max_speed: int = 75
    steer_target: float = 0.0
    steer_applied: float = 0.0
    steer_actual: float = 0.0
    steer_delta: float = 0.0
    hands_on: bool = False
    throttle: float = 0.0
    brake: float = 0.0
    gas_pressed: bool = False
    brake_pressed: bool = False
    a_ego: float = 0.0
    a_target: float = 0.0
    a_cmd: float = 0.0
    a_out: float = 0.0
    lead_dist: float = 55.0
    lead_speed: float = 40.0
    radar_pts: list = field(default_factory=list)
    confidence: float = 0.75
    engaged: bool = True
    engage_status: str = "engaged"   # engaged|lat_only|long_only|override|disengaged
    op_active: bool = True           # show model overlay (status != disengaged)
    alert1: str = ""
    alert2: str = ""
    alert_status: str = "normal"     # normal|userPrompt|critical
    alert_size: str = "none"         # none|small|mid|full
    left_blinker: bool = False
    right_blinker: bool = False
    # Real model trajectory (device frame, metres); empty => draw synthetic path
    path_x: list = field(default_factory=list)
    path_y: list = field(default_factory=list)
    path_z: list = field(default_factory=list)
    lead_y: float = 0.0
    model: dict = None               # full model snapshot (path/lanes/edges/leads)
    projector: object = None         # Projector for device->screen mapping
    # map / OSM
    road_name: str = ""
    map_speed_limit: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    gps_track: list = None           # [(t,lat,lon)] for the whole route (shared ref)
    nav_maneuver: str = ""
    nav_street: str = ""
    nav_dist: float = 0.0


class DemoGenerator:
    """Generates realistic-looking driving simulation data."""

    def __init__(self):
        self._static_radar = self._make_static_radar()

    def _make_static_radar(self):
        pts = []
        # Road-side objects along both sides
        for d in range(5, 120, 3):
            if random.random() < 0.6:
                offset = random.uniform(12, 50) * random.choice([-1, 1])
                pts.append((offset, d + random.uniform(-1, 1)))
        # Some clusters (buildings / parked cars)
        for _ in range(8):
            cx = random.uniform(15, 45) * random.choice([-1, 1])
            cy = random.uniform(10, 90)
            for _ in range(random.randint(3, 7)):
                pts.append((cx + random.uniform(-3, 3), cy + random.uniform(-3, 3)))
        return pts

    def generate(self, t: float) -> DrivingState:
        s = DrivingState()
        s.timestamp = t

        # Speed: cruising ~42 mph with gentle variation
        s.speed_mph = 42.0 + 5.0 * math.sin(t * 0.25) + 2.0 * math.sin(t * 0.67)
        s.speed_mph = max(28, min(55, s.speed_mph + 0.5 * math.sin(t * 1.3)))
        s.max_speed = 75

        # Steering: gentle lane keeping + occasional curves
        curve = 3.0 * math.sin(t * 0.18) + 1.5 * math.sin(t * 0.43)
        noise = 0.3 * math.sin(t * 2.1) + 0.15 * math.sin(t * 4.7)
        s.steer_target = curve + noise
        s.steer_applied = s.steer_target
        s.steer_actual = s.steer_target * 0.7 + 0.4 * math.sin(t * 0.9)
        s.steer_delta = s.steer_target - s.steer_actual
        s.hands_on = False

        # Throttle / Brake
        accel_desire = 0.8 * math.sin(t * 0.3) + 0.3 * math.sin(t * 0.8)
        s.throttle = max(0, accel_desire * 8)
        s.brake = max(0, -accel_desire * 10)

        # Acceleration
        s.a_ego = accel_desire * 0.5 + 0.1 * math.sin(t * 1.5)
        s.a_target = s.a_ego * 0.9
        s.a_cmd = s.a_ego * 0.75
        s.a_out = s.a_cmd

        # Lead car: oscillates in and out of range
        lead_cycle = math.sin(t * 0.12)
        if lead_cycle > -0.3:
            s.lead_dist = 45 + 25 * math.sin(t * 0.2) + 10 * math.sin(t * 0.55)
            s.lead_speed = s.speed_mph - 2 + 3 * math.sin(t * 0.4)
        else:
            s.lead_dist = 0
            s.lead_speed = 0

        # Radar points: static base + jitter + moving vehicles
        pts = []
        for bx, by in self._static_radar:
            jx = bx + random.uniform(-0.3, 0.3)
            jy = by + random.uniform(-0.3, 0.3)
            pts.append((jx, jy))

        # Lead vehicle as radar cluster
        if s.lead_dist > 0:
            for _ in range(5):
                pts.append((random.uniform(-1.5, 1.5), s.lead_dist + random.uniform(-1.5, 1.5)))

        # Adjacent lane traffic
        for i in range(3):
            adj_d = 30 + i * 25 + 10 * math.sin(t * 0.15 + i)
            adj_x = random.choice([-4, 4]) + random.uniform(-0.5, 0.5)
            for _ in range(3):
                pts.append((adj_x + random.uniform(-1, 1), adj_d + random.uniform(-1, 1)))

        s.radar_pts = pts
        s.confidence = 0.7 + 0.2 * math.sin(t * 0.3)
        s.engaged = True
        return s


class TileMap:
    """Downloads and caches OpenStreetMap tiles on the fly (background threads).

    Renders a slippy-map background under the GPS trace. Falls back silently to
    a trace-only view when offline. Tiles are cached on disk between runs.
    """
    TILE = 256
    ATTRIB = "© OpenStreetMap contributors"

    def __init__(self):
        import threading, queue, tempfile
        self._cache = {}        # (z,x,y) -> Surface | False(failed)
        self._pending = set()
        self._lock = threading.Lock()
        self._q = queue.Queue()
        self._dir = os.path.join(tempfile.gettempdir(), "comma4viz_tiles")
        try:
            os.makedirs(self._dir, exist_ok=True)
        except Exception:
            self._dir = None
        self._online = True
        for _ in range(3):
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            key = self._q.get()
            if key is None:
                return
            surf = self._load_or_fetch(*key)
            with self._lock:
                self._cache[key] = surf if surf is not None else False
                self._pending.discard(key)

    def _load_or_fetch(self, z, x, y):
        import io
        path = os.path.join(self._dir, f"{z}_{x}_{y}.png") if self._dir else None
        if path and os.path.exists(path):
            try:
                return pygame.image.load(path)
            except Exception:
                pass
        try:
            import urllib.request
            url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            req = urllib.request.Request(url, headers={"User-Agent": "comma4-visualizer/1.0 (drive review)"})
            data = urllib.request.urlopen(req, timeout=8).read()
            if path:
                try:
                    with open(path, "wb") as f:
                        f.write(data)
                except Exception:
                    pass
            self._online = True
            return pygame.image.load(io.BytesIO(data), "t.png")
        except Exception:
            self._online = False
            return None

    def get(self, z, x, y):
        key = (z, x, y)
        with self._lock:
            if key in self._cache:
                v = self._cache[key]
                return v if v else None
            if key not in self._pending:
                self._pending.add(key)
                self._q.put(key)
        return None

    @staticmethod
    def deg2tile(lat, lon, z):
        n = 2 ** z
        x = (lon + 180.0) / 360.0 * n
        la = math.radians(lat)
        y = (1.0 - math.log(math.tan(la) + 1.0 / math.cos(la)) / math.pi) / 2.0 * n
        return x, y


class MenuBar:
    """Clickable top menu bar. handle(ev,state)->action id or None; draw()."""

    MENUS = [
        ("File", [("open", "Open Route…"), ("quit", "Quit")]),
        ("Playback", [("playpause", "Play / Pause"), ("restart", "Restart"),
                      ("set_in", "Set In Point"), ("set_out", "Set Out Point"),
                      ("clear_sel", "Clear Selection"),
                      ("faster", "Speed +"), ("slower", "Speed -")]),
        ("View", [("toggle_osm", "Location Panel", "osm")]),
        ("Export", [("export", "Export Video…")]),
        ("Help", [("help", "Controls & Info")]),
    ]

    def __init__(self):
        self.font = pygame.font.SysFont("DejaVu Sans", 15)
        self.open_idx = None
        self._titles = []   # (rect, idx)
        self._items = []    # (rect, action)
        x = 6
        for i, (title, _items) in enumerate(self.MENUS):
            tw = self.font.size(title)[0] + 22
            self._titles.append((pygame.Rect(x, 0, tw, MENUBAR_H), i))
            x += tw

    def handle(self, ev):
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return None
        for rect, i in self._titles:
            if rect.collidepoint(ev.pos):
                self.open_idx = None if self.open_idx == i else i
                return None
        if self.open_idx is not None:
            for rect, action in self._items:
                if rect.collidepoint(ev.pos):
                    self.open_idx = None
                    return action
            self.open_idx = None
        return None

    def draw(self, screen, state):
        pygame.draw.rect(screen, (22, 28, 40), (0, 0, WINDOW_W, MENUBAR_H))
        pygame.draw.line(screen, (40, 50, 66), (0, MENUBAR_H), (WINDOW_W, MENUBAR_H))
        for rect, i in self._titles:
            if i == self.open_idx:
                pygame.draw.rect(screen, (45, 58, 78), rect)
            lbl = self.font.render(self.MENUS[i][0], True, WHITE)
            screen.blit(lbl, (rect.x + 11, 5))
        self._items = []
        if self.open_idx is not None:
            title_rect = self._titles[self.open_idx][0]
            items = self.MENUS[self.open_idx][1]
            iw = 210
            ih = 26
            x0 = title_rect.x
            y0 = MENUBAR_H
            box = pygame.Rect(x0, y0, iw, ih * len(items) + 6)
            pygame.draw.rect(screen, (28, 36, 50), box)
            pygame.draw.rect(screen, (60, 75, 95), box, 1)
            mouse = pygame.mouse.get_pos()
            for j, item in enumerate(items):
                action, label = item[0], item[1]
                ir = pygame.Rect(x0 + 2, y0 + 3 + j * ih, iw - 4, ih)
                if ir.collidepoint(mouse):
                    pygame.draw.rect(screen, (45, 60, 82), ir)
                check = ""
                if len(item) > 2 and state.get(item[2]):
                    check = "✓ "
                lbl = self.font.render(check + label, True, WHITE)
                screen.blit(lbl, (ir.x + 10, ir.y + 5))
                self._items.append((ir, action))


# ── Renderer ─────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.show_osm = True   # toggle the OSM map panel
        self.map_tiles = True  # use downloaded OSM tiles (vs trace only)
        self._tilemap = None
        self._init_fonts()

    def _init_fonts(self):
        candidates = ["Roboto", "Inter", "Segoe UI", "Helvetica Neue",
                       "DejaVu Sans", "Liberation Sans", "Arial", None]
        mono_cands = ["JetBrains Mono", "Roboto Mono", "Consolas",
                      "DejaVu Sans Mono", "Liberation Mono", "Courier New", None]

        def pick(names, size, bold=False):
            for n in names:
                f = pygame.font.SysFont(n, size, bold=bold)
                if f:
                    return f
            return pygame.font.Font(None, size)

        self.f_huge = pick(candidates, 54, True)
        self.f_large = pick(candidates, 38, True)
        self.f_med = pick(candidates, 22, True)
        self.f_body = pick(candidates, 18)
        self.f_small = pick(candidates, 15)
        self.f_tiny = pick(candidates, 13)
        self.f_mono = pick(mono_cands, 18, True)
        self.f_mono_sm = pick(mono_cands, 15)
        self.f_mono_lg = pick(mono_cands, 24, True)
        self.f_header = pick(candidates, 14)

    # ── Full frame ───────────────────────────────────────────────────────

    def draw(self, state: DrivingState, paused: bool, speed_mult: float,
             road_frame=None, wide_frame=None, route_info=None):
        route_mode = route_info is not None
        self.screen.fill(BG)
        self._header(state, route_info)
        self._camera(ROAD_RECT, "ROAD", False, state, road_frame, route_mode)
        self._camera(WIDE_RECT, "WIDE", True, state, wide_frame, route_mode)
        self._telemetry(TELEM_RECT, state)
        has_gps = bool(getattr(state, "gps_track", None))
        if route_mode and self.show_osm and has_gps:
            self._topdown(RADAR_RECT, state)
            self._map(MAP_RECT, state)
        else:
            self._topdown(TOPDOWN_RECT, state)
        self._footer(paused, speed_mult)
        self._controls_hint(paused, speed_mult)

    # ── Header ───────────────────────────────────────────────────────────

    def _header(self, st: DrivingState, route_info=None):
        y = MENUBAR_H + 8
        # Left: "UI ALT" + timestamp
        mode_label = "ROUTE" if route_info else "UI ALT"
        ui_lbl = self.f_small.render(mode_label, True, GREEN_T)
        self.screen.blit(ui_lbl, (MARGIN, y))
        mins = int(st.timestamp) // 60
        secs = int(st.timestamp) % 60
        total = int(st.timestamp)
        ts = self.f_med.render(f"{mins:02d}:{secs:02d} • {total}s", True, WHITE)
        self.screen.blit(ts, (MARGIN, y + 18))

        # Right: device + car info
        rx = WINDOW_W - MARGIN
        if route_info:
            car = route_info.get("car", "UNKNOWN")
            segs = route_info.get("segments", 0)
            dur = route_info.get("duration", 0)
            route_dir = route_info.get("route_dir", "")
            model = route_info.get("model", "")
            runner = route_info.get("runner", "")
            if model:
                model_line = f"model: {model}"
                if runner:
                    model_line += f"  ({runner})"
            else:
                model_line = "model: (not in log)"
            lines = [
                (f"comma 4  •  {car}", WHITE),
                (os.path.basename(route_dir), CYAN_T),
                (f"{segs} segments  •  {dur:.0f}s total", DIM),
                (model_line, GREEN_T if model else DIM),
                ("openpilot route playback", DIM),
            ]
        else:
            lines = [
                ("mici (comma 4)  •  FORD_BRONCO_SPORT_MK1", WHITE),
                ("5beb9b58bd12b691|0000010a--a51155e496", CYAN_T),
                ("segment Feb 25 2026 17:31 UTC  •  commit Feb 13 2026", DIM),
                ("openpilot  •  nightly  •  49a611df  •  clean", DIM),
                ("Model Commit: 5b6436a9 - CD210 model (#37050)", DIM),
            ]
        for i, (txt, col) in enumerate(lines):
            surf = self.f_header.render(txt, True, col)
            self.screen.blit(surf, (rx - surf.get_width(), y + i * 16))

        # Separator line
        sep_y = MENUBAR_H + HEADER_H - 2
        pygame.draw.line(self.screen, (35, 45, 60), (0, sep_y), (WINDOW_W, sep_y))

    # ── Camera View ──────────────────────────────────────────────────────

    def _camera(self, rect: pygame.Rect, label: str, wide: bool, st: DrivingState,
                video_frame=None, route_mode=False):
        w, h = rect.width, rect.height
        surf = pygame.Surface((w, h))
        fov = 1.4 if wide else 1.0

        if video_frame is not None:
            # Use real camera footage - scale to fit panel
            scaled = pygame.transform.scale(video_frame, (w, h))
            surf.blit(scaled, (0, 0))
            # Skip synthetic road rendering, jump to overlays
            self._camera_overlays(surf, w, h, fov, st, label)
            self.screen.blit(surf, rect.topleft)
            pygame.draw.rect(self.screen, CAM_BORDER, rect, 3, border_radius=4)
            return

        if route_mode:
            # Route loaded but no decoded frame for this moment - show placeholder,
            # never the synthetic animation. Telemetry stays real.
            surf.fill((16, 20, 28))
            msg = f"{label} camera — no video for this segment"
            lbl = self.f_small.render(msg, True, DIM)
            surf.blit(lbl, lbl.get_rect(center=(w // 2, h // 2)))
            self._camera_overlays(surf, w, h, fov, st, label)
            self.screen.blit(surf, rect.topleft)
            pygame.draw.rect(self.screen, CAM_BORDER, rect, 3, border_radius=4)
            return

        # Sky gradient
        horizon = int(h * 0.40)
        for y in range(horizon):
            t = y / max(1, horizon)
            r = int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t)
            g = int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t)
            b = int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (w, y))

        vp_x = w // 2 + int(st.steer_actual * -8)
        road_hw = int(w * 0.38 * fov)

        # Shoulder / grass
        left_grass = [(0, h), (0, horizon), (vp_x, horizon), (w // 2 - road_hw, h)]
        right_grass = [(w // 2 + road_hw, h), (vp_x, horizon), (w, horizon), (w, h)]
        pygame.draw.polygon(surf, SHOULDER, left_grass)
        pygame.draw.polygon(surf, SHOULDER, right_grass)

        # Road surface
        road_pts = [(w // 2 - road_hw, h), (vp_x - 1, horizon),
                     (vp_x + 1, horizon), (w // 2 + road_hw, h)]
        pygame.draw.polygon(surf, ROAD_COL, road_pts)

        # Horizon trees / foliage
        tree_col = (40, 58, 35)
        tree_col2 = (35, 52, 30)
        random.seed(42)
        for i in range(30):
            tx = int(random.uniform(0, w))
            tr = int(random.uniform(8, 22))
            ty = horizon - tr + random.randint(-3, 5)
            if abs(tx - vp_x) > 40:  # Keep clear around vanishing point
                pygame.draw.circle(surf, tree_col if i % 2 == 0 else tree_col2,
                                   (tx, ty), tr)
        random.seed(None)

        # Lane markings
        num_lanes = 5
        for li in range(num_lanes):
            lane_off = (li - 2) * (road_hw * 2 / 4)
            bx = w // 2 + int(lane_off)
            col = LANE_YELLOW if li == 2 else LANE_WHITE
            dashes = 20
            for di in range(dashes):
                t1 = di / dashes
                t2 = (di + 0.4) / dashes
                x1 = bx + (vp_x - bx) * t1
                y1 = h + (horizon - h) * t1
                x2 = bx + (vp_x - bx) * t2
                y2 = h + (horizon - h) * t2
                thick = max(1, int(2.5 * (1 - t1)))
                alpha = max(80, int(255 * (1 - t1 * 0.7)))
                c = tuple(min(255, int(cc * alpha / 255)) for cc in col)
                pygame.draw.line(surf, c, (int(x1), int(y1)), (int(x2), int(y2)), thick)

        # Yellow path overlay
        path_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        steer_off = st.steer_target * -12
        pts_l, pts_r = [], []
        segs = 35
        for i in range(segs + 1):
            t = i / segs
            pw = max(1, 75 * (1 - t) ** 1.7 * fov)
            cx = w // 2 + steer_off * t * t
            y = h + (horizon - h) * t
            if y < horizon:
                break
            pts_l.append((cx - pw, y))
            pts_r.append((cx + pw, y))
        poly = pts_l + list(reversed(pts_r))
        if len(poly) >= 3:
            pygame.draw.polygon(path_surf, PATH_FILL, poly)
        surf.blit(path_surf, (0, 0))

        # Chevron
        cy_base = h - 8
        cy_tip = h - 55
        cw = int(48 * fov)
        chevron_pts = [(w // 2 - cw, cy_base), (w // 2, cy_tip),
                       (w // 2 + cw, cy_base), (w // 2, cy_base - 20)]
        pygame.draw.polygon(surf, CHEVRON, chevron_pts)

        # Lead car bounding box
        if st.lead_dist > 0:
            lt = min(0.82, st.lead_dist / 120.0)
            ly = h + (horizon - h) * lt
            lw = max(16, int(110 * (1 - lt) * fov))
            lh = max(12, int(75 * (1 - lt)))
            lx = w // 2 - lw // 2 + int(steer_off * lt * 0.3)
            lr = pygame.Rect(lx, int(ly) - lh, lw, lh)
            # Simple car silhouette (darker than road so it's visible)
            pygame.draw.rect(surf, (48, 52, 60), lr)
            # Windshield highlight
            ws = pygame.Rect(lr.x + 3, lr.y + 2, lr.width - 6, lr.height // 3)
            pygame.draw.rect(surf, (70, 80, 95), ws)
            # Cyan bounding box (thick enough to see)
            pygame.draw.rect(surf, LEAD_CYAN, lr, max(2, int(3 * (1 - lt))))
            # Distance marker diamond at top-center
            dx, dy = lr.centerx, lr.top - 6
            ds = 4
            pygame.draw.polygon(surf, LEAD_CYAN, [(dx, dy - ds), (dx + ds, dy),
                                                   (dx, dy + ds), (dx - ds, dy)])

        # (Speed is shown in the telemetry panel, not on the camera, to match
        #  the comma UI and keep the sim recording clean.)

        # Max speed badge
        badge = pygame.Rect(12, 12, 52, 50)
        pygame.draw.rect(surf, (30, 38, 50, 200), badge, border_radius=6)
        mx_lbl = self.f_tiny.render("MAX", True, GREEN_T)
        mx_val = self.f_med.render(str(st.max_speed), True, WHITE)
        surf.blit(mx_lbl, mx_lbl.get_rect(centerx=badge.centerx, top=badge.top + 6))
        surf.blit(mx_val, mx_val.get_rect(centerx=badge.centerx, top=badge.top + 22))

        # Camera label
        cl = self.f_small.render(label, True, WHITE)
        surf.blit(cl, (14, 65))

        # Atom icon (top-right, openpilot engaged indicator)
        atom_col = GREEN_T if st.engaged else (ORANGE if st.op_active else DIM)
        self._draw_atom(surf, w - 38, 38, 22, atom_col)

        # Driver monitoring icon (bottom-left)
        self._draw_dm_icon(surf, 38, h - 38, 22, GREEN_T)

        # Blit + green border
        self.screen.blit(surf, rect.topleft)
        pygame.draw.rect(self.screen, CAM_BORDER, rect, 3, border_radius=4)

    def _ribbon(self, proj, pts, y_off, z_off, wide, max_x, w, h):
        """Project a model polyline (Nx3) as a screen polygon ribbon."""
        np = proj.np
        if pts is None or len(pts) < 2:
            return None
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] <= max_x)]
        if len(pts) < 2:
            return None
        left = proj.project(pts + np.array([0.0, -y_off, z_off]), wide)
        right = proj.project(pts + np.array([0.0, y_off, z_off]), wide)
        # Keep points where both edges are valid and within the panel (+margin),
        # matching openpilot's clip region so near off-screen points don't fan out.
        M = 250
        def inrange(a):
            return (~np.isnan(a).any(1)) & (a[:, 0] > -M) & (a[:, 0] < w + M) \
                & (a[:, 1] > -M) & (a[:, 1] < h + M)
        good = inrange(left) & inrange(right)
        left, right = left[good], right[good]
        if len(left) < 2:
            return None
        return [tuple(p) for p in left] + [tuple(p) for p in right[::-1]]

    def _draw_model(self, surf, w, h, st, wide):
        """Draw projected lane lines, road edges, path and lead from the model."""
        proj = st.projector
        m = st.model
        np = proj.np
        osurf = pygame.Surface((w, h), pygame.SRCALPHA)
        disengaged = not st.op_active

        max_x = 100.0
        # Lead shortens the path
        path = m.get("path")
        if path is not None and len(path):
            max_x = float(np.clip(path[-1, 0], 10.0, 100.0))

        # Lane lines (brighter, fuller — thicker ribbon so they read clearly)
        probs = m.get("lane_probs", [])
        for i, ll in enumerate(m.get("lanes", [])):
            prob = probs[i] if i < len(probs) else 0.0
            half = (0.22 if i in (1, 2) else 0.18) * max(prob, 0.25)
            poly = self._ribbon(proj, ll, half, 0.0, wide, max_x, w, h)
            if poly and len(poly) >= 3:
                a = int(np.clip(prob, 0.15, 1.0) * 235)
                col = (0, 0, 0, a) if disengaged else (240, 240, 245, a)
                pygame.draw.polygon(osurf, col, poly)

        # Road edges (solid red lines)
        stds = m.get("edge_stds", [])
        for i, re in enumerate(m.get("edges", [])):
            std = stds[i] if i < len(stds) else 1.0
            poly = self._ribbon(proj, re, 0.18, 0.0, wide, max_x, w, h)
            if poly and len(poly) >= 3:
                a = int(np.clip(1.0 - std, 0.25, 1.0) * 235)
                pygame.draw.polygon(osurf, (255, 70, 70, a), poly)

        # Path (offset down to the ground by camera height)
        if path is not None and len(path):
            # end the path at the lead if one is close
            pmax = max_x
            if st.lead_dist > 0:
                ld = st.lead_dist * 2.0
                pmax = float(np.clip(ld - min(ld * 0.35, 10.0), 0.0, max_x))
            poly = self._ribbon(proj, path, 0.9, proj.height, wide, pmax, w, h)
            if poly and len(poly) >= 3:
                if disengaged:
                    pygame.draw.polygon(osurf, (0, 0, 0, 90), poly)
                else:
                    pygame.draw.polygon(osurf, (16, 245, 120, 110), poly)

        # Detected vehicles: the model's lead predictions + the radar lead.
        # (openpilot logs leads ahead/in-path only; it does not track oncoming
        #  or cross traffic, so those can't be shown reliably.)
        drawn = []
        leads = m.get("leads", []) or []
        for prob, lx, ly in leads:
            if prob < 0.5 or lx <= 2:
                continue
            drawn.append((lx, ly, prob))
        # radar lead as a fallback if the model gave none
        if not drawn and st.lead_dist > 0:
            drawn.append((st.lead_dist + 1.52, st.lead_y, 1.0))

        for lx, ly, prob in drawn:
            self._draw_vehicle(osurf, proj, wide, lx, ly, prob, st.op_active, np)

        surf.blit(osurf, (0, 0))

    def _draw_vehicle(self, osurf, proj, wide, x, y, prob, active, np):
        """Draw a detected vehicle marker (box + chevron + distance) on the camera."""
        zoff = proj.height
        # 3D box corners around the vehicle (~1.8 m wide, ~1.4 m tall)
        hw, ht = 0.9, 1.4
        base = np.array([
            [x, y + hw, zoff], [x, y - hw, zoff],
            [x, y - hw, zoff - ht], [x, y + hw, zoff - ht],
        ])
        scr = proj.project(base, wide)
        if np.isnan(scr).any():
            return
        pts = [tuple(p) for p in scr]
        col = (40, 210, 235) if active else (180, 185, 195)
        pygame.draw.polygon(osurf, col, pts, 2)
        # chevron marker above the box
        cx = int(sum(p[0] for p in pts) / 4)
        top = int(min(p[1] for p in pts))
        s = max(5, int(14 - x * 0.08))
        pygame.draw.polygon(osurf, col, [(cx, top - s - 6), (cx - s, top - 6), (cx + s, top - 6)])
        # distance label
        d_m = int(round(x))
        lbl = self.f_tiny.render(f"{d_m} m", True, col)
        osurf.blit(lbl, (cx - lbl.get_width() // 2, top - s - 22))

    def _camera_overlays(self, surf, w, h, fov, st, label):
        """Draw HUD overlays on top of a real camera frame."""
        horizon = int(h * 0.40)
        wide = label == "WIDE"
        have_model = st.model is not None and st.projector is not None \
            and st.model.get("path") is not None and len(st.model.get("path"))

        if have_model:
            self._draw_model(surf, w, h, st, wide)
        else:
            # Synthetic fallback (demo mode): steer-driven path + chevron.
            steer_off = st.steer_target * -12
            path_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pts_l, pts_r = [], []
            segs = 35
            for i in range(segs + 1):
                t = i / segs
                pw = max(1, 75 * (1 - t) ** 1.7 * fov)
                cx = w // 2 + steer_off * t * t
                y = h + (horizon - h) * t
                if y < horizon:
                    break
                pts_l.append((cx - pw, y))
                pts_r.append((cx + pw, y))
            poly = pts_l + list(reversed(pts_r))
            if len(poly) >= 3:
                pygame.draw.polygon(path_surf, PATH_FILL, poly)
            surf.blit(path_surf, (0, 0))
            cw = int(48 * fov)
            pygame.draw.polygon(surf, CHEVRON, [
                (w // 2 - cw, h - 8), (w // 2, h - 55),
                (w // 2 + cw, h - 8), (w // 2, h - 28)])

        # (Speed is shown in the telemetry panel, not on the camera, to match
        #  the comma UI and keep the sim recording clean.)

        # Max speed badge
        if st.max_speed > 0:
            badge = pygame.Rect(12, 12, 52, 50)
            pygame.draw.rect(surf, (30, 38, 50, 200), badge, border_radius=6)
            mx_lbl = self.f_tiny.render("MAX", True, GREEN_T)
            mx_val = self.f_med.render(str(st.max_speed), True, WHITE)
            surf.blit(mx_lbl, mx_lbl.get_rect(centerx=badge.centerx, top=badge.top + 6))
            surf.blit(mx_val, mx_val.get_rect(centerx=badge.centerx, top=badge.top + 22))

        # Camera label
        cl = self.f_small.render(label, True, WHITE)
        surf.blit(cl, (14, 65 if st.max_speed > 0 else 14))

        # Atom icon + DM icon
        atom_col = GREEN_T if st.engaged else (ORANGE if st.op_active else DIM)
        self._draw_atom(surf, w - 38, 38, 22, atom_col)
        self._draw_dm_icon(surf, 38, h - 38, 22, GREEN_T)

        # Turn signals (blink like the comma UI)
        self._draw_blinkers(surf, w, h, st)

        # Alert / take-over banner (road camera only)
        if label == "ROAD":
            self._draw_alert(surf, w, h, st)

    def _draw_blinkers(self, surf, w, h, st):
        if not (st.left_blinker or st.right_blinker):
            return
        if int(time.time() * 2.5) % 2 == 1:   # ~1.25 Hz blink
            return
        cy = int(h * 0.16)
        col = (70, 230, 130)
        a = 16   # half-height of each chevron

        def chevrons(cx, d):
            # two stacked chevrons pointing in direction d (-1 left, +1 right)
            for k in (0, 1):
                ox = cx + d * k * 20
                pygame.draw.polygon(surf, col, [
                    (ox - d * a, cy - a), (ox + d * a, cy), (ox - d * a, cy + a)])
        if st.left_blinker:
            chevrons(w // 2 - 46, -1)
        if st.right_blinker:
            chevrons(w // 2 + 46, 1)

    def _draw_alert(self, surf, w, h, st):
        """Draw the openpilot alert banner (e.g. 'TAKE CONTROL') like the comma UI."""
        a1 = (st.alert1 or "").strip()
        a2 = (st.alert2 or "").strip()
        if not a1 and not a2:
            return
        if st.alert_size in ("none",) and not a1:
            return
        status = st.alert_status
        if status == "critical":
            bg, fg = (155, 20, 20), (255, 255, 255)
        elif status == "userPrompt":
            bg, fg = (90, 70, 20), (255, 240, 200)
        else:
            bg, fg = (30, 38, 52), (235, 240, 245)

        full = st.alert_size == "full"
        bh = int(h * (0.52 if full else 0.22))
        band = pygame.Surface((w, bh), pygame.SRCALPHA)
        band.fill((*bg, 230))
        # text
        f1 = self.f_large if full else self.f_med
        if a1:
            t1 = f1.render(a1[:42], True, fg)
            band.blit(t1, t1.get_rect(centerx=w // 2,
                                      centery=bh // 2 - (14 if a2 else 0)))
        if a2:
            t2 = self.f_body.render(a2[:52], True, fg)
            band.blit(t2, t2.get_rect(centerx=w // 2, centery=bh // 2 + 16))
        y = (h - bh) // 2 if full else (h - bh - 8)
        surf.blit(band, (0, y))

    def _draw_atom(self, surf, cx, cy, r, col):
        for deg in [0, 60, 120]:
            a = math.radians(deg)
            pts = []
            for t in range(0, 361, 8):
                tr = math.radians(t)
                ex = r * math.cos(tr)
                ey = r * 0.32 * math.sin(tr)
                rx = ex * math.cos(a) - ey * math.sin(a)
                ry = ex * math.sin(a) + ey * math.cos(a)
                pts.append((int(cx + rx), int(cy + ry)))
            if len(pts) >= 3:
                pygame.draw.lines(surf, col, True, pts, 2)
        pygame.draw.circle(surf, col, (cx, cy), 3)

    def _draw_dm_icon(self, surf, cx, cy, r, col):
        pygame.draw.circle(surf, col, (cx, cy), r, 2)
        # Simplified face: crosshairs
        pygame.draw.line(surf, col, (cx - 7, cy), (cx + 7, cy), 1)
        pygame.draw.line(surf, col, (cx, cy - 7), (cx, cy + 7), 1)
        pygame.draw.circle(surf, col, (cx, cy), 4, 1)

    # ── Telemetry Panel ──────────────────────────────────────────────────

    def _telemetry(self, rect: pygame.Rect, st: DrivingState):
        w, h = rect.width, rect.height
        surf = pygame.Surface((w, h))
        surf.fill(PANEL_BG)
        x0, y = 15, 10

        # Header + current speed (shown in the UI, not on the camera)
        tl = self.f_body.render("TELEMETRY", True, WHITE)
        surf.blit(tl, (x0, y))
        spd = self.f_med.render(f"{int(round(st.speed_mph))}", True, WHITE)
        mph = self.f_small.render("mph", True, DIM)
        surf.blit(spd, (w - spd.get_width() - mph.get_width() - 20, y - 2))
        surf.blit(mph, (w - mph.get_width() - 15, y + 4))
        y += 30

        # Steering wheel
        wh_cx = w // 2
        wh_cy = y + 105
        wh_r = 85
        self._draw_wheel(surf, wh_cx, wh_cy, wh_r, st.steer_actual)

        # Nav arrows
        arr_l = self.f_large.render("<", True, DIM)
        arr_r = self.f_large.render(">", True, DIM)
        surf.blit(arr_l, (x0, wh_cy - arr_l.get_height() // 2))
        surf.blit(arr_r, (w - arr_r.get_width() - 15, wh_cy - arr_r.get_height() // 2))

        y = wh_cy + wh_r + 20

        # Steering data
        rows = [
            ("TARGET", st.steer_target, ORANGE),
            ("APPLIED", st.steer_applied, ORANGE),
            ("DELTA", st.steer_delta, GREEN_T),
            ("ACTUAL", st.steer_actual, WHITE),
        ]
        for label, val, col in rows:
            ll = self.f_body.render(label, True, DIM)
            vt = self.f_mono.render(f"{val:+.1f} deg", True, col)
            surf.blit(ll, (x0, y))
            surf.blit(vt, (w // 2 + 10, y))
            y += 28

        # Hands
        hl = self.f_body.render("HANDS", True, DIM)
        hv = self.f_mono.render("ON WHEEL" if st.hands_on else "OFF WHEEL", True, WHITE)
        surf.blit(hl, (x0, y))
        surf.blit(hv, (w // 2 + 10, y))
        y += 38

        # Separator
        pygame.draw.line(surf, (35, 45, 60), (x0, y), (w - 15, y))
        y += 12

        # Driver / Openpilot status (sunnypilot MADS-aware)
        dl = self.f_body.render("DRIVER", True, DIM)
        status_map = {
            "engaged":    ("OPENPILOT", GREEN_T),
            "lat_only":   ("OP LATERAL", CYAN_T),
            "long_only":  ("OP LONG", CYAN_T),
            "override":   ("OVERRIDE", ORANGE),
            "disengaged": ("HUMAN", ORANGE),
        }
        txt, col = status_map.get(st.engage_status, ("HUMAN", ORANGE))
        dv = self.f_body.render(txt, True, col)
        surf.blit(dl, (x0, y))
        surf.blit(dv, (w // 2 + 10, y))
        y += 28

        # Gas / Throttle
        gl = self.f_body.render("GAS", True, DIM)
        gt = self.f_body.render("THROTTLE", True, DIM)
        gv = self.f_mono.render(f"{int(st.throttle)}%", True, WHITE)
        surf.blit(gl, (x0, y))
        surf.blit(gt, (w // 2 + 10, y))
        surf.blit(gv, (w - gv.get_width() - 15, y))
        y += 22
        # Gas digital state (gasPressed) + analog bar
        gs = self.f_small.render("PRESSED" if st.gas_pressed else "off", True,
                                 GREEN_T if st.gas_pressed else DIM)
        surf.blit(gs, (x0, y))
        bar_x = w // 2 + 10
        bar_w = w - bar_x - 15
        bar_h = 12
        pygame.draw.rect(surf, BAR_BG, (bar_x, y + 2, bar_w, bar_h), border_radius=2)
        fill_w = int(bar_w * min(1, st.throttle / 25))
        if fill_w > 0:
            pygame.draw.rect(surf, BAR_GREEN, (bar_x, y + 2, fill_w, bar_h), border_radius=2)
        y += 26

        # Brake
        bl = self.f_body.render("BRAKE", True, DIM)
        bt = self.f_body.render("BRAKE", True, DIM)
        bv = self.f_mono.render(f"{int(st.brake)}%", True, WHITE)
        surf.blit(bl, (x0, y))
        surf.blit(bt, (w // 2 + 10, y))
        surf.blit(bv, (w - bv.get_width() - 15, y))
        y += 22
        bs = self.f_small.render("PRESSED" if st.brake_pressed else "off", True,
                                 RED_T if st.brake_pressed else DIM)
        surf.blit(bs, (x0, y))
        pygame.draw.rect(surf, BAR_BG, (bar_x, y + 2, bar_w, bar_h), border_radius=2)
        bfill = int(bar_w * min(1, st.brake / 30))
        if bfill > 0:
            pygame.draw.rect(surf, BAR_ORANGE, (bar_x, y + 2, bfill, bar_h), border_radius=2)
        y += 32

        # Separator
        pygame.draw.line(surf, (35, 45, 60), (x0, y), (w - 15, y))
        y += 12

        # Acceleration row
        accel_items = [
            ("A EGO", st.a_ego),
            ("A TARGET", st.a_target),
            ("CMD", st.a_cmd),
            ("OUT", st.a_out),
        ]
        col_w = (w - 30) // 4
        for i, (lbl, val) in enumerate(accel_items):
            ax = x0 + i * col_w
            al_surf = self.f_tiny.render(lbl, True, DIM)
            av_surf = self.f_mono_sm.render(f"{val:+.2f}", True, WHITE)
            surf.blit(al_surf, (ax, y))
            surf.blit(av_surf, (ax, y + 16))

        self.screen.blit(surf, rect.topleft)

    def _draw_wheel(self, surf, cx, cy, r, angle_deg):
        # Negative: positive steeringAngleDeg is a left turn (wheel turns left)
        angle = math.radians(angle_deg * -4)  # Amplify visual rotation

        # Outer ring
        pygame.draw.circle(surf, WHEEL_COL, (cx, cy), r, 6)

        # Inner dark fill
        pygame.draw.circle(surf, WHEEL_BG, (cx, cy), r - 6)

        # Redraw rim on top
        pygame.draw.circle(surf, WHEEL_COL, (cx, cy), r, 5)

        # Three spokes
        hub_r = r * 0.28
        spoke_base_angles = [-90, 150, 30]
        for ba in spoke_base_angles:
            a = math.radians(ba) + angle
            x1 = cx + hub_r * math.cos(a)
            y1 = cy + hub_r * math.sin(a)
            x2 = cx + (r - 8) * math.cos(a)
            y2 = cy + (r - 8) * math.sin(a)
            # Draw as thick line
            perp = a + math.pi / 2
            hw = 7
            pts = [
                (x1 + hw * math.cos(perp), y1 + hw * math.sin(perp)),
                (x1 - hw * math.cos(perp), y1 - hw * math.sin(perp)),
                (x2 - hw * 0.6 * math.cos(perp), y2 - hw * 0.6 * math.sin(perp)),
                (x2 + hw * 0.6 * math.cos(perp), y2 + hw * 0.6 * math.sin(perp)),
            ]
            pygame.draw.polygon(surf, WHEEL_COL, [(int(p[0]), int(p[1])) for p in pts])

        # Center hub
        pygame.draw.circle(surf, WHEEL_COL, (cx, cy), int(hub_r), 0)
        pygame.draw.circle(surf, WHEEL_BG, (cx, cy), int(hub_r) - 4, 0)

        # Angle indicator dot on rim (top position + current angle)
        ind_a = math.radians(-90) + angle
        ix = cx + (r - 3) * math.cos(ind_a)
        iy = cy + (r - 3) * math.sin(ind_a)
        pygame.draw.circle(surf, CYAN_T, (int(ix), int(iy)), 5)

        # Opposite indicator (smaller)
        ind_b = ind_a + math.pi
        ix2 = cx + (r - 3) * math.cos(ind_b)
        iy2 = cy + (r - 3) * math.sin(ind_b)
        pygame.draw.circle(surf, CYAN_T, (int(ix2), int(iy2)), 3)

    # ── Top-Down Radar View ──────────────────────────────────────────────

    def _topdown(self, rect: pygame.Rect, st: DrivingState):
        w, h = rect.width, rect.height
        surf = pygame.Surface((w, h))
        surf.fill((10, 14, 22))

        # "CONFIDENCE" label
        cl = self.f_small.render("CONFIDENCE", True, DIM)
        surf.blit(cl, (w // 2 - cl.get_width() // 2, 6))

        # Confidence ball (vertical): high conf -> top, low -> bottom.
        # Colour matches the comma 4: green (high) / amber (mid) / red (low)
        # when engaged; grey/white for other states.
        bar_x = 22
        bar_top = 32
        bar_bot = h - 25
        bar_len = bar_bot - bar_top
        pygame.draw.line(surf, (35, 45, 55), (bar_x, bar_top), (bar_x, bar_bot), 2)
        conf = max(0.0, min(1.0, st.confidence))
        conf_y = int(bar_top + bar_len * (1 - conf))
        status = st.engage_status
        if status == "engaged":
            if conf > 0.5:
                ball = (0, 255, 120)        # green
            elif conf > 0.2:
                ball = (255, 180, 0)        # amber
            else:
                ball = (255, 30, 45)        # red
        elif status in ("lat_only", "long_only"):
            ball = CYAN_T
        elif status == "override":
            ball = (235, 235, 235)
        else:  # disengaged
            ball = (90, 95, 105)
            conf_y = bar_bot            # park at the bottom when disengaged
        pygame.draw.circle(surf, ball, (bar_x, conf_y), 9)
        pygame.draw.circle(surf, (10, 14, 22), (bar_x, conf_y), 9, 2)

        # Radar area
        radar_x0 = 48
        radar_w = w - radar_x0 - 8
        radar_h = h - 50
        radar_y0 = 30

        # Ego position (center-bottom of radar area)
        ego_px = radar_x0 + radar_w // 2
        ego_py = radar_y0 + radar_h - 35
        DRAW_M = 90.0  # metres shown ahead
        scale = (radar_h - 15) / DRAW_M

        clip = pygame.Rect(radar_x0, radar_y0, radar_w, radar_h)
        prevclip = surf.get_clip()
        surf.set_clip(clip)

        def bev(x_fwd, y_lat):
            return (int(ego_px + y_lat * scale), int(ego_py - x_fwd * scale))

        def polyline(pts3d, color, width):
            if pts3d is None or len(pts3d) < 2:
                return
            sp = [bev(p[0], p[1]) for p in pts3d if 0 <= p[0] <= DRAW_M]
            if len(sp) >= 2:
                pygame.draw.lines(surf, color, False, sp, width)

        model = st.model
        if model:
            # distance rings
            for d in (20, 40, 60, 80):
                ry = int(ego_py - d * scale)
                if ry > radar_y0:
                    pygame.draw.line(surf, (28, 36, 48), (radar_x0 + 4, ry),
                                     (radar_x0 + radar_w - 4, ry), 1)
            # road edges, lane lines, path (bird's eye)
            for e in model.get("edges", []):
                polyline(e, (170, 60, 60), 2)
            probs = model.get("lane_probs", [])
            for i, ll in enumerate(model.get("lanes", [])):
                p = probs[i] if i < len(probs) else 0.0
                c = int(90 + 140 * min(p, 1.0))
                polyline(ll, (c, c, c), 2)
            path = model.get("path")
            if path is not None and len(path) >= 2:
                band = []
                pts = [p for p in path if 0 <= p[0] <= DRAW_M]
                for p in pts:
                    band.append(bev(p[0], p[1] - 0.9))
                for p in reversed(pts):
                    band.append(bev(p[0], p[1] + 0.9))
                if len(band) >= 3:
                    ps = pygame.Surface((w, h), pygame.SRCALPHA)
                    col = (16, 245, 120, 90) if st.op_active else (90, 90, 90, 80)
                    pygame.draw.polygon(ps, col, band)
                    surf.blit(ps, (0, 0))
        else:
            # Demo fallback: scatter + synthetic trajectory
            for px, py in st.radar_pts:
                sx, sy = int(ego_px - px * scale), int(ego_py - py * scale)
                if clip.collidepoint(sx, sy):
                    pygame.draw.circle(surf, (170, 180, 195), (sx, sy), 2)

        # Lead vehicles (from radar), drawn as cars
        def draw_lead(dist, ylat, col):
            if dist <= 0:
                return
            sx, sy = bev(dist, ylat)
            if clip.collidepoint(sx, sy):
                pygame.draw.rect(surf, col, (sx - 7, sy - 10, 14, 20), border_radius=2)
                pygame.draw.rect(surf, (10, 14, 22), (sx - 7, sy - 10, 14, 20), 1, border_radius=2)
        if st.lead_dist > 0:
            draw_lead(st.lead_dist, st.lead_y, OTHER_CAR)

        surf.set_clip(prevclip)

        # Ego car (yellow)
        pygame.draw.rect(surf, EGO_YELLOW, (ego_px - 6, ego_py - 11, 12, 22), border_radius=2)

        # Road / intersection info along the bottom (from map + nav data)
        iy = h - 20
        info = ""
        if st.nav_maneuver:
            d = f"{int(st.nav_dist)} m  " if st.nav_dist > 0 else ""
            info = f"{d}{st.nav_maneuver}"
            if st.nav_street:
                info += f" → {st.nav_street}"
        elif st.road_name:
            info = st.road_name
            if st.map_speed_limit > 0:
                info += f"   •  limit {int(round(st.map_speed_limit * 0.621371))} mph"
        if info:
            t = self.f_tiny.render(info[:46], True, (150, 165, 185))
            surf.blit(t, (radar_x0, iy))

        self.screen.blit(surf, rect.topleft)

    # ── OSM Map ───────────────────────────────────────────────────────────

    def _map(self, rect: pygame.Rect, st: DrivingState):
        """Location panel: OSM map tiles (downloaded on the fly) under the GPS
        trace + current position. Falls back to a trace-only view offline."""
        surf = pygame.Surface((rect.width, rect.height))
        surf.fill((16, 22, 32))
        w, h = rect.width, rect.height

        title = self.f_small.render("LOCATION", True, GREEN_T)
        surf.blit(title, (10, 8))
        hint = self.f_tiny.render("M: hide", True, DIM)
        surf.blit(hint, (w - hint.get_width() - 10, 10))

        track = st.gps_track or []
        pad = 16
        mx0, my0 = pad, 30
        mw, mh = w - 2 * pad, h - 78
        map_area = pygame.Rect(mx0, my0, mw, mh)
        have_loc = len(track) >= 2 or (st.lat or st.lon)

        if not have_loc:
            nd = self.f_small.render("No GPS in log", True, DIM)
            surf.blit(nd, nd.get_rect(center=map_area.center))
        else:
            clat_lat = st.lat or (track[len(track) // 2][1] if track else 0.0)
            clon = st.lon or (track[len(track) // 2][2] if track else 0.0)
            drew_tiles = False
            if self.map_tiles and (clat_lat or clon):
                drew_tiles = self._draw_tiles(surf, map_area, clat_lat, clon, track, st)
            if not drew_tiles:
                self._draw_trace(surf, map_area, track, st)

        # road name / coordinates footer
        y = h - 40
        pygame.draw.line(surf, (35, 45, 60), (10, y), (w - 10, y))
        if st.road_name:
            rn = self.f_body.render(st.road_name[:30], True, WHITE)
            surf.blit(rn, (10, y + 6))
        if st.lat or st.lon:
            coord = self.f_tiny.render(f"{st.lat:.4f}, {st.lon:.4f}  ·  "
                                       f"© OpenStreetMap", True, DIM)
            surf.blit(coord, (10, y + 24))

        self.screen.blit(surf, rect.topleft)
        pygame.draw.rect(self.screen, (35, 45, 60), rect, 1)

    def _draw_tiles(self, surf, area, lat, lon, track, st):
        """Draw OSM tiles centered on (lat,lon). Returns True if any tile drawn."""
        if self._tilemap is None:
            self._tilemap = TileMap()
        tm = self._tilemap
        z = 15
        cen_x, cen_y = TileMap.deg2tile(lat, lon, z)
        T = TileMap.TILE
        prev = surf.get_clip()
        surf.set_clip(area)
        acx, acy = area.centerx, area.centery
        drew = False
        cols = int(area.width / T) + 2
        rows = int(area.height / T) + 2
        for dx in range(-cols, cols + 1):
            for dy in range(-rows, rows + 1):
                tx = int(cen_x) + dx
                ty = int(cen_y) + dy
                if tx < 0 or ty < 0 or tx >= 2 ** z or ty >= 2 ** z:
                    continue
                sx = acx + (tx - cen_x) * T
                sy = acy + (ty - cen_y) * T
                if sx > area.right or sy > area.bottom or sx + T < area.left or sy + T < area.top:
                    continue
                tile = tm.get(z, tx, ty)
                if tile is not None:
                    surf.blit(tile, (int(sx), int(sy)))
                    drew = True

        def ll_px(la, lo):
            x, y = TileMap.deg2tile(la, lo, z)
            return (int(acx + (x - cen_x) * T), int(acy + (y - cen_y) * T))

        if drew and len(track) >= 2:
            pts = [ll_px(p[1], p[2]) for p in track]
            pygame.draw.lines(surf, (40, 120, 230), False, pts, 3)
        if drew and (st.lat or st.lon):
            cp = ll_px(st.lat, st.lon)
            pygame.draw.circle(surf, EGO_YELLOW, cp, 7)
            pygame.draw.circle(surf, (20, 25, 35), cp, 7, 2)
        surf.set_clip(prev)
        return drew

    def _draw_trace(self, surf, area, track, st):
        """Fallback trace-only view when tiles are unavailable (offline)."""
        mx0, my0, mw, mh = area.x, area.y, area.width, area.height
        if len(track) >= 2:
            lats = [p[1] for p in track]
            lons = [p[2] for p in track]
            la0, la1 = min(lats), max(lats)
            lo0, lo1 = min(lons), max(lons)
            clat = math.cos(math.radians((la0 + la1) / 2))
            span_lo = max((lo1 - lo0) * clat, 1e-6)
            span_la = max(la1 - la0, 1e-6)
            sc = min(mw / span_lo, mh / span_la) * 0.9
            cx, cy = area.center
            mid_lo = (lo0 + lo1) / 2
            mid_la = (la0 + la1) / 2

            def to_px(la, lo):
                return (int(cx + (lo - mid_lo) * clat * sc), int(cy - (la - mid_la) * sc))
            pygame.draw.lines(surf, (70, 130, 180), False,
                              [to_px(p[1], p[2]) for p in track], 2)
            if st.lat or st.lon:
                cp = to_px(st.lat, st.lon)
                pygame.draw.circle(surf, EGO_YELLOW, cp, 6)
                pygame.draw.circle(surf, (20, 25, 35), cp, 6, 2)
        else:
            pygame.draw.circle(surf, EGO_YELLOW, area.center, 6)

    # ── Footer ───────────────────────────────────────────────────────────

    def _footer(self, paused: bool, speed: float):
        y = WINDOW_H - FOOTER_H + 5
        pygame.draw.line(self.screen, (35, 45, 60), (0, y - 5), (WINDOW_W, y - 5))

        # Attribution
        t1 = self.f_small.render("Comma 4 Drive Visualizer  •  inspired by", True, DIM)
        self.screen.blit(t1, (MARGIN, y + 5))

        badge = self.f_small.render(" ui-alt ", True, WHITE)
        bx = MARGIN + t1.get_width() + 6
        br = pygame.Rect(bx, y + 4, badge.get_width() + 4, badge.get_height() + 2)
        pygame.draw.rect(self.screen, (45, 55, 70), br, border_radius=3)
        self.screen.blit(badge, (bx + 2, y + 5))

        t2 = self.f_small.render("+ op-replay-clipper", True, DIM)
        self.screen.blit(t2, (br.right + 6, y + 5))

    def _controls_hint(self, paused: bool, speed: float):
        y = WINDOW_H - FOOTER_H + 5
        status_parts = []
        if paused:
            status_parts.append("PAUSED")
        if abs(speed - 1.0) > 0.01:
            status_parts.append(f"{speed:.1f}x")
        status_parts.append("Menus at top  •  Space play  •  drag timeline to scrub")
        status = "  •  ".join(status_parts)
        st = self.f_tiny.render(status, True, (80, 90, 105))
        self.screen.blit(st, (WINDOW_W - st.get_width() - MARGIN, y + 22))

    def draw_scrubber(self, t, total, hovering=False, in_pt=None, out_pt=None):
        """Draw the interactive timeline. Returns the clickable bar rect."""
        r = scrub_rect()
        total = max(total, 0.001)
        frac = max(0.0, min(1.0, t / total))

        pygame.draw.rect(self.screen, (28, 36, 50), r, border_radius=5)
        # In/out selection band
        if in_pt is not None or out_pt is not None:
            a = (in_pt or 0.0) / total
            b = (out_pt if out_pt is not None else total) / total
            sx = r.x + int(r.width * max(0, min(1, a)))
            ex = r.x + int(r.width * max(0, min(1, b)))
            pygame.draw.rect(self.screen, (60, 90, 120), (sx, r.y, max(2, ex - sx), r.height))
            for px in (sx, ex):
                pygame.draw.line(self.screen, WHITE, (px, r.y - 3), (px, r.bottom + 3), 2)
        fill = pygame.Rect(r.x, r.y, int(r.width * frac), r.height)
        pygame.draw.rect(self.screen, CAM_BORDER, fill, border_radius=5)

        hx = r.x + int(r.width * frac)
        pygame.draw.circle(self.screen, WHITE, (hx, r.centery), 8 if hovering else 6)
        pygame.draw.circle(self.screen, CAM_BORDER, (hx, r.centery), 8 if hovering else 6, 2)

        def fmt(s):
            s = max(0, int(s))
            return f"{s // 60:d}:{s % 60:02d}"
        cur = self.f_small.render(fmt(t), True, WHITE)
        tot = self.f_small.render(fmt(total), True, DIM)
        self.screen.blit(cur, cur.get_rect(midright=(r.x - 10, r.centery)))
        self.screen.blit(tot, tot.get_rect(midleft=(r.right + 10, r.centery)))
        return r


# ── Main Loop ────────────────────────────────────────────────────────────────

def _ffmpeg_exe():
    """Resolve an ffmpeg binary: bundled (imageio-ffmpeg) first, else PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


_GPU_ENCODER = None

def _encoder_runs(enc):
    """True if ffmpeg can actually encode one frame with `enc` on this machine.

    Listing an encoder (ffmpeg -encoders) does NOT mean it works: a build lists
    h264_nvenc even with no NVIDIA GPU. A tiny real encode is the only reliable
    cross-vendor test, so this works correctly on AMD / NVIDIA / Intel / Apple.
    """
    import subprocess
    cmd = [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=128x128:r=5:d=1",
           "-c:v", enc, "-frames:v", "3", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=25)
        return r.returncode == 0
    except Exception:
        return False


def gpu_encoder():
    """Return a hardware H.264 encoder that actually works here, or None (CPU).

    Probes each vendor encoder with a real 3-frame test encode and caches the
    first that succeeds — so it auto-adapts to whatever GPU is present.
    """
    global _GPU_ENCODER
    if _GPU_ENCODER is not None:
        return _GPU_ENCODER or None
    import subprocess
    try:
        listed = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-encoders"],
                                capture_output=True, text=True, timeout=20).stdout
    except Exception:
        _GPU_ENCODER = ""
        return None
    # Order by prevalence; each is verified to actually run before being chosen.
    for enc in ("h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox"):
        if enc in listed and _encoder_runs(enc):
            _GPU_ENCODER = enc
            return enc
    _GPU_ENCODER = ""
    return None


@dataclass
class ExportConfig:
    out_path: str = ""
    content: str = "dashboard"   # "dashboard" | "camera"
    camera: str = "road"         # for content=="camera": "road" | "wide"
    use_gpu: bool = True
    show_osm: bool = True
    scale: int = 2               # supersample factor for dashboard export
    t_start: float = 0.0
    t_end: float = 0.0           # 0 => full route
    max_perf: bool = True        # use all CPU/GPU for fastest render
    all_cores: bool = False      # multi-process render across all CPU cores


def _default_export_path(route_dir):
    base = os.path.basename(os.path.normpath(route_dir)) if route_dir else "comma4"
    docs = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(docs):
        docs = os.path.expanduser("~")
    return os.path.join(docs, f"{base}_comma4viz.mp4")


def _encoder_args(use_gpu, max_perf=True):
    """ffmpeg codec args for GPU (if available) or CPU, plus a human label."""
    if use_gpu:
        enc = gpu_encoder()
        if enc == "h264_amf":
            q = "speed" if max_perf else "quality"
            return ["-c:v", "h264_amf", "-quality", q, "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"], "GPU (AMD AMF)"
        if enc == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p4" if max_perf else "p6", "-cq", "20"], "GPU (NVENC)"
        if enc == "h264_qsv":
            return ["-c:v", "h264_qsv", "-global_quality", "20"], "GPU (QSV)"
        if enc == "h264_videotoolbox":
            return ["-c:v", "h264_videotoolbox", "-q:v", "55"], "GPU (VideoToolbox)"
    preset = "veryfast" if max_perf else "medium"
    return ["-threads", "0", "-c:v", "libx264", "-crf", "18", "-preset", preset], "CPU (x264)"


def _render_chunk_worker(task):
    """Worker process: render a time-chunk of a route to a temp mp4.

    Each worker loads its own route + decoders (processes can't share them) and
    encodes its slice with libx264 veryfast so chunks concat without re-encoding.
    Puts frame-progress ints on the queue. Returns (idx, out_path) or (idx, None).
    """
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import subprocess
    (idx, route_dir, route_info, cfg_d, c_t0, c_t1,
     src_w, src_h, out_w, out_h, tmp_path, q) = task
    try:
        pygame.init()
        screen = pygame.display.set_mode((src_w, src_h))
        cfg = ExportConfig(**cfg_d)
        source = RouteDataSource(route_dir, use_hw=cfg.use_gpu)
        renderer = Renderer(screen)
        renderer.show_osm = cfg.show_osm
        renderer.map_tiles = False  # avoid per-worker network stalls
        export_proj = None
        cam_surf = None
        if cfg.content == "camera":
            export_proj = Projector(source.route.calib, src_w, src_h, source.route.cam_config)
            cam_surf = pygame.Surface((src_w, src_h))

        cmd = [_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{src_w}x{src_h}", "-r", str(FPS), "-i", "-"]
        if (out_w, out_h) != (src_w, src_h):
            cmd += ["-vf", f"scale={out_w}:{out_h}:flags=bilinear"]
        cmd += ["-threads", "1", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", tmp_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        n = int(round((c_t1 - c_t0) * FPS))
        for i in range(n):
            t = c_t0 + i / FPS
            state = source.generate(t)
            rf = source.get_road_frame_exact(t)
            wf = source.get_wide_frame_exact(t)
            if cfg.content == "camera":
                wide = cfg.camera == "wide"
                vid = wf if wide else rf
                if vid is not None:
                    cam_surf.blit(pygame.transform.scale(vid, (src_w, src_h)), (0, 0))
                else:
                    cam_surf.fill((16, 20, 28))
                state.projector = export_proj
                renderer._camera_overlays(cam_surf, src_w, src_h,
                                          1.4 if wide else 1.0, state,
                                          "WIDE" if wide else "ROAD")
                proc.stdin.write(pygame.image.tostring(cam_surf, "RGB"))
            else:
                renderer.draw(state, False, 1.0, rf, wf, route_info)
                proc.stdin.write(pygame.image.tostring(screen, "RGB"))
            if q is not None and i % 3 == 0:
                try:
                    q.put(3)
                except Exception:
                    pass
        proc.stdin.close()
        proc.wait()
        pygame.quit()
        return (idx, tmp_path if proc.returncode in (0, None) else None)
    except Exception:
        return (idx, None)


def export_route_parallel(screen, source, route_info, max_time, cfg: ExportConfig):
    """Render across all CPU cores via worker processes, then concat. Fallback-safe."""
    import multiprocessing as mp
    import subprocess, tempfile, time as _time

    route_dir = source.route.route_dir
    t0 = max(0.0, cfg.t_start)
    t1 = cfg.t_end if cfg.t_end > t0 else max_time
    if cfg.content == "camera":
        nW, nH, _ = source.route.cam_config["wide" if cfg.camera == "wide" else "narrow"]
        src_w = out_w = 1920
        src_h = out_h = int(round(out_w * nH / nW / 2) * 2)
    else:
        src_w, src_h = WINDOW_W, WINDOW_H
        out_w, out_h = WINDOW_W * cfg.scale, WINDOW_H * cfg.scale

    ncpu = max(2, min(mp.cpu_count(), 8))
    total_frames = max(1, int((t1 - t0) * FPS))
    nchunks = min(ncpu, max(1, total_frames // 20))
    bounds = [t0 + (t1 - t0) * k / nchunks for k in range(nchunks + 1)]

    # 'spawn' = fresh processes (fork-after-SDL-init hangs); works in the frozen exe too.
    ctx = mp.get_context("spawn")
    tmpdir = tempfile.mkdtemp(prefix="comma4_par_")
    manager = ctx.Manager()
    q = manager.Queue()
    tasks = []
    cfg_d = cfg.__dict__.copy()
    for k in range(nchunks):
        tmp_path = os.path.join(tmpdir, f"chunk_{k:03d}.mp4")
        tasks.append((k, route_dir, route_info, cfg_d, bounds[k], bounds[k + 1],
                      src_w, src_h, out_w, out_h, tmp_path, q))

    start = _time.time()
    pool = ctx.Pool(processes=nchunks)
    result = pool.map_async(_render_chunk_worker, tasks)
    pool.close()

    done = 0
    while not result.ready():
        try:
            while True:
                done += q.get_nowait()
        except Exception:
            pass
        frac = min(0.98, done / total_frames)
        elapsed = _time.time() - start
        eta = (elapsed / frac - elapsed) if frac > 0.03 else 0
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pool.terminate()
                return False, "Export cancelled"
        _loading_screen(screen, "Exporting (all cores)…",
                        f"{done}/{total_frames} frames   {nchunks} cores   "
                        f"elapsed {int(elapsed)}s   ETA {int(eta)}s", frac)
        pygame.display.flip()
        _time.sleep(0.1)

    results = result.get()
    pool.join()
    parts = [p for _, p in sorted(results) if p]
    if len(parts) != nchunks:
        return False, "A render worker failed; try without 'all cores'."

    # concat chunks (stream copy)
    listf = os.path.join(tmpdir, "list.txt")
    with open(listf, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    concat = [_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", listf,
              "-c", "copy", cfg.out_path]
    r = subprocess.run(concat, capture_output=True)
    if r.returncode != 0:
        return False, "Concatenation failed; try without 'all cores'."
    return True, cfg.out_path


def export_route(screen, source, route_info, max_time, cfg: ExportConfig):
    """Render a route (or a range) to mp4 per the ExportConfig. Returns (ok, msg)."""
    import subprocess, time as _time

    if cfg.all_cores and hasattr(source, "route"):
        try:
            ok, msg = export_route_parallel(screen, source, route_info, max_time, cfg)
            if ok:
                return ok, msg
            # fall through to single-process on failure
        except Exception:
            pass

    t0 = max(0.0, cfg.t_start)
    t1 = cfg.t_end if cfg.t_end > t0 else max_time

    # Output geometry + a per-size projector for camera-only ("sim recording").
    if cfg.content == "camera":
        cam_cfg = source.route.cam_config
        nW, nH, _ = cam_cfg["wide" if cfg.camera == "wide" else "narrow"]
        out_w = 1920
        out_h = int(round(out_w * nH / nW / 2) * 2)
        src_w, src_h = out_w, out_h
        export_proj = Projector(source.route.calib, out_w, out_h, cam_cfg)
    else:
        src_w, src_h = WINDOW_W, WINDOW_H
        out_w, out_h = WINDOW_W * cfg.scale, WINDOW_H * cfg.scale
        export_proj = None

    codec, enc_label = _encoder_args(cfg.use_gpu, cfg.max_perf)
    cmd = [_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{src_w}x{src_h}", "-r", str(FPS), "-i", "-"]
    if (out_w, out_h) != (src_w, src_h):
        cmd += ["-vf", f"scale={out_w}:{out_h}:flags=lanczos"]
    cmd += codec + ["-pix_fmt", "yuv420p", cfg.out_path]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except Exception as e:
        return False, f"ffmpeg not available: {e}"

    renderer = Renderer(screen)
    renderer.show_osm = cfg.show_osm
    cam_surf = pygame.Surface((src_w, src_h)) if cfg.content == "camera" else None

    total = max(1, int((t1 - t0) * FPS))
    start_wall = _time.time()

    # Decode-ahead producer so GPU decode overlaps CPU render + GPU encode.
    import queue as _queue, threading as _threading
    need_road = cfg.content != "camera" or cfg.camera == "road"
    need_wide = cfg.content != "camera" or cfg.camera == "wide"
    fq = _queue.Queue(maxsize=16)
    stop_flag = {"stop": False}

    def _producer():
        for i in range(total + 1):
            if stop_flag["stop"]:
                break
            t = t0 + i / FPS
            rf = source.get_road_frame_exact(t) if need_road else None
            wf = source.get_wide_frame_exact(t) if need_wide else None
            fq.put((i, t, rf, wf))
        fq.put(None)

    _threading.Thread(target=_producer, daemon=True).start()

    try:
        while True:
            item = fq.get()
            if item is None:
                break
            i, t, rf, wf = item
            state = source.generate(t)

            if cfg.content == "camera":
                wide = cfg.camera == "wide"
                vid = wf if wide else rf
                if vid is not None:
                    cam_surf.blit(pygame.transform.scale(vid, (src_w, src_h)), (0, 0))
                else:
                    cam_surf.fill((16, 20, 28))
                state.projector = export_proj
                renderer._camera_overlays(cam_surf, src_w, src_h,
                                          1.4 if wide else 1.0, state,
                                          "WIDE" if wide else "ROAD")
                proc.stdin.write(pygame.image.tostring(cam_surf, "RGB"))
            else:
                renderer.draw(state, False, 1.0, rf, wf, route_info)
                proc.stdin.write(pygame.image.tostring(screen, "RGB"))

            if i % 2 == 0:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        stop_flag["stop"] = True
                        try:
                            while fq.get_nowait() is not None:
                                pass
                        except Exception:
                            pass
                        proc.stdin.close(); proc.wait()
                        return False, "Export cancelled"
                elapsed = _time.time() - start_wall
                frac = i / total
                if frac > 0.03:
                    eta = max(0, elapsed / frac - elapsed)
                    eta_s = f"ETA {int(eta)}s"
                else:
                    eta_s = "ETA estimating…"
                fps_now = i / elapsed if elapsed > 0.5 else 0
                clip_s = (t1 - t0)
                msg = (f"frame {i}/{total}  ({_fmt_t(t - t0)} / {_fmt_t(clip_s)})   "
                       f"{enc_label}   {fps_now:.0f} fps   "
                       f"elapsed {int(elapsed)}s   {eta_s}")
                _loading_screen(screen, "Exporting video…", msg, frac)
                pygame.display.flip()
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        try:
            proc.kill()
        except Exception:
            pass
        if cfg.use_gpu:
            return False, "GPU encoder failed to start. Re-open export (E) and switch Encoder to CPU."
        return False, "Encoder failed (broken pipe)."
    except Exception as e:
        try:
            proc.stdin.close(); proc.kill()
        except Exception:
            pass
        return False, f"Export failed: {e}"
    if proc.returncode not in (0, None):
        hint = "  Try switching Encoder to CPU in the export dialog." if cfg.use_gpu else ""
        return False, f"ffmpeg exited with code {proc.returncode}.{hint}"
    return True, cfg.out_path


def record_video(output_path="comma4_demo.mp4", duration=30.0):
    """Record a demo video (requires ffmpeg)."""
    import subprocess, tempfile, os, shutil

    tmpdir = tempfile.mkdtemp(prefix="comma4_")
    print(f"Recording {duration}s at {FPS}fps -> {output_path}")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

    gen = DemoGenerator()
    renderer = Renderer(screen)

    total_frames = int(duration * FPS)
    for i in range(total_frames):
        t = i / FPS
        state = gen.generate(t)
        renderer.draw(state, False, 1.0)
        frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
        pygame.image.save(screen, frame_path)
        if (i + 1) % 50 == 0 or i == total_frames - 1:
            print(f"  frame {i + 1}/{total_frames}")

    pygame.quit()

    print("Encoding with ffmpeg...")
    subprocess.run([
        _ffmpeg_exe(), "-y", "-framerate", str(FPS),
        "-i", os.path.join(tmpdir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        output_path
    ], check=True, capture_output=True)

    shutil.rmtree(tmpdir)
    print(f"Saved: {output_path}")


def _browse_folder():
    """Open a native folder picker dialog and return the selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select openpilot route folder")
        root.destroy()
        return folder if folder else None
    except Exception:
        return None


def _draw_button(screen, font, text, rect, mouse_pos, color=CAM_BORDER):
    """Draw a rounded button, highlighted when hovered."""
    hovered = rect.collidepoint(mouse_pos)
    fill = color if hovered else (30, 40, 55)
    border = color
    pygame.draw.rect(screen, fill, rect, border_radius=8)
    pygame.draw.rect(screen, border, rect, 2, border_radius=8)
    txt_col = (13, 18, 28) if hovered else color
    lbl = font.render(text, True, txt_col)
    screen.blit(lbl, lbl.get_rect(center=rect.center))
    return hovered


def _loading_screen(screen, title, message="", frac=None):
    """Show a loading screen with a progress bar.

    frac in [0,1] draws a determinate bar; None draws an indeterminate sweep.
    Pumps the event queue so the window stays responsive during a long load.
    """
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    f_lg = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
    f_sm = pygame.font.SysFont("DejaVu Sans", 18)

    screen.fill(BG)
    t = f_lg.render(title, True, WHITE)
    screen.blit(t, t.get_rect(centerx=cx, centery=cy - 48))

    # Progress bar
    bw, bh = 460, 16
    bx, by = cx - bw // 2, cy - 4
    pygame.draw.rect(screen, (30, 40, 55), (bx, by, bw, bh), border_radius=8)
    if frac is None:
        # Indeterminate: a moving chunk based on wall clock
        cw = int(bw * 0.28)
        span = bw - cw
        pos = int((math.sin(time.time() * 2.2) * 0.5 + 0.5) * span)
        pygame.draw.rect(screen, CAM_BORDER, (bx + pos, by, cw, bh), border_radius=8)
    else:
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(screen, CAM_BORDER, (bx, by, int(bw * frac), bh), border_radius=8)
        pct = f_sm.render(f"{int(frac * 100)}%", True, DIM)
        screen.blit(pct, pct.get_rect(centerx=cx, top=by + bh + 10))

    if message:
        m = f_sm.render(message, True, DIM)
        screen.blit(m, m.get_rect(centerx=cx, top=by + bh + 34))

    pygame.display.flip()


def export_dialog(screen, route_dir, max_time, sel_start=None, sel_end=None):
    """Interactive export settings. Returns ExportConfig or None if cancelled."""
    clock = pygame.time.Clock()
    f_lg = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
    f_md = pygame.font.SysFont("DejaVu Sans", 20)
    f_sm = pygame.font.SysFont("DejaVu Sans", 16)

    cfg = ExportConfig(out_path=_default_export_path(route_dir))
    enc = gpu_encoder()
    has_sel = sel_start is not None and sel_end is not None and sel_end > sel_start
    use_sel = has_sel

    cx = WINDOW_W // 2
    col_x = cx - 230
    val_x = cx + 20
    row_h = 46
    y0 = 180

    def rows():
        r = [
            ("Content", "Dashboard" if cfg.content == "dashboard" else "Camera only (sim recording)"),
        ]
        if cfg.content == "camera":
            r.append(("Camera", "Road (narrow)" if cfg.camera == "road" else "Wide"))
        else:
            r.append(("Supersample", f"{cfg.scale}x"))
        if has_sel:
            range_val = (f"Selection  {_fmt_t(sel_start)}–{_fmt_t(sel_end)}"
                         if use_sel else "Full drive  (click for selection)")
        else:
            range_val = "Full drive  —  set In/Out first (Playback menu)"
        r += [
            ("Range", range_val),
            ("Encoder", f"GPU  ({enc})" if (cfg.use_gpu and enc) else ("GPU (none found → CPU)" if cfg.use_gpu else "CPU (x264)")),
            ("Performance",
             "Max — ALL CORES  (fastest, PC very busy)" if cfg.all_cores
             else ("Max  (GPU + fast preset)" if cfg.max_perf else "Balanced")),
            ("Location overlay", "On" if cfg.show_osm else "Off  (privacy)"),
            ("Save to", _shorten_path(cfg.out_path)),
        ]
        return r

    start_rect = pygame.Rect(cx - 230, y0 + 8 * row_h + 10, 210, 46)
    cancel_rect = pygame.Rect(cx + 20, y0 + 8 * row_h + 10, 210, 46)

    while True:
        mouse = pygame.mouse.get_pos()
        rlist = rows()
        row_rects = [pygame.Rect(val_x, y0 + i * row_h - 4, 440, row_h - 8) for i in range(len(rlist))]

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return None
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for i, rr in enumerate(row_rects):
                    if rr.collidepoint(ev.pos):
                        label = rlist[i][0]
                        if label == "Content":
                            cfg.content = "camera" if cfg.content == "dashboard" else "dashboard"
                        elif label == "Camera":
                            cfg.camera = "wide" if cfg.camera == "road" else "road"
                        elif label == "Supersample":
                            cfg.scale = 1 if cfg.scale == 2 else 2
                        elif label == "Range" and has_sel:
                            use_sel = not use_sel
                        elif label == "Encoder":
                            cfg.use_gpu = not cfg.use_gpu
                        elif label == "Performance":
                            # cycle: Balanced -> Max -> Max all-cores
                            if not cfg.max_perf:
                                cfg.max_perf, cfg.all_cores = True, False
                            elif not cfg.all_cores:
                                cfg.all_cores = True
                            else:
                                cfg.max_perf, cfg.all_cores = False, False
                        elif label == "Location overlay":
                            cfg.show_osm = not cfg.show_osm
                        elif label == "Save to":
                            p = _browse_save(route_dir)
                            if p:
                                cfg.out_path = p
                if start_rect.collidepoint(ev.pos):
                    if use_sel:
                        cfg.t_start, cfg.t_end = sel_start, sel_end
                    else:
                        cfg.t_start, cfg.t_end = 0.0, max_time
                    return cfg
                if cancel_rect.collidepoint(ev.pos):
                    return None

        screen.fill(BG)
        title = f_lg.render("Export video", True, WHITE)
        screen.blit(title, title.get_rect(centerx=cx, top=120))
        for i, (label, val) in enumerate(rlist):
            yy = y0 + i * row_h
            ll = f_md.render(label, True, DIM)
            screen.blit(ll, (col_x, yy))
            rr = row_rects[i]
            hov = rr.collidepoint(mouse)
            pygame.draw.rect(screen, (30, 40, 55) if not hov else (45, 60, 80), rr, border_radius=6)
            pygame.draw.rect(screen, CAM_BORDER, rr, 1, border_radius=6)
            vv = f_md.render(val, True, WHITE)
            screen.blit(vv, (rr.x + 12, rr.y + 6))

        _draw_button(screen, f_md, "Start Export", start_rect, mouse)
        _draw_button(screen, f_md, "Cancel", cancel_rect, mouse, ORANGE)
        hint = f_sm.render("Click a value to change it  •  Esc to cancel", True, DIM)
        screen.blit(hint, hint.get_rect(centerx=cx, bottom=WINDOW_H - 30))
        pygame.display.flip()
        clock.tick(30)


def _fmt_t(s):
    if s is None:
        return "0:00"
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def _shorten_path(p, n=44):
    return p if len(p) <= n else "..." + p[-(n - 3):]


def _browse_save(route_dir):
    """Folder picker; returns a full mp4 path in the chosen folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        d = filedialog.askdirectory(title="Choose export folder")
        root.destroy()
        if d:
            base = os.path.basename(os.path.normpath(route_dir)) if route_dir else "comma4"
            return os.path.join(d, f"{base}_comma4viz.mp4")
    except Exception:
        pass
    return None


def _help_screen(screen):
    """Show controls and info; wait for a key/click."""
    f_lg = pygame.font.SysFont("DejaVu Sans", 28, bold=True)
    f_md = pygame.font.SysFont("DejaVu Sans", 18)
    f_sm = pygame.font.SysFont("DejaVu Sans", 16)
    cx = WINDOW_W // 2
    lines = [
        ("Everything is in the menu bar at the top.", DIM),
        ("", DIM),
        ("File → Open Route / Quit", WHITE),
        ("Playback → Play/Pause, Restart, Set In/Out, Speed", WHITE),
        ("View → Location Panel (privacy toggle)", WHITE),
        ("Export → Export Video (choose content, camera, range, GPU)", WHITE),
        ("", DIM),
        ("Shortcuts: Space play  •  ←→ scrub  •  I/O in-out  "
         "•  E export  •  M location  •  Q menu", DIM),
        ("Drag the timeline to scrub anywhere.", DIM),
    ]
    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        screen.fill(BG)
        t = f_lg.render("Controls & Info", True, WHITE)
        screen.blit(t, t.get_rect(centerx=cx, top=150))
        y = 220
        for txt, col in lines:
            if txt:
                s = f_md.render(txt, True, col)
                screen.blit(s, s.get_rect(centerx=cx, top=y))
            y += 30
        hint = f_sm.render("Press any key to go back", True, DIM)
        screen.blit(hint, hint.get_rect(centerx=cx, bottom=WINDOW_H - 40))
        pygame.display.flip()
        pygame.time.wait(20)


def _error_screen(screen, title, detail=""):
    """Show an error and wait for a keypress/click. Returns 'menu'."""
    f_lg = pygame.font.SysFont("DejaVu Sans", 28, bold=True)
    f_sm = pygame.font.SysFont("DejaVu Sans", 18)
    cx, cy = WINDOW_W // 2, WINDOW_H // 2

    # Wrap detail text to fit
    words = detail.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 70:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        screen.fill(BG)
        t = f_lg.render(title, True, RED_T)
        screen.blit(t, t.get_rect(centerx=cx, centery=cy - 60))
        for i, ln in enumerate(lines[:8]):
            m = f_sm.render(ln, True, WHITE)
            screen.blit(m, m.get_rect(centerx=cx, top=cy - 20 + i * 24))
        hint = f_sm.render("Press any key to go back", True, DIM)
        screen.blit(hint, hint.get_rect(centerx=cx, bottom=WINDOW_H - 40))
        pygame.display.flip()
        pygame.time.wait(20)
    return "menu"


def launcher_menu(screen):
    """GUI launcher menu. Returns (route_dir or None, quit)."""
    clock = pygame.time.Clock()
    cands = ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]
    font_lg = pygame.font.SysFont(None, 42)
    font_md = pygame.font.SysFont(None, 28)
    font_sm = pygame.font.SysFont(None, 20)
    for name in cands:
        try:
            font_lg = pygame.font.SysFont(name, 42, bold=True)
            font_md = pygame.font.SysFont(name, 28)
            font_sm = pygame.font.SysFont(name, 20)
            break
        except Exception:
            pass

    cx = WINDOW_W // 2
    cy = WINDOW_H // 2

    btn_w, btn_h = 320, 56
    load_rect = pygame.Rect(cx - btn_w // 2, cy - 30, btn_w, btn_h)
    demo_rect = pygame.Rect(cx - btn_w // 2, cy + 50, btn_w, btn_h)

    error_msg = ""
    error_time = 0

    while True:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None, True
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    return None, True
                elif ev.key == pygame.K_d:
                    return None, False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if load_rect.collidepoint(ev.pos):
                    folder = _browse_folder()
                    if folder:
                        if os.path.isdir(folder):
                            return folder, False
                        else:
                            error_msg = "Not a valid folder"
                            error_time = time.time()
                elif demo_rect.collidepoint(ev.pos):
                    return None, False

        screen.fill(BG)

        # Title
        title = font_lg.render("Comma 4 Drive Visualizer", True, WHITE)
        screen.blit(title, title.get_rect(centerx=cx, centery=cy - 140))

        sub = font_sm.render("openpilot route playback  +  ui-alt style", True, DIM)
        screen.blit(sub, sub.get_rect(centerx=cx, centery=cy - 100))

        # Separator
        pygame.draw.line(screen, (35, 45, 60), (cx - 180, cy - 72), (cx + 180, cy - 72))

        # Buttons
        _draw_button(screen, font_md, "Load Route Folder...", load_rect, mouse)
        _draw_button(screen, font_md, "Demo Mode", demo_rect, mouse, ORANGE)

        # Hint
        hint1 = font_sm.render("Load a route downloaded from your comma 4 via Cyberduck", True, DIM)
        screen.blit(hint1, hint1.get_rect(centerx=cx, centery=cy + 130))
        hint2 = font_sm.render("Route folder should contain segment subdirectories (0/, 1/, 2/...)", True, DIM)
        screen.blit(hint2, hint2.get_rect(centerx=cx, centery=cy + 155))
        hint3 = font_sm.render("Each with rlog.bz2  +  fcamera.hevc  +  ecamera.hevc", True, DIM)
        screen.blit(hint3, hint3.get_rect(centerx=cx, centery=cy + 180))

        # Error
        if error_msg and time.time() - error_time < 3.0:
            err = font_md.render(error_msg, True, RED_T)
            screen.blit(err, err.get_rect(centerx=cx, centery=cy + 220))

        # Footer
        ver = font_sm.render("SPACE:play/pause  Arrows:scrub  +/-:speed  R:reset  Q:quit", True, (60, 70, 85))
        screen.blit(ver, ver.get_rect(centerx=cx, bottom=WINDOW_H - 15))

        pygame.display.flip()
        clock.tick(30)


def run_playback(screen, route_dir):
    """Run the main playback loop for a route or demo."""
    use_route = route_dir is not None

    if use_route:
        title = f"Loading  {os.path.basename(route_dir)}"

        def on_progress(msg, frac=None):
            _loading_screen(screen, title, msg, frac)

        on_progress("Scanning for segments...", None)
        try:
            source = RouteDataSource(route_dir, progress_cb=on_progress, use_hw=USE_HW)
        except Exception as e:
            return _error_screen(screen, f"Could not load route:", str(e))
        route_info = {
            "car": source._car,
            "segments": source.route.num_segments,
            "duration": source.duration,
            "route_dir": route_dir,
            "model": source._model,
            "runner": source._runner,
        }
        max_time = source.duration
        caption = f"Comma 4 — {os.path.basename(route_dir)}"
    else:
        source = DemoGenerator()
        route_info = None
        max_time = float("inf")
        caption = "Comma 4 Drive Visualizer"

    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    renderer = Renderer(screen)
    menubar = MenuBar()

    sim_time = 0.0
    paused = False
    speed_mult = 1.0
    scrubbing = False
    resume_after_scrub = False
    in_pt = None
    out_pt = None

    def seek_to_mouse(mx):
        r = scrub_rect()
        frac = (mx - r.x) / max(1, r.width)
        return max(0.0, min(1.0, frac)) * max_time

    def do_export():
        cfg = export_dialog(screen, route_dir, max_time, in_pt, out_pt)
        if cfg is not None:
            ok, msg = export_route(screen, source, route_info, max_time, cfg)
            _error_screen(screen, "Exported video" if ok else "Export",
                          f"Saved to  {msg}" if ok else msg)

    while True:
        dt = clock.tick(FPS) / 1000.0
        scrub_hitbox = scrub_rect().inflate(20, 18)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"

            # Menu bar gets first crack at the mouse
            action = menubar.handle(ev)
            if action:
                if action == "quit":
                    return "quit"
                elif action == "open":
                    return "menu"
                elif action == "playpause":
                    paused = not paused
                elif action == "restart":
                    sim_time = 0.0
                elif action == "set_in" and use_route:
                    in_pt = sim_time
                    if out_pt is not None and out_pt <= in_pt:
                        out_pt = None
                elif action == "set_out" and use_route:
                    out_pt = sim_time
                    if in_pt is not None and in_pt >= out_pt:
                        in_pt = None
                elif action == "clear_sel":
                    in_pt = out_pt = None
                elif action == "faster":
                    speed_mult = min(5.0, speed_mult + 0.25)
                elif action == "slower":
                    speed_mult = max(0.25, speed_mult - 0.25)
                elif action == "toggle_osm":
                    renderer.show_osm = not renderer.show_osm
                elif action == "export" and use_route:
                    do_export()
                elif action == "help":
                    _help_screen(screen)
                continue
            if menubar.open_idx is not None and ev.type == pygame.MOUSEBUTTONDOWN:
                continue  # click was consumed by the (now closed) menu

            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "menu"
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    sim_time = min(sim_time + 1.0, max_time)
                elif ev.key == pygame.K_LEFT:
                    sim_time = max(0, sim_time - 1.0)
                elif ev.key == pygame.K_r:
                    sim_time = 0.0
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_mult = min(5.0, speed_mult + 0.25)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_mult = max(0.25, speed_mult - 0.25)
                elif ev.key == pygame.K_e and use_route:
                    do_export()
                elif ev.key == pygame.K_m:
                    renderer.show_osm = not renderer.show_osm
                elif ev.key == pygame.K_i and use_route:
                    in_pt = sim_time
                    if out_pt is not None and out_pt <= in_pt:
                        out_pt = None
                elif ev.key == pygame.K_o and use_route:
                    out_pt = sim_time
                    if in_pt is not None and in_pt >= out_pt:
                        in_pt = None
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if use_route and scrub_hitbox.collidepoint(ev.pos):
                    scrubbing = True
                    resume_after_scrub = not paused
                    paused = True
                    sim_time = seek_to_mouse(ev.pos[0])
            elif ev.type == pygame.MOUSEMOTION and scrubbing:
                sim_time = seek_to_mouse(ev.pos[0])
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and scrubbing:
                scrubbing = False
                if resume_after_scrub:
                    paused = False

        if not paused:
            sim_time += dt * speed_mult
            if use_route and sim_time >= max_time:
                sim_time = max_time - 0.05
                paused = True

        state = source.generate(sim_time)

        road_frame = None
        wide_frame = None
        if use_route:
            road_frame = source.get_road_frame(sim_time)
            wide_frame = source.get_wide_frame(sim_time)

        renderer.draw(state, paused, speed_mult, road_frame, wide_frame, route_info)
        if use_route:
            hovering = scrubbing or scrub_hitbox.collidepoint(pygame.mouse.get_pos())
            renderer.draw_scrubber(sim_time, max_time, hovering, in_pt, out_pt)
        menubar.draw(screen, {"osm": renderer.show_osm})
        pygame.display.flip()


def main():
    global USE_HW
    if "--no-gpu" in sys.argv:
        USE_HW = False

    if "--record" in sys.argv:
        dur = 30.0
        out = "comma4_demo.mp4"
        for i, a in enumerate(sys.argv):
            if a == "--duration" and i + 1 < len(sys.argv):
                dur = float(sys.argv[i + 1])
            if a == "--output" and i + 1 < len(sys.argv):
                out = sys.argv[i + 1]
        record_video(out, dur)
        return

    # CLI shortcut: pass a route folder directly
    route_dir = None
    for a in sys.argv[1:]:
        if not a.startswith("-") and os.path.isdir(a):
            route_dir = a
            break

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Comma 4 Drive Visualizer")

    if route_dir:
        result = run_playback(screen, route_dir)
    else:
        while True:
            route_dir, quit_app = launcher_menu(screen)
            if quit_app:
                break
            result = run_playback(screen, route_dir)
            if result == "quit":
                break
            route_dir = None

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required for parallel export in the .exe
    main()
