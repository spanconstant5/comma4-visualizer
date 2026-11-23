"""
Route loader for openpilot driving data (comma 4 / mici).
Parses rlog/qlog capnp files and stitches multi-segment routes.

Expects a directory containing segment folders downloaded via Cyberduck/SFTP:
  my_route/
    2024-01-15--12-30-45--0/   (or just "0/")
      rlog.bz2  (or rlog, qlog.bz2, qlog)
      fcamera.hevc
      ecamera.hevc
    2024-01-15--12-30-45--1/
      ...
"""

import os
import bz2
import struct
import math
import glob
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import capnp
    capnp.remove_import_hook()
    HAS_CAPNP = True
except ImportError:
    HAS_CAPNP = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── Schema Loading ───────────────────────────────────────────────────────────

_LOG_SCHEMA = None

def _resource_base():
    """Base directory for bundled resources (schemas).

    Works as a plain script and inside a PyInstaller onefile/onedir bundle,
    where data files are unpacked to sys._MEIPASS.
    """
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _get_schema():
    global _LOG_SCHEMA
    if _LOG_SCHEMA is not None:
        return _LOG_SCHEMA
    if not HAS_CAPNP:
        raise ImportError("pycapnp is required: pip install pycapnp")

    schema_dir = os.path.join(_resource_base(), "schemas")
    if not os.path.exists(os.path.join(schema_dir, "log.capnp")):
        raise FileNotFoundError(
            f"Schema files not found in {schema_dir}/\n"
            "Make sure the schemas/ directory is next to this file."
        )
    _LOG_SCHEMA = capnp.load(
        os.path.join(schema_dir, "log.capnp"),
        imports=[schema_dir]
    )
    return _LOG_SCHEMA


# ── Parsed Data Types ────────────────────────────────────────────────────────

@dataclass
class FrameData:
    """One time-slice of driving data for the visualizer."""
    t: float = 0.0           # seconds from route start
    mono_time: int = 0       # raw logMonoTime (ns)
    speed_ms: float = 0.0    # m/s
    speed_mph: float = 0.0
    steering_angle: float = 0.0
    steering_rate: float = 0.0
    steering_torque: float = 0.0
    gas_pressed: bool = False
    brake_pressed: bool = False
    standstill: bool = False
    cruise_speed: float = 0.0
    left_blinker: bool = False
    right_blinker: bool = False

    # controlsState / selfdriveState
    enabled: bool = False
    active: bool = False
    engage_status: str = "disengaged"  # engaged|lat_only|long_only|override|disengaged
    alert1: str = ""
    alert2: str = ""
    alert_status: str = "normal"       # normal|userPrompt|critical
    alert_size: str = "none"           # none|small|mid|full
    steer_target: float = 0.0
    steer_applied: float = 0.0
    curvature: float = 0.0
    desired_curvature: float = 0.0
    a_target: float = 0.0
    a_ego: float = 0.0

    # radarState
    lead_present: bool = False
    lead_dist: float = 0.0
    lead_v_rel: float = 0.0
    lead_v_lead: float = 0.0
    lead_y_rel: float = 0.0
    lead2_present: bool = False
    lead2_dist: float = 0.0

    # modelV2 - path (device frame: x fwd, y left, z up; metres)
    model_path_x: list = field(default_factory=list)
    model_path_y: list = field(default_factory=list)
    model_path_z: list = field(default_factory=list)
    lane_line_probs: list = field(default_factory=list)
    confidence: float = 0.0
    # Full model snapshot (shared reference across frames between model updates):
    # keys: path, lanes, lane_probs, edges, edge_stds, accel_x, leads, confidence
    model: dict = None

    # map / gps
    road_name: str = ""
    map_speed_limit: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    bearing: float = 0.0
    nav_maneuver: str = ""
    nav_street: str = ""
    nav_dist: float = 0.0

    # car info
    car_fingerprint: str = ""


# ── Segment Discovery ────────────────────────────────────────────────────────

def _is_segment_dir(d: str) -> bool:
    """A directory is a usable segment if it holds a log and/or a camera file."""
    return bool(_find_log_file(d)
                or _find_video_file(d, "road")
                or _find_video_file(d, "wide"))


def find_segments(route_dir: str) -> List[Tuple[int, str]]:
    """
    Discover segment directories anywhere under a route folder.
    Searches subfolders recursively so the user can select a parent folder
    (e.g. a route like "028/" whose segments live in nested subdirectories),
    an individual segment, or a folder of segment folders.
    Handles "N/", "routeid--N/" and arbitrary naming conventions.
    Returns a sorted list of (segment_number, full_path).
    """
    route_dir = os.path.abspath(route_dir)
    if not os.path.isdir(route_dir):
        raise FileNotFoundError(f"Route directory not found: {route_dir}")

    found: List[str] = []

    # The selected folder itself might be a single segment.
    if _is_segment_dir(route_dir):
        found.append(route_dir)

    # Walk the tree; collect every directory that looks like a segment.
    # Prune descent into a segment dir (segments don't nest inside segments).
    MAX_DEPTH = 6
    base_depth = route_dir.rstrip(os.sep).count(os.sep)
    for cur, subdirs, _files in os.walk(route_dir):
        depth = cur.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= MAX_DEPTH:
            subdirs[:] = []
        subdirs.sort()
        keep = []
        for name in subdirs:
            full = os.path.join(cur, name)
            if _is_segment_dir(full):
                found.append(full)
                # don't recurse into a segment
            else:
                keep.append(name)
        subdirs[:] = keep

    # Dedupe while preserving order.
    seen = set()
    uniq = []
    for p in found:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    # Assign segment numbers: prefer the number parsed from the folder name,
    # falling back to ordering by path so a full drive stays in sequence.
    parsed = []
    for p in uniq:
        parsed.append((_extract_seg_num(os.path.basename(p.rstrip(os.sep))), p))

    if all(n is not None for n, _ in parsed) and parsed:
        parsed.sort(key=lambda x: x[0])
        return [(n, p) for n, p in parsed]

    # Mixed / unparseable names: order by natural path sort, number sequentially.
    uniq.sort(key=_natural_key)
    return [(i, p) for i, p in enumerate(uniq)]


def _natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def _extract_seg_num(dirname: str) -> Optional[int]:
    try:
        return int(dirname)
    except ValueError:
        pass
    parts = dirname.split("--")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


def camera_config(device_type: str, sensor: str) -> dict:
    """Return {narrow,wide}:(W,H,focal_px) for the device's cameras.

    Mirrors openpilot's DEVICE_CAMERAS. The comma 4 (mici) ships the os04c10
    sensor, whose narrow focal/intrinsics differ a lot from the comma 3 family.
    """
    AR_OX = {"narrow": (1928, 1208, 2648.0), "wide": (1928, 1208, 567.0)}
    OS = {"narrow": (1344, 760, 1141.5), "wide": (1344, 760, 425.25)}
    if sensor == "os04c10":
        return OS
    if sensor in ("ar0231", "ox03c10"):
        return AR_OX
    if device_type == "mici":   # comma 4 default sensor is os04c10
        return OS
    return AR_OX


def engagement_status(sd: dict, sp: dict) -> str:
    """Replicate sunnypilot's UI engagement status from selfdriveState(+SP).

    Returns one of: engaged, lat_only, long_only, override, disengaged.
    Falls back to stock openpilot (engaged/disengaged) when no MADS data.
    """
    state = sd.get("state", "disabled")
    enabled = sd.get("enabled", False)
    mads_state = sp.get("mads_state")
    mads_enabled = sp.get("mads_enabled", False)
    mads_available = sp.get("mads_available", False)

    if state == "preEnabled":
        return "override"
    if state == "overriding" and not mads_available:
        return "override"
    if mads_state in ("paused", "overriding"):
        return "override"

    if not mads_available:
        return "engaged" if enabled else "disengaged"
    if not mads_enabled and not enabled:
        return "disengaged"
    if mads_enabled and enabled:
        return "engaged"
    if mads_enabled:
        return "lat_only"
    if enabled:
        return "long_only"
    return "disengaged"


def _find_log_file(seg_dir: str) -> Optional[str]:
    """Find the best available log file in a segment directory.

    Prefers the full rlog over qlog, and handles bz2 / zstd / uncompressed.
    Recent openpilot (comma 3X / comma 4) stores logs as rlog.zst.
    """
    for name in ["rlog.zst", "rlog.bz2", "rlog",
                 "qlog.zst", "qlog.bz2", "qlog"]:
        p = os.path.join(seg_dir, name)
        if os.path.exists(p):
            return p
    return None


def _find_video_file(seg_dir: str, camera: str = "road") -> Optional[str]:
    """Find camera video file. camera: 'road', 'wide', 'driver'."""
    name_map = {
        "road": ["fcamera.hevc", "video.hevc", "fcamera.mp4"],
        "wide": ["ecamera.hevc", "ecamera.mp4"],
        "driver": ["dcamera.hevc", "dcamera.mp4"],
    }
    for name in name_map.get(camera, []):
        p = os.path.join(seg_dir, name)
        if os.path.exists(p):
            return p
    return None


# ── Log Parsing ──────────────────────────────────────────────────────────────

def _open_log(filepath: str):
    """Open a log file, handling bz2 / zstd / uncompressed data.

    Returns a readable binary stream positioned at the start.
    """
    if filepath.endswith(".bz2"):
        return bz2.open(filepath, "rb")
    if filepath.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            raise ImportError(
                "This route uses zstd-compressed logs (rlog.zst). "
                "Install the decompressor: pip install zstandard"
            )
        import io
        with open(filepath, "rb") as fh:
            compressed = fh.read()
        # openpilot writes zst without a content-size frame header, so stream it.
        dctx = zstandard.ZstdDecompressor()
        data = dctx.stream_reader(io.BytesIO(compressed)).read()
        return io.BytesIO(data)
    return open(filepath, "rb")


def parse_segment_log(seg_dir: str) -> Dict[str, List]:
    """
    Parse a segment's log file and extract relevant events.
    Returns dict of event_type -> list of (mono_time_ns, data_dict).
    """
    log_file = _find_log_file(seg_dir)
    if not log_file:
        return {}

    schema = _get_schema()
    events = {
        "carState": [],
        "controlsState": [],
        "radarState": [],
        "modelV2": [],
        "drivingModelData": [],
        "selfdriveState": [],
        "selfdriveStateSP": [],
        "carParams": [],
        "modelManagerSP": [],
        "extrinsicsCalibration": [],
        "deviceState": [],
        "cameraSensor": [],
        "gps": [],
        "liveMapDataSP": [],
        "navInstruction": [],
    }

    with _open_log(log_file) as f:
        raw = f.read()

    # openpilot rlogs are a stream of packed capnp Events. Pass the raw bytes
    # to the *_bytes_* reader (the plain readers expect a real file object).
    try:
        reader = schema.Event.read_multiple_bytes_packed(raw)
        reader = list(reader)
    except Exception:
        # Fall back to unpacked encoding if this log isn't packed.
        reader = list(schema.Event.read_multiple_bytes(raw))

    for event in reader:
        try:
            mono = event.logMonoTime
            which = event.which()

            if which == "carState":
                cs = event.carState
                events["carState"].append((mono, {
                    "vEgo": cs.vEgo,
                    "aEgo": cs.aEgo,
                    "steeringAngleDeg": cs.steeringAngleDeg,
                    "steeringRateDeg": cs.steeringRateDeg,
                    "steeringTorque": cs.steeringTorque,
                    "gasPressed": cs.gasPressed,
                    "brakePressed": cs.brakePressed,
                    "standstill": cs.standstill,
                    "vCruise": (cs.vCruiseCluster if getattr(cs, 'vCruiseCluster', 0)
                                else getattr(cs, 'vCruise', 0)),
                    "leftBlinker": cs.leftBlinker,
                    "rightBlinker": cs.rightBlinker,
                    "steeringPressed": cs.steeringPressed,
                }))

            elif which == "controlsState":
                ctrl = event.controlsState
                events["controlsState"].append((mono, {
                    "curvature": ctrl.curvature,
                    "desiredCurvature": ctrl.desiredCurvature,
                }))

            elif which == "selfdriveState":
                sd = event.selfdriveState
                events["selfdriveState"].append((mono, {
                    "enabled": sd.enabled,
                    "active": sd.active,
                    "state": str(sd.state),
                    "alert1": str(sd.alertText1),
                    "alert2": str(sd.alertText2),
                    "alert_status": str(sd.alertStatus),
                    "alert_size": str(sd.alertSize),
                }))

            elif which == "selfdriveStateSP":
                try:
                    m = event.selfdriveStateSP.mads
                    events["selfdriveStateSP"].append((mono, {
                        "mads_state": str(m.state),
                        "mads_enabled": bool(m.enabled),
                        "mads_available": bool(m.available),
                    }))
                except Exception:
                    pass

            elif which == "radarState":
                rs = event.radarState
                lead1 = rs.leadOne
                lead2 = rs.leadTwo
                events["radarState"].append((mono, {
                    "lead1_present": lead1.present if hasattr(lead1, 'present') else (lead1.dRel > 0),
                    "lead1_dRel": lead1.dRel,
                    "lead1_yRel": lead1.yRel,
                    "lead1_vRel": lead1.vRel,
                    "lead1_vLead": lead1.vLead,
                    "lead2_present": lead2.present if hasattr(lead2, 'present') else False,
                    "lead2_dRel": lead2.dRel,
                }))

            elif which == "modelV2":
                import numpy as _np
                mv = event.modelV2

                def _xyz(obj):
                    return _np.array([list(obj.x), list(obj.y), list(obj.z)],
                                     dtype=_np.float32).T  # N x 3

                path = _xyz(mv.position) if mv.position and mv.position.x else _np.empty((0, 3), _np.float32)
                lanes = [_xyz(ll) for ll in mv.laneLines] if mv.laneLines else []
                edges = [_xyz(re) for re in mv.roadEdges] if mv.roadEdges else []
                accel_x = list(mv.acceleration.x) if mv.acceleration and mv.acceleration.x else []
                leads = []
                try:
                    for ld in mv.leadsV3:
                        if ld.x and ld.y:
                            leads.append((float(ld.prob), float(ld.x[0]), float(ld.y[0])))
                except Exception:
                    pass

                # comma 4 confidence = (1 - max brakeDisengageProb) * (1 - max steerOverrideProb)
                conf = 0.5
                try:
                    dp = mv.meta.disengagePredictions
                    bk = list(dp.brakeDisengageProbs) or [0.0]
                    so = list(dp.steerOverrideProbs) or [0.0]
                    conf = (1.0 - max(bk)) * (1.0 - max(so))
                except Exception:
                    try:
                        conf = mv.meta.engagedProb
                    except Exception:
                        pass

                events["modelV2"].append((mono, {
                    "path": path,
                    "lanes": lanes,
                    "lane_probs": list(mv.laneLineProbs) if mv.laneLineProbs else [],
                    "edges": edges,
                    "edge_stds": list(mv.roadEdgeStds) if mv.roadEdgeStds else [],
                    "accel_x": accel_x,
                    "leads": leads,
                    "confidence": conf,
                    # legacy scalar views kept for any older code paths
                    "path_x": list(path[:, 0]) if path.size else [],
                    "path_y": list(path[:, 1]) if path.size else [],
                    "path_z": list(path[:, 2]) if path.size else [],
                    "lane_line_probs": list(mv.laneLineProbs) if mv.laneLineProbs else [],
                }))

            elif which == "extrinsicsCalibration":
                try:
                    ec = event.extrinsicsCalibration
                    if len(ec.rpyCalib) == 3:
                        events["extrinsicsCalibration"].append((mono, {
                            "rpy": list(ec.rpyCalib),
                            "wide": list(ec.wideFromDeviceEuler) if len(ec.wideFromDeviceEuler) == 3 else [0.0, 0.0, 0.0],
                            "height": float(ec.height[0]) if ec.height else 1.22,
                        }))
                except Exception:
                    pass

            elif which == "carParams":
                cp = event.carParams
                events["carParams"].append((mono, {
                    "carFingerprint": str(cp.carFingerprint) if hasattr(cp, 'carFingerprint') else "",
                }))

            elif which == "modelManagerSP":
                try:
                    mm = event.modelManagerSP
                    b = mm.activeBundle if mm.activeBundle else mm.selectedBundle
                    name = b.displayName or b.internalName or ""
                    runner = ""
                    try:
                        runner = str(b.runner)
                    except Exception:
                        pass
                    events["modelManagerSP"].append((mono, {
                        "name": str(name),
                        "ref": str(b.ref) if hasattr(b, "ref") else "",
                        "runner": runner,
                        "is20hz": bool(getattr(b, "is20hz", False)),
                    }))
                except Exception:
                    pass

            elif which == "deviceState":
                if not events["deviceState"]:
                    try:
                        events["deviceState"].append((mono, {
                            "deviceType": str(event.deviceState.deviceType)}))
                    except Exception:
                        pass

            elif which == "narrowRoadCameraState":
                if not events["cameraSensor"]:
                    try:
                        events["cameraSensor"].append((mono, {
                            "sensor": str(event.narrowRoadCameraState.sensor)}))
                    except Exception:
                        pass

            elif which in ("gpsLocationExternal", "gpsLocation"):
                try:
                    g = getattr(event, which)
                    if g.latitude or g.longitude:
                        events["gps"].append((mono, {
                            "lat": float(g.latitude), "lon": float(g.longitude),
                            "bearing": float(getattr(g, "bearingDeg", 0.0)),
                        }))
                except Exception:
                    pass

            elif which == "liveMapDataSP":
                try:
                    lm = event.liveMapDataSP
                    events["liveMapDataSP"].append((mono, {
                        "road": str(lm.roadName),
                        "speed_limit": float(lm.speedLimit) if lm.speedLimitValid else 0.0,
                    }))
                except Exception:
                    pass

            elif which == "navInstruction":
                try:
                    ni = event.navInstruction
                    events["navInstruction"].append((mono, {
                        "maneuver": str(ni.maneuverPrimaryText),
                        "street": str(ni.maneuverSecondaryText),
                        "dist": float(ni.maneuverDistance),
                        "modifier": str(ni.maneuverModifier),
                    }))
                except Exception:
                    pass

        except Exception as e:
            continue

    return events


# ── Route Stitcher ───────────────────────────────────────────────────────────

class Route:
    """
    A complete openpilot route with stitched segments.
    Provides time-indexed access to driving data and video frames.
    """

    def __init__(self, route_dir: str, progress_cb=None):
        self.route_dir = os.path.abspath(route_dir)
        self.segments = find_segments(route_dir)
        if not self.segments:
            raise ValueError(f"No valid segments found in {route_dir}")

        self.car_fingerprint = ""
        self.model_name = ""
        self.model_runner = ""
        self.device_type = ""
        self.camera_sensor = ""
        self.cam_config = camera_config("mici", "")  # default comma 4
        # Calibration for projecting the model into the camera image
        self.calib = {"rpy": [0.0, 0.0, 0.0], "wide": [0.0, 0.0, 0.0], "height": 1.22}
        self._have_calib = False
        self.gps_track = []  # list of (t, lat, lon) for the OSM map
        self.frames: List[FrameData] = []
        # (seg_dir, t_start, t_end) in route time, aligned to the logs
        self.segment_spans: List[Tuple[str, float, float]] = []
        self._load_all(progress_cb)

    @property
    def duration(self) -> float:
        return self.frames[-1].t if self.frames else 0.0

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def get_frame(self, t: float) -> FrameData:
        """Get the driving data frame closest to time t."""
        if not self.frames:
            return FrameData()
        # Binary search
        lo, hi = 0, len(self.frames) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.frames[mid].t < t:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(self.frames[lo - 1].t - t) < abs(self.frames[lo].t - t):
            lo -= 1
        return self.frames[lo]

    def _load_all(self, progress_cb=None):
        """Load and stitch all segments."""
        all_car_state = []
        all_controls = []
        all_selfdrive = []
        all_sp = []
        all_radar = []
        all_model = []
        all_gps = []
        all_map = []
        all_nav = []
        time_offset = 0
        route_start_ns = None
        seg_marks = []  # (seg_dir, t_start) for segments that parsed, in order
        n = len(self.segments)

        def report(msg, frac):
            if progress_cb:
                progress_cb(msg, frac)

        for i, (seg_num, seg_dir) in enumerate(self.segments):
            # Loading logs is the bulk of the work -> map it to 0..0.9
            report(f"Reading segment {seg_num}  ({i + 1}/{n})", 0.9 * i / max(1, n))

            try:
                events = parse_segment_log(seg_dir)
            except Exception as e:
                print(f"  Warning: failed to parse segment {seg_num}: {e}")
                continue

            # Get car fingerprint from first available segment
            if not self.car_fingerprint and events.get("carParams"):
                self.car_fingerprint = events["carParams"][0][1].get("carFingerprint", "")

            # Capture the active driving model name if the log reports it
            if not self.model_name and events.get("modelManagerSP"):
                mm = events["modelManagerSP"][0][1]
                self.model_name = mm.get("name", "")
                self.model_runner = mm.get("runner", "")

            # Capture camera calibration (first valid wins)
            if not self._have_calib and events.get("extrinsicsCalibration"):
                self.calib = events["extrinsicsCalibration"][0][1]
                self._have_calib = True

            # Device type + camera sensor -> correct intrinsics
            if not self.device_type and events.get("deviceState"):
                self.device_type = events["deviceState"][0][1].get("deviceType", "")
            if not self.camera_sensor and events.get("cameraSensor"):
                self.camera_sensor = events["cameraSensor"][0][1].get("sensor", "")
            if self.device_type or self.camera_sensor:
                self.cam_config = camera_config(self.device_type, self.camera_sensor)

            # Determine time offset from first event in this segment
            all_times = []
            for ev_list in events.values():
                if ev_list:
                    all_times.append(ev_list[0][0])
            if not all_times:
                continue

            seg_start_ns = min(all_times)
            if route_start_ns is None:
                route_start_ns = seg_start_ns
                time_offset = 0.0
            else:
                time_offset = (seg_start_ns - route_start_ns) / 1e9

            # Record this segment's start time so video stays aligned to the logs
            if _find_video_file(seg_dir, "road") or _find_video_file(seg_dir, "wide"):
                seg_marks.append((seg_dir, time_offset))

            # Accumulate events with adjusted timestamps
            for mono, data in events.get("carState", []):
                t = (mono - route_start_ns) / 1e9
                all_car_state.append((t, data))

            for mono, data in events.get("controlsState", []):
                t = (mono - route_start_ns) / 1e9
                all_controls.append((t, data))

            for mono, data in events.get("selfdriveState", []):
                t = (mono - route_start_ns) / 1e9
                all_selfdrive.append((t, data))

            for mono, data in events.get("selfdriveStateSP", []):
                t = (mono - route_start_ns) / 1e9
                all_sp.append((t, data))

            for mono, data in events.get("radarState", []):
                t = (mono - route_start_ns) / 1e9
                all_radar.append((t, data))

            for mono, data in events.get("modelV2", []):
                t = (mono - route_start_ns) / 1e9
                all_model.append((t, data))

            for mono, data in events.get("gps", []):
                t = (mono - route_start_ns) / 1e9
                all_gps.append((t, data))

            for mono, data in events.get("liveMapDataSP", []):
                t = (mono - route_start_ns) / 1e9
                all_map.append((t, data))

            for mono, data in events.get("navInstruction", []):
                t = (mono - route_start_ns) / 1e9
                all_nav.append((t, data))

        # Build segment spans: each span runs until the next segment starts,
        # with the last one defaulting to a 60s comma segment.
        for idx, (seg_dir, t_start) in enumerate(seg_marks):
            if idx + 1 < len(seg_marks):
                t_end = seg_marks[idx + 1][1]
            else:
                t_end = t_start + 60.0
            self.segment_spans.append((seg_dir, t_start, t_end))

        # Sort all events by time
        all_car_state.sort(key=lambda x: x[0])
        all_controls.sort(key=lambda x: x[0])
        all_selfdrive.sort(key=lambda x: x[0])
        all_sp.sort(key=lambda x: x[0])
        all_radar.sort(key=lambda x: x[0])
        all_model.sort(key=lambda x: x[0])
        all_gps.sort(key=lambda x: x[0])
        all_map.sort(key=lambda x: x[0])
        all_nav.sort(key=lambda x: x[0])

        # Decimated GPS track (~2 Hz) for the map panel
        self.gps_track = []
        last_gt = -1.0
        for t, g in all_gps:
            if t - last_gt >= 0.5:
                self.gps_track.append((t, g["lat"], g["lon"]))
                last_gt = t

        report("Building frame timeline...", 0.92)

        # Build frame timeline from carState (highest frequency ~100Hz)
        # Downsample to 20Hz for the visualizer
        target_dt = 1.0 / 20.0
        ctrl_idx = 0
        sd_idx = 0
        sp_idx = 0
        radar_idx = 0
        model_idx = 0
        gps_idx = 0
        map_idx = 0

        last_ctrl = {}
        last_sd = {}
        last_sp = {}
        last_radar = {}
        last_model = {}
        last_gps = {}
        last_map = {}
        last_nav = {}
        nav_idx = 0

        last_frame_t = -1.0

        for t, cs_data in all_car_state:
            if t - last_frame_t < target_dt:
                continue
            last_frame_t = t

            # Advance controls to current time
            while ctrl_idx < len(all_controls) and all_controls[ctrl_idx][0] <= t:
                last_ctrl = all_controls[ctrl_idx][1]
                ctrl_idx += 1

            while sd_idx < len(all_selfdrive) and all_selfdrive[sd_idx][0] <= t:
                last_sd = all_selfdrive[sd_idx][1]
                sd_idx += 1

            while sp_idx < len(all_sp) and all_sp[sp_idx][0] <= t:
                last_sp = all_sp[sp_idx][1]
                sp_idx += 1

            while radar_idx < len(all_radar) and all_radar[radar_idx][0] <= t:
                last_radar = all_radar[radar_idx][1]
                radar_idx += 1

            while model_idx < len(all_model) and all_model[model_idx][0] <= t:
                last_model = all_model[model_idx][1]
                model_idx += 1

            while gps_idx < len(all_gps) and all_gps[gps_idx][0] <= t:
                last_gps = all_gps[gps_idx][1]
                gps_idx += 1

            while map_idx < len(all_map) and all_map[map_idx][0] <= t:
                last_map = all_map[map_idx][1]
                map_idx += 1

            while nav_idx < len(all_nav) and all_nav[nav_idx][0] <= t:
                last_nav = all_nav[nav_idx][1]
                nav_idx += 1

            frame = FrameData(
                t=t,
                speed_ms=cs_data.get("vEgo", 0),
                speed_mph=cs_data.get("vEgo", 0) * 2.23694,
                steering_angle=cs_data.get("steeringAngleDeg", 0),
                steering_rate=cs_data.get("steeringRateDeg", 0),
                steering_torque=cs_data.get("steeringTorque", 0),
                gas_pressed=cs_data.get("gasPressed", False),
                brake_pressed=cs_data.get("brakePressed", False),
                standstill=cs_data.get("standstill", False),
                cruise_speed=cs_data.get("vCruise", 0),
                left_blinker=cs_data.get("leftBlinker", False),
                right_blinker=cs_data.get("rightBlinker", False),
                a_ego=cs_data.get("aEgo", 0),
                # controlsState
                curvature=last_ctrl.get("curvature", 0),
                desired_curvature=last_ctrl.get("desiredCurvature", 0),
                # selfdriveState
                enabled=last_sd.get("enabled", False),
                active=last_sd.get("active", False),
                engage_status=engagement_status(last_sd, last_sp),
                alert1=last_sd.get("alert1", ""),
                alert2=last_sd.get("alert2", ""),
                alert_status=last_sd.get("alert_status", "normal"),
                alert_size=last_sd.get("alert_size", "none"),
                # radarState
                lead_present=last_radar.get("lead1_present", False),
                lead_dist=last_radar.get("lead1_dRel", 0),
                lead_v_rel=last_radar.get("lead1_vRel", 0),
                lead_v_lead=last_radar.get("lead1_vLead", 0),
                lead_y_rel=last_radar.get("lead1_yRel", 0),
                lead2_present=last_radar.get("lead2_present", False),
                lead2_dist=last_radar.get("lead2_dRel", 0),
                # modelV2
                model_path_x=last_model.get("path_x", []),
                model_path_y=last_model.get("path_y", []),
                model_path_z=last_model.get("path_z", []),
                lane_line_probs=last_model.get("lane_line_probs", []),
                confidence=last_model.get("confidence", 0.5),
                model=last_model if last_model else None,
                road_name=last_map.get("road", ""),
                map_speed_limit=last_map.get("speed_limit", 0.0),
                lat=last_gps.get("lat", 0.0),
                lon=last_gps.get("lon", 0.0),
                bearing=last_gps.get("bearing", 0.0),
                nav_maneuver=last_nav.get("maneuver", ""),
                nav_street=last_nav.get("street", ""),
                nav_dist=last_nav.get("dist", 0.0),
                car_fingerprint=self.car_fingerprint,
            )
            self.frames.append(frame)

        report(f"Loaded {len(self.frames)} frames, {self.duration:.1f}s, "
               f"{len(self.segments)} segments", 1.0)


# ── Video Reader ─────────────────────────────────────────────────────────────

VIDEO_FPS = 20.0   # comma records road/wide cameras at 20 fps

# Decide once whether OpenCV was built with hardware-accelerated decoding.
_HW_ACCEL = None

def _hw_accel_params():
    """Return VideoCapture open params requesting GPU decode, or [] if N/A.

    On Windows this lets OpenCV's FFmpeg backend use D3D11VA / DXVA2, which on
    an AMD GPU (e.g. Radeon RX 9060 XT) offloads HEVC decode from the CPU.
    Falls back silently to software decode when unavailable.
    """
    global _HW_ACCEL
    if _HW_ACCEL is not None:
        return _HW_ACCEL
    _HW_ACCEL = []
    try:
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
            _HW_ACCEL = [
                int(cv2.CAP_PROP_HW_ACCELERATION), int(cv2.VIDEO_ACCELERATION_ANY),
                int(cv2.CAP_PROP_HW_ACCELERATION_USE_OPENCL), 0,
            ]
    except Exception:
        _HW_ACCEL = []
    return _HW_ACCEL


class _SegmentDecoder:
    """Sequential HEVC decoder for one segment's raw .hevc stream.

    Raw elementary streams have no reliable frame index, so we never seek by
    frame (OpenCV can't). Instead we decode forward and cache the last frame;
    a backwards request simply reopens and fast-forwards. Normal playback only
    ever reads the next frame, so it stays smooth.
    """

    def __init__(self, path: str, use_hw: bool):
        self.path = path
        self.use_hw = use_hw
        self._hw_disabled = False  # set if GPU decode opened but yielded nothing
        self.cap = None
        self.pos = -1          # index of the last decoded frame
        self.last = None       # last decoded frame (RGB), cached

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        params = _hw_accel_params() if (self.use_hw and not self._hw_disabled) else []
        try:
            if params:
                self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG, params)
            else:
                self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        except Exception:
            self.cap = cv2.VideoCapture(self.path)
        self.pos = -1
        self.last = None

    def frame_at(self, idx: int):
        if idx < 0:
            idx = 0
        if self.cap is None or idx < self.pos:
            self._open()
        if idx == self.pos and self.last is not None:
            return self.last
        frame = self.last
        while self.pos < idx:
            ok, bgr = self.cap.read()
            if not ok:
                break  # past end of stream -> keep last good frame
            self.pos += 1
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Cross-GPU safety: if hardware decode opened but produced nothing on the
        # very first attempt, disable it and retry once in software.
        if (frame is None and self.use_hw and not self._hw_disabled
                and self.pos < 0):
            self._hw_disabled = True
            self._open()
            while self.pos < idx:
                ok, bgr = self.cap.read()
                if not ok:
                    break
                self.pos += 1
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.last = frame
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoReader:
    """Serves video frames across stitched segments, aligned to log time.

    Keeps a bounded LRU cache of recently decoded frames (keyed by segment +
    frame index) so paused frames, replays and backward scrubs don't re-decode.
    """

    def __init__(self, spans: List[Tuple[str, float, float]], camera: str = "road",
                 use_hw: bool = True, cache_frames: int = 300):
        import collections
        self.camera = camera
        self.use_hw = use_hw
        self._segs = []  # (t_start, t_end, decoder)
        self._cache = collections.OrderedDict()
        self._cache_max = cache_frames

        if not HAS_CV2:
            print("Warning: opencv not available, video frames disabled")
            return

        for seg_dir, t_start, t_end in spans:
            vf = _find_video_file(seg_dir, camera)
            if not vf:
                continue
            self._segs.append((t_start, t_end, _SegmentDecoder(vf, use_hw)))

    def get_frame(self, t: float):
        """Get the video frame at route time t. Returns an RGB array or None."""
        for si, (t_start, t_end, dec) in enumerate(self._segs):
            if t_start <= t < t_end:
                idx = int((t - t_start) * VIDEO_FPS)
                key = (si, idx)
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    return hit
                frame = dec.frame_at(idx)
                if frame is not None:
                    self._cache[key] = frame
                    if len(self._cache) > self._cache_max:
                        self._cache.popitem(last=False)
                return frame
        return None

    @property
    def available(self) -> bool:
        return len(self._segs) > 0

    def close(self):
        for _, _, dec in self._segs:
            dec.release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ThreadedCamera:
    """Decodes a camera off the main thread so playback never blocks on HEVC.

    The worker continuously decodes toward the requested time; the UI reads the
    latest available frame. For exact frames (export) use exact(), which takes
    the decode lock so it never races the worker.
    """

    def __init__(self, reader: 'VideoReader'):
        import threading
        self.reader = reader
        self._lock = threading.Lock()
        self._target = 0.0
        self._last_decoded_t = None
        self._latest = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import time as _t
        while not self._stop:
            t = self._target
            if t == self._last_decoded_t:
                _t.sleep(0.02)
                continue
            with self._lock:
                fr = self.reader.get_frame(t)
            if fr is not None:
                self._latest = fr
            self._last_decoded_t = t
            _t.sleep(0.001)

    def set_target(self, t):
        self._target = t

    def frame(self):
        return self._latest

    def exact(self, t):
        with self._lock:
            return self.reader.get_frame(t)

    @property
    def available(self):
        return self.reader.available

    def close(self):
        self._stop = True


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python route_loader.py <route_directory>")
        print("\nProvide the path to a folder containing openpilot segment directories.")
        print("Example: python route_loader.py ~/drives/2024-01-15--12-30-45/")
        sys.exit(1)

    route_dir = sys.argv[1]
    print(f"Loading route from: {route_dir}")

    segments = find_segments(route_dir)
    print(f"Found {len(segments)} segments:")
    for num, path in segments:
        log = _find_log_file(path)
        road_vid = _find_video_file(path, "road")
        wide_vid = _find_video_file(path, "wide")
        print(f"  Segment {num}: log={'yes' if log else 'no'}"
              f"  road_vid={'yes' if road_vid else 'no'}"
              f"  wide_vid={'yes' if wide_vid else 'no'}")

    print("\nParsing logs...")
    route = Route(route_dir, progress_cb=print)
    print(f"\nCar: {route.car_fingerprint}")
    print(f"Duration: {route.duration:.1f}s ({route.duration / 60:.1f} min)")
    print(f"Frames: {len(route.frames)}")

    if route.frames:
        f0 = route.frames[0]
        fm = route.frames[len(route.frames) // 2]
        print(f"\nFirst frame: speed={f0.speed_mph:.1f}mph steer={f0.steering_angle:.1f}°")
        print(f"Mid frame:   speed={fm.speed_mph:.1f}mph steer={fm.steering_angle:.1f}°"
              f" lead={fm.lead_dist:.1f}m engaged={fm.enabled}")
