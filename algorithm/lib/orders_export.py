import math
from typing import List, Dict, Tuple
from algorithm.lib.path import DubinsPath, DubinsPathType
from algorithm.config import config

def _norm(a: float) -> float:
    while a < 0: a += 2*math.pi
    while a >= 2*math.pi: a -= 2*math.pi
    return a

def _circle_center(point: Tuple[float, float], heading: float, turn: str, R: float) -> Tuple[float, float]:
    x, y = point
    if turn.upper() == 'L':
        return (x - R * math.sin(heading), y + R * math.cos(heading))
    else:
        return (x + R * math.sin(heading), y - R * math.cos(heading))

def _arc_delta(a0: float, a1: float, turn: str) -> float:
    a0 = _norm(a0); a1 = _norm(a1)
    if turn.lower() == 'l':   # CCW
        if a1 <= a0: a1 += 2*math.pi
        return a1 - a0
    else:                     # CW
        if a0 <= a1: a0 += 2*math.pi
        return a0 - a1

def _heading_tangent(cx: float, cy: float, px: float, py: float, turn: str) -> float:
    ang = math.atan2(py - cy, px - cx)
    return ang + (math.pi/2 if turn.lower() == 'l' else -math.pi/2)

def _pose_dict(x: float, y: float, theta: float) -> Dict:
    return {"x": round(x, 2), "y": round(y, 2),
            "theta": round(theta, 4), "theta_deg": round(math.degrees(theta), 1)}

def export_compact_orders_from_paths(paths: List[DubinsPath]) -> Dict:
    """
    Convert planned segments into compact orders.

    Rules:
      • 'S' only if seg.metadata['scan_stop'] is True
      • 'B' only for retreat line segments (seg.metadata['retreat'] or len(waypoints)==2)
      • Normal Dubins segments are decomposed into L/R arcs and an F straight.
      • RLR/LRL middle piece falls back to a straight chord (no center available).
    """
    R = float(config.car.turning_radius)
    v_lin = float(config.car.linear_speed_cm_s)
    v_arc = float(R * getattr(config.car, "angular_speed_rad_s", 0.0)) or v_lin
    scan_sec = float(getattr(config.vision, "scan_seconds", 0.0))

    def _emit_F(a: Tuple[float,float], b: Tuple[float,float]) -> Dict:
        dx, dy = b[0]-a[0], b[1]-a[1]
        dist = math.hypot(dx, dy)
        th = math.atan2(dy, dx)
        return {
            "op": "F",
            "value_cm": round(dist, 1),
            "time_s": round(dist / v_lin, 3),
            "pose": _pose_dict(b[0], b[1], th)
        }

    def _emit_arc(seg: DubinsPath, i: int, a: Tuple[float,float], b: Tuple[float,float], turn: str) -> Dict:
        if i == 0:
            cx, cy = _circle_center(a, seg.start_state.theta, turn, R)
            a0 = math.atan2(a[1]-cy, a[0]-cx)
            a1 = math.atan2(b[1]-cy, b[0]-cx)
            dth = _arc_delta(a0, a1, turn)
            end_th = _heading_tangent(cx, cy, b[0], b[1], turn)
        else:
            cx, cy = _circle_center(b, seg.end_state.theta, turn, R)
            a0 = math.atan2(a[1]-cy, a[0]-cx)
            a1 = math.atan2(b[1]-cy, b[0]-cx)
            dth = _arc_delta(a0, a1, turn)
            end_th = seg.end_state.theta
        return {
            "op": turn.upper(),
            "angle_deg": round(math.degrees(dth), 1),
            "radius_cm": round(R, 2),
            "time_s": round(R * dth / v_arc, 3),
            "pose": _pose_dict(b[0], b[1], end_th)
        }

    orders: List[Dict] = []

    for seg in (paths or []):
        w = list(seg.waypoints or [])
        if len(w) < 2:
            continue

        meta = getattr(seg, "metadata", {}) or {}
        is_scan_stop   = bool(meta.get("scan_stop"))
        is_retreat_seg = bool(meta.get("retreat")) or (len(w) == 2)

        # --- Retreat line: output only B (no S here)
        if is_retreat_seg and len(w) == 2:
            (ax, ay), (bx, by) = w[0], w[1]
            dist = math.hypot(bx - ax, by - ay)
            orders.append({
                "op": "B",
                "value_cm": round(dist, 1),
                "time_s": round(dist / v_lin, 3),
                # keep end pose exactly as planned
                "pose": _pose_dict(seg.end_state.x, seg.end_state.y, seg.end_state.theta),
            })
            continue

        # --- Normal Dubins segment
        pat = (seg.path_type.value if isinstance(seg.path_type, DubinsPathType)
               else str(seg.path_type)).lower()

        for i in range(len(w) - 1):
            sub = pat[i] if i < len(pat) else 's'
            a, b = w[i], w[i+1]
            if sub == 's':
                orders.append(_emit_F(a, b))
            elif sub in ('l', 'r'):
                # middle arc in 3-piece paths -> fallback to straight chord
                if (len(w) - 1) == 3 and i == 1:
                    orders.append(_emit_F(a, b))
                else:
                    orders.append(_emit_arc(seg, i, a, b, sub))
            else:
                orders.append(_emit_F(a, b))

        # scan only when this segment ends at a target
        if is_scan_stop:
            s = {"op": "S",
                 "pose": _pose_dict(seg.end_state.x, seg.end_state.y, seg.end_state.theta)}
            if scan_sec > 0:
                s["scan_seconds"] = round(scan_sec, 2)
            orders.append(s)

    # Coalesce consecutive F’s
    coalesced: List[Dict] = []
    for cmd in orders:
        if coalesced and cmd["op"] == "F" and coalesced[-1]["op"] == "F":
            coalesced[-1]["value_cm"] = round(coalesced[-1]["value_cm"] + cmd["value_cm"], 1)
            coalesced[-1]["time_s"] = round(coalesced[-1]["time_s"] + cmd["time_s"], 3)
            coalesced[-1]["pose"] = cmd["pose"]
        else:
            coalesced.append(cmd)

    meta_out = {
        "assumed_constant_speeds": True,
        "linear_speed_cm_s": v_lin,
        "turn_linear_cm_s": v_arc,
        "turning_radius_cm": R,
    }

    pretty = []
    for c in coalesced:
        if c["op"] in ("L", "R"):
            pretty.append(f"{c['op']} {c['angle_deg']}°")
        elif c["op"] in ("F", "B"):
            pretty.append(f"{c['op']} {c['value_cm']}cm")
        else:
            pretty.append("S" if "scan_seconds" not in c else f"S ({c['scan_seconds']}s)")

    return {"meta": meta_out, "orders": coalesced, "orders_text": pretty, "status": "success"}
