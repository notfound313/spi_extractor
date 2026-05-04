from __future__ import annotations
import math
from typing import Callable, List, Tuple

import cv2
import numpy as np

from config import LEGEND


Geo2Px = Callable[[float, float], Tuple[float, float]]


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _nice_interval(span: float, target_lines: int = 6) -> float:
    raw   = span / target_lines
    mag   = 10 ** math.floor(math.log10(raw))
    for step in (1, 2, 2.5, 5, 10):
        if raw <= step * mag:
            return step * mag
    return 10 * mag


def _dms(deg: float, is_lat: bool) -> str:
    sign   = deg < 0
    d      = abs(deg)
    deg_i  = int(d)
    min_f  = (d - deg_i) * 60
    min_i  = int(min_f)
    sec    = (min_f - min_i) * 60

    if sec >= 59.5:
        min_i += 1; sec = 0
    if min_i == 60:
        deg_i += 1; min_i = 0

    if sec > 0.5:
        s = f"{deg_i}\u00b0{min_i:02d}'{sec:04.1f}\""
    elif min_i > 0:
        s = f"{deg_i}\u00b0{min_i:02d}'"
    else:
        s = f"{deg_i}\u00b0"

    if is_lat:
        s += "S" if sign else "N"
    else:
        s += "W" if sign else "E"
    return s


def _draw_graticule(
    output: np.ndarray,
    geo2px: Geo2Px,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    scale: int,
    margin: int,
) -> None:
    H, W = output.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.30, min(0.45, W / 3000))
    thickness  = 3
    grid_color = (140, 140, 140)
    tick_color = (80, 80, 80)
    text_color = (30, 30, 30)
    alpha      = 0.30

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lon_step = _nice_interval(lon_span, 7)
    lat_step = _nice_interval(lat_span, 5)

    lon_start = math.ceil(lon_min / lon_step) * lon_step
    lat_start = math.ceil(lat_min / lat_step) * lat_step

    overlay = output.copy()
    tick_len = max(6, margin // 3)

    lon_lines = []
    v = lon_start
    while v <= lon_max + 1e-9:
        lon_lines.append(round(v, 8))
        v += lon_step
        v  = round(v, 8)

    lat_lines = []
    v = lat_start
    while v <= lat_max + 1e-9:
        lat_lines.append(round(v, 8))
        v += lat_step
        v  = round(v, 8)

    for lon in lon_lines:
        x_top_f, _    = geo2px(lon, lat_max)
        x_bot_f, _    = geo2px(lon, lat_min)
        x_top = int(round(x_top_f * scale))
        x_bot = int(round(x_bot_f * scale))

        if not (margin <= x_top <= W - margin or margin <= x_bot <= W - margin):
            continue

        cv2.line(overlay, (x_top, margin), (x_bot, H - margin), grid_color, 1)
        cv2.line(output,  (x_top, margin), (x_top, margin + tick_len), tick_color, 1)
        cv2.line(output,  (x_bot, H - margin - tick_len), (x_bot, H - margin), tick_color, 1)

        label = _dms(lon, is_lat=False)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx_top = max(margin, min(W - margin - tw, x_top - tw // 2))
        cv2.putText(output, label, (tx_top, margin + tick_len + th + 2),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        tx_bot = max(margin, min(W - margin - tw, x_bot - tw // 2))
        cv2.putText(output, label, (tx_bot, H - margin - tick_len - 4),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

    for lat in lat_lines:
        _, y_left_f   = geo2px(lon_min, lat)
        _, y_right_f  = geo2px(lon_max, lat)
        y_left  = int(round(y_left_f  * scale))
        y_right = int(round(y_right_f * scale))

        if not (margin <= y_left <= H - margin or margin <= y_right <= H - margin):
            continue

        cv2.line(overlay, (margin, y_left), (W - margin, y_right), grid_color, 1)
        cv2.line(output,  (margin, y_left), (margin + tick_len, y_left), tick_color, 1)
        cv2.line(output,  (W - margin - tick_len, y_right), (W - margin, y_right), tick_color, 1)

        label = _dms(lat, is_lat=True)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(output, label,
                    (margin + tick_len + 3, y_left + th // 2),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(output, label,
                    (W - margin - tick_len - tw - 3, y_right + th // 2),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.rectangle(output, (margin, margin), (W - margin, H - margin), tick_color, 1)


def build_output_image(
    sea: np.ndarray,
    features: List[dict],
    island_rings: list,
    h: int,
    w: int,
    geo2px: Geo2Px,
    scale: int = 3,
    gcps: list | None = None,
) -> np.ndarray:
    LEGEND_BGR    = {k: (v["rgb"][2], v["rgb"][1], v["rgb"][0]) for k, v in LEGEND.items()}
    LEGEND_STROKE = {k: _hex_to_bgr(v["stroke"]) for k, v in LEGEND.items()}

    S      = scale
    margin = max(28, int(min(h, w) * S * 0.04))

    output = np.full((h * S, w * S, 3), 230, dtype=np.uint8)

    sea_big = cv2.resize(sea, (w * S, h * S), interpolation=cv2.INTER_NEAREST)
    output[sea_big == 255] = (190, 168, 140)

    cls_name_to_id = {v["name"]: k for k, v in LEGEND.items()}
    for feat in features:
        p   = feat["properties"]
        cls = cls_name_to_id[p["spi_category"]]
        parts = p["grid_id"].split("_")
        x1, y1 = int(parts[1]) * S, int(parts[2]) * S
        cv2.rectangle(output, (x1, y1), (x1 + 3 * S, y1 + 3 * S), LEGEND_BGR[cls], -1)
        cv2.rectangle(output, (x1, y1), (x1 + 3 * S, y1 + 3 * S), LEGEND_STROKE[cls], 1)

    for ring in island_rings:
        pts = [
            [int(round(geo2px(pt[0], pt[1])[0] * S)),
             int(round(geo2px(pt[0], pt[1])[1] * S))]
            for pt in ring
        ]
        cv2.polylines(output, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                      True, (50, 50, 50), 2)

    if gcps is not None and len(gcps) >= 2:
        lons = [g["lon"] for g in gcps]
        lats = [g["lat"] for g in gcps]
        _draw_graticule(
            output, geo2px,
            lon_min=min(lons), lon_max=max(lons),
            lat_min=min(lats), lat_max=max(lats),
            scale=S, margin=margin,
        )
    elif features:
        lons = [f["properties"]["lon_center"] for f in features]
        lats = [f["properties"]["lat_center"] for f in features]
        _draw_graticule(
            output, geo2px,
            lon_min=min(lons), lon_max=max(lons),
            lat_min=min(lats), lat_max=max(lats),
            scale=S, margin=margin,
        )

    lx = w * S - 222
    ly = margin + 6
    cv2.rectangle(output, (lx - 6, ly - 6), (w * S - margin + 4, ly + 206), (255, 255, 255), -1)
    cv2.rectangle(output, (lx - 6, ly - 6), (w * S - margin + 4, ly + 206), (100, 100, 100), 1)

    cv2.putText(output, "Nilai SPI (Kekeringan)",
                (lx, ly + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)

    for i, (k, v) in enumerate(LEGEND.items()):
        yy = ly + 28 + i * 25
        cv2.rectangle(output, (lx, yy), (lx + 22, yy + 18), LEGEND_BGR[k], -1)
        cv2.rectangle(output, (lx, yy), (lx + 22, yy + 18), LEGEND_STROKE[k], 1)
        cv2.putText(output, f"{v['name']}  ({v['spi_range']})",
                    (lx + 28, yy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)

    return output
