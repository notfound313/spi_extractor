from __future__ import annotations
import math
from typing import Callable, Tuple

import cv2
import numpy as np

from config import LEGEND


Px2Geo = Callable[[float, float], Tuple[float, float]]
Geo2Px = Callable[[float, float], Tuple[float, float]]


def _scale_gcps(gcps: list[dict], sx: float, sy: float) -> list[dict]:
    return [
        {**g, "px": g["px"] * sx, "py": g["py"] * sy}
        for g in gcps
    ]


def build_homography(
    gcps: list[dict],
    img_w: int,
    img_h: int,
    master_w: int,
    master_h: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    sx = img_w / img_w
    sy = img_h / img_h
    scaled = _scale_gcps(gcps, sx, sy)

    src = np.float64([[g["px"], g["py"]] for g in scaled])
    dst = np.float64([[g["lon"], g["lat"]] for g in scaled])

    H_px2geo, _ = cv2.findHomography(src, dst)
    H_geo2px, _ = cv2.findHomography(dst, src)

    return H_px2geo, H_geo2px, sx, sy


def make_px2geo(H: np.ndarray) -> Px2Geo:
    def _fn(px: float, py: float) -> Tuple[float, float]:
        pt  = np.float64([[[px, py]]])
        res = cv2.perspectiveTransform(pt, H)
        return float(res[0][0][0]), float(res[0][0][1])
    return _fn


def make_geo2px(H_inv: np.ndarray) -> Geo2Px:
    def _fn(lon: float, lat: float) -> Tuple[float, float]:
        pt  = np.float64([[[lon, lat]]])
        res = cv2.perspectiveTransform(pt, H_inv)
        return float(res[0][0][0]), float(res[0][0][1])
    return _fn


def compute_resolution_m(
    gcps: list[dict],
    sx: float,
    sy: float,
) -> Tuple[float, float]:
    nw = next(g for g in gcps if g["px"] == min(g2["px"] for g2 in gcps) and
              g["py"] == min(g2["py"] for g2 in gcps if g2["px"] == min(g3["px"] for g3 in gcps)))
    ne = next(g for g in gcps if g["px"] == max(g2["px"] for g2 in gcps) and
              g["py"] == min(g2["py"] for g2 in gcps if g2["px"] == max(g3["px"] for g3 in gcps)))
    sw = next(g for g in gcps if g["px"] == min(g2["px"] for g2 in gcps) and
              g["py"] == max(g2["py"] for g2 in gcps if g2["px"] == min(g3["px"] for g3 in gcps)))

    lat_mid    = (nw["lat"] + sw["lat"]) / 2
    dx_geo_m   = abs(ne["lon"] - nw["lon"]) * 111_000 * math.cos(math.radians(abs(lat_mid)))
    dx_px      = abs(ne["px"] - nw["px"]) * sx
    dy_geo_m   = abs(sw["lat"] - nw["lat"]) * 111_000
    dy_px      = abs(sw["py"] - nw["py"]) * sy

    m_per_px_x = dx_geo_m / dx_px if dx_px else 1.0
    m_per_px_y = dy_geo_m / dy_px if dy_px else 1.0
    return m_per_px_x, m_per_px_y


def compute_grid_km(
    gcps: list[dict],
    sx: float,
    sy: float,
    grid_px: int,
) -> Tuple[float, float]:
    m_per_px_x, m_per_px_y = compute_resolution_m(gcps, sx, sy)
    return (m_per_px_x * grid_px) / 1_000, (m_per_px_y * grid_px) / 1_000


def gcp_bbox(gcps: list[dict]) -> Tuple[float, float, float, float]:
    lons = [g["lon"] for g in gcps]
    lats = [g["lat"] for g in gcps]
    return min(lons), min(lats), max(lons), max(lats)


def build_palette_lab() -> dict[int, np.ndarray]:
    palette: dict[int, np.ndarray] = {}
    for k, v in LEGEND.items():
        bgr = np.array([[[v["rgb"][2], v["rgb"][1], v["rgb"][0]]]], dtype=np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3).astype(float)
        palette[k] = lab
    return palette


def classify_lab_with_conf(
    lab_color: np.ndarray,
    palette: dict[int, np.ndarray],
) -> Tuple[int, float]:
    dists   = {k: float(np.linalg.norm(lab_color - ref)) for k, ref in palette.items()}
    best    = min(dists, key=dists.get)
    sorted_d = sorted(dists.values())
    conf    = round(1.0 - sorted_d[0] / (sorted_d[0] + sorted_d[1] + 1e-9), 4)
    return best, conf
