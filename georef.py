from __future__ import annotations
from typing import Callable, Tuple
import math

import cv2
import numpy as np

from config import LEGEND
Px2Geo = Callable[[float, float], Tuple[float, float]]
Geo2Px = Callable[[float, float], Tuple[float, float]]


def build_transform(georef: dict) -> Tuple[float, float, float, float]:
    """
    Hitung skala dan offset linear dari konfigurasi georeferensi.

    Returns    
    (lon_scale, lon_offset, lat_scale, lat_offset)
    """
    lp = georef["lon"]
    ap = georef["lat"]

    lon_scale = (lp["lon2"] - lp["lon1"]) / (lp["px2"] - lp["px1"])
    lon_offset = lp["lon1"] - lp["px1"] * lon_scale

    lat_scale = (ap["lat2"] - ap["lat1"]) / (ap["py2"] - ap["py1"])
    lat_offset = ap["lat1"] - ap["py1"] * lat_scale

    return lon_scale, lon_offset, lat_scale, lat_offset


def make_px2geo(ls: float, lo: float, as_: float, ao: float) -> Px2Geo:
    """Buat fungsi konversi piksel -> (lon, lat)."""
    return lambda px, py: (ls * px + lo, as_ * py + ao)


def make_geo2px(ls: float, lo: float, as_: float, ao: float) -> Geo2Px:
    """Buat fungsi konversi (lon, lat) -> piksel."""
    return lambda lon, lat: ((lon - lo) / ls, (lat - ao) / as_)



def compute_resolution_m(lon_scale: float, lat_scale: float, lat_mid: float) -> Tuple[float, float]:
    """
    Hitung resolusi satu piksel dalam meter.

    Returns
    -------
    (m_per_px_x, m_per_px_y)
    """
    m_per_px_x = abs(lon_scale) * 111_000 * math.cos(math.radians(abs(lat_mid)))
    m_per_px_y = abs(lat_scale) * 111_000
    return m_per_px_x, m_per_px_y


def compute_grid_km(lon_scale: float, lat_scale: float, lat_mid: float, grid_px: int = 3) -> Tuple[float, float]:
    """
    Hitung ukuran satu sel grid dalam km.

    Returns
    -------
    (km_x, km_y)
    """
    km_x = abs(lon_scale) * 111 * math.cos(math.radians(abs(lat_mid))) * grid_px
    km_y = abs(lat_scale) * 111 * grid_px
    return km_x, km_y



def build_palette_lab() -> dict[int, np.ndarray]:
    """
    Konversi warna legenda SPI ke ruang warna LAB untuk klasifikasi
    yang lebih akurat secara perseptual.
    """
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
    """
    Klasifikasikan warna LAB ke kategori SPI terdekat.

    Returns
    -------
    (class_id, confidence 0-1)
    """
    dists = {k: float(np.linalg.norm(lab_color - ref)) for k, ref in palette.items()}
    best = min(dists, key=dists.get)
    sorted_d = sorted(dists.values())
    conf = round(1.0 - sorted_d[0] / (sorted_d[0] + sorted_d[1] + 1e-9), 4)
    return best, conf
