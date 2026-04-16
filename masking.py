from __future__ import annotations
from typing import Callable, List, Tuple
import math

import cv2
import numpy as np


Px2Geo = Callable[[float, float], Tuple[float, float]]

def build_sea_mask(img: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Buat masker piksel laut dengan flood-fill dari tepi gambar.

    Returns
    -------
    Masker uint8 (255 = laut, 0 = bukan laut).
    """
    h, w = img.shape[:2]
   
    candidate = cv2.inRange(hsv, (90, 15, 50), (120, 210, 240))

    padded = cv2.copyMakeBorder(candidate, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood = padded.copy()

   
    seed_points = [
        (0, 0), (w // 2, 0), (w - 1, 0),
        (0, h // 2), (w - 1, h // 2),
        (0, h - 1), (w // 2, h - 1), (w - 1, h - 1),
    ]
    for bx, by in seed_points:
        if flood[by + 1, bx + 1] == 255:
            cv2.floodFill(flood, None, (bx + 1, by + 1), 128)

    sea = ((flood[1 : h + 1, 1 : w + 1]) == 128).astype(np.uint8) * 255
    sea = cv2.morphologyEx(sea, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return sea


def build_land_mask(
    img: np.ndarray,
    hsv: np.ndarray,
    georef_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Buat masker daratan dan mengembalikan pasangan (land, sea).

    Piksel di border gambar dan area margin konfigurasi dihapus untuk
    menghindari artefak frame peta.

    Returns
    -------
    (land_mask, sea_mask)  — keduanya uint8 (255 / 0).
    """
    h, w = img.shape[:2]
    sea = build_sea_mask(img, hsv)
    land = cv2.bitwise_not(sea)

    
    B = 10
    land[:B, :]  = 0
    land[-B:, :] = 0
    land[:, :B]  = 0
    land[:, -B:] = 0

    
    px_min = georef_cfg.get("px_min", 0)
    py_min = georef_cfg.get("py_min", 0)
    if px_min > 0:
        land[:, :px_min] = 0
    if py_min > 0:
        land[:py_min, :] = 0

    
    cnts, _ = cv2.findContours(land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(land)
    for c in cnts:
        if cv2.contourArea(c) >= 25:
            cv2.drawContours(clean, [c], -1, 255, -1)

    return clean, sea



def extract_island_polygons(
    land: np.ndarray,
    px2geo: Px2Geo,
    lat_scale: float,
    lon_scale: float,
    lat_mid: float = -9.0,
) -> List[List[Tuple[float, float]]]:
    """
    Ekstrak ring poligon pulau dari masker daratan.

    Returns
    -------
    Daftar ring, masing-masing berupa list koordinat [lon, lat].
    """
    cnts, _ = cv2.findContours(land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings: List[List] = []

    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < 25:
            continue
        approx = cv2.approxPolyDP(
            cnt,
            max(1.0, 0.004 * cv2.arcLength(cnt, True)),
            True,
        )
        ring = [
            [
                round(px2geo(float(pt[0]), float(pt[1]))[0], 6),
                round(px2geo(float(pt[0]), float(pt[1]))[1], 6),
            ]
            for pt in approx.reshape(-1, 2)
        ]
        if len(ring) < 3:
            continue
        ring.append(ring[0])
        rings.append(ring)

    return rings



def pip(lon: float, lat: float, ring: list) -> bool:
    """
    Algoritma ray-casting: cek apakah titik (lon, lat) berada di dalam ring.
    """
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside
