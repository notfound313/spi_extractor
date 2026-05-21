from __future__ import annotations
from typing import Callable, List, Tuple

import cv2
import numpy as np



Px2Geo = Callable[[float, float], Tuple[float, float]]


_MIN_LAND_AREA_PX  = 25       
_EPSILON_MAIN      = 0.0008   
_EPSILON_SMALL     = 0.0020   
_MIN_APPROX_PTS    = 30      
_MAX_RAW_PTS       = 1_500    
_SMOOTH_WINDOW     = 9
_CLOSE_ITER_MAIN   = 3        
_CLOSE_ITER_SEA    = 2       


def _smooth_contour_gaussian(cnt: np.ndarray, window: int = 9) -> np.ndarray:
    
    pts = cnt.reshape(-1, 2).astype(np.float32)
    n   = len(pts)
    if n < window * 3:          
        return cnt.astype(np.float32)

    half   = window // 2
    kernel = np.ones(window, dtype=np.float32) / window  


    pad_x = np.pad(pts[:, 0], (half, half), mode="wrap")
    pad_y = np.pad(pts[:, 1], (half, half), mode="wrap")

    sx = np.convolve(pad_x, kernel, mode="valid")[:n]
    sy = np.convolve(pad_y, kernel, mode="valid")[:n]

    return np.stack([sx, sy], axis=1).reshape(-1, 1, 2)


def _fill_land_holes(land: np.ndarray) -> np.ndarray:  
    h, w  = land.shape
    inv   = cv2.bitwise_not(land)

    padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1,
                                 cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 128)
    
    interior = flooded[1:h + 1, 1:w + 1]
    holes    = (interior != 128).astype(np.uint8) * 255
    return cv2.bitwise_or(land, holes)


def build_sea_mask(img: np.ndarray, hsv: np.ndarray) -> np.ndarray: 
    h, w = img.shape[:2]

    
    sea_blue  = cv2.inRange(hsv, ( 88,  12,  40), (126, 230, 255))
    # White legend / margins
    sea_white = cv2.inRange(hsv, (  0,   0, 165), (180,  35, 255))
    # Light-grey axis ticks
    sea_grey  = cv2.inRange(hsv, (  0,   0, 140), (180,  20, 200))

    candidate = cv2.bitwise_or(sea_blue,
                cv2.bitwise_or(sea_white, sea_grey))

    k5        = np.ones((5, 5), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,
                                  k5, iterations=_CLOSE_ITER_SEA)
    
    padded = cv2.copyMakeBorder(candidate, 1, 1, 1, 1,
                                 cv2.BORDER_CONSTANT, value=255)
    flood  = padded.copy()

    # 16 seeds spread across all four edges
    seeds = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        seeds += [
            (int(frac * (w - 1)), 0),
            (int(frac * (w - 1)), h - 1),
            (0,     int(frac * (h - 1))),
            (w - 1, int(frac * (h - 1))),
        ]

    for bx, by in seeds:
        if flood[by + 1, bx + 1] == 255:
            cv2.floodFill(flood, None, (bx + 1, by + 1), 128)

    sea = ((flood[1:h + 1, 1:w + 1]) == 128).astype(np.uint8) * 255

   
    sea = cv2.morphologyEx(sea, cv2.MORPH_CLOSE, k5, iterations=2)
    return sea


def build_land_mask(
    img: np.ndarray,
    hsv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:   
    h, w = img.shape[:2]
    sea  = build_sea_mask(img, hsv)
    land = cv2.bitwise_not(sea)
   
    k7   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, k7,
                             iterations=_CLOSE_ITER_MAIN)

    
    land = _fill_land_holes(land)

    
    margin = max(2, min(h, w) // 100)
    land[:margin, :]  = 0
    land[-margin:, :] = 0
    land[:, :margin]  = 0
    land[:, -margin:] = 0

   
    cnts, _ = cv2.findContours(land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(land)
    for c in cnts:
        if cv2.contourArea(c) >= _MIN_LAND_AREA_PX:
            cv2.drawContours(clean, [c], -1, 255, -1)

    return clean, sea


def extract_island_polygons(
    land: np.ndarray,
    px2geo: Px2Geo,
    *,
    main_epsilon_factor: float = _EPSILON_MAIN,
    small_epsilon_factor: float = _EPSILON_SMALL,
    min_points: int = _MIN_APPROX_PTS,
    max_raw_points: int = _MAX_RAW_PTS,
    smoothing_window: int = _SMOOTH_WINDOW,
) -> List[List]:    
    cnts, _ = cv2.findContours(land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []

    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rings: List[List] = []

    for idx, cnt in enumerate(sorted_cnts):
        if cv2.contourArea(cnt) < _MIN_LAND_AREA_PX:
            continue
       
        smoothed = _smooth_contour_gaussian(cnt, window=smoothing_window)
       
        eps_factor = main_epsilon_factor if idx == 0 else small_epsilon_factor
        arc_len    = cv2.arcLength(smoothed, True)
        epsilon    = max(0.5, eps_factor * arc_len)   
        approx     = cv2.approxPolyDP(smoothed, epsilon, True)

        
        if len(approx) < min_points:
            raw  = smoothed.reshape(-1, 2)
            step = max(1, len(raw) // max_raw_points)
            approx = raw[::step].reshape(-1, 1, 2)

        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue

       
        ring = [
            [round(px2geo(float(p[0]), float(p[1]))[0], 6),
             round(px2geo(float(p[0]), float(p[1]))[1], 6)]
            for p in pts
        ]

       
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        if len(ring) < 4:   
            continue

        rings.append(ring)

    return rings


def pip(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j      = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside