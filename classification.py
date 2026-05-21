from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np
import cv2

from config import LEGEND
from georef import classify_lab_with_conf
from spatial import find_province, ProvinceRecord


Px2Geo = Callable[[float, float], Tuple[float, float]]

_LAND_RATIO_THRESHOLD = 0.25
_MIN_CLEAN_PIXELS: int = 2

_ADMIN_LINE_BGR = np.array(
    [[217,  90,   5],  
     [103, 125, 124],
     [151, 141, 117],
     [197, 90, 14],
     [140, 89, 50],
     [173, 86, 30],
     [93, 112, 111],
     [99,110,110]], 
    dtype=np.uint8,
).reshape(-1, 1, 3)

_ADMIN_LINE_LAB: np.ndarray = (
    cv2.cvtColor(_ADMIN_LINE_BGR, cv2.COLOR_BGR2LAB)
    .reshape(-1, 3)
    .astype(np.float32)
)
_ADMIN_DIST_THRESHOLD: float = 18
_ADMIN_DIST_THRESHOLD_SQ: float = _ADMIN_DIST_THRESHOLD ** 2

def _build_admin_mask(img_lab: np.ndarray) -> np.ndarray:   
    h, w   = img_lab.shape[:2]
    lab_f  = img_lab.reshape(-1, 3).astype(np.float32)   
    min_sq = np.full(h * w, np.inf, dtype=np.float32)  
 
    for admin_lab in _ADMIN_LINE_LAB:                    
        diff = lab_f - admin_lab                         
        sq   = (diff * diff).sum(axis=1)       
        np.minimum(min_sq, sq, out=min_sq)              
 
    is_admin = (min_sq <= _ADMIN_DIST_THRESHOLD_SQ).reshape(h, w).astype(np.uint8)
 
   
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    is_admin = cv2.dilate(is_admin, kernel, iterations=1)
 
    return is_admin.astype(bool)   


def classify_grid(
    img_lab: np.ndarray,
    land: np.ndarray,
    palette: dict,
    px2geo: Px2Geo,
    processed_provs: List[ProvinceRecord],
    grid: int = 3,
    progress_callback=None,
) -> Tuple[List[dict], dict]:
    h, w = land.shape
    features: List[dict] = []
    class_count: dict[int, int] = {k: 0 for k in LEGEND}
    admin_mask = _build_admin_mask(img_lab)

    ys          = range(0, h, grid)
    xs          = range(0, w, grid)
    total_cells = len(ys) * len(xs)
    processed   = 0

    for y in ys:
        for x in xs:
            processed += 1
            if progress_callback and processed % 500 == 0:
                progress_callback(processed, total_cells)

            pm = land[y : y + grid, x : x + grid]

            land_pixels = int(np.sum(pm == 255))
            total_pixels = pm.size
            if total_pixels == 0 or land_pixels / total_pixels < _LAND_RATIO_THRESHOLD:
                continue

            am_patch = admin_mask[y : y + grid, x : x + grid]  
            usable   = (pm == 255) & ~am_patch                 
 
            lpx = img_lab[y : y + grid, x : x + grid][usable]
            if len(lpx) < _MIN_CLEAN_PIXELS:              
                continue

            lpx_f    = lpx.astype(np.float32)                             
            diffs    = lpx_f[:, np.newaxis, :] - _ADMIN_LINE_LAB          
            sq_dists = (diffs * diffs).sum(axis=2)                      
            min_sq   = sq_dists.min(axis=1)                           
            lpx      = lpx[min_sq > _ADMIN_DIST_THRESHOLD_SQ]
            if len(lpx) == 0:
                continue

            med = np.median(lpx, axis=0)
            cls, conf = classify_lab_with_conf(med, palette)

            lon_c, lat_c = px2geo(x + grid / 2, y + grid / 2)

            prov_name = find_province(lon_c, lat_c, processed_provs) if processed_provs else "Tidak Diketahui"

            if prov_name == "Luar Batas / Pesisir":
                continue

            class_count[cls] += 1

            lon_w, lat_n = px2geo(x,        y)
            lon_e, lat_s = px2geo(x + grid, y + grid)

            t = conf
            r0, g0, b0 = LEGEND[cls]["rgb"]
            rs, gs, bs = LEGEND[cls]["spin_rgb"]
            fill_color = (
                f"#{int(rs*t + r0*(1-t)):02X}"
                f"{int(gs*t + g0*(1-t)):02X}"
                f"{int(bs*t + b0*(1-t)):02X}"
            )

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [round(lon_w, 6), round(lat_n, 6)],
                                [round(lon_e, 6), round(lat_n, 6)],
                                [round(lon_e, 6), round(lat_s, 6)],
                                [round(lon_w, 6), round(lat_s, 6)],
                                [round(lon_w, 6), round(lat_n, 6)],
                            ]
                        ],
                    },
                    "properties": {
                        "grid_id":        f"G_{x}_{y}",
                        "provinsi":       prov_name,
                        "spi_category":   LEGEND[cls]["name"],
                        "spi_range":      LEGEND[cls]["spi_range"],
                        "spi_value":      LEGEND[cls]["spi_value"],
                        "lon_center":     round(lon_c, 6),
                        "lat_center":     round(lat_c, 6),
                        "fill":           fill_color,
                        "fill-opacity":   round(LEGEND[cls]["fill_opacity"] * (0.7 + 0.3 * conf), 3),
                        "stroke":         LEGEND[cls]["stroke"],
                        "stroke-width":   LEGEND[cls]["stroke_width"],
                        "stroke-opacity": 0.9,
                        "title": (
                            f"Daerah: {prov_name} | "
                            f"SPI: {LEGEND[cls]['spi_value']} ({LEGEND[cls]['name']})"
                        ),
                    },
                }
            )

    return features, class_count


def build_geojson(features: List[dict], slug: str, image_path: str) -> dict:
    return {
        "type": "FeatureCollection",
        "name": f"{slug}_spi_grid",
        "metadata": {
            "description": "Grid klasifikasi SPI dengan anotasi nama daerah (provinsi)",
            "source": image_path,
            "total_grids": len(features),
        },
        "features": features,
    }
