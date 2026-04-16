from __future__ import annotations
from typing import Callable, List, Tuple

import cv2
import numpy as np

from config import LEGEND


Geo2Px = Callable[[float, float], Tuple[float, float]]


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:    
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def build_output_image(
    sea: np.ndarray,
    features: List[dict],
    island_rings: list,
    h: int,
    w: int,
    geo2px: Geo2Px,
    scale: int = 3,
) -> np.ndarray:
    """    
    sea         : masker laut (uint8)
    features    : list GeoJSON Feature dari classify_grid()
    island_rings: list ring poligon pulau
    h, w        : tinggi dan lebar gambar asli (piksel)
    geo2px      : fungsi (lon, lat) -> (px_x, px_y)
    scale       : faktor upscale output (default 3x)    

    """
    LEGEND_BGR    = {k: (v["rgb"][2], v["rgb"][1], v["rgb"][0]) for k, v in LEGEND.items()}
    LEGEND_STROKE = {k: _hex_to_bgr(v["stroke"]) for k, v in LEGEND.items()}

    S = scale
    output = np.full((h * S, w * S, 3), 215, dtype=np.uint8)

    sea_big = cv2.resize(sea, (w * S, h * S), interpolation=cv2.INTER_NEAREST)
    output[sea_big == 255] = (190, 168, 140)


    cls_name_to_id = {v["name"]: k for k, v in LEGEND.items()}
    for feat in features:
        p = feat["properties"]
        cls = cls_name_to_id[p["spi_category"]]
        parts = p["grid_id"].split("_")
        x1, y1 = int(parts[1]) * S, int(parts[2]) * S
        cv2.rectangle(output, (x1, y1), (x1 + 3 * S, y1 + 3 * S), LEGEND_BGR[cls], -1)
        cv2.rectangle(output, (x1, y1), (x1 + 3 * S, y1 + 3 * S), LEGEND_STROKE[cls], 1)

   
    for ring in island_rings:
        pts = [
            [
                int(round(geo2px(pt[0], pt[1])[0] * S)),
                int(round(geo2px(pt[0], pt[1])[1] * S)),
            ]
            for pt in ring
        ]
        cv2.polylines(
            output,
            [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
            True,
            (50, 50, 50),
            2,
        )

   
    lx = w * S - 218
    ly = 12
   
    cv2.rectangle(output, (lx - 6, ly - 6), (w * S - 4, ly + 202), (255, 255, 255), -1)
    cv2.rectangle(output, (lx - 6, ly - 6), (w * S - 4, ly + 202), (100, 100, 100), 1)

    cv2.putText(
        output, "Nilai SPI (Kekeringan)",
        (lx, ly + 13),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA,
    )

    for i, (k, v) in enumerate(LEGEND.items()):
        yy = ly + 28 + i * 25
        cv2.rectangle(output, (lx, yy), (lx + 22, yy + 18), LEGEND_BGR[k], -1)
        cv2.rectangle(output, (lx, yy), (lx + 22, yy + 18), LEGEND_STROKE[k], 1)
        label = f"{v['name']}  ({v['spi_range']})"
        cv2.putText(
            output, label,
            (lx + 28, yy + 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA,
        )

    return output
