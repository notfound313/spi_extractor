from __future__ import annotations
import json
import urllib.request
from typing import List, Optional

from config import ISLAND_PROVINCES, GEOJSON_URLS
from masking import pip



ProvinceRecord = dict  


def download_and_prepare_provinces(
    island_key: str,
    progress_callback=None,
) -> List[ProvinceRecord]:
    
    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    _log("Mengunduh batas provinsi...")
    prov_data: Optional[dict] = None

    for url in GEOJSON_URLS:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                prov_data = json.loads(resp.read().decode("utf-8"))
            _log(f"Data batas provinsi berhasil diunduh.")
            break
        except Exception as exc:
            _log(f"Gagal unduh dari {url}: {exc}")

    if prov_data is None:
        _log("Gagal mengunduh batas provinsi. Koneksi internet bermasalah.")
        return []

    target_provs = ISLAND_PROVINCES.get(island_key, [])
    processed: List[ProvinceRecord] = []

    for feat in prov_data.get("features", []):
        props = feat.get("properties", {})
        name = (
            props.get("Propinsi")
            or props.get("state")
            or props.get("NAME_1")
            or "Unknown"
        )
        if name not in target_provs:
            continue

        geom = feat.get("geometry", {})
        rings: list = []
        if geom["type"] == "Polygon":
            rings.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                rings.append(poly[0])

        if not rings:
            continue

        min_lon = min(min(pt[0] for pt in r) for r in rings)
        max_lon = max(max(pt[0] for pt in r) for r in rings)
        min_lat = min(min(pt[1] for pt in r) for r in rings)
        max_lat = max(max(pt[1] for pt in r) for r in rings)

        processed.append(
            {
                "name": name,
                "bbox": (min_lon, min_lat, max_lon, max_lat),
                "rings": rings,
            }
        )

    _log(f"Dimuat {len(processed)} poligon provinsi untuk {island_key}.")
    return processed


def find_province(
    lon: float,
    lat: float,
    processed_provs: List[ProvinceRecord],
) -> str:
    """
    Cari nama provinsi dari koordinat (lon, lat).
    Menggunakan optimasi bounding-box sebelum cek PIP penuh.

    Returns
    -------
    Nama provinsi, atau "Luar Batas / Pesisir" jika tidak ditemukan.
    """
    for p in processed_provs:
        min_lon, min_lat, max_lon, max_lat = p["bbox"]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            if any(pip(lon, lat, ring) for ring in p["rings"]):
                return p["name"]
    return "Luar Batas / Pesisir"
