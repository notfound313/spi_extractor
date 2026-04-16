from __future__ import annotations
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


from config import ISLAND_GEOREF, LEGEND, GRID_PX
from georef import (
    build_transform,
    build_palette_lab,
    compute_grid_km,
    compute_resolution_m,
    make_geo2px,
    make_px2geo,
)
from masking import build_land_mask, extract_island_polygons
from classification import build_geojson, classify_grid
from spatial import download_and_prepare_provinces
from visualization import build_output_image



st.set_page_config(
    page_title="SPI Grid Extractor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── font & background ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #8b949e !important;
    }

    /* main area */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* stat card */
    .stat-block {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        color: #e6edf3;
        font-weight: 600;
    }
    .stat-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #6e7681;
        margin-top: 0.1rem;
    }

    /* section header */
    .section-head {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
        margin-top: 1.4rem;
    }

    /* legend row */
    .legend-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #c9d1d9;
    }
    .legend-swatch {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid rgba(255,255,255,0.15);
        flex-shrink: 0;
    }
    .legend-count {
        margin-left: auto;
        color: #6e7681;
    }

    /* download button override */
    .stDownloadButton > button {
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        padding: 0.35rem 0.9rem;
        width: 100%;
        margin-bottom: 0.3rem;
    }
    .stDownloadButton > button:hover {
        background: #30363d;
        border-color: #8b949e;
        color: #e6edf3;
    }

    /* progress / info */
    .stProgress > div > div {
        background: #388bfd;
    }
    div[data-testid="stStatusWidget"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## SPI Grid Extractor")
    st.markdown(
        "<span style='font-size:0.75rem;color:#6e7681'>"
        "Standarized Precipitation Index<br>dari citra peta raster"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    target_island = st.selectbox(
        "Pulau Target",
        options=list(ISLAND_GEOREF.keys()),
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Unggah Citra Peta",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Gambar peta SPI dalam format raster.",
    )

    st.divider()

    grid_px = st.slider(
        "Ukuran Grid (piksel)",
        min_value=2,
        max_value=6,
        value=GRID_PX,
        step=1,
        help="Satu piksel ≈ 2–3 km tergantung pulau.",
    )

    use_province = st.checkbox(
        "Spatial Join Provinsi",
        value=True,
        help="Unduh batas provinsi dan tandai setiap grid dengan nama provinsinya.",
    )

    run_btn = st.button("Proses Ekstraksi", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  AREA UTAMA
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='font-family:IBM Plex Mono,monospace;font-size:1.4rem;"
    "color:#e6edf3;font-weight:600;margin-bottom:0.2rem'>"
    "SPI Grid Extractor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e;font-size:0.85rem;margin-top:0'>"
    "Konversi peta raster SPI ke grid GeoJSON bergeoreferensi dengan anotasi provinsi."
    "</p>",
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("Unggah citra peta dan pilih pulau target di sidebar, lalu tekan Proses Ekstraksi.")
    st.stop()

if not run_btn:
    # Tampilkan preview gambar saja
    img_bytes = uploaded_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    preview = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_prev, w_prev = preview.shape[:2]
    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption=f"{uploaded_file.name}   {w_prev}×{h_prev}px",
        use_container_width=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE PEMROSESAN
# ══════════════════════════════════════════════════════════════════════

slug = target_island.lower().replace(" ", "_")
georef_cfg = ISLAND_GEOREF[target_island]

log_area    = st.empty()
prog_bar    = st.progress(0, text="Memulai proses...")
status_area = st.empty()

def _log(msg: str):
    status_area.markdown(
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
        f"color:#8b949e'>{msg}</span>",
        unsafe_allow_html=True,
    )

t_start = time.perf_counter()

prog_bar.progress(5, text="Membaca gambar...")
_log("Membaca gambar...")

img_bytes = uploaded_file.read()
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
if img is None:
    st.error("Gagal membaca gambar. Pastikan format file valid.")
    st.stop()

h_img, w_img = img.shape[:2]
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

prog_bar.progress(10, text="Membangun transform georeferensi...")
_log("Membangun transform georeferensi...")

lon_scale, lon_offset, lat_scale, lat_offset = build_transform(georef_cfg)
px2geo = make_px2geo(lon_scale, lon_offset, lat_scale, lat_offset)
geo2px = make_geo2px(lon_scale, lon_offset, lat_scale, lat_offset)

bbox = georef_cfg["bbox_geo"]  # [lon_min, lat_min, lon_max, lat_max]
lat_mid = (bbox[1] + bbox[3]) / 2

m_per_px_x, m_per_px_y = compute_resolution_m(lon_scale, lat_scale, lat_mid)
km_x, km_y             = compute_grid_km(lon_scale, lat_scale, lat_mid, grid_px)


prog_bar.progress(20, text="Memisahkan daratan dari lautan...")
_log("Memisahkan daratan dari lautan...")

land, sea = build_land_mask(img, img_hsv, georef_cfg)
island_rings = extract_island_polygons(land, px2geo, lat_scale, lon_scale, lat_mid)

processed_provs = []
if use_province:
    prog_bar.progress(30, text="Mengunduh batas provinsi...")

    def _prov_log(msg: str):
        _log(msg)

    processed_provs = download_and_prepare_provinces(target_island, _prov_log)

prog_bar.progress(40, text="Mengklasifikasikan grid SPI...")
_log("Mengklasifikasikan grid SPI... (ini membutuhkan waktu)")

palette = build_palette_lab()

def _cls_progress(cur, tot):
    pct = int(40 + (cur / max(tot, 1)) * 45)
    prog_bar.progress(min(pct, 85), text=f"Klasifikasi grid {cur:,} / {tot:,}...")

features, class_count = classify_grid(
    img_lab, land, island_rings, palette, px2geo,
    processed_provs if use_province else [],
    grid_px,
    progress_callback=_cls_progress,
)

prog_bar.progress(88, text="Menghitung statistik...")
_log("Menghitung statistik...")

total_grids    = len(features)
cell_area_km2  = km_x * km_y
coverage_km2   = total_grids * cell_area_km2

all_lons = [f["properties"]["lon_center"] for f in features]
all_lats = [f["properties"]["lat_center"] for f in features]
lon_min_actual = min(all_lons) if all_lons else bbox[0]
lon_max_actual = max(all_lons) if all_lons else bbox[2]
lat_min_actual = min(all_lats) if all_lats else bbox[1]
lat_max_actual = max(all_lats) if all_lats else bbox[3]

prog_bar.progress(92, text="Merender peta output...")
_log("Merender peta output...")

vis_img = build_output_image(sea, features, island_rings, h_img, w_img, geo2px, scale=3)
_, vis_png_buf = cv2.imencode(".png", vis_img)
vis_png_bytes  = vis_png_buf.tobytes()

prog_bar.progress(96, text="Menyusun GeoJSON...")
_log("Menyusun GeoJSON...")

geojson_obj   = build_geojson(features, slug, uploaded_file.name)
geojson_str   = json.dumps(geojson_obj, separators=(",", ":"))
geojson_bytes = geojson_str.encode("utf-8")

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr(f"{slug}_grid_spi.geojson", geojson_bytes)
    zf.writestr(f"{slug}_classified_spi_map.png", vis_png_bytes)
zip_buffer.seek(0)

t_elapsed = time.perf_counter() - t_start

prog_bar.progress(100, text="Selesai.")
_log(f"Proses selesai dalam {t_elapsed:.1f} detik.")
time.sleep(0.4)
prog_bar.empty()
status_area.empty()


col_map, col_info = st.columns([3, 1], gap="large")

with col_map:
    st.markdown("<div class='section-head'>Peta Klasifikasi SPI</div>", unsafe_allow_html=True)
    st.image(
        cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB),
        caption=(
            f"{uploaded_file.name}   {w_img}×{h_img}px  |  "
            f"{m_per_px_x:.0f}m/px × {m_per_px_y:.0f}m/px"
        ),
        use_container_width=True,
    )

with col_info:
    st.markdown("<div class='section-head'>Ringkasan</div>", unsafe_allow_html=True)

    def _stat(label: str, value: str, sub: str = ""):
        sub_html = f"<div class='stat-sub'>{sub}</div>" if sub else ""
        st.markdown(
            f"<div class='stat-block'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            f"{sub_html}</div>",
            unsafe_allow_html=True,
        )

    _stat("Pulau Target",  target_island)
    _stat("Dimensi Gambar",
          f"{w_img} × {h_img} px",
          f"{m_per_px_x:.0f} m/px × {m_per_px_y:.0f} m/px")
    _stat("Ukuran Grid",
          f"{grid_px}px ≈ {km_x:.1f} × {km_y:.1f} km")
    _stat("Grid Cells",   f"{total_grids:,}")
    _stat("Coverage",     f"~{coverage_km2:,.0f} km²")
    _stat("Koordinat",
          f"{lon_min_actual:.4f}–{lon_max_actual:.4f}°E",
          f"{lat_min_actual:.4f}–{lat_max_actual:.4f}°S")

    
    st.markdown("<div class='section-head'>Distribusi SPI</div>", unsafe_allow_html=True)

    for k, v in LEGEND.items():
        count = class_count.get(k, 0)
        if count == 0:
            continue
        pct = count / max(total_grids, 1) * 100
        hex_color = v["hex"]
        stroke    = v["stroke"]
        st.markdown(
            f"<div class='legend-row'>"
            f"<div class='legend-swatch' style='background:{hex_color};border-color:{stroke}'></div>"
            f"<span>{v['name']}</span>"
            f"<span class='legend-count'>{count:,} ({pct:.1f}%)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    
    st.markdown("<div class='section-head'>Unduh Hasil</div>", unsafe_allow_html=True)

    st.download_button(
        label="Peta PNG",
        data=vis_png_bytes,
        file_name=f"{slug}_classified_spi_map.png",
        mime="image/png",
    )

    st.download_button(
        label="Grid GeoJSON",
        data=geojson_bytes,
        file_name=f"{slug}_grid_spi.geojson",
        mime="application/json",
    )

    st.download_button(
        label="Semua File (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"{slug}_spi_output.zip",
        mime="application/zip",
    )

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
        f"color:#6e7681;margin-top:0.6rem'>"
        f"Diproses dalam {t_elapsed:.1f}s &nbsp;|&nbsp; "
        f"{total_grids:,} fitur &nbsp;|&nbsp; "
        f"{len(geojson_bytes)/1024:.0f} KB GeoJSON"
        f"</div>",
        unsafe_allow_html=True,
    )
