from __future__ import annotations
import csv
import io
import json
import time
import zipfile

import cv2
import numpy as np
import streamlit as st

from config import ISLAND_GEOREF, LEGEND, GRID_PX
from georef import (
    build_homography, build_palette_lab,
    compute_grid_km, compute_resolution_m,
    gcp_bbox, make_geo2px, make_px2geo,
)
from masking import build_land_mask, extract_island_polygons
from classification import build_geojson, classify_grid
from spatial import download_and_prepare_provinces
from visualization import build_output_image
from export import features_to_csv_bytes, features_to_records
from calibration import render_calibration


st.set_page_config(
    page_title="SPI Grid Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
section[data-testid="stSidebar"]{background:#0f1117;border-right:1px solid #1e2130}
section[data-testid="stSidebar"] *{color:#c9d1d9 !important}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label{
  font-size:.78rem;letter-spacing:.05em;text-transform:uppercase;color:#8b949e !important}
.main .block-container{padding-top:1.6rem;padding-bottom:2rem;max-width:1200px}
.stat-block{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:.9rem 1.1rem;margin-bottom:.5rem}
.stat-label{font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.15rem}
.stat-value{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;color:#e6edf3;font-weight:600}
.stat-sub{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#6e7681;margin-top:.1rem}
.section-head{
  font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#8b949e;
  text-transform:uppercase;letter-spacing:.12em;
  border-bottom:1px solid #21262d;padding-bottom:.3rem;
  margin-bottom:.8rem;margin-top:1.4rem}
.step-indicator{display:flex;align-items:center;gap:.5rem;font-family:'IBM Plex Mono',monospace;font-size:.75rem;margin-bottom:1.2rem}
.step-pill{padding:4px 14px;border-radius:20px;font-size:.72rem;font-family:'IBM Plex Mono',monospace}
.step-active{background:#1a3a5c;color:#58a6ff;border:1px solid #388bfd}
.step-done{background:#0d2d10;color:#3fb950;border:1px solid #2ea043}
.step-idle{background:#161b22;color:#6e7681;border:1px solid #21262d}
.step-arrow{color:#30363d;font-size:.9rem}
.legend-row{display:flex;align-items:center;gap:.5rem;padding:.25rem 0;font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#c9d1d9}
.legend-count{margin-left:auto;color:#6e7681}
.stDownloadButton>button{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  border-radius:6px;font-family:'IBM Plex Mono',monospace;
  font-size:.78rem;padding:.35rem .9rem;width:100%;margin-bottom:.3rem}
.stDownloadButton>button:hover{background:#30363d;border-color:#8b949e;color:#e6edf3}
.stProgress>div>div{background:#388bfd}
div[data-testid="stStatusWidget"]{display:none}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## SPI Grid Extractor")
    st.markdown(
        "<span style='font-size:.75rem;color:#6e7681'>"
        "Standardized Precipitation Index<br>dari citra peta raster</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    target_island = st.selectbox("Pulau Target", options=list(ISLAND_GEOREF.keys()), index=0)
    uploaded_file = st.file_uploader(
        "Unggah Citra Peta",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    st.divider()
    grid_px      = st.slider("Ukuran Grid (piksel)", min_value=2, max_value=6, value=GRID_PX, step=1)
    use_province = st.checkbox("Spatial Join Provinsi", value=True)

    app_step = st.session_state.get("app_step", "calibrate")
    st.divider()

    if app_step == "process":
        if st.button("Ubah Kalibrasi", use_container_width=True):
            for k in [
                "app_step", "has_result", "cached_file_id",
                "vis_png_bytes", "geojson_bytes", "csv_bytes",
                "zip_bytes", "df_records", "class_count", "slug", "stats", "img_name",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

    if st.session_state.get("has_result"):
        if st.button("Kembali / Reset Hasil", use_container_width=True):
            for k in [
                "has_result", "cached_file_id",
                "vis_png_bytes", "geojson_bytes", "csv_bytes",
                "zip_bytes", "df_records", "class_count", "slug", "stats", "img_name",
            ]:
                st.session_state.pop(k, None)
            st.rerun()


st.markdown(
    "<h1 style='font-family:IBM Plex Mono,monospace;font-size:1.4rem;"
    "color:#e6edf3;font-weight:600;margin-bottom:.5rem'>SPI Grid Extractor</h1>",
    unsafe_allow_html=True,
)

file_sig = (
    f"{uploaded_file.name}_{uploaded_file.size}_{target_island}"
    if uploaded_file else None
)

if file_sig and st.session_state.get("file_sig") != file_sig:
    for k in list(st.session_state.keys()):
        if k not in ("custom_georef",):
            del st.session_state[k]
    st.session_state["file_sig"] = file_sig
    st.session_state["app_step"] = "calibrate"

app_step = st.session_state.get("app_step", "calibrate")

cal_done  = bool(st.session_state.get("custom_georef", {}).get(target_island))
proc_done = st.session_state.get("has_result", False)

s1_cls = "step-done" if cal_done else ("step-active" if app_step == "calibrate" else "step-idle")
s2_cls = "step-active" if (app_step == "process" and not proc_done) else ("step-done" if proc_done else "step-idle")

st.markdown(
    f"<div class='step-indicator'>"
    f"<span class='step-pill {s1_cls}'>1. Kalibrasi Georef (4 GCP)</span>"
    f"<span class='step-arrow'>›</span>"
    f"<span class='step-pill {s2_cls}'>2. Proses Ekstraksi</span>"
    f"</div>",
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("Unggah citra peta dan pilih pulau target di sidebar untuk memulai.")
    st.stop()

if st.session_state.get("file_sig") == file_sig and "img_bytes_cache" in st.session_state:
    img_bytes = st.session_state["img_bytes_cache"]
else:
    img_bytes = uploaded_file.read()
    st.session_state["img_bytes_cache"] = img_bytes


# ============================================================================
# STEP 1 — KALIBRASI
# ============================================================================

if app_step == "calibrate":
    st.markdown("<div class='section-head'>Kalibrasi Georeferensi  —  4-Point Homography</div>",
                unsafe_allow_html=True)
    render_calibration(img_bytes, target_island)
    st.stop()


# ============================================================================
# STEP 2 — PROSES EKSTRAKSI
# ============================================================================

st.markdown("<div class='section-head'>Proses Ekstraksi SPI</div>", unsafe_allow_html=True)

georef_cfg = (
    st.session_state.get("custom_georef", {}).get(target_island)
    or ISLAND_GEOREF[target_island]
)

gcps       = georef_cfg["gcps"]
master_w, master_h = georef_cfg["master_size"]

if not st.session_state.get("has_result"):
    arr      = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h_p, w_p = arr.shape[:2]

    sx = w_p / master_w
    sy = h_p / master_h

    gcp_summary = "  |  ".join(
        f"GCP{i+1} px={round(g['px']*sx)} py={round(g['py']*sy)} → ({g['lon']:.2f}°, {g['lat']:.2f}°)"
        for i, g in enumerate(gcps)
    )
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;"
        f"color:#c9d1d9;background:#0d1117;border:1px solid #21262d;"
        f"border-radius:6px;padding:.6rem 1rem;margin-bottom:1rem;overflow-x:auto;white-space:nowrap'>"
        f"<span style='color:#3fb950'>Homography GCPs aktif</span>  "
        f"(sx={sx:.3f} sy={sy:.3f})  &nbsp;  {gcp_summary}</div>",
        unsafe_allow_html=True,
    )

    col_prev, col_settings = st.columns([3, 1], gap="large")
    with col_prev:
        st.image(
            cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
            caption=f"{uploaded_file.name}   {w_p}×{h_p}px",
            use_container_width=True,
        )
    with col_settings:
        st.markdown("<div class='section-head'>Pengaturan</div>", unsafe_allow_html=True)

        def _sbox(label, value):
            st.markdown(
                f"<div class='stat-block'><div class='stat-label'>{label}</div>"
                f"<div class='stat-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )

        _sbox("Pulau",       target_island)
        _sbox("Ukuran Grid", f"{grid_px}px")
        _sbox("Skala Auto",  f"sx={sx:.3f} / sy={sy:.3f}")
        _sbox("Spatial Join", "Aktif" if use_province else "Nonaktif")

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.button("Mulai Proses Ekstraksi", type="primary", use_container_width=True):
            st.session_state["run_process"] = True
            st.rerun()

    if not st.session_state.get("run_process"):
        st.stop()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

if st.session_state.get("run_process") and not st.session_state.get("has_result"):
    slug    = target_island.lower().replace(" ", "_")
    file_id = f"{file_sig}_{grid_px}_{use_province}"

    prog_bar    = st.progress(0, text="Memulai proses...")
    status_area = st.empty()

    def _log(msg: str):
        status_area.markdown(
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.78rem;"
            f"color:#8b949e'>{msg}</span>",
            unsafe_allow_html=True,
        )

    t0 = time.perf_counter()

    prog_bar.progress(5, text="Membaca gambar...")
    _log("Membaca gambar...")
    nparr        = np.frombuffer(img_bytes, np.uint8)
    img          = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_img, w_img = img.shape[:2]
    img_hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    prog_bar.progress(10, text="Homography + auto-scaling...")
    _log("Membangun homografi 4-titik & skala otomatis...")

    gcps       = georef_cfg["gcps"]
    master_w, master_h = georef_cfg["master_size"]

    H_px2geo, H_geo2px, sx, sy = build_homography(gcps, w_img, h_img, master_w, master_h)
    px2geo = make_px2geo(H_px2geo)
    geo2px = make_geo2px(H_geo2px)

    lon_min, lat_min, lon_max, lat_max = gcp_bbox(gcps)
    lat_mid = (lat_min + lat_max) / 2

    m_per_px_x, m_per_px_y = compute_resolution_m(gcps, sx, sy)
    km_x, km_y             = compute_grid_km(gcps, sx, sy, grid_px)

    prog_bar.progress(20, text="Masking daratan...")
    _log("Memisahkan daratan dari lautan...")
    land, sea    = build_land_mask(img, img_hsv)
    island_rings = extract_island_polygons(land, px2geo)

    processed_provs = []
    if use_province:
        prog_bar.progress(30, text="Mengunduh batas provinsi...")
        processed_provs = download_and_prepare_provinces(target_island, _log)

    prog_bar.progress(40, text="Klasifikasi grid SPI...")
    _log("Mengklasifikasikan grid SPI...")
    palette = build_palette_lab()

    def _cls_prog(cur, tot):
        pct = int(40 + cur / max(tot, 1) * 45)
        prog_bar.progress(min(pct, 85), text=f"Klasifikasi grid {cur:,} / {tot:,}...")

    features, class_count = classify_grid(
        img_lab, land, palette, px2geo,
        processed_provs if use_province else [],
        grid_px, progress_callback=_cls_prog,
    )

    prog_bar.progress(88, text="Statistik...")
    _log("Menghitung statistik...")
    total_grids  = len(features)
    coverage_km2 = total_grids * km_x * km_y
    all_lons     = [f["properties"]["lon_center"] for f in features]
    all_lats     = [f["properties"]["lat_center"] for f in features]

    prog_bar.progress(92, text="Render peta...")
    _log("Merender peta output...")
    vis_img       = build_output_image(sea, features, island_rings, h_img, w_img, geo2px, scale=3)
    _, vis_buf    = cv2.imencode(".png", vis_img)
    vis_png_bytes = vis_buf.tobytes()

    prog_bar.progress(95, text="GeoJSON...")
    _log("Menyusun GeoJSON...")
    geojson_obj   = build_geojson(features, slug, uploaded_file.name)
    geojson_bytes = json.dumps(geojson_obj, separators=(",", ":")).encode("utf-8")

    prog_bar.progress(97, text="CSV...")
    _log("Mengekspor CSV...")
    csv_bytes  = features_to_csv_bytes(features)
    df_records = features_to_records(features)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{slug}_classified_spi_map.png", vis_png_bytes)
        zf.writestr(f"{slug}_grid_spi.geojson", geojson_bytes)
        zf.writestr(f"{slug}_grid_spi.csv", csv_bytes)

    t_elapsed = time.perf_counter() - t0

    st.session_state.update({
        "has_result":     True,
        "run_process":    False,
        "cached_file_id": file_id,
        "vis_png_bytes":  vis_png_bytes,
        "geojson_bytes":  geojson_bytes,
        "csv_bytes":      csv_bytes,
        "zip_bytes":      zip_buf.getvalue(),
        "df_records":     df_records,
        "class_count":    class_count,
        "slug":           slug,
        "img_name":       uploaded_file.name,
        "stats": {
            "w_img":        w_img,
            "h_img":        h_img,
            "m_per_px_x":   m_per_px_x,
            "m_per_px_y":   m_per_px_y,
            "km_x":         km_x,
            "km_y":         km_y,
            "grid_px":      grid_px,
            "sx":           sx,
            "sy":           sy,
            "total_grids":  total_grids,
            "coverage_km2": coverage_km2,
            "lon_min":      min(all_lons) if all_lons else lon_min,
            "lon_max":      max(all_lons) if all_lons else lon_max,
            "lat_min":      min(all_lats) if all_lats else lat_min,
            "lat_max":      max(all_lats) if all_lats else lat_max,
            "t_elapsed":    t_elapsed,
        },
    })

    prog_bar.progress(100, text="Selesai.")
    time.sleep(0.3)
    prog_bar.empty()
    status_area.empty()
    st.rerun()


# ---------------------------------------------------------------------------
# Tampilan hasil
# ---------------------------------------------------------------------------

if not st.session_state.get("has_result"):
    st.stop()

vis_png_bytes = st.session_state["vis_png_bytes"]
geojson_bytes = st.session_state["geojson_bytes"]
csv_bytes     = st.session_state["csv_bytes"]
zip_bytes     = st.session_state["zip_bytes"]
df_records    = st.session_state["df_records"]
class_count   = st.session_state["class_count"]
slug          = st.session_state["slug"]
img_name      = st.session_state["img_name"]
s             = st.session_state["stats"]


def _stat(label, value, sub=""):
    sub_html = f"<div class='stat-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='stat-block'><div class='stat-label'>{label}</div>"
        f"<div class='stat-value'>{value}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )


col_map, col_info = st.columns([3, 1], gap="large")

with col_map:
    vis_rgb = cv2.imdecode(np.frombuffer(vis_png_bytes, np.uint8), cv2.IMREAD_COLOR)
    st.image(
        cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB),
        caption=(
            f"{img_name}   {s['w_img']}×{s['h_img']}px  |  "
            f"{s['m_per_px_x']:.0f}m/px × {s['m_per_px_y']:.0f}m/px\n"
            f"Grid {s['grid_px']}px ≈ {s['km_x']:.1f}km × {s['km_y']:.1f}km  |  "
            f"Scale sx={s['sx']:.3f} sy={s['sy']:.3f}"
        ),
        use_container_width=True,
    )

with col_info:
    st.markdown("<div class='section-head'>Ringkasan</div>", unsafe_allow_html=True)
    _stat("Pulau Target",   target_island)
    _stat("Dimensi Gambar",
          f"{s['w_img']} × {s['h_img']} px",
          f"{s['m_per_px_x']:.0f} m/px × {s['m_per_px_y']:.0f} m/px")
    _stat("Skala Otomatis", f"sx={s['sx']:.3f}  sy={s['sy']:.3f}")
    _stat("Ukuran Grid",    f"{s['grid_px']}px ≈ {s['km_x']:.1f} × {s['km_y']:.1f} km")
    _stat("Grid Cells",     f"{s['total_grids']:,}")
    _stat("Coverage",       f"~{s['coverage_km2']:,.0f} km²")
    _stat("Koordinat",
          f"{s['lon_min']:.4f}–{s['lon_max']:.4f}°E",
          f"{s['lat_min']:.4f}–{s['lat_max']:.4f}°S")

    st.markdown("<div class='section-head'>Distribusi SPI</div>", unsafe_allow_html=True)
    for k, v in LEGEND.items():
        count = class_count.get(k, 0)
        if count == 0:
            continue
        pct = count / max(s["total_grids"], 1) * 100
        st.markdown(
            f"<div class='legend-row'>"
            f"<span style='width:14px;height:14px;border-radius:3px;flex-shrink:0;"
            f"background:{v['hex']};border:1px solid {v['stroke']};display:inline-block'></span>"
            f"<span>{v['name']}</span>"
            f"<span class='legend-count'>{count:,} ({pct:.1f}%)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-head'>Unduh Hasil</div>", unsafe_allow_html=True)
    st.download_button("Peta PNG",         vis_png_bytes, f"{slug}_classified_spi_map.png", "image/png")
    st.download_button("Grid GeoJSON",     geojson_bytes, f"{slug}_grid_spi.geojson",       "application/json")
    st.download_button("Data CSV",         csv_bytes,     f"{slug}_grid_spi.csv",            "text/csv")
    st.download_button("Semua File (ZIP)", zip_bytes,     f"{slug}_spi_output.zip",          "application/zip")

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.68rem;"
        f"color:#6e7681;margin-top:.6rem'>"
        f"Diproses dalam {s['t_elapsed']:.1f}s &nbsp;|&nbsp; "
        f"{s['total_grids']:,} fitur &nbsp;|&nbsp; "
        f"{len(geojson_bytes)//1024} KB GeoJSON &nbsp;|&nbsp; "
        f"{len(csv_bytes)//1024} KB CSV</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tabel data CSV
# ---------------------------------------------------------------------------

HEADER_LABELS = {
    "provinsi":     "Provinsi",
    "spi_category": "Kategori SPI",
    "spi_range":    "Rentang SPI",
    "spi_value":    "Nilai SPI",
    "lon_center":   "Longitude",
    "lat_center":   "Latitude",
}
SPI_HEX = {v["name"]: (v["hex"], v["stroke"]) for v in LEGEND.values()}

st.markdown("<div class='section-head'>Data Grid SPI</div>", unsafe_allow_html=True)

ctrl_l, ctrl_r = st.columns([2, 1])
with ctrl_l:
    filter_cat = st.selectbox(
        "Filter",
        options=["Semua"] + [v["name"] for v in LEGEND.values()],
        index=0, label_visibility="collapsed",
    )
with ctrl_r:
    page_size = st.selectbox(
        "Baris", options=[25, 50, 100, 250, 500],
        index=1, label_visibility="collapsed",
    )

filtered       = df_records if filter_cat == "Semua" else [r for r in df_records if r["spi_category"] == filter_cat]
total_filtered = len(filtered)
total_pages    = max(1, (total_filtered + page_size - 1) // page_size)

pg_col, info_col = st.columns([1, 3])
with pg_col:
    page_num = st.number_input(
        "Hal", min_value=1, max_value=total_pages, value=1, step=1,
        label_visibility="collapsed",
    )
with info_col:
    s_row = (page_num - 1) * page_size + 1
    e_row = min(page_num * page_size, total_filtered)
    note  = f"  (filter: {filter_cat})" if filter_cat != "Semua" else ""
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.75rem;"
        f"color:#8b949e;padding-top:.45rem'>"
        f"Menampilkan {s_row:,}–{e_row:,} dari {total_filtered:,} baris{note}</div>",
        unsafe_allow_html=True,
    )

page_data = filtered[(page_num - 1) * page_size : page_num * page_size]

header_html = "".join(
    f"<th style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#8b949e;"
    f"text-transform:uppercase;letter-spacing:.07em;padding:.4rem .8rem;"
    f"border-bottom:1px solid #21262d;text-align:left;white-space:nowrap'>{lbl}</th>"
    for lbl in HEADER_LABELS.values()
)

rows_html = ""
for rec in page_data:
    cat             = rec.get("spi_category", "")
    hex_c, stroke_c = SPI_HEX.get(cat, ("#444", "#666"))
    swatch = (
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
        f"background:{hex_c};border:1px solid {stroke_c};"
        f"margin-right:6px;vertical-align:middle'></span>"
    )
    cells = ""
    for col in HEADER_LABELS:
        val = rec.get(col, "")
        if col == "spi_category":
            content = swatch + str(val)
        elif col in ("lon_center", "lat_center"):
            content = f"{val:.6f}" if isinstance(val, float) else str(val)
        elif col == "spi_value":
            content = f"{val:+.2f}" if isinstance(val, (int, float)) else str(val)
        else:
            content = str(val) if val else '<span style="color:#6e7681">—</span>'
        cells += (
            f"<td style='font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#c9d1d9;"
            f"padding:.32rem .8rem;border-bottom:1px solid #161b22;"
            f"white-space:nowrap'>{content}</td>"
        )
    rows_html += f"<tr style='background:#0d1117'>{cells}</tr>"

st.markdown(
    f"<div style='overflow-x:auto;border:1px solid #21262d;border-radius:6px;"
    f"background:#0d1117;margin-bottom:1rem'>"
    f"<table style='border-collapse:collapse;width:100%;min-width:640px'>"
    f"<thead><tr style='background:#161b22'>{header_html}</tr></thead>"
    f"<tbody>{rows_html}</tbody></table></div>",
    unsafe_allow_html=True,
)

dl_col, _ = st.columns([1, 3])
with dl_col:
    if filter_cat != "Semua":
        buf = io.StringIO()
        w   = csv.DictWriter(
            buf, fieldnames=list(HEADER_LABELS.keys()),
            extrasaction="ignore", lineterminator="\n",
        )
        w.writeheader()
        w.writerows(filtered)
        st.download_button(
            f"Unduh CSV — {filter_cat} ({total_filtered:,} baris)",
            buf.getvalue().encode("utf-8"),
            f"{slug}_{filter_cat.lower().replace(' ','_')}_spi.csv",
            "text/csv",
        )
    else:
        st.download_button(
            f"Unduh CSV — Semua Data ({total_filtered:,} baris)",
            csv_bytes,
            f"{slug}_grid_spi.csv",
            "text/csv",
        )
