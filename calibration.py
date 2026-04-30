from __future__ import annotations
import base64
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from config import ISLAND_GEOREF

_DIR = os.path.join(tempfile.gettempdir(), "spi_zoomcal_v3")
os.makedirs(_DIR, exist_ok=True)

_COMPONENT_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117;
  font-family: 'IBM Plex Mono', monospace;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 100vh;
}
#toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #161b22;
  border-bottom: 1px solid #21262d;
  padding: 5px 10px;
  height: 38px;
  flex-shrink: 0;
}
.tbtn {
  background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
  border-radius: 4px; padding: 2px 11px; font-family: inherit;
  font-size: 14px; cursor: pointer; line-height: 1.6; user-select: none;
}
.tbtn:hover { background: #30363d; border-color: #8b949e; }
#zoom-val { font-size: 12px; color: #8b949e; min-width: 46px; text-align: center; }
#step-lbl {
  margin-left: auto; font-size: 11px; padding: 2px 14px;
  border-radius: 10px; white-space: nowrap;
}
.s0 { background:#3a1010; color:#e63946; border:1px solid #e63946; }
.s1 { background:#0a1e3a; color:#2196f3; border:1px solid #2196f3; }
.s2 { background:#1a2a0a; color:#8bc34a; border:1px solid #8bc34a; }
.s3 { background:#2a1a0a; color:#ff9800; border:1px solid #ff9800; }
.sd { background:#0a2010; color:#3fb950; border:1px solid #2ea043; }

#viewport {
  position: relative; width: 100%; flex: 1;
  overflow: hidden; cursor: crosshair; background: #0d1117;
  min-height: 200px;
}
#scene {
  position: absolute; top: 0; left: 0;
  transform-origin: 0 0; will-change: transform;
}
#scene img {
  display: block; max-width: none;
  user-select: none; -webkit-user-drag: none;
}
#badge {
  position: absolute; top: 8px; left: 8px;
  background: rgba(0,0,0,.82); color: #e6edf3;
  padding: 4px 12px; border-radius: 4px; font-size: 13px;
  pointer-events: none; border: 1px solid rgba(255,255,255,.12);
  z-index: 30; white-space: nowrap; letter-spacing: .04em;
}
#hint {
  position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
  font-size: 11px; padding: 3px 14px; border-radius: 10px;
  pointer-events: none; white-space: nowrap; z-index: 30;
}
#chH, #chV {
  position: absolute; pointer-events: none; z-index: 20; display: none;
}
#chH { height: 1px; left: 0; right: 0; background: rgba(255,255,255,.25); }
#chV { width: 1px; top: 0; bottom: 0; background: rgba(255,255,255,.25); }

.dot {
  position: absolute; width: 22px; height: 22px; border-radius: 50%;
  border: 2px solid #fff; transform: translate(-50%, -50%);
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; color: #fff; pointer-events: none;
  box-shadow: 0 2px 10px rgba(0,0,0,.7); z-index: 15;
}
.dot-lbl {
  position: absolute; font-size: 9px; pointer-events: none;
  background: rgba(0,0,0,.7); padding: 2px 5px; border-radius: 3px;
  white-space: nowrap; z-index: 15;
}
.guide {
  position: absolute; width: 18px; height: 18px; border-radius: 50%;
  border: 2px dashed rgba(255,255,255,.45); transform: translate(-50%, -50%);
  pointer-events: none; z-index: 12;
}
.guide-lbl {
  position: absolute; font-size: 9px; color: rgba(255,255,255,.6);
  background: rgba(0,0,0,.55); padding: 1px 4px; border-radius: 2px;
  white-space: nowrap; pointer-events: none; z-index: 12;
}
</style>
</head>
<body>
<div id="toolbar">
  <button class="tbtn" id="btn-zin">+</button>
  <span id="zoom-val">100%</span>
  <button class="tbtn" id="btn-zout">−</button>
  <button class="tbtn" id="btn-reset">Reset</button>
  <span style="font-size:11px;color:#6e7681">Scroll: zoom &nbsp;|&nbsp; Ctrl+drag: geser</span>
  <span id="step-lbl" class="s0">Klik GCP 1 (NW)</span>
</div>
<div id="viewport">
  <div id="badge">X: —  |  Y: —</div>
  <div id="chH"></div>
  <div id="chV"></div>
  <div id="hint" class="s0">Klik Sudut Kiri Atas (Barat Laut)</div>
  <div id="scene">
    <img id="map" src="" alt="" draggable="false"/>
  </div>
</div>
<script>
const ST = {
  RENDER_EVENT: "streamlit:render",
  events: {
    addEventListener(ev, cb) {
      window.addEventListener("message", e => {
        if (e.data && e.data.type === ev) cb({ detail: e.data });
      });
    }
  },
  setComponentReady() {
    window.parent.postMessage({ isStreamlitMessage:true, type:"streamlit:componentReady", apiVersion:1 }, "*");
  },
  setComponentValue(val) {
    window.parent.postMessage({ isStreamlitMessage:true, type:"streamlit:setComponentValue", value:val }, "*");
  },
  setFrameHeight(h) {
    window.parent.postMessage({ isStreamlitMessage:true, type:"streamlit:setFrameHeight", height:Math.round(h) }, "*");
  }
};

const viewport = document.getElementById("viewport");
const scene    = document.getElementById("scene");
const mapImg   = document.getElementById("map");
const badge    = document.getElementById("badge");
const chH      = document.getElementById("chH");
const chV      = document.getElementById("chV");
const hint     = document.getElementById("hint");
const stepLbl  = document.getElementById("step-lbl");
const zoomVal  = document.getElementById("zoom-val");

const STEP_CFG = [
  { cls:"s0", step:"Klik GCP 1 (NW)", hint:"Klik Sudut Kiri Atas (Barat Laut)",  color:"#e63946" },
  { cls:"s1", step:"Klik GCP 2 (NE)", hint:"Klik Sudut Kanan Atas (Timur Laut)", color:"#2196f3" },
  { cls:"s2", step:"Klik GCP 3 (SW)", hint:"Klik Sudut Kiri Bawah (Barat Daya)", color:"#8bc34a" },
  { cls:"s3", step:"Klik GCP 4 (SE)", hint:"Klik Sudut Kanan Bawah (Tenggara)",  color:"#ff9800" },
  { cls:"sd", step:"Selesai",          hint:"4 GCP terekam — isi lon/lat & simpan", color:"#3fb950" },
];

let scale = 1, panX = 0, panY = 0;
let clicks = [];
let isPan = false, panOX = 0, panOY = 0, panMX = 0, panMY = 0;
let imgLoaded = false;
const MIN_S = 0.15, MAX_S = 14, VP_H = 520;

viewport.style.height = VP_H + "px";
ST.setFrameHeight(VP_H + 38 + 4);

function applyTransform() {
  scene.style.transform = `translate(${panX}px,${panY}px) scale(${scale})`;
  zoomVal.textContent   = Math.round(scale * 100) + "%";
}

function clampPan() {
  if (!imgLoaded) return;
  const sw = mapImg.naturalWidth * scale, sh = mapImg.naturalHeight * scale;
  const vw = viewport.clientWidth || 800, vh = VP_H;
  panX = sw <= vw ? (vw - sw) / 2 : Math.min(0, Math.max(panX, vw - sw));
  panY = sh <= vh ? (vh - sh) / 2 : Math.min(0, Math.max(panY, vh - sh));
}

function toImg(cx, cy) {
  const vr = viewport.getBoundingClientRect();
  return {
    x: Math.round((cx - vr.left - panX) / scale),
    y: Math.round((cy - vr.top  - panY) / scale),
  };
}

function inBounds(x, y) {
  return x >= 0 && y >= 0 && x <= mapImg.naturalWidth && y <= mapImg.naturalHeight;
}

function updateStep() {
  const n  = Math.min(clicks.length, 4);
  const cf = STEP_CFG[n];
  stepLbl.textContent = cf.step; stepLbl.className = cf.cls;
  hint.textContent    = cf.hint; hint.className    = cf.cls;
}

function addDot(x, y, n) {
  const col = STEP_CFG[n - 1].color;
  const dot = document.createElement("div");
  dot.className   = "dot";
  dot.style.left  = x + "px"; dot.style.top = y + "px";
  dot.style.background = col; dot.textContent = n;
  scene.appendChild(dot);
  const lbl = document.createElement("div");
  lbl.className   = "dot-lbl";
  lbl.style.left  = (x + 13) + "px"; lbl.style.top = (y - 14) + "px";
  lbl.style.color = col;
  lbl.textContent = ["NW","NE","SW","SE"][n - 1] + ` (${x},${y})`;
  scene.appendChild(lbl);
}

function drawGuides(guides) {
  document.querySelectorAll(".guide,.guide-lbl").forEach(el => el.remove());
  if (!guides) return;
  guides.forEach(g => {
    const el = document.createElement("div");
    el.className = "guide"; el.style.left = g.x + "px"; el.style.top = g.y + "px";
    scene.appendChild(el);
    const lbl = document.createElement("div");
    lbl.className = "guide-lbl";
    lbl.style.left = (g.x + 11) + "px"; lbl.style.top = (g.y - 8) + "px";
    lbl.textContent = g.label;
    scene.appendChild(lbl);
  });
}

viewport.addEventListener("mousemove", e => {
  if (isPan) return;
  const vr = viewport.getBoundingClientRect();
  const { x, y } = toImg(e.clientX, e.clientY);
  if (inBounds(x, y)) {
    badge.textContent = `X: ${x}   |   Y: ${y}`;
    chH.style.top   = (e.clientY - vr.top)  + "px"; chH.style.display = "block";
    chV.style.left  = (e.clientX - vr.left) + "px"; chV.style.display = "block";
  } else {
    badge.textContent = "X: —  |  Y: —";
    chH.style.display = chV.style.display = "none";
  }
});

viewport.addEventListener("mouseleave", () => {
  if (!isPan) { badge.textContent = "X: —  |  Y: —"; chH.style.display = chV.style.display = "none"; }
});

viewport.addEventListener("mousedown", e => {
  if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
    isPan = true; panOX = panX; panOY = panY; panMX = e.clientX; panMY = e.clientY;
    viewport.style.cursor = "grabbing"; e.preventDefault();
  }
});

window.addEventListener("mousemove", e => {
  if (!isPan) return;
  panX = panOX + (e.clientX - panMX);
  panY = panOY + (e.clientY - panMY);
  clampPan(); applyTransform();
});

window.addEventListener("mouseup", () => {
  if (isPan) { isPan = false; viewport.style.cursor = "crosshair"; }
});

viewport.addEventListener("click", e => {
  if (e.button !== 0 || e.ctrlKey || isPan) return;
  if (clicks.length >= 4) return;
  const { x, y } = toImg(e.clientX, e.clientY);
  if (!inBounds(x, y)) return;
  clicks.push({ x, y });
  addDot(x, y, clicks.length);
  updateStep();
  ST.setComponentValue({ clicks });
});

viewport.addEventListener("wheel", e => {
  e.preventDefault();
  const f  = e.deltaY < 0 ? 1.18 : 1 / 1.18;
  const nr = Math.max(MIN_S, Math.min(MAX_S, scale * f));
  const vr = viewport.getBoundingClientRect();
  const mx = e.clientX - vr.left, my = e.clientY - vr.top;
  panX = mx - (mx - panX) * (nr / scale);
  panY = my - (my - panY) * (nr / scale);
  scale = nr; clampPan(); applyTransform();
}, { passive: false });

let _pinchD = 0, _pinchS = 1;
viewport.addEventListener("touchstart", e => {
  if (e.touches.length === 2) {
    const t = e.touches;
    _pinchD = Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    _pinchS = scale; e.preventDefault();
  }
}, { passive: false });

viewport.addEventListener("touchmove", e => {
  if (e.touches.length === 2) {
    const t  = e.touches;
    const d  = Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const nr = Math.max(MIN_S, Math.min(MAX_S, _pinchS * (d / _pinchD)));
    const mx = (t[0].clientX + t[1].clientX) / 2, my = (t[0].clientY + t[1].clientY) / 2;
    const vr = viewport.getBoundingClientRect();
    const px = mx - vr.left, py = my - vr.top;
    panX = px - (px - panX) * (nr / scale);
    panY = py - (py - panY) * (nr / scale);
    scale = nr; clampPan(); applyTransform(); e.preventDefault();
  }
}, { passive: false });

function zoomBy(f) {
  const vw = viewport.clientWidth / 2, vh = VP_H / 2;
  const nr = Math.max(MIN_S, Math.min(MAX_S, scale * f));
  panX = vw - (vw - panX) * (nr / scale);
  panY = vh - (vh - panY) * (nr / scale);
  scale = nr; clampPan(); applyTransform();
}

document.getElementById("btn-zin").onclick   = () => zoomBy(1.4);
document.getElementById("btn-zout").onclick  = () => zoomBy(1 / 1.4);
document.getElementById("btn-reset").onclick = () => { scale = 1; panX = 0; panY = 0; clampPan(); applyTransform(); };

mapImg.addEventListener("load", () => {
  imgLoaded = true;
  const fh = Math.min(Math.max(mapImg.naturalHeight, 200), 600);
  viewport.style.height = fh + "px";
  ST.setFrameHeight(fh + 38 + 6);
  clampPan(); applyTransform();
});

let _lastSrc = "";
ST.events.addEventListener(ST.RENDER_EVENT, e => {
  const a = e.detail.args || {};
  if (a.image_src && a.image_src !== _lastSrc) { _lastSrc = a.image_src; imgLoaded = false; mapImg.src = a.image_src; }
  if (a.guides) drawGuides(a.guides);
});

ST.setComponentReady();
</script>
</body>
</html>
"""

with open(os.path.join(_DIR, "index.html"), "w", encoding="utf-8") as _fh:
    _fh.write(_COMPONENT_HTML)

_picker = components.declare_component("spi_zoomcal_v3", path=_DIR)

GCP_LABELS = ["NW — Kiri Atas (Barat Laut)", "NE — Kanan Atas (Timur Laut)",
               "SW — Kiri Bawah (Barat Daya)", "SE — Kanan Bawah (Tenggara)"]
GCP_COLORS = ["#e63946", "#2196f3", "#8bc34a", "#ff9800"]


def _build_custom_georef(clicks: list, gcps_default: list, lon_vals: list, lat_vals: list,
                          master_w: int, master_h: int) -> dict:
    result_gcps = []
    for i, gcp in enumerate(gcps_default):
        px = clicks[i]["x"] if i < len(clicks) else gcp["px"]
        py = clicks[i]["y"] if i < len(clicks) else gcp["py"]
        result_gcps.append({"px": px, "py": py, "lon": lon_vals[i], "lat": lat_vals[i]})
    return {"master_size": (master_w, master_h), "gcps": result_gcps}


def render_calibration(img_bytes: bytes, island_key: str) -> None:
    default      = ISLAND_GEOREF[island_key]
    gcps_default = default["gcps"]
    master_w, master_h = default["master_size"]

    arr          = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h_img, w_img = arr.shape[:2]

    sx = w_img / master_w
    sy = h_img / master_h

    rst_key     = f"cal_rst_{island_key}"
    clicks_key  = f"cal_clicks_{island_key}"
    img_b64_key = f"cal_b64_{island_key}"

    if rst_key    not in st.session_state: st.session_state[rst_key]    = 0
    if clicks_key not in st.session_state: st.session_state[clicks_key] = []

    if img_b64_key not in st.session_state:
        _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        st.session_state[img_b64_key] = (
            "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        )

    img_b64 = st.session_state[img_b64_key]
    clicks  = st.session_state[clicks_key]

    guides = [
        {"x": g["px"] * sx, "y": g["py"] * sy, "label": lbl.split("—")[0].strip()}
        for g, lbl in zip(gcps_default, GCP_LABELS)
    ]

    st.markdown(
        f"<p style='color:#8b949e;font-size:.82rem;margin-bottom:.8rem'>"
        f"Gambar: <b>{w_img}×{h_img}px</b>  |  Skala otomatis dari master "
        f"<b>{master_w}×{master_h}px</b>  →  sx={sx:.3f}, sy={sy:.3f}<br>"
        f"Lingkaran putus-putus = posisi GCP default yang sudah discale. "
        f"Klik <b>4 sudut</b> secara berurutan: NW → NE → SW → SE."
        f"</p>",
        unsafe_allow_html=True,
    )

    col_img, col_form = st.columns([3, 2], gap="large")

    with col_img:
        rc, _ = st.columns([1, 3])
        with rc:
            if st.button("Ulangi Titik", key=f"rst_{island_key}"):
                st.session_state[clicks_key] = []
                st.session_state[rst_key]   += 1
                st.rerun()

        result = _picker(
            image_src=img_b64,
            guides=guides,
            key=f"zc_{island_key}_{st.session_state[rst_key]}",
        )

        if result and result.get("clicks"):
            incoming = result["clicks"]
            if len(incoming) > len(clicks):
                st.session_state[clicks_key] = incoming
                st.rerun()

        n_recorded = len(clicks)
        progress_html = ""
        for i, (lbl, col) in enumerate(zip(GCP_LABELS, GCP_COLORS)):
            if i < n_recorded:
                c = clicks[i]
                tag = f"px={c['x']} py={c['y']}"
            else:
                g   = gcps_default[i]
                tag = f"px={round(g['px']*sx)} py={round(g['py']*sy)}  (default)"
                col = "#6e7681"
            progress_html += (
                f"<div style='font-size:.68rem;color:{col};"
                f"font-family:IBM Plex Mono,monospace;margin-top:3px'>"
                f"{'✓' if i < n_recorded else '○'} GCP{i+1} {lbl.split('—')[0].strip()}  {tag}</div>"
            )
        st.markdown(
            f"<div style='background:#0d1117;border:1px solid #21262d;"
            f"border-radius:5px;padding:8px 12px;margin-top:6px'>{progress_html}</div>",
            unsafe_allow_html=True,
        )

    with col_form:
        lon_vals = []
        lat_vals = []

        for i, (gcp, lbl, col) in enumerate(zip(gcps_default, GCP_LABELS, GCP_COLORS)):
            clicked = i < len(clicks)
            px_show = clicks[i]["x"] if clicked else round(gcp["px"] * sx)
            py_show = clicks[i]["y"] if clicked else round(gcp["py"] * sy)
            c = col if clicked else "#6e7681"

            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.7rem;"
                f"color:{c};background:#0d1117;border:1px solid #21262d;"
                f"border-radius:4px;padding:4px 10px;margin-bottom:4px;margin-top:{'1rem' if i>0 else '0'}'>"
                f"<b>GCP{i+1} {lbl}</b><br>"
                f"px={px_show}  py={py_show}{'  ✓' if clicked else '  (default)'}</div>",
                unsafe_allow_html=True,
            )
            ca, cb = st.columns(2)
            with ca:
                lon = st.number_input(f"Lon GCP{i+1}", value=float(gcp["lon"]),
                                      step=0.01, format="%.4f", key=f"lon{i}_{island_key}")
            with cb:
                lat = st.number_input(f"Lat GCP{i+1}", value=float(gcp["lat"]),
                                      step=0.01, format="%.4f", key=f"lat{i}_{island_key}")
            lon_vals.append(lon)
            lat_vals.append(lat)

        n_clicked = len(clicks)
        georef = _build_custom_georef(
            clicks, gcps_default, lon_vals, lat_vals,
            w_img, h_img,
        )

        gcps_preview = georef["gcps"]
        lons = [g["lon"] for g in gcps_preview]
        lats = [g["lat"] for g in gcps_preview]
        bbox_str = f"[{min(lons):.4f}, {min(lats):.4f}, {max(lons):.4f}, {max(lats):.4f}]"

        preview_lines = "\n".join(
            f'  GCP{i+1}: px={g["px"]} py={g["py"]}  →  ({g["lon"]:.4f}°, {g["lat"]:.4f}°)'
            for i, g in enumerate(gcps_preview)
        )
        st.markdown(
            f"<pre style='background:#0d1117;border:1px solid #21262d;border-radius:6px;"
            f"padding:.6rem 1rem;font-family:IBM Plex Mono,monospace;font-size:.68rem;"
            f"color:#79c0ff;overflow-x:auto;line-height:1.7;margin-top:.8rem'>"
            f"master_size: {w_img} × {h_img}\n"
            f"{preview_lines}\n"
            f"bbox_geo: {bbox_str}</pre>",
            unsafe_allow_html=True,
        )

        btn_label = (
            f"Simpan Kalibrasi ({n_clicked}/4 titik diklik) & Lanjut"
            if n_clicked < 4
            else "Simpan Kalibrasi 4-GCP & Lanjut Proses"
        )
        if st.button(btn_label, type="primary", use_container_width=True, key=f"save_{island_key}"):
            if "custom_georef" not in st.session_state:
                st.session_state["custom_georef"] = {}
            st.session_state["custom_georef"][island_key] = georef
            st.session_state["app_step"]    = "process"
            st.session_state.pop("has_result",     None)
            st.session_state.pop("cached_file_id", None)
            st.session_state.pop(img_b64_key,      None)
            st.rerun()
