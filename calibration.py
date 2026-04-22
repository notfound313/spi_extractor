from __future__ import annotations
import base64
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from config import ISLAND_GEOREF

_DIR = os.path.join(tempfile.gettempdir(), "spi_zoomcal_v2")
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

/* ── toolbar ── */
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
  background: #21262d;
  border: 1px solid #30363d;
  color: #c9d1d9;
  border-radius: 4px;
  padding: 2px 11px;
  font-family: inherit;
  font-size: 14px;
  cursor: pointer;
  line-height: 1.6;
  user-select: none;
}
.tbtn:hover { background: #30363d; border-color: #8b949e; }
#zoom-val { font-size: 12px; color: #8b949e; min-width: 46px; text-align: center; }
#step-lbl {
  margin-left: auto;
  font-size: 11px;
  padding: 2px 12px;
  border-radius: 10px;
  white-space: nowrap;
}
.s1 { background: #3a1010; color: #e63946; border: 1px solid #e63946; }
.s2 { background: #0a1e3a; color: #2196f3; border: 1px solid #2196f3; }
.sd { background: #0a2010; color: #3fb950; border: 1px solid #2ea043; }

/* ── viewport: clips the zoomed scene, holds overlay elements ── */
#viewport {
  position: relative;
  width: 100%;
  flex: 1;
  overflow: hidden;
  cursor: crosshair;
  background: #0d1117;
  min-height: 200px;
}

/* ── scene: the transformed (zoom+pan) container ── */
#scene {
  position: absolute;
  top: 0; left: 0;
  transform-origin: 0 0;
  will-change: transform;
}
#scene img {
  display: block;
  max-width: none;
  user-select: none;
  -webkit-user-drag: none;
}

/* ── overlays inside VIEWPORT (not scene) — viewport-space coordinates ── */
#badge {
  position: absolute;
  top: 8px; left: 8px;
  background: rgba(0,0,0,.82);
  color: #e6edf3;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 13px;
  pointer-events: none;
  border: 1px solid rgba(255,255,255,.12);
  z-index: 30;
  white-space: nowrap;
  letter-spacing: .04em;
}
#hint {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  padding: 3px 14px;
  border-radius: 10px;
  pointer-events: none;
  white-space: nowrap;
  z-index: 30;
}
/* Crosshairs live in VIEWPORT space — never inside #scene */
#chH, #chV {
  position: absolute;
  pointer-events: none;
  z-index: 20;
  display: none;
}
#chH { height: 1px; left: 0; right: 0; background: rgba(255,255,255,.25); }
#chV { width: 1px;  top: 0; bottom: 0; background: rgba(255,255,255,.25); }

/* ── elements inside SCENE (image-space coordinates, scaled with image) ── */
.dot {
  position: absolute;
  width: 24px; height: 24px;
  border-radius: 50%;
  border: 2.5px solid #fff;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  pointer-events: none;
  box-shadow: 0 2px 10px rgba(0,0,0,.7);
  z-index: 15;
}
.dot-lbl {
  position: absolute;
  font-size: 10px;
  pointer-events: none;
  background: rgba(0,0,0,.7);
  padding: 2px 6px;
  border-radius: 3px;
  white-space: nowrap;
  z-index: 15;
}
.guide {
  position: absolute;
  width: 20px; height: 20px;
  border-radius: 50%;
  border: 2px dashed rgba(255,255,255,.45);
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 12;
}
.guide-lbl {
  position: absolute;
  font-size: 9px;
  color: rgba(255,255,255,.6);
  background: rgba(0,0,0,.55);
  padding: 1px 5px;
  border-radius: 2px;
  white-space: nowrap;
  pointer-events: none;
  z-index: 12;
}
</style>
</head>
<body>

<div id="toolbar">
  <button class="tbtn" id="btn-zin"   title="Zoom In">+</button>
  <span   id="zoom-val">100%</span>
  <button class="tbtn" id="btn-zout"  title="Zoom Out">−</button>
  <button class="tbtn" id="btn-reset" title="Reset view">Reset</button>
  <span style="font-size:11px;color:#6e7681">
    Scroll: zoom &nbsp;|&nbsp; Ctrl+drag / tengah: geser
  </span>
  <span id="step-lbl" class="s1">Klik Titik 1</span>
</div>

<!-- viewport: clips scene, hosts badge/crosshair/hint in viewport space -->
<div id="viewport">
  <div id="badge">X: —  |  Y: —</div>
  <div id="chH"></div>
  <div id="chV"></div>
  <div id="hint" class="s1">Klik Sudut Kiri Atas Peta (Barat Laut)</div>

  <!-- scene: zoom+pan transform applied here; dots/guides live here -->
  <div id="scene">
    <img id="map" src="" alt="" draggable="false"/>
  </div>
</div>

<script>
/* ── Streamlit protocol (inlined — no CDN dependency) ── */
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
    window.parent.postMessage(
      { isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }, "*"
    );
  },
  setComponentValue(val) {
    window.parent.postMessage(
      { isStreamlitMessage: true, type: "streamlit:setComponentValue", value: val }, "*"
    );
  },
  setFrameHeight(h) {
    window.parent.postMessage(
      { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: Math.round(h) }, "*"
    );
  }
};

/* ── DOM refs ── */
const viewport  = document.getElementById("viewport");
const scene     = document.getElementById("scene");
const mapImg    = document.getElementById("map");
const badge     = document.getElementById("badge");
const chH       = document.getElementById("chH");
const chV       = document.getElementById("chV");
const hint      = document.getElementById("hint");
const stepLbl   = document.getElementById("step-lbl");
const zoomVal   = document.getElementById("zoom-val");

/* ── state ── */
let scale   = 1;
let panX    = 0;
let panY    = 0;
let clicks  = [];
let isPan   = false;
let panOX   = 0;   // panX when pan started
let panOY   = 0;
let panMX   = 0;   // mouse X when pan started
let panMY   = 0;
let imgLoaded = false;

const MIN_SCALE = 0.2;
const MAX_SCALE = 12;
const VIEWPORT_H = 520;   // fixed height — avoids clientHeight=0 bug

/* ── set fixed height immediately so frame never collapses ── */
viewport.style.height = VIEWPORT_H + "px";
ST.setFrameHeight(VIEWPORT_H + 38 + 4);

/* ── transform helpers ── */
function applyTransform() {
  scene.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  zoomVal.textContent   = Math.round(scale * 100) + "%";
}

function clampPan() {
  if (!imgLoaded) return;
  const sw = mapImg.naturalWidth  * scale;
  const sh = mapImg.naturalHeight * scale;
  const vw = viewport.clientWidth  || 800;
  const vh = VIEWPORT_H;

  if (sw <= vw) { panX = (vw - sw) / 2; }
  else          { panX = Math.min(0, Math.max(panX, vw - sw)); }

  if (sh <= vh) { panY = (vh - sh) / 2; }
  else          { panY = Math.min(0, Math.max(panY, vh - sh)); }
}

/* ── image→viewport coordinate conversion ── */
function toImg(clientX, clientY) {
  const vr = viewport.getBoundingClientRect();
  return {
    x: Math.round((clientX - vr.left - panX) / scale),
    y: Math.round((clientY - vr.top  - panY) / scale)
  };
}

function inBounds(x, y) {
  return x >= 0 && y >= 0
      && x <= mapImg.naturalWidth
      && y <= mapImg.naturalHeight;
}

/* ── step UI ── */
function updateStep() {
  const n = clicks.length;
  if (n === 0) {
    stepLbl.textContent = "Klik Titik 1";       stepLbl.className = "s1";
    hint.textContent    = "Klik Sudut Kiri Atas Peta (Barat Laut)"; hint.className = "s1";
  } else if (n === 1) {
    stepLbl.textContent = "Klik Titik 2";       stepLbl.className = "s2";
    hint.textContent    = "Klik Sudut Kanan Bawah Peta (Tenggara)"; hint.className = "s2";
  } else {
    stepLbl.textContent = "Selesai";            stepLbl.className = "sd";
    hint.textContent    = "2 titik terekam — isi lon/lat lalu simpan"; hint.className = "sd";
  }
}

/* ── add dot+label inside SCENE (image-space coords) ── */
function addDot(x, y, n) {
  const dot     = document.createElement("div");
  dot.className = "dot";
  dot.style.left       = x + "px";
  dot.style.top        = y + "px";
  dot.style.background = n === 1 ? "#e63946" : "#2196f3";
  dot.textContent      = n;
  scene.appendChild(dot);

  const lbl     = document.createElement("div");
  lbl.className = "dot-lbl";
  lbl.style.left  = (x + 14) + "px";
  lbl.style.top   = (y - 16) + "px";
  lbl.style.color = n === 1 ? "#e63946" : "#2196f3";
  lbl.textContent = `P${n} (${x}, ${y})`;
  scene.appendChild(lbl);
}

/* ── guide markers inside SCENE (image-space coords) ── */
function drawGuides(guides) {
  document.querySelectorAll(".guide, .guide-lbl").forEach(el => el.remove());
  if (!guides) return;
  guides.forEach(g => {
    const el     = document.createElement("div");
    el.className = "guide";
    el.style.left = g.x + "px";
    el.style.top  = g.y + "px";
    scene.appendChild(el);

    const lbl     = document.createElement("div");
    lbl.className = "guide-lbl";
    lbl.style.left = (g.x + 13) + "px";
    lbl.style.top  = (g.y - 9)  + "px";
    lbl.textContent = g.label;
    scene.appendChild(lbl);
  });
}

/* ════════════════════════════════════════════════════
   EVENTS
   ════════════════════════════════════════════════════ */

/* ── HOVER: only on viewport, positions chH/chV in VIEWPORT space ── */
viewport.addEventListener("mousemove", e => {
  if (isPan) return;   // handled separately by window.mousemove
  const vr  = viewport.getBoundingClientRect();
  const rx  = e.clientX - vr.left;
  const ry  = e.clientY - vr.top;
  const { x, y } = toImg(e.clientX, e.clientY);

  if (inBounds(x, y)) {
    badge.textContent   = `X: ${x}   |   Y: ${y}`;
    chH.style.top       = ry + "px";
    chV.style.left      = rx + "px";
    chH.style.display   = "block";
    chV.style.display   = "block";
  } else {
    badge.textContent   = "X: —  |  Y: —";
    chH.style.display   = "none";
    chV.style.display   = "none";
  }
});

viewport.addEventListener("mouseleave", () => {
  if (isPan) return;
  badge.textContent = "X: —  |  Y: —";
  chH.style.display = "none";
  chV.style.display = "none";
});

/* ── PAN start: Ctrl+LMB or middle button ── */
viewport.addEventListener("mousedown", e => {
  if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
    isPan  = true;
    panOX  = panX;
    panOY  = panY;
    panMX  = e.clientX;
    panMY  = e.clientY;
    viewport.style.cursor = "grabbing";
    e.preventDefault();
  }
});

/* ── PAN move: on window so drag outside viewport still works ── */
window.addEventListener("mousemove", e => {
  if (!isPan) return;
  panX = panOX + (e.clientX - panMX);
  panY = panOY + (e.clientY - panMY);
  clampPan();
  applyTransform();
});

/* ── PAN end ── */
window.addEventListener("mouseup", e => {
  if (!isPan) return;
  isPan = false;
  viewport.style.cursor = "crosshair";
});

/* ── CLICK: record point (only when not panning, not Ctrl) ── */
viewport.addEventListener("click", e => {
  if (e.button !== 0 || e.ctrlKey) return;
  if (clicks.length >= 2) return;
  const { x, y } = toImg(e.clientX, e.clientY);
  if (!inBounds(x, y)) return;
  clicks.push({ x, y });
  addDot(x, y, clicks.length);
  updateStep();
  ST.setComponentValue({ clicks });
});

/* ── ZOOM: mouse wheel, centered on cursor ── */
viewport.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.18 : (1 / 1.18);
  const nr     = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
  const vr     = viewport.getBoundingClientRect();
  const mx     = e.clientX - vr.left;
  const my     = e.clientY - vr.top;
  panX = mx - (mx - panX) * (nr / scale);
  panY = my - (my - panY) * (nr / scale);
  scale = nr;
  clampPan();
  applyTransform();
}, { passive: false });

/* ── TOUCH pinch zoom ── */
let _pinchDist = 0, _pinchScale = 1;
viewport.addEventListener("touchstart", e => {
  if (e.touches.length === 2) {
    const t = e.touches;
    _pinchDist  = Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    _pinchScale = scale;
    e.preventDefault();
  }
}, { passive: false });

viewport.addEventListener("touchmove", e => {
  if (e.touches.length === 2) {
    const t  = e.touches;
    const d  = Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const nr = Math.max(MIN_SCALE, Math.min(MAX_SCALE, _pinchScale * (d / _pinchDist)));
    const mx = (t[0].clientX + t[1].clientX) / 2;
    const my = (t[0].clientY + t[1].clientY) / 2;
    const vr = viewport.getBoundingClientRect();
    const px = mx - vr.left, py = my - vr.top;
    panX  = px - (px - panX) * (nr / scale);
    panY  = py - (py - panY) * (nr / scale);
    scale = nr;
    clampPan();
    applyTransform();
    e.preventDefault();
  }
}, { passive: false });

/* ── toolbar buttons ── */
function zoomBy(factor) {
  const vw = viewport.clientWidth  / 2;
  const vh = VIEWPORT_H / 2;
  const nr = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
  panX  = vw - (vw - panX) * (nr / scale);
  panY  = vh - (vh - panY) * (nr / scale);
  scale = nr;
  clampPan();
  applyTransform();
}

document.getElementById("btn-zin").onclick   = () => zoomBy(1.4);
document.getElementById("btn-zout").onclick  = () => zoomBy(1 / 1.4);
document.getElementById("btn-reset").onclick = () => {
  scale = 1; panX = 0; panY = 0;
  clampPan();
  applyTransform();
};

/* ── image load ── */
mapImg.addEventListener("load", () => {
  imgLoaded = true;
  clampPan();
  applyTransform();
  const natH  = mapImg.naturalHeight;
  const finalH = Math.min(Math.max(natH, 200), 600);
  viewport.style.height = finalH + "px";
  ST.setFrameHeight(finalH + 38 + 6);
});

/* ── Streamlit RENDER_EVENT ── */
let _lastSrc = "";
ST.events.addEventListener(ST.RENDER_EVENT, e => {
  const a = e.detail.args || {};

  if (a.image_src && a.image_src !== _lastSrc) {
    _lastSrc      = a.image_src;
    imgLoaded     = false;
    mapImg.src    = a.image_src;
  }

  if (a.guides) drawGuides(a.guides);
});

/* ── ready: announce immediately with fixed height ── */
ST.setComponentReady();
</script>
</body>
</html>
"""

with open(os.path.join(_DIR, "index.html"), "w", encoding="utf-8") as _fh:
    _fh.write(_COMPONENT_HTML)

_picker = components.declare_component("spi_zoomcal_v2", path=_DIR)


def _make_georef(px1, py1, lon1, lat1, px2, py2, lon2, lat2, px_min, py_min) -> dict:
    return {
        "lon": {"px1": int(px1), "lon1": float(lon1), "px2": int(px2), "lon2": float(lon2)},
        "lat": {"py1": int(py1), "lat1": float(lat1), "py2": int(py2), "lat2": float(lat2)},
        "px_min":  int(px_min),
        "py_min":  int(py_min),
        "bbox_geo": [
            min(float(lon1), float(lon2)), min(float(lat1), float(lat2)),
            max(float(lon1), float(lon2)), max(float(lat1), float(lat2)),
        ],
    }


def render_calibration(img_bytes: bytes, island_key: str) -> None:
    default      = ISLAND_GEOREF[island_key]
    arr          = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h_img, w_img = arr.shape[:2]

    rst_key    = f"cal_rst_{island_key}"
    clicks_key = f"cal_clicks_{island_key}"
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
        {"x": default["lon"]["px1"], "y": default["lat"]["py1"], "label": "Default P1"},
        {"x": default["lon"]["px2"], "y": default["lat"]["py2"], "label": "Default P2"},
    ]

    st.markdown(
        "<p style='color:#8b949e;font-size:.82rem;margin-bottom:.8rem'>"
        "Gunakan <b>scroll</b> untuk zoom terpusat pada kursor, "
        "<b>Ctrl+drag</b> atau <b>klik-tengah</b> untuk menggeser. "
        "Koordinat X/Y tampil <b>real-time</b> mengikuti posisi kursor."
        "</p>",
        unsafe_allow_html=True,
    )

    col_img, col_form = st.columns([3, 2], gap="large")

    with col_img:
        rst_col, _ = st.columns([1, 3])
        with rst_col:
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
                clicks = incoming
                st.rerun()

        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.68rem;"
            f"color:#6e7681;margin-top:.3rem'>"
            f"Gambar: {w_img} × {h_img} px  |  Titik terekam: {len(clicks)}/2"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_form:
        has_p1 = len(clicks) > 0
        has_p2 = len(clicks) > 1

        px1 = clicks[0]["x"] if has_p1 else default["lon"]["px1"]
        py1 = clicks[0]["y"] if has_p1 else default["lat"]["py1"]
        px2 = clicks[1]["x"] if has_p2 else default["lon"]["px2"]
        py2 = clicks[1]["y"] if has_p2 else default["lat"]["py2"]

        def _badge(label, px, py, color, clicked):
            c   = color if clicked else "#6e7681"
            sfx = "" if clicked else "  (default)"
            return (
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{c};"
                f"background:#0d1117;border:1px solid #21262d;border-radius:4px;"
                f"padding:4px 10px;margin-bottom:.5rem'>"
                f"<b>{label}</b>  px={px}  py={py}{sfx}</div>"
            )

        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#8b949e;"
            "text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #21262d;"
            "padding-bottom:.3rem;margin-bottom:.6rem'>"
            "Titik 1  —  Kiri Atas (Barat Laut)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_badge("P1", px1, py1, "#e63946", has_p1), unsafe_allow_html=True)
        c1a, c1b = st.columns(2)
        with c1a:
            lon1 = st.number_input("Longitude 1 (°E)", value=float(default["lon"]["lon1"]),
                                   step=0.01, format="%.4f", key=f"lon1_{island_key}")
        with c1b:
            lat1 = st.number_input("Latitude 1 (°)", value=float(default["lat"]["lat1"]),
                                   step=0.01, format="%.4f", key=f"lat1_{island_key}")

        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#8b949e;"
            "text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #21262d;"
            "padding-bottom:.3rem;margin-bottom:.6rem;margin-top:.9rem'>"
            "Titik 2  —  Kanan Bawah (Tenggara)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_badge("P2", px2, py2, "#2196f3", has_p2), unsafe_allow_html=True)
        c2a, c2b = st.columns(2)
        with c2a:
            lon2 = st.number_input("Longitude 2 (°E)", value=float(default["lon"]["lon2"]),
                                   step=0.01, format="%.4f", key=f"lon2_{island_key}")
        with c2b:
            lat2 = st.number_input("Latitude 2 (°)", value=float(default["lat"]["lat2"]),
                                   step=0.01, format="%.4f", key=f"lat2_{island_key}")

        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#8b949e;"
            "text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #21262d;"
            "padding-bottom:.3rem;margin-bottom:.6rem;margin-top:.9rem'>"
            "Margin Gambar</div>",
            unsafe_allow_html=True,
        )
        cm1, cm2 = st.columns(2)
        with cm1:
            px_min = st.number_input("px_min", value=int(default.get("px_min", 0)),
                                     min_value=0, step=1, key=f"pxmin_{island_key}")
        with cm2:
            py_min = st.number_input("py_min", value=int(default.get("py_min", 0)),
                                     min_value=0, step=1, key=f"pymin_{island_key}")

        georef  = _make_georef(px1, py1, lon1, lat1, px2, py2, lon2, lat2, px_min, py_min)
        preview = (
            f'"{island_key}": {{\n'
            f'  "lon": {{"px1":{px1}, "lon1":{lon1:.4f},\n'
            f'           "px2":{px2}, "lon2":{lon2:.4f}}},\n'
            f'  "lat": {{"py1":{py1}, "lat1":{lat1:.4f},\n'
            f'           "py2":{py2}, "lat2":{lat2:.4f}}},\n'
            f'  "px_min":{px_min}, "py_min":{py_min},\n'
            f'  "bbox_geo": [{georef["bbox_geo"][0]:.4f}, {georef["bbox_geo"][1]:.4f},\n'
            f'               {georef["bbox_geo"][2]:.4f}, {georef["bbox_geo"][3]:.4f}]\n'
            f'}}'
        )
        st.markdown(
            f"<pre style='background:#0d1117;border:1px solid #21262d;border-radius:6px;"
            f"padding:.6rem 1rem;font-family:IBM Plex Mono,monospace;font-size:.7rem;"
            f"color:#79c0ff;overflow-x:auto;line-height:1.6;margin-top:.8rem'>{preview}</pre>",
            unsafe_allow_html=True,
        )

        btn_label = (
            "Simpan Kalibrasi & Lanjut Proses" if (has_p1 and has_p2)
            else "Simpan Nilai Manual & Lanjut Proses"
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
