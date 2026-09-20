"""
EcoTriBlend — Multi-Agent Environmental Monitoring Dashboard
Bengaluru Smart City Initiative
"""

import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import pydeck as pdk

load_dotenv()

import orchestrator
from orchestrator import ZONES

import base64

def get_base64_logo() -> str:
    icon_path = PROJECT_ROOT / "ecotriblend_icon.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    logo_path = PROJECT_ROOT / "ecotriblend_logo.jpg"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    return ""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EcoTriBlend | Three Signals. One Combined View.",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — Futuristic Dark Glassmorphism UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.cdnfonts.com/css/cooper-black');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* Apply Cooper Black cleanly to text headings and main UI elements */
html, body, p, h1, h2, h3, h4, h5, h6, a, label, .card-title, .card-value, .card-note, .header-title, .header-subtitle, .pill-row, .control-bar-title, .tech-badge, .risk-level, .reasoning-text, .action-text {
    font-family: 'Cooper Black', 'Cooper Hewitt', 'Bookman Old Style', serif, sans-serif !important;
}

/* Restore Material Symbols font ligatures for all Streamlit internal icons (prevents arrow_right text display) */
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
[data-testid="stIcon"],
[data-testid="stIcon"] *,
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
span[translate="no"],
[data-testid*="Icon"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    font-feature-settings: "liga" 1 !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
}

/* Expander layout fix */
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.6rem !important;
}

[data-testid="stExpander"] summary > div:first-child {
    display: flex !important;
    align-items: center !important;
}

.stApp {
    background: #000000;
    color: #f1f5f9;
}

/* ── Header Wrapper ── */
.header-wrapper {
    text-align: center;
    padding: 1.2rem 1rem 0.8rem;
    margin-bottom: 1.2rem;
}
.header-title {
    font-size: 2.7rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
}
.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Metallic Dark Cards ── */
.glass-card {
    background: linear-gradient(180deg, #2a2a32 0%, #16161c 50px, #0e0e12 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 1.25rem;
    height: 100%;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.8);
}
.card-header-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Glossy Metadata Pill Rows ── */
.pill-list {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    margin-top: 0.8rem;
}
.pill-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.1rem;
    border-radius: 999px;
    font-size: 0.98rem;
    font-weight: 700;
    color: #ffffff;
    box-shadow: inset 0 2px 3px rgba(255, 255, 255, 0.35), 0 4px 14px rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.pill-left {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 1.0rem;
}
.pill-badge {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.8px;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.pill-red {
    background: linear-gradient(90deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
}
.pill-orange {
    background: linear-gradient(90deg, #ea580c 0%, #c2410c 50%, #9a3412 100%);
}
.pill-yellow {
    background: linear-gradient(90deg, #ca8a04 0%, #a16207 50%, #854d0e 100%);
}
.pill-green {
    background: linear-gradient(90deg, #059669 0%, #047857 50%, #065f46 100%);
}
.pill-purple {
    background: linear-gradient(90deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%);
}

/* ── System Control Bar ── */
.system-control-bar {
    background: linear-gradient(180deg, #1f1f28 0%, #0e0e14 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.0rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1.2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
}
.control-bar-left {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}
.control-bar-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
}
.control-bar-glow-line {
    height: 3px;
    width: 150px;
    background: #10b981;
    border-radius: 999px;
    box-shadow: 0 0 10px #10b981;
}
.control-bar-right {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

/* Hide Mapbox/Carto attribution watermarks */
.mapboxgl-ctrl-bottom-right,
.mapboxgl-ctrl-bottom-left,
.mapboxgl-ctrl-attrib,
.mapboxgl-ctrl-logo,
[class*="mapboxgl-ctrl"],
[class*="carto-attribution"],
.deck-tooltip,
div[style*="font-family: Helvetica, Arial, sans-serif;"] {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

/* 3D MAP RADAR (PyDeck) */
.stPyDeckChart {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.tech-badge {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 999px;
    padding: 0.4rem 1.0rem;
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.led-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 12px #10b981;
    margin-left: 0.4rem;
}

/* ── Primary Action Button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 999px !important;
    padding: 0.95rem 2.8rem !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.6) !important;
}

/* ── Agent & Coordinator Cards ── */
.agent-card {
    background: linear-gradient(180deg, #26262e 0%, #0e0e12 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 1.5rem;
    height: 100%;
}
.card-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.card-title { font-size: 0.92rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; margin-bottom: 0.4rem; }
.card-value { font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem; }
.card-note  { font-size: 0.98rem; color: #cbd5e1; line-height: 1.6; }

.coordinator-panel { border-radius: 24px; padding: 2rem; margin: 2rem 0; border: 1px solid; }
.coordinator-panel.risk-high    { background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.05));   border-color: rgba(239,68,68,0.3); }
.coordinator-panel.risk-medium  { background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.05));  border-color: rgba(245,158,11,0.3); }
.coordinator-panel.risk-low     { background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.05));  border-color: rgba(16,185,129,0.3); }
.coordinator-panel.risk-unknown { background: linear-gradient(135deg, rgba(148,163,184,0.08), rgba(100,116,139,0.04)); border-color: rgba(148,163,184,0.2); }
.risk-label { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #94a3b8; margin-bottom: 0.5rem; }
.risk-level { font-size: 3.8rem; font-weight: 800; letter-spacing: -2px; line-height: 1; margin-bottom: 1.2rem; }
.risk-high   .risk-level { color: #fca5a5; }
.risk-medium .risk-level { color: #fde047; }
.risk-low    .risk-level { color: #6ee7b7; }
.risk-unknown .risk-level { color: #94a3b8; }
.reasoning-box { background: rgba(0, 0, 0, 0.4); border-radius: 14px; padding: 1.2rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.06); }
.reasoning-label { font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; margin-bottom: 0.5rem; }
.reasoning-text  { font-size: 1.05rem; color: #e2e8f0; line-height: 1.7; }
.action-box  { background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 14px; padding: 1rem 1.2rem; }
.action-text { font-size: 1.02rem; color: #93c5fd; font-weight: 500; line-height: 1.5; }

.section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 2rem 0; }

div[data-testid="stSelectbox"] > div > div {
    background: linear-gradient(180deg, #374151 0%, #1f2937 100%);
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px; color: #f1f5f9;
}
/* Fix Streamlit File Uploader button text overlap */
div[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px dashed rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 0.6rem 0.8rem !important;
}

div[data-testid="stFileUploader"] label {
    display: none !important;
}

div[data-testid="stFileUploader"] section {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 0.8rem !important;
    padding: 0.2rem !important;
}

div[data-testid="stFileUploader"] button {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    padding: 0.35rem 1rem !important;
    margin: 0 !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] section > div {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
}

div[data-testid="stFileUploader"] section > div * {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers & Metadata
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444", "unknown": "#94a3b8"}
RISK_EMOJIS     = {"low": "🟢", "medium": "🟡", "high": "🔴", "unknown": "⚪"}

ZONE_META = {
    "Zone 1 - Riverside Industrial": {
        "coords": "12.9141°N, 77.6432°E",
        "lat": 12.9141, "lon": 77.6432,
        "area": "Bellandur / Varthur belt",
        "expected": "🔴 High risk (hero demo zone)",
        "temp": "High risk",
        "wind1": "12.9141°N, 77.6432°E",
        "wind2": "12.9141°N, 77.6432°E",
        "air": "Air risk",
        "conservation": "High risk",
    },
    "Zone 2 - Central Market": {
        "coords": "12.9634°N, 77.5760°E",
        "lat": 12.9634, "lon": 77.5760,
        "area": "KR Market, Central Bengaluru",
        "expected": "🟢 Low risk (contrast zone)",
        "temp": "Low risk",
        "wind1": "12.9634°N, 77.5760°E",
        "wind2": "12.9634°N, 77.5760°E",
        "air": "Clear",
        "conservation": "Low risk",
    },
    "Zone 3 - Lakeside Residential": {
        "coords": "12.9784°N, 77.6183°E",
        "lat": 12.9784, "lon": 77.6183,
        "area": "Ulsoor Lake neighbourhood",
        "expected": "🟡 Medium risk",
        "temp": "Medium risk",
        "wind1": "12.9784°N, 77.6183°E",
        "wind2": "12.9784°N, 77.6183°E",
        "air": "Moderate risk",
        "conservation": "Medium risk",
    },
    "Zone 4 - Outer Ring Road": {
        "coords": "12.9352°N, 77.6940°E",
        "lat": 12.9352, "lon": 77.6940,
        "area": "Marathahalli / ORR junction",
        "expected": "🟡 Low-medium risk",
        "temp": "Low risk",
        "wind1": "12.9352°N, 77.6940°E",
        "wind2": "12.9352°N, 77.6940°E",
        "air": "Low risk",
        "conservation": "Low risk",
    },
}

SAMPLE_IMAGES = {
    "Sample 1 - Heavy Litter": "sample_litter_1.jpg",
    "Sample 2 - Market street": "sample_litter_2.jpg",
    "Sample 3 - Clean sidewalk": "sample_litter_3.jpg",
}

PROJECT_ROOT = Path(__file__).parent


def draw_bounding_boxes(image_path: str, boxes: list) -> Image.Image:
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    palette = ["#ef4444","#f59e0b","#10b981","#3b82f6","#8b5cf6","#ec4899","#06b6d4","#84cc16"]
    label_colors: dict[str, str] = {}
    color_idx = 0
    for box in boxes:
        x1, y1, x2, y2, label, conf = box
        if label not in label_colors:
            label_colors[label] = palette[color_idx % len(palette)]
            color_idx += 1
        color = label_colors[label]
        for offset in range(3):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)
        caption   = f"{label} {conf:.0%}"
        text_bbox = draw.textbbox((x1, y1-18), caption)
        draw.rectangle([text_bbox[0]-3, text_bbox[1]-2, text_bbox[2]+3, text_bbox[3]+2], fill=color)
        draw.text((x1, y1-18), caption, fill="white")
    return img


def render_agent_card(icon: str, title: str, result: dict, unit: str = ""):
    severity = result.get("severity", "unknown")
    value    = result.get("value", 0)
    note     = result.get("note", "")
    color    = SEVERITY_COLORS.get(severity, "#94a3b8")
    st.markdown(f"""
    <div class="agent-card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <span class="badge-tag tag-{severity}">{severity.upper()}</span>
        <div class="card-value" style="color:{color};">{value}{unit}</div>
        <div class="card-note">{note}</div>
    </div>""", unsafe_allow_html=True)


def render_coordinator(result: dict):
    risk      = result.get("overall_risk", "unknown").lower()
    reasoning = result.get("reasoning", "No reasoning provided.")
    action    = result.get("recommended_action", "No action recommended.")
    emoji     = RISK_EMOJIS.get(risk, "⚪")
    st.markdown(f"""
    <div class="coordinator-panel risk-{risk}">
        <div class="risk-label">🤖 AI Coordinator Verdict — Groq LLaMA 3</div>
        <div class="risk-level">{emoji} {risk.upper()} RISK</div>
        <div class="reasoning-box">
            <div class="reasoning-label">📊 Cross-Signal Reasoning</div>
            <div class="reasoning-text">{reasoning}</div>
        </div>
        <div class="action-box">
            <div class="action-text">⚡ <strong>Recommended Action:</strong> {action}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "sample_image_path" not in st.session_state:
    st.session_state.sample_image_path = str(PROJECT_ROOT / "sample_litter_1.jpg")
if "sample_image_label" not in st.session_state:
    st.session_state.sample_image_label = "Sample 1 - Heavy Litter"

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
logo_b64 = get_base64_logo()
st.markdown(f"""
<div class="header-wrapper">
    <div style="display: flex; align-items: center; justify-content: center; gap: 1.2rem; margin-bottom: 0.5rem;">
        <img src="{logo_b64}" style="height: 80px; width: 80px; object-fit: contain; filter: drop-shadow(0 0 25px rgba(52, 211, 153, 0.5));" />
        <div style="text-align: left;">
            <h1 class="header-title" style="font-size: 2.5rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #34d399 0%, #10b981 40%, #fbbf24 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EcoTriBlend</h1>
            <div style="font-size: 0.95rem; font-weight: 600; color: #a7f3d0; letter-spacing: 0.5px;">Three Signals. One Combined View.</div>
        </div>
    </div>
    <div class="header-subtitle" style="margin-top: 0.2rem;"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main 3-Column Interface Layout
# ---------------------------------------------------------------------------
col_loc, col_map, col_img = st.columns([1, 1.15, 1])

import requests

def get_coordinates_for_location(loc_str: str, preset_zone: str) -> tuple[float, float, str]:
    preset_meta = ZONE_META.get(preset_zone, {})
    if not loc_str or not loc_str.strip():
        return preset_meta.get("lat", 12.9716), preset_meta.get("lon", 77.5946), preset_meta.get("area", preset_zone)
    
    clean_str = loc_str.strip()
    
    for z_name, z_data in ZONE_META.items():
        if clean_str.lower() in z_name.lower() or clean_str.lower() in z_data.get("area", "").lower():
            return z_data["lat"], z_data["lon"], z_data.get("area", z_name)

    if "," in clean_str:
        try:
            parts = clean_str.split(",")
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            return lat, lon, f"Custom ({lat:.4f}°N, {lon:.4f}°E)"
        except ValueError:
            pass

    try:
        query = clean_str
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
                "namedetails": 1,
            },
            headers={"User-Agent": "EcoTriBlend/1.0"},
            timeout=5
        )
        if resp.status_code == 200 and resp.json():
            # Prefer the result with the highest importance score for accuracy
            results = resp.json()
            best = max(results, key=lambda r: float(r.get("importance", 0)))
            lat, lon = float(best["lat"]), float(best["lon"])
            display_name = best.get("display_name", clean_str).split(",")[0]
            return lat, lon, display_name
    except Exception:
        pass

    return preset_meta.get("lat", 12.9716), preset_meta.get("lon", 77.5946), clean_str

def deg_to_cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25) / 22.5)
    return dirs[ix % 16]

def get_live_weather_and_wind(lat: float, lon: float) -> dict:
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            cw = resp.json().get("current_weather", {})
            temp = cw.get("temperature", 25.0)
            wspeed = cw.get("windspeed", 10.0)
            wdeg = cw.get("winddirection", 0.0)
            cardinal = deg_to_cardinal(wdeg)
            return {
                "temp_c": temp,
                "wind_speed_kmh": wspeed,
                "wind_deg": wdeg,
                "wind_cardinal": cardinal,
                "wind_str": f"{cardinal} {int(wdeg)}° ({wspeed:.1f} km/h)",
                "temp_str": f"{temp:.1f}°C"
            }
    except Exception:
        pass
    return {
        "temp_c": 26.0,
        "wind_speed_kmh": 12.0,
        "wind_deg": 135.0,
        "wind_cardinal": "SE",
        "wind_str": "SE 135° (12.0 km/h)",
        "temp_str": "26.0°C"
    }

# ── COLUMN 1: LOCATION CARD ────────────────────────────────────────────────
with col_loc:
    st.markdown('<div class="card-header-title">📍 Location & Monitoring Zone</div>', unsafe_allow_html=True)
    
    manual_loc = st.text_input(
        "Manual Location Input",
        value="",
        placeholder="Search any global city/location (e.g. Tokyo, Paris, New York, 48.85, 2.35)...",
        key="manual_location_input",
        label_visibility="collapsed"
    )

    zone_options = ZONES
    zone_display = []
    for z in zone_options:
        if "Zone 1" in z:
            zone_display.append(z + " ✦ hero")
        else:
            zone_display.append(z)

    selected_idx = st.selectbox(
        "Select Zone",
        options=range(len(zone_options)),
        format_func=lambda i: zone_display[i],
        index=0,
        key="zone_select",
        label_visibility="collapsed"
    )
    selected_zone = zone_options[selected_idx]

    center_lat, center_lon, active_area_name = get_coordinates_for_location(manual_loc, selected_zone)
    effective_zone = active_area_name if (manual_loc and manual_loc.strip()) else selected_zone
    meta = ZONE_META.get(selected_zone, {})
    live_w = get_live_weather_and_wind(center_lat, center_lon)

    # Retrieve live pipeline state if available
    pipe_st = st.session_state.get("pipeline_state", {}) or {}
    air_res = pipe_st.get("air_result", {}) or {}
    water_res = pipe_st.get("water_result", {}) or {}
    litter_res = pipe_st.get("litter_result", {}) or {}
    coord_res = pipe_st.get("coordinator_result", {}) or {}

    # 1. Location Risk Badge
    loc_risk = coord_res.get("overall_risk", "high").lower()
    loc_pill_class = "pill-red" if loc_risk == "high" else ("pill-orange" if loc_risk == "medium" else "pill-green")
    loc_badge_text = f"⚠️ {loc_risk.upper()} RISK" if loc_risk in ("high", "medium") else f"🌿 {loc_risk.upper()} RISK"

    # 2. Temperature Badge
    temp_val = live_w["temp_c"]
    temp_pill_class = "pill-orange" if temp_val > 28 else "pill-green"
    temp_badge_text = f"🌡️ {live_w['temp_str']}"

    # 3. Wind Direction Badge (Real-time live wind vector)
    wind_badge_text = f"🚩 {live_w['wind_str']}"

    # 4. Air Quality Badge (Live Agent Severity)
    air_sev = air_res.get("severity", "low").lower()
    air_val = air_res.get("value", 45)
    air_pill_class = "pill-red" if air_sev == "high" else ("pill-yellow" if air_sev == "medium" else "pill-green")
    air_badge_text = f"⚠️ HIGH RISK (AQI {air_val})" if air_sev == "high" else (f"⚠️ MODERATE (AQI {air_val})" if air_sev == "medium" else f"🌿 GOOD (AQI {air_val})")

    # 5. Conservation / Waste Badge (Live Agent Severity)
    litter_sev = litter_res.get("severity", "high").lower()
    box_cnt = len(litter_res.get("boxes", []))
    litter_pill_class = "pill-purple" if litter_sev == "high" else ("pill-orange" if litter_sev == "medium" else "pill-green")
    litter_badge_text = f"⚠️ HIGH RISK ({box_cnt} items)" if litter_sev == "high" else (f"⚠️ MODERATE ({box_cnt} items)" if litter_sev == "medium" else f"🌿 SAFE ({box_cnt} items)")

    # Styled metadata glossy pill rows dynamically updating from live data
    st.markdown(f"""
    <div class="pill-list">
        <div class="pill-row {loc_pill_class}">
            <div class="pill-left">
                <span>✚</span>
                <span>{effective_zone}</span>
            </div>
            <div class="pill-badge">{loc_badge_text}</div>
        </div>
        <div class="pill-row {temp_pill_class}">
            <div class="pill-left">
                <span>🌡️</span>
                <span>Temperature</span>
            </div>
            <div class="pill-badge">{temp_badge_text}</div>
        </div>
        <div class="pill-row pill-yellow">
            <div class="pill-left">
                <span>🚩</span>
                <span>Wind direction</span>
            </div>
            <div class="pill-badge">{wind_badge_text}</div>
        </div>
        <div class="pill-row {air_pill_class}">
            <div class="pill-left">
                <span>🌳</span>
                <span>Air Quality</span>
            </div>
            <div class="pill-badge">{air_badge_text}</div>
        </div>
        <div class="pill-row {litter_pill_class}">
            <div class="pill-left">
                <span>♻️</span>
                <span>Conservation</span>
            </div>
            <div class="pill-badge">{litter_badge_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── COLUMN 2: MAP CARD ─────────────────────────────────────────────────────
with col_map:
    st.markdown('<div class="card-header-title">🗺️ Live Geographic Map</div>', unsafe_allow_html=True)
    
    # Use robust Scatterplot layers for perfect concentric circles (avoids polygon glitching)
    map_rings_df = pd.DataFrame([
        # Outer Ring (Purple, 30% opacity)
        {"name": "Outer Bounds", "lat": center_lat, "lon": center_lon, "color": [124, 58, 237, 76], "radius": 3200},
        # Mid Ring (Green, 30% opacity)
        {"name": "Mid Zone", "lat": center_lat, "lon": center_lon, "color": [16, 185, 129, 76], "radius": 2000},
        # Inner Ring (Red, 30% opacity)
        {"name": "Core Area", "lat": center_lat, "lon": center_lon, "color": [239, 68, 68, 76], "radius": 1000},
        # Solid Center Point
        {"name": effective_zone, "lat": center_lat, "lon": center_lon, "color": [239, 68, 68, 255], "radius": 150},
    ])

    scatter_layer = pdk.Layer(
        "ScatterplotLayer", map_rings_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_line_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
    )

    zoom_level = 14.0 if manual_loc and manual_loc.strip() else 12.0

    view_state = pdk.ViewState(
        latitude=center_lat, longitude=center_lon, zoom=zoom_level, pitch=25,
    )

    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        map_provider=None,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}"}
    )
    
    st.pydeck_chart(deck, use_container_width=True, height=330)

# ── COLUMN 3: IMAGE UPLOAD CARD ──────────────────────────────────
with col_img:
    st.markdown('<div class="card-header-title">📷 Scene Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Scene Image",
        type=["jpg","jpeg","png","webp","bmp"],
        label_visibility="collapsed",
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption=f"📷 Uploaded: {uploaded_file.name}", use_container_width=True)

# ---------------------------------------------------------------------------
# System Control Bar Below Top Grid
# ---------------------------------------------------------------------------
st.markdown("""
<div class="system-control-bar">
    <div class="control-bar-left">
        <div class="control-bar-title">Multi-Agent Environmental Monitoring System</div>
        <div class="control-bar-glow-line"></div>
    </div>
    <div class="control-bar-right">
        <div class="tech-badge">🔬 YOLOv8</div>
        <div class="tech-badge">🧠 Groq AI</div>
        <div class="tech-badge">⚡ LangGraph</div>
        <div class="led-dot"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prominent Run Button & Status Section
# ---------------------------------------------------------------------------
st.markdown('<div style="margin-top: 1.8rem;"></div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 1.6, 1])
with btn_col:
    run_clicked = st.button("Run Environmental Analysis", use_container_width=True)

# Status bar & leaf icon below button
st.markdown(f"""
<div class="status-bar-container">
    <div class="status-icon-circle">🌿</div>
    <div class="status-progress-track">
        <div class="status-progress-fill"></div>
    </div>
    <div class="status-text">EcoTriBlend Active: Monitoring {effective_zone}</div>
</div>
""", unsafe_allow_html=True)

# City skyline SVG graphic at bottom of header section
st.markdown("""
<svg viewBox="0 0 1200 120" preserveAspectRatio="none" style="width: 100%; height: 70px; opacity: 0.25; margin-top: -1.5rem; fill: #10b981;">
  <path d="M0 120 L0 90 L20 90 L20 120 L40 120 L40 70 L70 70 L70 120 L90 120 L90 50 L130 50 L130 120 L150 120 L150 80 L180 80 L180 120 L210 120 L210 30 L250 30 L250 120 L280 120 L280 95 L310 95 L310 120 L350 120 L350 40 L400 40 L400 120 L430 120 L430 75 L470 75 L470 120 L510 120 L510 20 L560 20 L560 120 L600 120 L600 65 L640 65 L640 120 L680 120 L680 35 L730 35 L730 120 L770 120 L770 85 L810 85 L810 120 L850 120 L850 55 L900 55 L900 120 L940 120 L940 70 L980 70 L980 120 L1020 120 L1020 45 L1070 45 L1070 120 L1110 120 L1110 80 L1150 80 L1150 120 L1200 120 L1200 120 Z"></path>
</svg>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Execution & Analysis Results
# ---------------------------------------------------------------------------
if run_clicked or "pipeline_state" in st.session_state:
    if run_clicked:
        image_path_to_use = None
        tmp_path = None

        if uploaded_file is not None:
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            image_path_to_use = tmp_path
        elif st.session_state.sample_image_path:
            image_path_to_use = st.session_state.sample_image_path
        else:
            st.warning("Please upload an image or select a sample image.")
            st.stop()

        with st.spinner("Running multi-agent pipeline… Air · Water · Litter · Coordinator"):
            try:
                state = orchestrator.run_pipeline(
                    zone=effective_zone,
                    image_path=image_path_to_use,
                )
                st.session_state.pipeline_state = state
                st.session_state.current_image = image_path_to_use
                
                # Polish & Delight: UI feedback on completion
                st.toast("✅ Analysis complete! Scroll down to view the Coordinator's verdict.", icon="✨")
                if state.get("coordinator_result", {}).get("overall_risk", "").lower() == "low":
                    st.balloons()
                    
            except Exception as exc:
                st.error(f"Pipeline Error: {exc}")
                st.stop()
    
    state = st.session_state.pipeline_state
    image_path_to_use = st.session_state.get("current_image", st.session_state.sample_image_path)

    air_result    = state.get("air_result", {})
    water_result  = state.get("water_result", {})
    litter_result = state.get("litter_result", {})
    coord_result  = state.get("coordinator_result", {})
    pipe_errors   = state.get("errors", [])

    # Results Header
    st.markdown(
        f"<h3 style='color:#ffffff; margin-bottom:0.2rem;'>📡 Results for {effective_zone}</h3>"
        f"<p style='color:#94a3b8; font-size:0.95rem; margin-top:0;'>📍 Location: {active_area_name} ({center_lat:.4f}°N, {center_lon:.4f}°E)</p>",
        unsafe_allow_html=True,
    )

    # Specialist Agent Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        render_agent_card("💨", "Air Quality Agent",   air_result,    " AQI")
    with c2:
        render_agent_card("🌊", "Water Quality Agent", water_result,  " NTU")
    with c3:
        render_agent_card("🗑️", "Litter Detection",    litter_result, " objects")

    if pipe_errors:
        with st.expander("⚠️ Pipeline Warnings", expanded=False):
            for err in pipe_errors:
                st.warning(err)

    # Coordinator Verdict
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_coordinator(coord_result)

    # Scene Analysis (YOLO Bounding Boxes)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Litter Detection — Scene Analysis")

    boxes = litter_result.get("boxes", [])
    img_col, meta_col = st.columns([3, 2])

    with img_col:
        if image_path_to_use and os.path.exists(image_path_to_use):
            annotated = draw_bounding_boxes(image_path_to_use, boxes)
            st.image(
                annotated,
                caption=f"Detected {len(boxes)} object(s) · Severity: {litter_result.get('severity','unknown').upper()}",
                use_container_width=True,
            )

    with meta_col:
        severity = litter_result.get("severity","unknown")
        st.markdown(f'<span class="badge-tag tag-{severity}">{severity.upper()}</span>', unsafe_allow_html=True)
        st.metric("Total Objects Detected", len(boxes))
        if boxes:
            st.markdown("**Detected Labels:**")
            label_counts: dict[str, int] = {}
            for box in boxes:
                lbl = box[4]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
                st.markdown(f"- `{lbl}` × {cnt}")
        else:
            st.info("No objects detected in this image.")

    # Water Quality Breakdown Expander
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    with st.expander("💧 Water Quality Details", expanded=False):
        w1, w2, w3, w4 = st.columns(4)
        with w1: st.metric("pH", water_result.get("ph", "N/A"))
        with w2: st.metric("Turbidity (NTU)", water_result.get("turbidity_ntu", "N/A"))
        with w3: st.metric("Coliform", "⚠️ Detected" if water_result.get("coliform_detected") else "✅ Clear")
        with w4: st.metric("WQI Score", f"{water_result.get('wqi_score', 'N/A')}/100")

    # Air Quality Pollutant Breakdown Expander
    st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    if air_result.get("components"):
        with st.expander("💨 Air Pollutant Breakdown (live OWM API)", expanded=False):
            comps = air_result["components"]
            st.markdown('<div style="margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, (p, v) in enumerate(comps.items()):
                with cols[i % 4]:
                    st.metric(p.upper(), f"{v:.2f}")
            st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
            st.caption(f"📍 Coordinates: {air_result.get('lat', 0):.4f}°N, {air_result.get('lon', 0):.4f}°E  ·  "
                       f"🌡️ {air_result.get('temperature_c', 'N/A')}°C  ·  "
                       f"💨 Wind {air_result.get('wind_speed_ms', 'N/A')} m/s {air_result.get('wind_direction', '')}")

else:
    # Initial state prompt
    st.markdown(f"""
    <div style="text-align:center; padding:2rem 1rem; color:#64748b;">
        <h4 style="color:#94a3b8; font-weight:600; font-size:1.2rem;">Ready to Monitor {effective_zone}</h4>
        <p style="font-size:0.98rem; max-width:550px; margin:0 auto; line-height:1.6;">
            Select a monitoring zone or enter a custom location, choose or upload a scene image, and click
            <strong style="color:#60a5fa;">Run Environmental Analysis</strong> to evaluate multi-agent environmental risk.
        </p>
    </div>
    """, unsafe_allow_html=True)
