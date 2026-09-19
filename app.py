"""
EcoWatch — Multi-Agent Environmental Monitoring Dashboard
Run with: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

import orchestrator
from orchestrator import ZONES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EcoWatch | Environmental Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2e 50%, #091529 100%);
    color: #e2e8f0;
}

/* ── Header ── */
.eco-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(59,130,246,0.08) 100%);
    border-radius: 20px;
    border: 1px solid rgba(16,185,129,0.15);
    margin-bottom: 2rem;
}
.eco-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; letter-spacing: -1px;
}
.eco-subtitle { font-size: 1.05rem; color: #94a3b8; margin-top: 0.5rem; }

/* ── Zone meta strip ── */
.zone-meta {
    display: flex; gap: 1.5rem; flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.9rem 1.2rem;
    margin-bottom: 1.2rem; font-size: 0.82rem; color: #94a3b8;
}
.zone-meta span { display: flex; align-items: center; gap: 0.4rem; }
.zone-meta strong { color: #e2e8f0; }

/* ── Input panel ── */
.input-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem;
}

/* ── Sample photo buttons ── */
.sample-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: #475569; margin-bottom: 0.5rem;
}

/* ── Agent cards ── */
.agent-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem; height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.agent-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,0.3); }
.card-icon  { font-size: 2.2rem; margin-bottom: 0.6rem; }
.card-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: #64748b; margin-bottom: 0.4rem; }
.card-value { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.3rem; }
.card-note  { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }

/* ── Severity badges ── */
.badge { display: inline-block; padding: 0.2rem 0.9rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; }
.badge-low    { background: rgba(16,185,129,0.2);  color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-medium { background: rgba(245,158,11,0.2);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-high   { background: rgba(239,68,68,0.2);   color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-unknown{ background: rgba(148,163,184,0.2); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

/* ── Coordinator panel ── */
.coordinator-panel { border-radius: 20px; padding: 2rem; margin: 2rem 0; border: 1px solid; }
.coordinator-panel.risk-high    { background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));   border-color: rgba(239,68,68,0.3); }
.coordinator-panel.risk-medium  { background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.06));  border-color: rgba(245,158,11,0.3); }
.coordinator-panel.risk-low     { background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));  border-color: rgba(16,185,129,0.3); }
.coordinator-panel.risk-unknown { background: linear-gradient(135deg, rgba(148,163,184,0.1), rgba(100,116,139,0.05)); border-color: rgba(148,163,184,0.2); }
.risk-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #64748b; margin-bottom: 0.5rem; }
.risk-level { font-size: 3.5rem; font-weight: 800; letter-spacing: -2px; line-height: 1; margin-bottom: 1.2rem; }
.risk-high   .risk-level { color: #ef4444; }
.risk-medium .risk-level { color: #f59e0b; }
.risk-low    .risk-level { color: #10b981; }
.risk-unknown .risk-level { color: #94a3b8; }
.reasoning-box { background: rgba(0,0,0,0.25); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.06); }
.reasoning-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #475569; margin-bottom: 0.5rem; }
.reasoning-text  { font-size: 0.92rem; color: #cbd5e1; line-height: 1.7; }
.action-box  { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); border-radius: 12px; padding: 1rem 1.2rem; }
.action-text { font-size: 0.88rem; color: #93c5fd; font-weight: 500; line-height: 1.5; }

/* ── Divider ── */
.section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); margin: 2rem 0; }

/* ── Hero tag ── */
.hero-tag {
    display: inline-block; background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; padding: 0.15rem 0.7rem; border-radius: 999px;
    margin-left: 0.5rem; vertical-align: middle;
}

/* ── Streamlit overrides ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #10b981, #3b82f6);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2rem; font-weight: 600; font-size: 0.95rem;
    transition: opacity 0.2s; width: 100%;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; border: none; }

div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px; color: #e2e8f0;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 12px; padding: 0.5rem;
}
label { color: #94a3b8 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444", "unknown": "#94a3b8"}
RISK_EMOJIS     = {"low": "🟢", "medium": "🟡", "high": "🔴", "unknown": "⚪"}

# Zone metadata shown in the info strip
ZONE_META = {
    "Zone 1 - Riverside Industrial": {"coords": "12.9141°N, 77.6432°E", "area": "Bellandur / Varthur belt", "expected": "🔴 High risk (hero demo zone)"},
    "Zone 2 - Central Market":       {"coords": "12.9634°N, 77.5760°E", "area": "KR Market, Central Bengaluru", "expected": "🟢 Low risk (contrast zone)"},
    "Zone 3 - Lakeside Residential": {"coords": "12.9784°N, 77.6183°E", "area": "Ulsoor Lake neighbourhood",  "expected": "🟡 Medium risk"},
    "Zone 4 - Outer Ring Road":      {"coords": "12.9352°N, 77.6940°E", "area": "Marathahalli / ORR junction", "expected": "🟡 Low-medium risk"},
}

SAMPLE_IMAGES = {
    "🏭 Sample 1 — Riverside (heavy litter)": "sample_litter_1.jpg",
    "🛒 Sample 2 — Market street (moderate)": "sample_litter_2.jpg",
    "🌿 Sample 3 — Clean sidewalk (low)":     "sample_litter_3.jpg",
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
        <span class="badge badge-{severity}">{severity.upper()}</span>
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
# Session state — for sample image selection
# ---------------------------------------------------------------------------
if "sample_image_path" not in st.session_state:
    st.session_state.sample_image_path = None
if "sample_image_label" not in st.session_state:
    st.session_state.sample_image_label = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="eco-header">
    <h1 class="eco-title">🌍 EcoWatch</h1>
    <p class="eco-subtitle">Multi-Agent Environmental Monitoring · Bengaluru Pilot · YOLOv8 · Groq AI · LangGraph</p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

col_zone, col_img = st.columns([2, 3])

with col_zone:
    # Zone dropdown — Zone 1 is default (hero demo)
    zone_options = ZONES
    zone_display = []
    for z in zone_options:
        if "Zone 1" in z:
            zone_display.append(z + "  ★ hero")
        else:
            zone_display.append(z)

    selected_idx = st.selectbox(
        "📍 Select Monitoring Zone",
        options=range(len(zone_options)),
        format_func=lambda i: zone_display[i],
        index=0,
        key="zone_select",
    )
    selected_zone = zone_options[selected_idx]

    # Zone info strip
    meta = ZONE_META.get(selected_zone, {})
    st.markdown(f"""
    <div class="zone-meta">
        <span>📌 <strong>{meta.get('area','')}</strong></span>
        <span>🗺️ {meta.get('coords','')}</span>
        <span>Expected: {meta.get('expected','')}</span>
    </div>""", unsafe_allow_html=True)

with col_img:
    uploaded_file = st.file_uploader(
        "📸 Upload Scene Image for Litter Detection",
        type=["jpg","jpeg","png","webp","bmp"],
        help="Or use a sample image below",
    )

    # Sample image quick-load buttons
    st.markdown('<div class="sample-label">🖼️ Or use a sample image:</div>', unsafe_allow_html=True)
    btn_cols = st.columns(3)
    for idx, (label, filename) in enumerate(SAMPLE_IMAGES.items()):
        with btn_cols[idx]:
            sample_path = PROJECT_ROOT / filename
            disabled    = not sample_path.exists()
            if st.button(label, key=f"sample_{idx}", disabled=disabled):
                st.session_state.sample_image_path  = str(sample_path)
                st.session_state.sample_image_label = label

    # Show which sample is active
    if st.session_state.sample_image_path and uploaded_file is None:
        st.caption(f"✅ Using: {st.session_state.sample_image_label}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
_, run_col, _ = st.columns([1, 2, 1])
with run_col:
    run_clicked = st.button("🚀 Run Environmental Analysis", use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_clicked:
    # Resolve image path: uploaded file > sample selection
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

    with st.spinner("🔄 Running multi-agent pipeline… Air · Water · Litter · Coordinator"):
        try:
            state = orchestrator.run_pipeline(
                zone=selected_zone,
                image_path=image_path_to_use,
            )
        except Exception as exc:
            st.error(f"❌ Pipeline Error: {exc}")
            st.stop()

    air_result    = state.get("air_result",    {})
    water_result  = state.get("water_result",  {})
    litter_result = state.get("litter_result", {})
    coord_result  = state.get("coordinator_result", {})
    pipe_errors   = state.get("errors", [])

    # ── Zone header ────────────────────────────────────────────────────────
    hero_tag = '<span class="hero-tag">HERO ZONE</span>' if "Zone 1" in selected_zone else ""
    st.markdown(
        f"<h3 style='color:#e2e8f0; margin-bottom:0.2rem;'>📡 Results for {selected_zone}{hero_tag}</h3>"
        f"<p style='color:#64748b; font-size:0.85rem; margin-top:0;'>Bengaluru · {ZONE_META.get(selected_zone,{}).get('area','')}</p>",
        unsafe_allow_html=True,
    )

    # ── Specialist agent cards ─────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        render_agent_card("💨", "Air Quality Agent",   air_result,    " AQI")
    with c2:
        render_agent_card("🌊", "Water Quality Agent", water_result,  " NTU")
    with c3:
        render_agent_card("🗑️", "Litter Detection",    litter_result, " objects")

    # ── Pipeline warnings ──────────────────────────────────────────────────
    if pipe_errors:
        with st.expander("⚠️ Pipeline Warnings", expanded=False):
            for err in pipe_errors:
                st.warning(err)

    # ── Coordinator verdict ────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_coordinator(coord_result)

    # ── Litter image panel ─────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Litter Detection — Scene Analysis")

    boxes = litter_result.get("boxes", [])
    img_col, meta_col = st.columns([3, 2])

    with img_col:
        annotated = draw_bounding_boxes(image_path_to_use, boxes)
        st.image(
            annotated,
            caption=f"Detected {len(boxes)} object(s) · Severity: {litter_result.get('severity','unknown').upper()}",
            use_container_width=True,
        )

    with meta_col:
        severity = litter_result.get("severity","unknown")
        color    = SEVERITY_COLORS.get(severity,"#94a3b8")
        st.markdown(f'<span class="badge badge-{severity}">{severity.upper()}</span>', unsafe_allow_html=True)
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

    # ── Water detail expander ──────────────────────────────────────────────
    with st.expander("💧 Water Quality Details (mock_zones.csv)", expanded=False):
        w1, w2, w3 = st.columns(3)
        with w1: st.metric("pH",            water_result.get("ph","N/A"))
        with w2: st.metric("Turbidity NTU", water_result.get("turbidity_ntu","N/A"))
        with w3: st.metric("Coliform",      "⚠️ Detected" if water_result.get("coliform_detected") else "✅ Clear")

    # ── Air detail expander ────────────────────────────────────────────────
    if air_result.get("components"):
        with st.expander("💨 Air Pollutant Breakdown (live OWM API)", expanded=False):
            comps = air_result["components"]
            cols  = st.columns(4)
            for i, (p, v) in enumerate(comps.items()):
                with cols[i % 4]:
                    st.metric(p.upper(), f"{v:.2f}")
            st.caption(f"📍 Coordinates used: {air_result.get('lat'):.4f}°N, {air_result.get('lon'):.4f}°E")

    # Cleanup temp file
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

else:
    # ── Empty state ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#475569;">
        <div style="font-size:4rem; margin-bottom:1rem;">🌿</div>
        <h3 style="color:#64748b; font-weight:600;">Ready to Monitor Bengaluru</h3>
        <p style="max-width:520px; margin:0 auto; line-height:1.7; font-size:0.9rem;">
            Select a monitoring zone, pick a sample image (or upload your own), and click
            <strong>Run Environmental Analysis</strong> to fire the four-agent pipeline.
        </p>
        <br/>
        <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap; font-size:0.82rem; color:#334155; margin-top:0.5rem;">
            <span>💨 Air · Live OWM AQI</span>
            <span>🌊 Water · mock_zones.csv</span>
            <span>🗑️ Litter · YOLOv8 Nano</span>
            <span>🤖 Coordinator · Groq LLaMA 3</span>
        </div>
        <br/>
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:1rem 1.5rem; max-width:480px; margin:1rem auto; font-size:0.82rem; color:#fca5a5;">
            ⭐ <strong>Hero demo:</strong> Select <em>Zone 1 — Riverside Industrial</em> + Sample 1 for the full multi-signal HIGH RISK story.
        </div>
    </div>""", unsafe_allow_html=True)
