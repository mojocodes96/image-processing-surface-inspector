"""
Surface Defect Inspector v2
ML-based cosmetic quality inspection for consumer hardware surfaces.
Author: Marcus Alvarez | Quality Engineering Portfolio
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import io
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Surface Defect Inspector", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d0d0f; color: #e8e8ed; }
section[data-testid="stSidebar"] { background-color: #111114; border-right: 1px solid #222228; }
h1, h2, h3 { color: #f5f5f7; letter-spacing: -0.02em; }
.card { background: #1a1a1f; border: 1px solid #2a2a32; border-radius: 12px; padding: 20px 24px; margin-bottom: 12px; }
.verdict-pass { background: linear-gradient(135deg,#0a2e1a,#0f3d22); border:1px solid #1a7a3a; border-radius:16px; padding:28px 32px; text-align:center; }
.verdict-fail { background: linear-gradient(135deg,#2e0a0a,#3d0f0f); border:1px solid #7a1a1a; border-radius:16px; padding:28px 32px; text-align:center; }
.eyebrow { font-size:11px; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; color:#636366; margin-bottom:8px; }
.feat-row { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #1e1e24; font-size:13px; }
.ctq-tag { display:inline-block; background:#1c1c28; border:1px solid #2e2e40; border-radius:6px; padding:4px 10px; font-size:12px; font-family:'JetBrains Mono',monospace; color:#a0a0b8; margin:3px; }
.stButton > button { background:#2563eb; color:white; border:none; border-radius:10px; font-weight:600; font-size:14px; width:100%; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_gray_f32(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return gray, gray.astype(np.float32)


def extract_features(img_array):
    gray, gray_f32 = to_gray_f32(img_array)
    h, w = gray.shape

    lap = cv2.Laplacian(gray_f32, cv2.CV_32F)
    lap_var = float(lap.var())
    lap_max = float(np.abs(lap).max())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.sum()) / (255.0 * h * w)
    surface_std = float(gray.std())

    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray_f32)))
    cy, cx = h // 2, w // 2
    yi, xi = np.ogrid[:h, :w]
    hf_mask = (yi - cy)**2 + (xi - cx)**2 > (min(h, w) // 6)**2
    hf_energy = float(fft_mag[hf_mask].mean()) / (float(fft_mag.mean()) + 1e-8)

    _, dark = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = float(dark.sum()) / (255.0 * h * w)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                             minLineLength=max(10, min(h,w)//8), maxLineGap=10)
    line_count = len(lines) if lines is not None else 0

    mean_brightness = float(gray.mean())
    quads = [gray[:h//2,:w//2], gray[:h//2,w//2:], gray[h//2:,:w//2], gray[h//2:,w//2:]]
    brightness_uniformity = float(np.std([q.mean() for q in quads]))
    channel_divergence = float(np.std([img_array[:,:,i].mean() for i in range(3)]))

    return {
        "laplacian_variance":    round(lap_var, 3),
        "laplacian_max":         round(lap_max, 3),
        "edge_density":          round(edge_density, 5),
        "surface_std":           round(surface_std, 3),
        "hf_energy_ratio":       round(hf_energy, 4),
        "dark_spot_ratio":       round(dark_ratio, 5),
        "scratch_line_count":    int(line_count),
        "brightness_uniformity": round(brightness_uniformity, 3),
        "channel_divergence":    round(channel_divergence, 3),
        "mean_brightness":       round(mean_brightness, 2),
    }


@st.cache_resource(show_spinner=False)
def build_model():
    rng = np.random.default_rng(42)
    n = 1000
    clean = np.column_stack([
        rng.normal(80,20,n), rng.normal(30,8,n), rng.uniform(0.001,0.02,n),
        rng.normal(12,4,n), rng.normal(1.2,0.2,n), rng.uniform(0,0.005,n),
        rng.integers(0,3,n), rng.normal(4,2,n), rng.normal(3,1.5,n), rng.normal(150,30,n),
    ])
    defect = np.column_stack([
        rng.normal(600,200,n), rng.normal(180,40,n), rng.uniform(0.04,0.18,n),
        rng.normal(40,10,n), rng.normal(3.5,0.8,n), rng.uniform(0.01,0.08,n),
        rng.integers(3,20,n), rng.normal(18,6,n), rng.normal(12,4,n), rng.normal(120,40,n),
    ])
    X = np.vstack([clean, defect])
    y = np.array([0]*n + [1]*n)
    scaler = StandardScaler()
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.08, subsample=0.8, random_state=42)
    clf.fit(scaler.fit_transform(X), y)
    return clf, scaler


def generate_heatmap(img_array):
    _, gray_f32 = to_gray_f32(img_array)
    h, w = gray_f32.shape
    stride = max(4, min(h, w) // 32)
    win = stride * 4
    heat = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - win, stride):
        for x in range(0, w - win, stride):
            patch = gray_f32[y:y+win, x:x+win]
            lap = cv2.Laplacian(patch, cv2.CV_32F)
            score = float(lap.var())
            heat[y:y+win, x:x+win] = np.maximum(heat[y:y+win, x:x+win], score)
    heat = cv2.GaussianBlur(heat, (21, 21), 0)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


def overlay_heatmap(img_array, heat, alpha=0.45):
    colored = (plt.get_cmap("plasma")(heat)[:,:,:3] * 255).astype(np.uint8)
    return (img_array*(1-alpha) + colored*alpha).clip(0,255).astype(np.uint8)


def make_demo_image():
    rng = np.random.default_rng(7)
    base = rng.integers(190, 220, (400,400,3), dtype=np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(4,8))):
        x0,y0 = int(rng.integers(20,350)), int(rng.integers(20,350))
        x1 = x0 + int(rng.integers(40,160)) * int(rng.choice([-1,1]))
        y1 = y0 + int(rng.integers(5,30))  * int(rng.choice([-1,1]))
        draw.line([(x0,y0),(x1,y1)], fill=(80,80,85), width=int(rng.integers(1,3)))
    for _ in range(int(rng.integers(2,5))):
        cx,cy = int(rng.integers(30,370)), int(rng.integers(30,370))
        r = int(rng.integers(4,16))
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(60,60,64))
    return img.filter(ImageFilter.GaussianBlur(0.6))


# ── Sidebar ────────────────────────────────────────────────────────────────────
CTQS = ["Scratch / Linear Mark Density","Surface Texture Variance","Edge Artifact Density",
        "Dark Blemish Coverage","Luminance Uniformity","Chroma / Stain Indicator"]

with st.sidebar:
    st.markdown("## 🔬 Surface Defect Inspector")
    st.markdown('<div class="eyebrow">Configuration</div>', unsafe_allow_html=True)
    aql_level = st.selectbox("AQL Level",["AQL 0.65 — Critical","AQL 1.0 — Major","AQL 2.5 — Minor"], index=1)
    threshold = {"AQL 0.65 — Critical":0.25,"AQL 1.0 — Major":0.42,"AQL 2.5 — Minor":0.60}[aql_level]
    st.markdown("---")
    grade = st.radio("Surface Grade",["Grade A (Display/External)","Grade B (Internal)","Grade C (Non-visible)"])
    st.markdown("---")
    st.markdown('<div class="eyebrow">CTQs Monitored</div>', unsafe_allow_html=True)
    for c in CTQS:
        st.markdown(f'<span class="ctq-tag">{c}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("GradientBoosting · 10 features · Portfolio demonstration")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# Surface Defect Inspector")
st.markdown('<p style="color:#636366;font-size:15px;margin-top:-12px;margin-bottom:24px;">ML-powered cosmetic quality inspection · Consumer hardware surfaces</p>', unsafe_allow_html=True)

c1, c2 = st.columns([3,1])
with c1:
    uploaded = st.file_uploader("Upload surface image", type=["png","jpg","jpeg","webp"], label_visibility="collapsed")
with c2:
    demo_clicked = st.button("⚡ Run demo image", use_container_width=True)

if demo_clicked:
    buf = io.BytesIO()
    make_demo_image().save(buf, format="PNG")
    buf.seek(0)
    uploaded = buf
    st.info("Running synthetic defective surface image.")

if uploaded:
    img_array = np.array(Image.open(uploaded).convert("RGB").resize((400,400), Image.LANCZOS))
    clf, scaler = build_model()

    with st.spinner("Extracting surface features…"):
        time.sleep(0.3)
        feats = extract_features(img_array)

    with st.spinner("Running defect classifier…"):
        time.sleep(0.2)
        vec = np.array([[feats["laplacian_variance"], feats["laplacian_max"],
                         feats["edge_density"], feats["surface_std"],
                         feats["hf_energy_ratio"], feats["dark_spot_ratio"],
                         feats["scratch_line_count"], feats["brightness_uniformity"],
                         feats["channel_divergence"], feats["mean_brightness"]]])
        prob = float(clf.predict_proba(scaler.transform(vec))[0][1])
        verdict = "FAIL" if prob >= threshold else "PASS"

    with st.spinner("Generating heatmap…"):
        time.sleep(0.2)
        overlay = overlay_heatmap(img_array, generate_heatmap(img_array))

    st.markdown("---")
    ci, ch, cv = st.columns([2,2,1.6])

    with ci:
        st.markdown('<div class="eyebrow">Input Surface</div>', unsafe_allow_html=True)
        st.image(img_array, use_container_width=True)

    with ch:
        st.markdown('<div class="eyebrow">Defect Heatmap</div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True, caption="High-activation = potential defects")

    with cv:
        st.markdown('<div class="eyebrow">Verdict</div>', unsafe_allow_html=True)
        vc = "verdict-pass" if verdict=="PASS" else "verdict-fail"
        pc = "#30d158" if verdict=="PASS" else "#ff453a"
        tc = "#30d158" if prob < threshold else "#ff453a"
        st.markdown(f"""
        <div class="{vc}">
            <div style="font-size:13px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;opacity:.6;margin-bottom:8px;">Verdict</div>
            <div style="font-size:48px;font-weight:700;color:{pc};line-height:1;">{verdict}</div>
            <div style="margin-top:16px;font-size:13px;color:#8e8e93;">Defect confidence</div>
            <div style="font-size:28px;font-weight:700;color:{tc};">{prob*100:.1f}%</div>
            <div style="font-size:12px;color:#636366;margin-top:6px;">Threshold {threshold*100:.0f}% · {aql_level.split('—')[0].strip()}</div>
        </div>
        <br><div class="card"><span style="font-size:13px;color:#8e8e93;">{grade}</span></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    FLABELS = {
        "laplacian_variance":"Surface texture variance","laplacian_max":"Peak edge intensity",
        "edge_density":"Edge artifact density","surface_std":"Pixel std deviation",
        "hf_energy_ratio":"High-freq energy ratio","dark_spot_ratio":"Dark blemish coverage",
        "scratch_line_count":"Detected linear marks","brightness_uniformity":"Luminance uniformity",
        "channel_divergence":"Chroma divergence","mean_brightness":"Mean brightness",
    }

    cf, cq = st.columns(2)

    with cf:
        st.markdown('<div class="eyebrow">Feature Vector</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k,v in feats.items():
            st.markdown(f'<div class="feat-row"><span style="color:#8e8e93;font-family:monospace;">{FLABELS[k]}</span><span style="color:#f5f5f7;font-weight:500;">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    CTQ_ITEMS = [
        ("Scratch / Linear Marks",   feats["scratch_line_count"],         3,  15,  "lines"),
        ("Surface Texture Variance", feats["laplacian_variance"],         150,500, "Lap var"),
        ("Edge Artifact Density",    feats["edge_density"]*1000,          20, 80,  "×10⁻³"),
        ("Dark Blemish Coverage",    feats["dark_spot_ratio"]*1000,       5,  30,  "×10⁻³"),
        ("Luminance Uniformity",     feats["brightness_uniformity"],      8,  20,  "std"),
        ("Chroma Divergence",        feats["channel_divergence"],         6,  15,  "RGB"),
    ]

    with cq:
        st.markdown('<div class="eyebrow">CTQ Risk Assessment</div>', unsafe_allow_html=True)
        for label,val,wt,ft,unit in CTQ_ITEMS:
            color = "#30d158" if val<wt else ("#ffd60a" if val<ft else "#ff453a")
            status = "✓ Pass" if val<wt else ("⚠ Watch" if val<ft else "✗ Fail")
            bar_w = int(min(val/(ft*1.2),1.0)*100)
            st.markdown(f"""
            <div class="card" style="padding:14px 18px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:13px;color:#c7c7cc;">{label}</span>
                    <span style="font-size:13px;font-weight:600;color:{color};">{status}</span>
                </div>
                <div style="font-size:11px;color:#48484a;margin-top:2px;">{val:.2f} {unit}</div>
                <div style="background:#111114;border-radius:3px;height:5px;margin-top:8px;">
                    <div style="width:{bar_w}%;height:5px;background:{color};border-radius:3px;opacity:.7;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="eyebrow">FMEA-Aligned Summary</div>', unsafe_allow_html=True)

    ACTIONS = {
        "Scratch / Linear Marks":  "Inspect handling fixtures; initiate supplier cosmetic audit",
        "Surface Texture Variance":"Review polishing process; check abrasive media lot",
        "Edge Artifact Density":   "Evaluate masking process; check tooling wear",
        "Dark Blemish Coverage":   "Review cleaning protocol; check contamination control",
        "Luminance Uniformity":    "Check coating uniformity; review lamp calibration",
        "Chroma Divergence":       "Inspect anodization bath; check material lot",
    }
    flagged = [(l,"HIGH" if v>=ft else "MEDIUM",f"{v:.2f} {u}") for l,v,wt,ft,u in CTQ_ITEMS if v>=wt]

    if not flagged:
        st.markdown('<div class="verdict-pass" style="text-align:left;padding:20px 24px;"><span style="color:#30d158;font-weight:600;">No CTQ violations detected.</span> Surface meets cosmetic acceptance criteria.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns([3,1.2,2,3])
        for col,h in zip(cols,["CTQ","Severity","Value","Recommended Action"]):
            col.markdown(f'<span style="font-size:11px;color:#48484a;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">{h}</span>', unsafe_allow_html=True)
        for ctq,sev,val in flagged:
            sc = "#ff453a" if sev=="HIGH" else "#ffd60a"
            cs = st.columns([3,1.2,2,3])
            cs[0].markdown(f'<span style="font-size:13px;color:#e8e8ed;">{ctq}</span>', unsafe_allow_html=True)
            cs[1].markdown(f'<span style="font-size:12px;font-weight:700;color:{sc};">{sev}</span>', unsafe_allow_html=True)
            cs[2].markdown(f'<span style="font-size:13px;color:#8e8e93;font-family:monospace;">{val}</span>', unsafe_allow_html=True)
            cs[3].markdown(f'<span style="font-size:12px;color:#636366;">{ACTIONS.get(ctq,"Investigate and escalate")}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"AQL: {aql_level} · Grade: {grade} · Defect prob: {prob*100:.2f}% · Threshold: {threshold*100:.0f}%")

else:
    st.markdown("""
    <div style="background:#111114;border:1px dashed #2a2a32;border-radius:16px;padding:60px 40px;text-align:center;">
        <div style="font-size:40px;margin-bottom:16px;">🔬</div>
        <div style="font-size:18px;font-weight:500;color:#636366;margin-bottom:8px;">Upload a surface image to begin inspection</div>
        <div style="font-size:13px;color:#48484a;">PNG · JPG · WEBP &nbsp;|&nbsp; Or click "Run demo image"</div>
    </div>
    """, unsafe_allow_html=True)
