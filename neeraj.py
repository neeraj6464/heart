import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import joblib
import pickle
import time
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0d14 0%, #14101f 50%, #0d1420 100%);
    color: #e8e4f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13102a 0%, #0f1824 100%);
    border-right: 1px solid rgba(220,80,100,0.25);
}
section[data-testid="stSidebar"] * { color: #d4ceea !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #1e0a1a 0%, #2a0d22 40%, #0e1a2e 100%);
    border: 1px solid rgba(220,80,100,0.35);
    border-radius: 18px;
    padding: 2.2rem 2.8rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🫀";
    position: absolute;
    right: 2rem; top: 50%; transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.08;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ff6b7a;
    margin: 0 0 .35rem;
    line-height: 1.15;
}
.hero p { color: #b0a8cc; font-size: .95rem; margin: 0; }

/* ── Section labels ── */
.section-label {
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #ff6b7a;
    margin-bottom: .6rem;
}

/* ── Risk gauge wrapper ── */
.gauge-wrap {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(220,80,100,0.2);
    border-radius: 16px;
    padding: 1.4rem;
    text-align: center;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: rgba(255,107,122,0.5); }
.metric-card .label { font-size: .72rem; color: #9b92b8; letter-spacing:.06em; text-transform:uppercase; }
.metric-card .value { font-size: 1.7rem; font-weight: 700; color: #ff9aa5; margin-top:.2rem; }

/* ── Advice cards ── */
.advice-card {
    background: rgba(255,255,255,0.035);
    border-left: 4px solid #ff6b7a;
    border-radius: 0 12px 12px 0;
    padding: .85rem 1.1rem;
    margin-bottom: .6rem;
    font-size: .88rem;
    color: #ccc4e0;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c0254a, #e8435a) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .75rem 2rem !important;
    letter-spacing: .04em !important;
    transition: opacity .2s !important;
}
div[data-testid="stButton"] > button:hover { opacity: .88 !important; }

/* ── Divider tweak ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Slider accent ── */
[data-testid="stSlider"] [class*="thumb"] { background: #ff6b7a !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,#c0254a,#ff9aa5) !important; }

/* ── Expander ── */
[data-testid="stExpander"] { border-color: rgba(255,255,255,0.1) !important; border-radius: 12px !important; }

/* ── History table ── */
.history-row {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: .5rem .8rem;
    margin-bottom: .35rem;
    font-size: .82rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.badge-high { background: rgba(220,50,70,.25); color:#ff8a96; padding:2px 10px; border-radius:20px; font-size:.75rem; }
.badge-low  { background: rgba(50,200,100,.2);  color:#6ee89e; padding:2px 10px; border-radius:20px; font-size:.75rem; }

/* ── Info box ── */
.info-box {
    background: rgba(80,130,220,0.1);
    border: 1px solid rgba(80,130,220,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: .85rem;
    color: #aac3ee;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# ─────────────────────────────────────────────
# MODEL LOADER  (cached + graceful fallback)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open("logist_heart.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("columns.pkl", "rb") as f:
            cols = pickle.load(f)

        return model, scaler, cols, True

    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, False

model, scaler, expected_columns, model_loaded = load_models()
# ─────────────────────────────────────────────
# SIDEBAR — PATIENT PROFILE + HISTORY
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Patient Profile")
    patient_name = st.text_input("Patient Name", placeholder="e.g. Rajesh Sharma")
    patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-2024-001")
    st.markdown("---")

    st.markdown("### 📋 Session History")
    if st.session_state.history:
        for i, rec in enumerate(reversed(st.session_state.history[-6:])):
            badge = "badge-high" if rec["risk"] == "High" else "badge-low"
            st.markdown(
                f'<div class="history-row">'
                f'<span>#{len(st.session_state.history)-i} &nbsp; {rec["time"]}</span>'
                f'<span class="{badge}">{rec["risk"]} {rec["prob"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

    st.markdown("---")
    st.markdown(
        '<div class="info-box">⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Always consult a qualified physician for medical decisions.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    f"""<div class="hero">
        <h1>CardioAI</h1>
        <p>Advanced heart disease risk assessment powered by machine learning &nbsp;·&nbsp; 
           {datetime.now().strftime("%d %B %Y")}</p>
    </div>""",
    unsafe_allow_html=True,
)

if not model_loaded:
    st.warning("⚠️ Model files not found (`logist_heart.pkl`, `scaler.pkl`, `columns.pkl`). Running in **demo mode** with mock predictions.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_predict, tab_info, tab_compare = st.tabs(["🔍 Predict", "📖 Feature Guide", "📊 Risk Factors"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:

    st.markdown('<p class="section-label">Clinical Parameters</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age        = st.slider("Age (years)", 18, 100, 52)
        sex        = st.selectbox("Biological Sex", ["M", "F"], format_func=lambda x: "Male" if x=="M" else "Female")
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 130, step=1)

    with col2:
        st.markdown("**Cardiac Markers**")
        cholesterol    = st.number_input("Cholesterol (mg/dL)", 100, 600, 245, step=5)
        max_hr         = st.slider("Max Heart Rate Achieved", 60, 220, 162)
        oldpeak        = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)

    with col3:
        st.markdown("**ECG / Stress Test**")
        chest_pain      = st.selectbox("Chest Pain Type",    ["ATA", "NAP", "TA", "ASY"],
                                       help="ATA=Atypical, NAP=Non-anginal, TA=Typical, ASY=Asymptomatic")
        resting_ecg     = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise-induced Angina", ["N", "Y"])
        st_slope        = st.selectbox("ST Slope",           ["Up", "Flat", "Down"])
        fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                       format_func=lambda x: "Yes" if x else "No")

    st.divider()

    # ── Predict button ──
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button("🫀 Analyse Heart Risk", use_container_width=True)

    if predict_btn:
        with st.spinner("Analysing cardiac parameters…"):
            time.sleep(0.8)  # brief UX pause

        # ── Build feature vector ──
        input_data = {
            "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "MaxHR": max_hr, "Oldpeak": oldpeak,
            f"Sex_{sex}": 1,
            f"ChestPainType_{chest_pain}": 1,
            f"RestingECG_{resting_ecg}": 1,
            f"ExerciseAngina_{exercise_angina}": 1,
            f"ST_Slope_{st_slope}": 1,
        }
        df_input = pd.DataFrame([input_data])

        if model_loaded:
            for c in expected_columns:
                if c not in df_input.columns:
                    df_input[c] = 0
            df_input  = df_input[expected_columns]
            scaled    = scaler.transform(df_input)
            prediction  = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][1]
        else:
            # Demo mode
            probability = float(np.clip(
                0.05*age/100 + 0.2*(sex=="M") + 0.15*(chest_pain=="ASY") +
                0.1*(oldpeak/6) + 0.1*(fasting_bs) + np.random.uniform(0,.15), 0, 1))
            prediction = 1 if probability >= 0.5 else 0

        # ── Save to history ──
        st.session_state.history.append({
            "time":  datetime.now().strftime("%H:%M"),
            "risk":  "High" if prediction == 1 else "Low",
            "prob":  f"{probability*100:.1f}%",
            "age":   age,
            "name":  patient_name or "—",
        })

        # ── Result banner ──
        st.divider()
        if prediction == 1:
            st.markdown(
                f"""<div style="background:rgba(200,40,60,.15);border:1px solid rgba(200,40,60,.5);
                border-radius:16px;padding:1.3rem 1.6rem;display:flex;align-items:center;gap:1.2rem;">
                <span style="font-size:2.2rem">⚠️</span>
                <div><strong style="font-size:1.2rem;color:#ff6b7a;">High Risk Detected</strong><br>
                <span style="color:#c9bedd;font-size:.9rem;">This patient profile shows elevated cardiovascular risk indicators.</span></div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style="background:rgba(40,180,90,.12);border:1px solid rgba(40,180,90,.4);
                border-radius:16px;padding:1.3rem 1.6rem;display:flex;align-items:center;gap:1.2rem;">
                <span style="font-size:2.2rem">✅</span>
                <div><strong style="font-size:1.2rem;color:#5ee88a;">Low Risk Profile</strong><br>
                <span style="color:#c9bedd;font-size:.9rem;">Current indicators suggest a lower cardiovascular risk.</span></div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Key Metrics Row ──
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Patient Age", f"{age} yrs"),
            ("Max Heart Rate", f"{max_hr} bpm"),
            ("Cholesterol", f"{cholesterol} mg/dL"),
            ("Risk Score", f"{probability*100:.1f}%"),
        ]
        for col, (lbl, val) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge + Bar Chart ──
        g_col, b_col = st.columns([1, 1])

        with g_col:
            st.markdown('<p class="section-label">Risk Probability Gauge</p>', unsafe_allow_html=True)

            fig_gauge, ax = plt.subplots(figsize=(5, 3), facecolor="none")
            ax.set_facecolor("none")
            fig_gauge.patch.set_alpha(0)

            theta = np.linspace(np.pi, 0, 200)
            for i, (start, end, color) in enumerate([
                (np.pi, np.pi*0.67, "#2dbd6e"),
                (np.pi*0.67, np.pi*0.33, "#f0a500"),
                (np.pi*0.33, 0, "#e03050"),
            ]):
                t = np.linspace(start, end, 60)
                ax.plot(np.cos(t), np.sin(t), color=color, linewidth=18, solid_capstyle="butt", alpha=.85)

            needle_angle = np.pi * (1 - probability)
            ax.annotate("", xy=(0.62*np.cos(needle_angle), 0.62*np.sin(needle_angle)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#f0e8ff", lw=2, mutation_scale=18))
            ax.plot(0, 0, "o", color="#f0e8ff", markersize=9, zorder=5)
            ax.text(0, -0.28, f"{probability*100:.1f}%",
                    ha="center", va="center", fontsize=20, fontweight="bold", color="#ff9aa5")
            ax.text(0, -0.48, "Risk Score",
                    ha="center", va="center", fontsize=9, color="#9b92b8")
            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 1.15)
            ax.axis("off")
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

        with b_col:
            st.markdown('<p class="section-label">Risk Distribution</p>', unsafe_allow_html=True)

            fig_bar, ax2 = plt.subplots(figsize=(5, 3), facecolor="none")
            ax2.set_facecolor("none")
            fig_bar.patch.set_alpha(0)
            bars = ax2.bar(
                ["Low Risk", "High Risk"],
                [1 - probability, probability],
                color=["#2dbd6e", "#e03050"],
                edgecolor="none", width=0.5
            )
            for bar, val in zip(bars, [1-probability, probability]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val*100:.1f}%", ha="center", fontsize=10,
                         color="#e8e4f0", fontweight="600")
            ax2.set_ylim(0, 1.18)
            ax2.set_ylabel("Probability", color="#9b92b8", fontsize=9)
            ax2.tick_params(colors="#9b92b8")
            ax2.spines[:].set_visible(False)
            ax2.set_facecolor("none")
            for spine in ax2.spines.values():
                spine.set_edgecolor("none")
            ax2.yaxis.set_tick_params(labelcolor="#9b92b8")
            ax2.xaxis.set_tick_params(labelcolor="#c0b8d8")
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        # ── Risk Progress Bar ──
        st.markdown('<p class="section-label">Overall Risk Level</p>', unsafe_allow_html=True)
        st.progress(probability)

        # ── Feature Contribution (manual importance proxy) ──
        st.markdown('<p class="section-label">Top Contributing Factors</p>', unsafe_allow_html=True)
        factors = {
            "Age":           min(age / 100, 1),
            "Chest Pain":    0.85 if chest_pain == "ASY" else 0.3,
            "Oldpeak":       oldpeak / 6,
            "Max HR":        1 - (max_hr / 220),
            "Cholesterol":   min(cholesterol / 600, 1),
            "Fasting BS":    0.6 if fasting_bs else 0.1,
            "ST Slope":      0.8 if st_slope == "Down" else (0.5 if st_slope == "Flat" else 0.2),
        }
        fig_feat, ax3 = plt.subplots(figsize=(7, 2.8), facecolor="none")
        ax3.set_facecolor("none")
        fig_feat.patch.set_alpha(0)
        names  = list(factors.keys())
        values = list(factors.values())
        colors = ["#e03050" if v > 0.6 else "#f0a500" if v > 0.35 else "#2dbd6e" for v in values]
        bars3  = ax3.barh(names, values, color=colors, edgecolor="none", height=0.55)
        ax3.set_xlim(0, 1.15)
        ax3.set_xlabel("Relative Influence", color="#9b92b8", fontsize=8)
        ax3.tick_params(colors="#c0b8d8", labelsize=8)
        for spine in ax3.spines.values(): spine.set_visible(False)
        for bar, val in zip(bars3, values):
            ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:.0%}", va="center", fontsize=8, color="#e8e4f0")
        patches = [
            mpatches.Patch(color="#2dbd6e", label="Low"),
            mpatches.Patch(color="#f0a500", label="Moderate"),
            mpatches.Patch(color="#e03050", label="High"),
        ]
        ax3.legend(handles=patches, loc="lower right", framealpha=0, fontsize=7,
                   labelcolor="#c0b8d8")
        st.pyplot(fig_feat, use_container_width=True)
        plt.close(fig_feat)

        # ── Personalised Advice ──
        st.markdown('<p class="section-label">Personalised Recommendations</p>', unsafe_allow_html=True)
        advice = []
        if age > 55:
            advice.append("🩺 Annual cardiac screening recommended for patients over 55.")
        if cholesterol > 240:
            advice.append("🥗 Cholesterol is elevated. Consider dietary changes and consult a lipid specialist.")
        if resting_bp > 140:
            advice.append("💊 Resting BP is high. Monitor regularly and discuss antihypertensive options with your doctor.")
        if fasting_bs == 1:
            advice.append("🍬 Elevated fasting blood sugar. Diabetes screening and glucose management are advised.")
        if exercise_angina == "Y":
            advice.append("🏃 Exercise-induced angina noted. Avoid strenuous activity without medical clearance.")
        if st_slope == "Down" or st_slope == "Flat":
            advice.append("📈 Abnormal ST slope detected. Further stress-ECG evaluation may be warranted.")
        if max_hr < 100:
            advice.append("❤️ Reduced max heart rate. Cardiopulmonary fitness assessment recommended.")
        if not advice:
            advice.append("✅ No major individual risk flags detected. Maintain a balanced diet and regular exercise.")

        for tip in advice:
            st.markdown(f'<div class="advice-card">{tip}</div>', unsafe_allow_html=True)

        # ── Print / export summary ──
        with st.expander("📄 Full Report Summary"):
            report_name = patient_name or "Patient"
            st.markdown(f"""
**Patient:** {report_name} &nbsp;&nbsp; **ID:** {patient_id or '—'} &nbsp;&nbsp; **Date:** {datetime.now().strftime("%d %B %Y %H:%M")}

| Parameter | Value |
|-----------|-------|
| Age | {age} years |
| Sex | {"Male" if sex=="M" else "Female"} |
| Resting BP | {resting_bp} mm Hg |
| Cholesterol | {cholesterol} mg/dL |
| Fasting BS >120 | {"Yes" if fasting_bs else "No"} |
| Max Heart Rate | {max_hr} bpm |
| Chest Pain Type | {chest_pain} |
| Resting ECG | {resting_ecg} |
| Exercise Angina | {"Yes" if exercise_angina=="Y" else "No"} |
| ST Slope | {st_slope} |
| Oldpeak | {oldpeak} |
| **Risk Score** | **{probability*100:.2f}%** |
| **Prediction** | **{"⚠️ High Risk" if prediction==1 else "✅ Low Risk"}** |
""")
            st.download_button(
                "⬇️ Download Report (CSV)",
                data=pd.DataFrame([{
                    "Name": report_name, "ID": patient_id or "", "Date": datetime.now().isoformat(),
                    "Age": age, "Sex": sex, "RestingBP": resting_bp, "Cholesterol": cholesterol,
                    "FastingBS": fasting_bs, "MaxHR": max_hr, "ChestPain": chest_pain,
                    "RestingECG": resting_ecg, "ExerciseAngina": exercise_angina,
                    "STSlope": st_slope, "Oldpeak": oldpeak,
                    "RiskScore_%": round(probability*100, 2),
                    "Prediction": "High Risk" if prediction==1 else "Low Risk",
                }]).to_csv(index=False),
                file_name=f"cardioai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════
# TAB 2 — FEATURE GUIDE
# ══════════════════════════════════════════════
with tab_info:
    st.markdown('<p class="section-label">Understanding Each Parameter</p>', unsafe_allow_html=True)
    guide = {
        "🩺 Age": "Cardiovascular risk increases with age. Men over 45 and women over 55 are at higher baseline risk.",
        "⚧ Sex": "Biological sex influences risk distribution. Men generally show earlier onset of coronary artery disease.",
        "💢 Chest Pain Type":
            "**ATA** (Atypical Angina): Unusual chest discomfort. "
            "**NAP** (Non-Anginal Pain): Chest pain not related to the heart. "
            "**TA** (Typical Angina): Classic pressure/tightening. "
            "**ASY** (Asymptomatic): No chest pain despite possible disease.",
        "🩸 Resting BP": "Normal resting BP is below 120/80 mm Hg. Persistent values above 140 indicate hypertension.",
        "🧪 Cholesterol": "LDL cholesterol above 200 mg/dL raises plaque buildup risk. Above 240 is considered high.",
        "🍬 Fasting Blood Sugar": "Blood glucose >120 mg/dL after fasting suggests pre-diabetes or diabetes, a major cardiac risk factor.",
        "📊 Resting ECG":
            "**Normal**: No abnormalities. "
            "**ST**: ST-T wave changes suggesting ischemia. "
            "**LVH**: Left Ventricular Hypertrophy — enlarged heart muscle.",
        "❤️ Max Heart Rate": "Lower-than-expected maximum heart rate during exertion may indicate reduced cardiac reserve.",
        "🏃 Exercise Angina": "Chest pain triggered by physical activity is a strong indicator of obstructive coronary artery disease.",
        "📉 Oldpeak (ST Depression)": "Measures ST segment depression during exercise. Higher values indicate greater ischemic burden.",
        "📈 ST Slope":
            "**Up**: Normal response during exercise. "
            "**Flat**: Borderline, warrants monitoring. "
            "**Down**: Abnormal — associated with higher risk of coronary disease.",
    }
    for title, desc in guide.items():
        with st.expander(title):
            st.markdown(desc)

# ══════════════════════════════════════════════
# TAB 3 — RISK FACTOR OVERVIEW
# ══════════════════════════════════════════════
with tab_compare:
    st.markdown('<p class="section-label">Population-Level Risk Factors</p>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)

    with rc1:
        # Age vs risk
        fig_age, ax_a = plt.subplots(figsize=(5, 3.2), facecolor="none")
        ax_a.set_facecolor("none"); fig_age.patch.set_alpha(0)
        ages  = np.arange(20, 85, 5)
        risk  = 1 / (1 + np.exp(-0.1*(ages - 55)))
        ax_a.fill_between(ages, risk, alpha=0.25, color="#e03050")
        ax_a.plot(ages, risk, color="#ff6b7a", linewidth=2.5)
        ax_a.set_xlabel("Age", color="#9b92b8", fontsize=9)
        ax_a.set_ylabel("Relative Risk", color="#9b92b8", fontsize=9)
        ax_a.set_title("Age vs Cardiovascular Risk", color="#d4ceea", fontsize=10)
        ax_a.tick_params(colors="#9b92b8", labelsize=8)
        for s in ax_a.spines.values(): s.set_edgecolor("#333")
        st.markdown("**Age Risk Curve**")
        st.pyplot(fig_age, use_container_width=True)
        plt.close(fig_age)

    with rc2:
        # Cholesterol distribution
        fig_ch, ax_c = plt.subplots(figsize=(5, 3.2), facecolor="none")
        ax_c.set_facecolor("none"); fig_ch.patch.set_alpha(0)
        low_chol  = np.random.normal(180, 25, 400)
        high_chol = np.random.normal(260, 35, 400)
        ax_c.hist(low_chol,  bins=30, alpha=0.6, color="#2dbd6e", label="Low Risk Group",  edgecolor="none")
        ax_c.hist(high_chol, bins=30, alpha=0.6, color="#e03050", label="High Risk Group", edgecolor="none")
        ax_c.set_xlabel("Cholesterol (mg/dL)", color="#9b92b8", fontsize=9)
        ax_c.set_ylabel("Frequency", color="#9b92b8", fontsize=9)
        ax_c.set_title("Cholesterol Distribution by Risk", color="#d4ceea", fontsize=10)
        ax_c.tick_params(colors="#9b92b8", labelsize=8)
        ax_c.legend(fontsize=8, framealpha=0, labelcolor="#c0b8d8")
        for s in ax_c.spines.values(): s.set_edgecolor("#333")
        st.markdown("**Cholesterol Distribution**")
        st.pyplot(fig_ch, use_container_width=True)
        plt.close(fig_ch)

    st.markdown("---")
    st.markdown(
        '<div class="info-box">📚 <strong>Reference ranges:</strong> '
        'BP &lt; 120 mm Hg (normal), Cholesterol &lt; 200 mg/dL (desirable), '
        'Resting HR 60–100 bpm (normal), Fasting glucose &lt; 100 mg/dL (normal). '
        'Sources: ACC/AHA 2023 Guidelines, WHO Cardiovascular Risk Charts.</div>',
        unsafe_allow_html=True,
    )
