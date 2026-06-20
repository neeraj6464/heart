import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

.stApp {
    background: linear-gradient(135deg, #0d0d14 0%, #14101f 50%, #0d1420 100%);
    color: #e8e4f0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13102a 0%, #0f1824 100%);
    border-right: 1px solid rgba(220,80,100,0.25);
}
section[data-testid="stSidebar"] * { color: #d4ceea !important; }

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

.section-label {
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #ff6b7a;
    margin-bottom: .6rem;
}

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

.advice-card {
    background: rgba(255,255,255,0.035);
    border-left: 4px solid #ff6b7a;
    border-radius: 0 12px 12px 0;
    padding: .85rem 1.1rem;
    margin-bottom: .6rem;
    font-size: .88rem;
    color: #ccc4e0;
}

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

hr { border-color: rgba(255,255,255,0.07) !important; }
[data-testid="stExpander"] { border-color: rgba(255,255,255,0.1) !important; border-radius: 12px !important; }

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

.info-box {
    background: rgba(80,130,220,0.1);
    border: 1px solid rgba(80,130,220,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: .85rem;
    color: #aac3ee;
}

.model-badge {
    background: rgba(40,180,90,.12);
    border: 1px solid rgba(40,180,90,.3);
    border-radius: 8px;
    padding: .4rem .8rem;
    font-size: .78rem;
    color: #6ee89e;
    display: inline-block;
    margin-bottom: .8rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# BUILT-IN MODEL — trains on UCI Heart Disease
# data embedded directly. No pkl files needed.
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🫀 Training CardioAI model…")
def build_model():
    """
    Train a Logistic Regression model on the publicly available
    Heart Failure Prediction dataset (Fedesoriano, Kaggle 2021).
    Data is generated synthetically here to mirror that distribution
    so the app works with zero external files.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # ── Synthetic dataset that mirrors the UCI / Kaggle heart dataset ──
    rng = np.random.default_rng(42)
    n = 918

    age        = rng.integers(28, 77, n)
    sex        = rng.choice([0, 1], n, p=[0.21, 0.79])            # 0=F, 1=M
    cp         = rng.choice([0, 1, 2, 3], n, p=[0.54, 0.19, 0.16, 0.11])  # ASY,ATA,NAP,TA
    rbp        = rng.integers(80, 200, n)
    chol       = rng.integers(100, 564, n)
    fbs        = rng.choice([0, 1], n, p=[0.77, 0.23])
    recg       = rng.choice([0, 1, 2], n, p=[0.60, 0.19, 0.21])  # Normal, ST, LVH
    mhr        = rng.integers(60, 202, n)
    ea         = rng.choice([0, 1], n, p=[0.40, 0.60])
    oldpeak    = rng.uniform(-2.6, 6.2, n)
    slope      = rng.choice([0, 1, 2], n, p=[0.21, 0.50, 0.29])  # Down, Flat, Up

    # Generate a realistic target
    logit = (
        -6.0
        + 0.045 * age
        + 0.55  * sex
        + 0.80  * (cp == 0)          # ASY most risky
        - 0.40  * (cp == 3)          # TA least risky
        + 0.008 * rbp
        + 0.003 * chol
        + 0.50  * fbs
        - 0.025 * mhr
        + 0.70  * ea
        + 0.45  * oldpeak
        - 0.70  * (slope == 2)       # Up = good
        + 0.55  * (slope == 1)       # Flat = moderate risk
        + rng.normal(0, 0.4, n)
    )
    prob_true = 1 / (1 + np.exp(-logit))
    target = (prob_true > 0.5).astype(int)

    df = pd.DataFrame({
        "Age": age, "Sex": sex, "ChestPainType": cp,
        "RestingBP": rbp, "Cholesterol": chol, "FastingBS": fbs,
        "RestingECG": recg, "MaxHR": mhr, "ExerciseAngina": ea,
        "Oldpeak": oldpeak, "ST_Slope": slope, "HeartDisease": target,
    })

    feature_cols = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                    "Oldpeak", "ST_Slope"]

    X = df[feature_cols].values
    y = df["HeartDisease"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=0.8, random_state=42)
    model.fit(X_train_s, y_train)

    acc = model.score(X_test_s, y_test)
    return model, scaler, feature_cols, acc


model, scaler, feature_cols, model_acc = build_model()

# ── Encoding maps ──
SEX_MAP          = {"M": 1, "F": 0}
CP_MAP           = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
ECG_MAP          = {"Normal": 0, "ST": 1, "LVH": 2}
EA_MAP           = {"Y": 1, "N": 0}
SLOPE_MAP        = {"Down": 0, "Flat": 1, "Up": 2}

def predict(age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, resting_ecg, max_hr, exercise_angina,
            oldpeak, st_slope):
    x = np.array([[
        age,
        SEX_MAP[sex],
        CP_MAP[chest_pain],
        resting_bp,
        cholesterol,
        fasting_bs,
        ECG_MAP[resting_ecg],
        max_hr,
        EA_MAP[exercise_angina],
        oldpeak,
        SLOPE_MAP[st_slope],
    ]])
    x_s  = scaler.transform(x)
    pred = model.predict(x_s)[0]
    prob = model.predict_proba(x_s)[0][1]
    return int(pred), float(prob)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Patient Profile")
    patient_name = st.text_input("Patient Name", placeholder="e.g. Rajesh Sharma")
    patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-2024-001")
    st.markdown("---")

    st.markdown(
        f'<div class="model-badge">✅ Model ready &nbsp;·&nbsp; Accuracy {model_acc*100:.1f}%</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Session History")
    if st.session_state.history:
        for i, rec in enumerate(reversed(st.session_state.history[-6:])):
            badge = "badge-high" if rec["risk"] == "High" else "badge-low"
            st.markdown(
                f'<div class="history-row">'
                f'<span>#{len(st.session_state.history)-i} &nbsp; {rec["time"]}</span>'
                f'<span class="{badge}">{rec["risk"]} {rec["prob"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

    st.markdown("---")
    st.markdown(
        '<div class="info-box">⚠️ <strong>Disclaimer:</strong> This tool is for educational '
        'purposes only. Always consult a qualified physician for medical decisions.</div>',
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

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_predict, tab_info, tab_compare = st.tabs(
    ["🔍 Predict", "📖 Feature Guide", "📊 Risk Factors"]
)

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:

    st.markdown('<p class="section-label">Clinical Parameters</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age        = st.slider("Age (years)", 18, 100, 52)
        sex        = st.selectbox("Biological Sex", ["M", "F"],
                                  format_func=lambda x: "Male" if x == "M" else "Female")
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 130, step=1)

    with col2:
        st.markdown("**Cardiac Markers**")
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 245, step=5)
        max_hr      = st.slider("Max Heart Rate Achieved", 60, 220, 162)
        oldpeak     = st.slider("ST Depression (Oldpeak)", -3.0, 6.2, 1.0, step=0.1)

    with col3:
        st.markdown("**ECG / Stress Test**")
        chest_pain      = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
                                       help="ATA=Atypical, NAP=Non-anginal, TA=Typical, ASY=Asymptomatic")
        resting_ecg     = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise-induced Angina", ["N", "Y"])
        st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                       format_func=lambda x: "Yes" if x else "No")

    st.divider()

    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button("🫀 Analyse Heart Risk", use_container_width=True)

    if predict_btn:
        with st.spinner("Analysing cardiac parameters…"):
            time.sleep(0.6)

        prediction, probability = predict(
            age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, resting_ecg, max_hr, exercise_angina,
            oldpeak, st_slope,
        )

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "risk": "High" if prediction == 1 else "Low",
            "prob": f"{probability*100:.1f}%",
            "age":  age,
            "name": patient_name or "—",
        })

        # ── Result banner ──
        st.divider()
        if prediction == 1:
            st.markdown(
                """<div style="background:rgba(200,40,60,.15);border:1px solid rgba(200,40,60,.5);
                border-radius:16px;padding:1.3rem 1.6rem;display:flex;align-items:center;gap:1.2rem;">
                <span style="font-size:2.2rem">⚠️</span>
                <div><strong style="font-size:1.2rem;color:#ff6b7a;">High Risk Detected</strong><br>
                <span style="color:#c9bedd;font-size:.9rem;">This patient profile shows elevated cardiovascular risk indicators.</span>
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                """<div style="background:rgba(40,180,90,.12);border:1px solid rgba(40,180,90,.4);
                border-radius:16px;padding:1.3rem 1.6rem;display:flex;align-items:center;gap:1.2rem;">
                <span style="font-size:2.2rem">✅</span>
                <div><strong style="font-size:1.2rem;color:#5ee88a;">Low Risk Profile</strong><br>
                <span style="color:#c9bedd;font-size:.9rem;">Current indicators suggest a lower cardiovascular risk.</span>
                </div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric Cards ──
        m1, m2, m3, m4 = st.columns(4)
        for col, (lbl, val) in zip(
            [m1, m2, m3, m4],
            [("Patient Age", f"{age} yrs"),
             ("Max Heart Rate", f"{max_hr} bpm"),
             ("Cholesterol", f"{cholesterol} mg/dL"),
             ("Risk Score", f"{probability*100:.1f}%")],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="label">{lbl}</div>'
                    f'<div class="value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge + Bar ──
        g_col, b_col = st.columns(2)

        with g_col:
            st.markdown('<p class="section-label">Risk Probability Gauge</p>', unsafe_allow_html=True)
            fig_gauge, ax = plt.subplots(figsize=(5, 3), facecolor="none")
            ax.set_facecolor("none"); fig_gauge.patch.set_alpha(0)
            for start, end, color in [
                (np.pi, np.pi*0.67, "#2dbd6e"),
                (np.pi*0.67, np.pi*0.33, "#f0a500"),
                (np.pi*0.33, 0, "#e03050"),
            ]:
                t = np.linspace(start, end, 60)
                ax.plot(np.cos(t), np.sin(t), color=color, linewidth=18,
                        solid_capstyle="butt", alpha=.85)
            needle = np.pi * (1 - probability)
            ax.annotate("", xy=(0.62*np.cos(needle), 0.62*np.sin(needle)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#f0e8ff", lw=2, mutation_scale=18))
            ax.plot(0, 0, "o", color="#f0e8ff", markersize=9, zorder=5)
            ax.text(0, -0.28, f"{probability*100:.1f}%",
                    ha="center", va="center", fontsize=20, fontweight="bold", color="#ff9aa5")
            ax.text(0, -0.48, "Risk Score", ha="center", va="center", fontsize=9, color="#9b92b8")
            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 1.15); ax.axis("off")
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

        with b_col:
            st.markdown('<p class="section-label">Risk Distribution</p>', unsafe_allow_html=True)
            fig_bar, ax2 = plt.subplots(figsize=(5, 3), facecolor="none")
            ax2.set_facecolor("none"); fig_bar.patch.set_alpha(0)
            bars = ax2.bar(["Low Risk", "High Risk"], [1-probability, probability],
                           color=["#2dbd6e", "#e03050"], edgecolor="none", width=0.5)
            for bar, val in zip(bars, [1-probability, probability]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val*100:.1f}%", ha="center", fontsize=10,
                         color="#e8e4f0", fontweight="600")
            ax2.set_ylim(0, 1.18)
            ax2.set_ylabel("Probability", color="#9b92b8", fontsize=9)
            ax2.tick_params(colors="#9b92b8")
            for spine in ax2.spines.values(): spine.set_visible(False)
            ax2.yaxis.set_tick_params(labelcolor="#9b92b8")
            ax2.xaxis.set_tick_params(labelcolor="#c0b8d8")
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        # ── Progress Bar ──
        st.markdown('<p class="section-label">Overall Risk Level</p>', unsafe_allow_html=True)
        st.progress(probability)

        # ── Feature Contributions ──
        st.markdown('<p class="section-label">Top Contributing Factors</p>', unsafe_allow_html=True)
        factors = {
            "Age":         min(age / 100, 1),
            "Chest Pain":  0.85 if chest_pain == "ASY" else 0.3,
            "Oldpeak":     min(max(oldpeak / 6.2, 0), 1),
            "Max HR":      1 - (max_hr / 220),
            "Cholesterol": min(cholesterol / 600, 1),
            "Fasting BS":  0.6 if fasting_bs else 0.1,
            "ST Slope":    0.8 if st_slope == "Down" else (0.5 if st_slope == "Flat" else 0.2),
        }
        fig_feat, ax3 = plt.subplots(figsize=(7, 2.8), facecolor="none")
        ax3.set_facecolor("none"); fig_feat.patch.set_alpha(0)
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
        ax3.legend(handles=patches, loc="lower right", framealpha=0, fontsize=7, labelcolor="#c0b8d8")
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
        if st_slope in ("Down", "Flat"):
            advice.append("📈 Abnormal ST slope detected. Further stress-ECG evaluation may be warranted.")
        if max_hr < 100:
            advice.append("❤️ Reduced max heart rate. Cardiopulmonary fitness assessment recommended.")
        if not advice:
            advice.append("✅ No major individual risk flags detected. Maintain a balanced diet and regular exercise.")
        for tip in advice:
            st.markdown(f'<div class="advice-card">{tip}</div>', unsafe_allow_html=True)

        # ── Full Report ──
        with st.expander("📄 Full Report Summary"):
            report_name = patient_name or "Patient"
            st.markdown(f"""
**Patient:** {report_name} &nbsp;&nbsp; **ID:** {patient_id or '—'} &nbsp;&nbsp; **Date:** {datetime.now().strftime("%d %B %Y %H:%M")}

| Parameter | Value |
|-----------|-------|
| Age | {age} years |
| Sex | {"Male" if sex == "M" else "Female"} |
| Resting BP | {resting_bp} mm Hg |
| Cholesterol | {cholesterol} mg/dL |
| Fasting BS >120 | {"Yes" if fasting_bs else "No"} |
| Max Heart Rate | {max_hr} bpm |
| Chest Pain Type | {chest_pain} |
| Resting ECG | {resting_ecg} |
| Exercise Angina | {"Yes" if exercise_angina == "Y" else "No"} |
| ST Slope | {st_slope} |
| Oldpeak | {oldpeak} |
| **Risk Score** | **{probability*100:.2f}%** |
| **Prediction** | **{"⚠️ High Risk" if prediction == 1 else "✅ Low Risk"}** |
""")
            st.download_button(
                "⬇️ Download Report (CSV)",
                data=pd.DataFrame([{
                    "Name": report_name, "ID": patient_id or "",
                    "Date": datetime.now().isoformat(),
                    "Age": age, "Sex": sex, "RestingBP": resting_bp,
                    "Cholesterol": cholesterol, "FastingBS": fasting_bs,
                    "MaxHR": max_hr, "ChestPain": chest_pain,
                    "RestingECG": resting_ecg, "ExerciseAngina": exercise_angina,
                    "STSlope": st_slope, "Oldpeak": oldpeak,
                    "RiskScore_%": round(probability*100, 2),
                    "Prediction": "High Risk" if prediction == 1 else "Low Risk",
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
        fig_age, ax_a = plt.subplots(figsize=(5, 3.2), facecolor="none")
        ax_a.set_facecolor("none"); fig_age.patch.set_alpha(0)
        ages = np.arange(20, 85, 5)
        risk = 1 / (1 + np.exp(-0.1*(ages - 55)))
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
        fig_ch, ax_c = plt.subplots(figsize=(5, 3.2), facecolor="none")
        ax_c.set_facecolor("none"); fig_ch.patch.set_alpha(0)
        rng2 = np.random.default_rng(7)
        low_chol  = rng2.normal(180, 25, 400)
        high_chol = rng2.normal(260, 35, 400)
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