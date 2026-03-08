"""
CreditAI — Intelligent Credit Risk Scoring System
Multi-page Streamlit application.

Pages
-----
🏠 Home            — Project overview and architecture.
📊 Risk Assessment  — ML prediction with SHAP explainability (Milestone 1).
🤖 AI Assistant     — LangGraph agentic lending memo + PDF export (Milestone 2).
📈 Portfolio        — Batch analysis and portfolio risk distribution.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="CreditAI — Intelligent Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

/* ── Backgrounds ── */
.stApp {
    background: linear-gradient(135deg, #060d1a 0%, #0a1628 55%, #07111f 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #080f1e 100%);
    border-right: 1px solid rgba(0,217,255,0.12);
}

/* ── Hero title ── */
.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #00d9ff 0%, #7c3aed 60%, #00d9ff 100%);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    animation: shimmer 4s infinite linear;
}
@keyframes shimmer { to { background-position: 200% center; } }

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    transition: border-color .25s, transform .25s;
}
.glass-card:hover {
    border-color: rgba(0,217,255,0.25);
    transform: translateY(-2px);
}

/* ── Section header ── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #00d9ff;
    padding-left: 12px;
    margin: 20px 0 14px;
}

/* ── Risk badges ── */
.risk-high {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: #fff;
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    animation: pulse-red 2s infinite;
}
.risk-low {
    background: linear-gradient(135deg, #22c55e, #15803d);
    color: #fff;
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%      { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
}

/* ── Decision chips ── */
.chip-approve     { background:linear-gradient(135deg,#22c55e,#15803d); color:#fff; }
.chip-conditional { background:linear-gradient(135deg,#f59e0b,#b45309); color:#fff; }
.chip-decline     { background:linear-gradient(135deg,#ef4444,#b91c1c); color:#fff; }
.decision-chip {
    padding: 14px 36px;
    border-radius: 50px;
    font-weight: 800;
    font-size: 1.35rem;
    display: inline-block;
    letter-spacing: 0.06em;
}

/* ── Metric card ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,217,255,0.15);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 1.9rem; font-weight: 700; color: #00d9ff; }
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }

/* ── Report sections ── */
.report-section {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.report-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #00d9ff;
    font-weight: 700;
    margin-bottom: 10px;
}
.report-content { color: #cbd5e1; line-height: 1.7; font-size: 0.94rem; }

/* ── Disclaimer box ── */
.disclaimer-box {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 8px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d9ff, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: .55rem 1.4rem !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover {
    opacity: .9 !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #00d9ff !important; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"]  input { border-radius: 8px !important; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp .45s ease-out; }

hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.pkl")

FEATURE_DISPLAY: dict[str, str] = {
    "rev_util":    "Revolving Utilization",
    "age":         "Age",
    "late_30_59":  "30–59 Days Late",
    "debt_ratio":  "Debt Ratio",
    "monthly_inc": "Monthly Income",
    "open_credit": "Open Credit Lines",
    "late_90":     "90+ Days Late",
    "real_estate": "Real Estate Loans",
    "late_60_89":  "60–89 Days Late",
    "dependents":  "Dependents",
}

_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
)

# ─── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("page",            "🏠 Home"),
    ("last_prediction", None),
    ("last_report",     None),
    ("groq_key",        ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model() -> dict | None:
    """Load the persisted model artifact (backward-compatible with bare .pkl)."""
    try:
        artifact = joblib.load(MODEL_PATH)
        if hasattr(artifact, "predict"):          # old bare-model format
            from src.preprocess import FEATURE_COLS
            return {"model": artifact, "feature_names": FEATURE_COLS, "metrics": {}, "model_type": "RandomForestClassifier"}
        return artifact
    except Exception as exc:
        st.error(f"⚠️  Model not loaded: {exc}\n\nRun `python3 src/train_model.py` first.")
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:20px 0 10px">
            <div style="font-size:2.6rem">🏦</div>
            <div style="font-size:1.15rem;font-weight:800;color:#00d9ff;letter-spacing:.02em">CreditAI</div>
            <div style="font-size:.72rem;color:#475569;letter-spacing:.04em">INTELLIGENT RISK SCORING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    pages = ["🏠 Home", "📊 Risk Assessment", "🤖 AI Assistant", "📈 Portfolio"]
    for p in pages:
        active = st.session_state.page == p
        if st.button(p, key=f"nav_{p}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = p
            st.session_state.last_report = None
            st.rerun()

    st.markdown("---")

    # Model KPIs
    artifact = load_model()
    if artifact:
        m = artifact.get("metrics", {})
        st.markdown("**📊 Model Performance**")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC",   f"{m.get('roc_auc', 0):.3f}")
        c1.metric("Recall",    f"{m.get('recall',  0):.3f}")
        c2.metric("Precision", f"{m.get('precision', 0):.3f}")
        c2.metric("F1 Score",  f"{m.get('f1', 0):.3f}")

    st.markdown("---")

    # Groq key input
    st.markdown("**🤖 Groq API Key**")
    groq_inp = st.text_input(
        "API Key", value=st.session_state.groq_key, type="password",
        placeholder="gsk_…", label_visibility="collapsed",
        help="Free key at console.groq.com — enables LLM-powered agent",
    )
    if groq_inp != st.session_state.groq_key:
        st.session_state.groq_key = groq_inp
    if st.session_state.groq_key:
        st.success("Key set ✓")
    else:
        st.caption("Get a free key → [console.groq.com](https://console.groq.com)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home() -> None:
    st.markdown('<div class="hero-title">Intelligent Credit Risk<br>Scoring System</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;font-size:1.05rem;margin-top:6px;max-width:680px">'
        "End-to-end AI platform combining a Random Forest scoring engine with an "
        "autonomous LangGraph agent for structured, policy-grounded lending decisions."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature tiles
    tiles = [
        ("📊", "ML Risk Prediction",  "Random Forest with 5-fold cross-validated ROC-AUC and class-imbalance handling"),
        ("🔍", "SHAP Explainability",  "Per-prediction feature attribution for regulatory adverse-action compliance"),
        ("🤖", "Agentic AI Assistant", "LangGraph + RAG + Groq LLaMA-3.3-70B generates structured lending memos"),
        ("📄", "PDF Export",           "Download a professional lending assessment report with one click"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, tiles):
        with col:
            st.markdown(
                f"""<div class="glass-card" style="text-align:center;min-height:190px">
                    <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
                    <div style="font-weight:700;color:#e2e8f0;margin-bottom:8px;font-size:.95rem">{title}</div>
                    <div style="font-size:.82rem;color:#64748b;line-height:1.5">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(
            """<div class="glass-card" style="line-height:2;font-size:.93rem;color:#94a3b8">
            <b style="color:#00d9ff">Milestone 1 — ML Pipeline</b><br>
            📥 CSV / Manual Input → Preprocessing <i>(median imputation, outlier capping)</i><br>
            → Feature Engineering → Random Forest <i>(class_weight=balanced, n=200)</i><br>
            → Risk Score + SHAP Attribution → Streamlit UI<br><br>
            <b style="color:#7c3aed">Milestone 2 — Agentic AI</b><br>
            🤖 ML Score + SHAP Features → LangGraph State Machine<br>
            &nbsp;&nbsp;∟ Node 1: FAISS RAG — <i>retrieves relevant lending policy chunks</i><br>
            &nbsp;&nbsp;∟ Node 2: Groq / LLaMA-3.3-70B — <i>generates structured lending memo</i><br>
            → Structured Assessment Report → PDF Download
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """<div class="glass-card" style="font-size:.88rem;color:#94a3b8;line-height:2.1">
            <b style="color:#e2e8f0">Tech Stack</b><br>
            🐍 Python 3.10+<br>🤖 scikit-learn<br>🔍 SHAP<br>
            🧠 LangGraph + Groq<br>🗄️ FAISS + SentenceT<br>
            📊 Streamlit + Plotly<br>📄 ReportLab
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Attribute": ["Source", "Records", "Input Features", "Target Label", "Class Imbalance", "Imbalance Strategy"],
            "Detail":    [
                "Kaggle — Credit Risk Benchmark Dataset",
                "≈ 150,000 borrower records",
                "10 engineered features (delinquency, income, utilization …)",
                "dlq_2yrs — serious delinquency within 2 years (binary)",
                "≈ 6.7% positive rate",
                "class_weight='balanced' + Stratified 80/20 split",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
def _shap_bar(shap_df: pd.DataFrame) -> go.Figure:
    top   = shap_df.head(8)
    names = [FEATURE_DISPLAY.get(f, f) for f in top["feature"]]
    vals  = top["shap_value"].tolist()
    colors_ = ["#ef4444" if v >= 0 else "#22c55e" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors_,
        text=[f"{v:+.4f}" for v in vals], textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(
        **_PLOTLY_BASE,
        title=dict(text="Feature Contributions — red = increases risk · green = decreases risk",
                   font=dict(color="#64748b", size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=310, margin=dict(t=36, b=16, l=0, r=90),
    )
    return fig


def _gauge(probability: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number=dict(suffix="%", font=dict(color=color, size=30)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#64748b"),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)",
            steps=[
                dict(range=[0,  15],  color="rgba(34,197,94,0.18)"),
                dict(range=[15, 30],  color="rgba(34,197,94,0.08)"),
                dict(range=[30, 50],  color="rgba(245,158,11,0.14)"),
                dict(range=[50, 100], color="rgba(239,68,68,0.18)"),
            ],
            threshold=dict(line=dict(color=color, width=3), value=probability * 100),
        ),
        title=dict(text="Default Probability", font=dict(color="#64748b", size=13)),
    ))
    fig.update_layout(**_PLOTLY_BASE, height=230, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def page_risk_assessment() -> None:
    st.markdown('<div class="hero-title" style="font-size:2rem">📊 Credit Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;margin-top:4px">Enter borrower attributes to predict delinquency risk and explain contributing factors.</p>', unsafe_allow_html=True)

    artifact = load_model()
    if not artifact:
        return

    model         = artifact["model"]
    feature_names = artifact["feature_names"]

    tab1, tab2 = st.tabs(["✏️  Manual Entry", "📁  Batch CSV Upload"])

    # ── Manual Entry ──────────────────────────────────────────────────────────
    with tab1:
        with st.form("risk_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**👤 Borrower Profile**")
                age         = st.number_input("Age",              18, 120,    40,  help="Years")
                monthly_inc = st.number_input("Monthly Income ($)", 0, 100_000, 5_000, step=500)
                dependents  = st.number_input("Dependents",        0,  20,     0)

            with col2:
                st.markdown("**💳 Credit Profile**")
                rev_util    = st.number_input("Revolving Utilization", 0.0, 20_000.0,  0.30, step=0.01, format="%.2f",
                                               help="Ratio of revolving credit used (0–1 typical, but raw feature may exceed 1)")
                debt_ratio  = st.number_input("Debt Ratio",            0.0,    50.0,   0.35, step=0.01, format="%.2f",
                                               help="Total monthly debt / gross income")
                open_credit = st.number_input("Open Credit Lines",     0,      50,      6)
                real_estate = st.number_input("Real Estate Loans",     0,      10,      1)

            with col3:
                st.markdown("**⚠️ Delinquency History**")
                late_30_59 = st.number_input("30–59 Days Late",  0, 20, 0, help="Times past due 30–59 days in last 2 years")
                late_60_89 = st.number_input("60–89 Days Late",  0, 20, 0, help="Times past due 60–89 days in last 2 years")
                late_90    = st.number_input("90+ Days Late",     0, 20, 0, help="Times 90+ days past due in last 2 years")

            submitted = st.form_submit_button("🔮  Predict Credit Risk", use_container_width=True)

        if submitted:
            input_data = pd.DataFrame([{
                "rev_util": rev_util, "age": age, "late_30_59": late_30_59,
                "debt_ratio": debt_ratio, "monthly_inc": monthly_inc,
                "open_credit": open_credit, "late_90": late_90,
                "real_estate": real_estate, "late_60_89": late_60_89,
                "dependents": dependents,
            }])[feature_names]

            with st.spinner("Analysing credit profile …"):
                prediction  = model.predict(input_data)[0]
                probability = float(model.predict_proba(input_data)[0][1])
                risk_label  = "High Risk" if prediction == 1 else "Low Risk"

            # ── Results ───────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="fade-up">', unsafe_allow_html=True)

            r1, r2 = st.columns([1, 2])
            with r1:
                badge = "risk-high" if prediction == 1 else "risk-low"
                color  = "#ef4444" if prediction == 1 else "#22c55e"
                st.markdown(
                    f"""<div style="text-align:center;padding:28px 0">
                        <div class="metric-label" style="margin-bottom:12px">Risk Classification</div>
                        <div class="{badge}">{risk_label}</div>
                        <div style="margin-top:24px">
                            <span class="metric-value">{probability*100:.1f}%</span><br>
                            <span class="metric-label">Probability of Default</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r2:
                st.plotly_chart(_gauge(probability, color), use_container_width=True)

            # ── SHAP plot ─────────────────────────────────────────────────────
            st.markdown('<div class="section-header">🔍 Risk Factor Analysis (SHAP)</div>', unsafe_allow_html=True)

            with st.spinner("Computing SHAP values …"):
                try:
                    from src.explain_model import get_shap_values
                    shap_df = get_shap_values(input_data)
                except Exception:
                    imps = model.feature_importances_
                    shap_df = pd.DataFrame({
                        "feature": feature_names, "shap_value": imps, "abs_shap": imps,
                    }).sort_values("abs_shap", ascending=False)

            st.plotly_chart(_shap_bar(shap_df), use_container_width=True)

            # Save to session state for AI Assistant
            st.session_state.last_prediction = {
                "borrower_profile": {
                    "age": age, "monthly_inc": monthly_inc, "dependents": dependents,
                    "rev_util": rev_util, "debt_ratio": debt_ratio,
                    "open_credit": open_credit, "real_estate": real_estate,
                    "late_30_59": late_30_59, "late_60_89": late_60_89, "late_90": late_90,
                },
                "risk_score":   probability,
                "risk_label":   risk_label,
                "top_features": list(zip(shap_df["feature"].tolist(), shap_df["shap_value"].tolist())),
            }
            st.markdown("</div>", unsafe_allow_html=True)
            st.info("✅  Prediction saved — go to **🤖 AI Assistant** to generate a full lending assessment report.")

    # ── Batch CSV ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Upload a CSV file containing borrower rows to score in bulk.")
        uploaded = st.file_uploader("Upload Borrower CSV", type=["csv"])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.write("**Preview (first 5 rows):**")
            st.dataframe(batch_df.head(), use_container_width=True)

            missing = [c for c in feature_names if c not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}\n\nExpected: {feature_names}")
            else:
                if st.button("⚡  Run Batch Predictions", use_container_width=True):
                    with st.spinner("Running batch predictions …"):
                        X = batch_df[feature_names].fillna(batch_df[feature_names].median())
                        preds = model.predict(X)
                        probs = model.predict_proba(X)[:, 1]

                    batch_df["Risk_Prediction"]    = ["High Risk" if p == 1 else "Low Risk" for p in preds]
                    batch_df["Default_Probability"] = (probs * 100).round(2)

                    h = int((preds == 1).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Applicants", f"{len(preds):,}")
                    c2.metric("High Risk",         f"{h:,}",           f"{h/len(preds)*100:.1f}%")
                    c3.metric("Low Risk",           f"{len(preds)-h:,}", f"{(len(preds)-h)/len(preds)*100:.1f}%")

                    st.dataframe(
                        batch_df[["Risk_Prediction", "Default_Probability"] + list(feature_names)],
                        use_container_width=True,
                    )
                    csv_out = batch_df.to_csv(index=False).encode()
                    st.download_button("⬇️  Download Results CSV", csv_out,
                                       "batch_predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
def _decision_chip(decision: str) -> str:
    d = decision.strip().upper()
    if "DECLINE" in d:
        cls = "chip-decline"
    elif "CONDITIONAL" in d:
        cls = "chip-conditional"
    else:
        cls = "chip-approve"
    return f'<div class="decision-chip {cls}">{d}</div>'


def page_ai_assistant() -> None:
    st.markdown('<div class="hero-title" style="font-size:2rem">🤖 AI Lending Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;margin-top:4px">'
        "LangGraph agent retrieves policy context via RAG, then uses Groq LLaMA-3.3-70B "
        "to produce a structured lending memo with PDF export."
        "</p>",
        unsafe_allow_html=True,
    )

    last = st.session_state.get("last_prediction")
    if not last:
        st.warning("No prediction loaded. Go to **📊 Risk Assessment**, run a prediction, then return here.")
        return

    # Borrower summary
    st.success(
        f"Using prediction — **{last['risk_label']}** ({last['risk_score']*100:.1f}% default probability)"
    )
    with st.expander("📋  Borrower Profile", expanded=False):
        bp_items = list(last["borrower_profile"].items())
        cols = st.columns(5)
        for i, (k, v) in enumerate(bp_items):
            cols[i % 5].metric(FEATURE_DISPLAY.get(k, k), v)

    st.markdown("---")

    # API key status
    api_key = st.session_state.get("groq_key") or os.environ.get("GROQ_API_KEY", "")
    if api_key:
        st.markdown(
            '<div style="color:#22c55e;font-size:.88rem;margin-bottom:8px">✅  Groq API key detected — LLM-powered agent active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "ℹ️  No Groq API key provided. A policy-grounded template report will be generated instead. "
            "Add your free key in the sidebar for LLM-powered reasoning."
        )

    if st.button("📋  Generate Lending Assessment", use_container_width=True):
        with st.spinner("🤖  Agent running: retrieving policies → reasoning → drafting report …"):
            try:
                from src.agent import run_lending_agent
                report = run_lending_agent(
                    borrower_profile=last["borrower_profile"],
                    risk_score=last["risk_score"],
                    risk_label=last["risk_label"],
                    top_features=last["top_features"],
                    groq_api_key=api_key or None,
                )
                st.session_state.last_report = report
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                return

    report = st.session_state.get("last_report")
    if not report:
        return

    # ── Display report ─────────────────────────────────────────────────────────
    st.success("✅  Assessment complete!")
    st.markdown("<br>", unsafe_allow_html=True)

    # Decision chip
    decision = report.get("decision", "").strip().upper()
    st.markdown(
        f'<div style="text-align:center;padding:16px 0">'
        f'<div class="metric-label" style="margin-bottom:12px">Recommended Decision</div>'
        f"{_decision_chip(decision)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    sections = [
        ("👤  Borrower Profile Summary",    "profile_summary"),
        ("📈  Credit Risk Analysis",          "risk_analysis"),
        ("⚖️  Decision Rationale",            "decision_rationale"),
        ("🛡️  Risk Mitigation Suggestions",   "risk_mitigation"),
        ("📚  References & Sources",          "references"),
    ]
    for label, key in sections:
        content = report.get(key, "N/A")
        st.markdown(
            f"""<div class="report-section fade-up">
                <div class="report-label">{label}</div>
                <div class="report-content">{content}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    disclaimer = report.get("disclaimer", "")
    if disclaimer:
        st.markdown(
            f"""<div class="disclaimer-box">
                <div class="report-label" style="color:#f59e0b">⚠️  Legal Disclaimer</div>
                <div style="font-size:.82rem;color:#92400e;line-height:1.6">{disclaimer}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── PDF download ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        from src.pdf_report import generate_pdf_report
        pdf_bytes = generate_pdf_report(
            report=report,
            borrower_profile=last["borrower_profile"],
            risk_score=last["risk_score"],
            risk_label=last["risk_label"],
        )
        st.download_button(
            "📄  Download PDF Report",
            data=pdf_bytes,
            file_name="lending_assessment_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.caption(f"PDF unavailable: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
def page_portfolio() -> None:
    st.markdown('<div class="hero-title" style="font-size:2rem">📈 Portfolio Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;margin-top:4px">Analyse the credit risk distribution across an entire borrower portfolio.</p>', unsafe_allow_html=True)

    artifact = load_model()
    if not artifact:
        return

    model         = artifact["model"]
    feature_names = artifact["feature_names"]

    uploaded = st.file_uploader("Upload Portfolio CSV (or view demo below)", type=["csv"])

    if not uploaded:
        st.info("No file uploaded — displaying demo with 500 synthetic borrowers.")
        np.random.seed(42)
        n = 500
        portfolio = pd.DataFrame({
            "rev_util":    np.random.exponential(0.4, n).clip(0, 3),
            "age":         np.random.randint(22, 72, n),
            "late_30_59":  np.random.poisson(0.3, n),
            "debt_ratio":  np.random.exponential(0.35, n).clip(0, 3),
            "monthly_inc": np.random.lognormal(8.5, 0.7, n).clip(1_000, 50_000),
            "open_credit": np.random.randint(1, 18, n),
            "late_90":     np.random.poisson(0.08, n),
            "real_estate": np.random.randint(0, 4, n),
            "late_60_89":  np.random.poisson(0.08, n),
            "dependents":  np.random.randint(0, 5, n),
        })
    else:
        portfolio = pd.read_csv(uploaded)

    missing = [c for c in feature_names if c not in portfolio.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    X     = portfolio[feature_names].fillna(portfolio[feature_names].median())
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    portfolio["Default_Prob"] = probs
    portfolio["Risk_Tier"] = pd.cut(
        probs,
        bins=[0, 0.15, 0.30, 0.50, 1.0],
        labels=["Low (<15%)", "Moderate (15–30%)", "Elevated (30–50%)", "High (>50%)"],
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    h = int((preds == 1).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Size",     f"{len(probs):,}")
    c2.metric("High Risk Count",    f"{h:,}", f"{h/len(probs)*100:.1f}%")
    c3.metric("Avg Default Prob",   f"{probs.mean()*100:.1f}%")
    c4.metric("Expected Loss Rate", f"{probs[probs > 0.30].mean()*100:.1f}%" if (probs > 0.30).any() else "0%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        tier_counts = portfolio["Risk_Tier"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=tier_counts.index, values=tier_counts.values,
            hole=0.5,
            marker_colors=["#22c55e", "#84cc16", "#f59e0b", "#ef4444"],
        ))
        fig_pie.update_layout(**_PLOTLY_BASE, title="Risk Tier Distribution",
                              height=320, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_hist = go.Figure(go.Histogram(x=probs * 100, nbinsx=40,
                                           marker_color="#00d9ff", opacity=0.8))
        fig_hist.update_layout(
            **_PLOTLY_BASE, title="Default Probability Distribution",
            xaxis=dict(title="Default Probability (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Count",                   gridcolor="rgba(255,255,255,0.05)"),
            height=320, margin=dict(t=40, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Feature correlation bar
    corr = {
        FEATURE_DISPLAY.get(f, f): float(abs(pd.Series(X[f].values).corr(pd.Series(probs))))
        for f in feature_names
    }
    corr_df = (
        pd.DataFrame(corr.items(), columns=["Feature", "Correlation"])
        .sort_values("Correlation", ascending=True)
    )
    fig_corr = go.Figure(go.Bar(
        x=corr_df["Correlation"], y=corr_df["Feature"],
        orientation="h", marker_color="#7c3aed",
    ))
    fig_corr.update_layout(
        **_PLOTLY_BASE, title="Feature Correlation with Default Probability",
        xaxis=dict(title="Pearson |r|", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=340, margin=dict(t=40, b=40, l=0, r=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    csv_out = portfolio.to_csv(index=False).encode()
    st.download_button("⬇️  Download Portfolio Results", csv_out,
                       "portfolio_risk_scores.csv", "text/csv")


# ─── Router ───────────────────────────────────────────────────────────────────
_PAGES = {
    "🏠 Home":            page_home,
    "📊 Risk Assessment": page_risk_assessment,
    "🤖 AI Assistant":    page_ai_assistant,
    "📈 Portfolio":       page_portfolio,
}
_PAGES[st.session_state.page]()