"""
app.py — PhishGuard Streamlit Web Application (Layer 4)

Layout:
    - Branded header with gradient title
    - URL text input + Analyze button
    - Example URL buttons (phishing / borderline / legitimate)
    - Verdict badge (color-coded red/green)
    - Probability metrics + progress bar gauge
    - SHAP feature contribution horizontal bar chart
    - Plain-English explanation text
    - Expandable detailed feature breakdown table
    - Expandable full feature vector

Run with:
    streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="PhishGuard — URL Phishing Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header { text-align: center; padding: 2rem 0 1rem; }
.main-header h1 {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.main-header p { color: #64748b; font-size: 1.05rem; line-height: 1.5; }

.verdict-phishing {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white; padding: 1.2rem 2rem; border-radius: 12px;
    text-align: center; font-size: 1.5rem; font-weight: 700; margin: 1.5rem 0;
    box-shadow: 0 8px 24px rgba(239,68,68,0.3);
    letter-spacing: 0.5px;
}
.verdict-legitimate {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white; padding: 1.2rem 2rem; border-radius: 12px;
    text-align: center; font-size: 1.5rem; font-weight: 700; margin: 1.5rem 0;
    box-shadow: 0 8px 24px rgba(16,185,129,0.3);
    letter-spacing: 0.5px;
}
.explanation-box {
    background: #f8fafc; border-left: 4px solid #6366f1;
    padding: 1.1rem 1.3rem; border-radius: 0 10px 10px 0;
    color: #1e293b; font-size: 0.95rem; line-height: 1.6; margin: 1.2rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the best model once per session (cached)."""
    path = Path("models/best_model.joblib")
    return joblib.load(path) if path.exists() else None


def plot_contributions(top_features: list) -> plt.Figure:
    """Horizontal bar chart of top contributing features, colored by direction."""
    names = [f["feature"].replace("_", " ").title() for f in top_features]
    values = [f["contribution"] for f in top_features]
    colors = ["#ef4444" if f["direction"] == "phishing" else "#10b981" for f in top_features]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="none", height=0.55)
    ax.axvline(x=0, color="#64748b", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Risk Contribution (SHAP value)", fontsize=9.5, color="#64748b")
    ax.set_title("Top Feature Risk Contributions for this URL", fontsize=11.5, fontweight="bold", color="#0f172a", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    red = mpatches.Patch(color="#ef4444", label="Increases phishing risk")
    green = mpatches.Patch(color="#10b981", label="Reduces phishing risk")
    ax.legend(handles=[red, green], loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ PhishGuard</h1>
        <p>Explainable ML-Based Phishing URL Detector — Real-time classification
        with feature-level risk justification.</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    model = load_model()
    if model is None:
        st.error("⚠️ Model not found. Run `python src/train.py` first.")
        st.stop()

    # ── Example Buttons ───────────────────────────────────────────────────
    st.markdown("**📋 Try a quick example URL:**")
    col1, col2, col3, col4 = st.columns(4)
    examples = {
        "🔴 Phishing (TK)": "http://paypal-secure-login.verify-account.tk/signin",
        "🔴 IP-based":       "http://192.168.1.1/bank/login",
        "🟡 Borderline":     "http://paypal.com.secure-login.net/verify",
        "🟢 Legitimate":     "https://www.github.com",
    }
    
    # Initialize session state for URL if not present
    if "selected_url" not in st.session_state:
        st.session_state.selected_url = ""

    for col, (label, url) in zip([col1, col2, col3, col4], examples.items()):
        with col:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state.selected_url = url

    # ── URL Input ─────────────────────────────────────────────────────────
    url_input = st.text_input(
        "Enter or paste URL to analyze:",
        value=st.session_state.selected_url,
        placeholder="https://example.com/login or paste any suspicious link",
        key="url_field",
    )
    analyze_btn = st.button("🔍 Analyze URL", use_container_width=True)

    final_url = url_input.strip()

    # ── Analysis ──────────────────────────────────────────────────────────
    if analyze_btn or (final_url and final_url == st.session_state.selected_url):
        if not final_url:
            st.warning("Please enter a URL to analyze.")
            return

        with st.spinner("Extracting 19 features and calculating SHAP contributions..."):
            try:
                from src.explain import predict_and_explain
                result = predict_and_explain(final_url, model=model)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        st.divider()

        # Verdict badge
        if result["prediction"] == "phishing":
            st.markdown(
                f'<div class="verdict-phishing">🚨 PHISHING DETECTED — {result["risk_label"].upper()}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="verdict-legitimate">✅ LEGITIMATE — {result["risk_label"].upper()}</div>',
                unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Phishing Probability", f"{result['probability']*100:.1f}%")
        c2.metric("Risk Level", result["risk_label"])
        c3.metric("Verdict", result["prediction"].capitalize())

        prob = result["probability"]
        if prob > 0.5:
            st.error(f"Confidence: {prob*100:.1f}% phishing")
        else:
            st.success(f"Confidence: {(1-prob)*100:.1f}% legitimate")
        st.progress(prob)

        st.divider()

        # Feature contribution chart
        st.subheader("🔎 Feature-Level Risk Justification")
        st.pyplot(plot_contributions(result["top_contributing_features"]),
                  use_container_width=True)

        # Explanation text
        st.markdown(
            f'<div class="explanation-box">💡 <strong>Natural Language Explanation:</strong><br><br>'
            f'{result["explanation_text"]}</div>',
            unsafe_allow_html=True)

        # Detailed breakdown
        with st.expander("📊 Detailed Feature Breakdown (Top 5 Contributions)"):
            tbl = pd.DataFrame(result["top_contributing_features"])[
                ["feature", "value", "contribution", "direction", "description"]
            ]
            tbl.columns = ["Feature", "Observed Value", "SHAP Contribution", "Risk Direction", "Security Meaning"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Full feature vector
        with st.expander("🔬 Complete Extracted Feature Vector (All 19 Features)"):
            fv_df = pd.DataFrame([{"Feature": k, "Extracted Value": v}
                                   for k, v in result["feature_vector"].items()])
            st.dataframe(fv_df, use_container_width=True, hide_index=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 1rem 0;">
        <strong>PhishGuard</strong> — Real-Time Explainable ML Phishing URL Detection<br>
        19 Handcrafted Features · XGBoost + TreeSHAP · Coursework Assignment
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
