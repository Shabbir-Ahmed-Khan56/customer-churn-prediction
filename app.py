"""
ChurnSense — AI-Powered Customer Churn Prediction Platform
==========================================================

Production Streamlit SaaS application that runs predictions through a
pre-trained machine learning pipeline (customer_churn_model.pkl).

The pipeline bundles preprocessing + estimator together, so the app
feeds raw customer attributes (the exact 19 training features) directly
into the model and reads predict_proba() for the churn probability.

Author: ChurnSense
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Page configuration (must run before any other Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnSense — AI Churn Prediction Platform",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.pkl")
STYLESHEET = os.path.join(BASE_DIR, "styles", "main.css")


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
def load_css() -> None:
    """Inject the external stylesheet into the Streamlit app."""
    if os.path.exists(STYLESHEET):
        with open(STYLESHEET, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# -------------------------------------------------------------------
# Model integration layer
# -------------------------------------------------------------------
@dataclass
class ChurnResult:
    """Container for a single churn prediction."""

    probability: float   # P(churn) in [0.0, 1.0]
    prediction: int      # 1 = churn, 0 = retain
    risk_label: str      # Low / Medium / High / Critical
    confidence: float    # model confidence = max(p, 1-p)


@st.cache_resource(show_spinner=False)
def load_model() -> Optional[Any]:
    """
    Load the pre-trained churn pipeline from disk using joblib.

    The pipeline bundles preprocessing + estimator together, so no
    separate scaler / encoder files are needed.

    Returns:
        The loaded pipeline object, or None if the file is missing.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_churn(model: Any, features: Dict[str, Any]) -> Optional[ChurnResult]:
    """
    Run a churn prediction through the loaded pipeline.

    `features` is a flat dict of the 19 raw customer attributes with the
    exact column names and categorical values the pipeline was trained on.
    A single-row DataFrame preserves column names + order for the pipeline.

    Returns:
        ChurnResult on success, or None on failure (error surfaced to UI).
    """
    if model is None:
        return None
    try:
        # Preserve the exact feature order the pipeline was trained with.
        feature_order = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges",
        ]
        X = pd.DataFrame([[features[f] for f in feature_order]],
                         columns=feature_order)

        proba = model.predict_proba(X)[0]
        churn_p = float(proba[1])
        # predict() may return numeric (0/1) or string ('No'/'Yes') class
        # labels depending on how the target was encoded during training.
        raw_pred = model.predict(X)[0]
        if isinstance(raw_pred, (int, float, bool)):
            prediction = int(raw_pred)
        else:
            prediction = 1 if str(raw_pred) == "Yes" else 0
        confidence = max(churn_p, 1.0 - churn_p)

        return ChurnResult(
            probability=churn_p,
            prediction=prediction,
            risk_label=_risk_band(churn_p),
            confidence=confidence,
        )
    except Exception as err:  # noqa: BLE001  (surfaced to the UI)
        st.session_state["prediction_error"] = str(err)
        return None


def _risk_band(probability: float) -> str:
    """Map a churn probability to a human-readable risk band."""
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.25:
        return "Medium"
    return "Low"


# -------------------------------------------------------------------
# Recommendation engine
# -------------------------------------------------------------------
def recommend(risk: str, features: Dict[str, Any]) -> str:
    """Return a business recommendation tailored to the risk band."""
    tenure = features.get("tenure", 0)
    contract = features.get("Contract", "")

    if risk == "Critical":
        return (
            "Immediate retention intervention required. Assign a dedicated "
            "account manager, offer a loyalty discount, and schedule a "
            "win-back call within 48 hours."
        )
    if risk == "High":
        return (
            "Proactive outreach recommended. Offer a targeted promotion or "
            "upgrade, and monitor engagement signals over the next billing cycle."
        )
    if risk == "Medium":
        return (
            "Keep the customer engaged with onboarding nudges, usage tips, "
            "and a quarterly check-in to prevent escalation."
        )
    # Low
    if tenure and float(tenure) < 6 and contract == "Month-to-month":
        return (
            "Customer is stable but new and on a flexible contract. Reinforce "
            "value early with guided onboarding and milestone rewards to lock "
            "in long-term loyalty."
        )
    return (
        "Healthy account. Continue standard engagement and consider "
        "upsell opportunities to expand lifetime value."
    )


# -------------------------------------------------------------------
# UI components
# -------------------------------------------------------------------
def render_header() -> None:
    """Top navigation bar with logo and brand name."""
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    logo_html = ""
    if os.path.exists(logo_path):
        import base64

        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{encoded}" '
            'class="cs-logo" alt="ChurnSense logo" />'
        )

    st.markdown(
        f"""
        <header class="cs-header">
            <div class="cs-header-inner">
                <div class="cs-brand">
                    {logo_html}
                    <span class="cs-brand-name">ChurnSense</span>
                    <span class="cs-brand-tag">AI Churn Intelligence</span>
                </div>
                <nav class="cs-nav">
                    <span class="cs-nav-item">Dashboard</span>
                    <span class="cs-nav-item">Insights</span>
                    <span class="cs-nav-item cs-nav-item--accent">Predict</span>
                </nav>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Hero section with headline and supporting copy."""
    st.markdown(
        """
        <section class="cs-hero">
            <div class="cs-hero-badge">
                <span class="cs-hero-dot"></span>
                Production-grade churn intelligence
            </div>
            <h1 class="cs-hero-title">
                AI-Powered Customer<br/>
                <span class="cs-hero-title-accent">Churn Prediction Platform</span>
            </h1>
            <p class="cs-hero-subtitle">
                Identify at-risk customers before they leave. Turn retention
                into a measurable, data-driven practice with ChurnSense.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_form() -> tuple[Dict[str, Any], bool]:
    """
    Render the customer information input card and return collected values.

    Collects exactly the 19 features the trained pipeline expects, using
    the original training categories. No extra fields, no missing fields.

    Uses st.form() so native Streamlit widgets are properly nested
    inside the card. A hidden marker div lets CSS identify and style the
    container as a glass card via the :has() selector.
    """
    st.markdown(
        '<div class="cs-form-title">Customer Information</div>',
        unsafe_allow_html=True,
    )

    with st.form("customer_form", clear_on_submit=False):
        st.markdown('<div class="cs-form-marker"></div>',
                    unsafe_allow_html=True)

        # ---- Account & demographics ----
        st.markdown(
            '<div class="cs-group-label cs-group-label--first">'
            "Account &amp; Demographics</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"])
        with c4:
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        # ---- Services ----
        st.markdown(
            '<div class="cs-group-label">Services</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with c2:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
        with c3:
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
        with c4:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        # ---- Add-ons ----
        st.markdown(
            '<div class="cs-group-label">Add-ons</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
        with c2:
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
        with c3:
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
        with c4:
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
        with c5:
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )

        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

        # ---- Contract & billing ----
        st.markdown(
            '<div class="cs-group-label">Contract &amp; Billing</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
        with c2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with c3:
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        c1, c2 = st.columns(2)
        with c1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 0.0, 500.0, 50.0, step=0.5
            )
        with c2:
            total_charges = st.number_input(
                "Total Charges ($)", 0.0, 20000.0, 500.0, step=1.0
            )

        st.markdown("<div class='cs-predict-wrap'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Churn Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    # Map UI selections to the exact training schema.
    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }, submitted


def render_result(result: ChurnResult, features: Dict[str, Any]) -> None:
    """Render the prediction result section."""
    prob_pct = result.probability * 100
    retain_pct = 100 - prob_pct
    risk_class = result.risk_label.lower()

    st.markdown(
        f"""
<div class="cs-section-title">Prediction Result</div>
<div class="cs-card cs-card--result">
    <div class="cs-result-top">
        <div class="cs-result-label">Churn Risk</div>
        <span class="cs-badge cs-badge--{risk_class}">{result.risk_label}</span>
    </div>
    <div class="cs-prob-row">
        <div class="cs-prob-block cs-prob-block--churn">
            <div class="cs-prob-value">{prob_pct:.1f}%</div>
            <div class="cs-prob-caption">Churn Probability</div>
        </div>
        <div class="cs-prob-block cs-prob-block--retain">
            <div class="cs-prob-value">{retain_pct:.1f}%</div>
            <div class="cs-prob-caption">Retention Probability</div>
        </div>
    </div>
    <div class="cs-meter">
        <div class="cs-meter-fill cs-meter-fill--{risk_class}" style="width: {prob_pct:.1f}%"></div>
    </div>
    <div class="cs-result-meta">
        <span>Model confidence: <strong>{result.confidence*100:.1f}%</strong></span>
        <span>Prediction: <strong>{"Churn" if result.prediction else "Retain"}</strong></span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="cs-section-title">Business Recommendation</div>
<div class="cs-card cs-card--recommendation">
    <div class="cs-recommendation-icon">&#9733;</div>
    <p class="cs-recommendation-text">{recommend(result.risk_label, features)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(model: Optional[Any]) -> None:
    """Show a small status pill indicating whether the model is loaded."""
    if model is not None:
        st.markdown(
            '<div class="cs-status cs-status--ok">'
            '<span class="cs-status-dot"></span>'
            '<span>Model pipeline loaded</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cs-status cs-status--error">'
            '<span class="cs-status-dot cs-status-dot--error"></span>'
            '<span>Model file not found. Place '
            '<code>customer_churn_model.pkl</code> '
            "in the project root to enable predictions.</span>"
            '</div>',
            unsafe_allow_html=True,
        )


def render_error(message: str) -> None:
    """Render a prediction error card."""
    st.markdown(
        f"""
<div class="cs-section-title">Prediction Error</div>
<div class="cs-card cs-card--error">
    <div class="cs-error-icon">!</div>
    <div>
        <div class="cs-error-title">Prediction failed</div>
        <p class="cs-error-text">{message}</p>
        <p class="cs-error-hint">Check that the input values match the training schema and that the pipeline is intact.</p>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Professional footer."""
    st.markdown(
        """
<footer class="cs-footer">
    <div class="cs-footer-inner">
        <div class="cs-footer-brand">
            <strong>ChurnSense</strong>
            <span>AI-Powered Customer Churn Prediction Platform</span>
        </div>
        <div class="cs-footer-links">
            <span>Privacy</span>
            <span>Terms</span>
            <span>Security</span>
            <span>Contact</span>
        </div>
        <div class="cs-footer-copy">&copy; 2025 ChurnSense. All rights reserved.</div>
    </div>
</footer>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main() -> None:
    render_header()
    render_hero()

    model = load_model()
    render_model_status(model)

    features, submitted = render_input_form()

    if submitted:
        if model is None:
            st.session_state["prediction_error"] = (
                "Model pipeline is not loaded."
            )
        else:
            with st.spinner("Analyzing customer profile…"):
                result = predict_churn(model, features)
                if result is not None:
                    st.session_state["last_result"] = result
                    st.session_state["last_features"] = features
                    st.session_state.pop("prediction_error", None)
                # error path is handled below

    if st.session_state.get("prediction_error"):
        render_error(st.session_state["prediction_error"])

    last = st.session_state.get("last_result")
    last_features = st.session_state.get("last_features", features)
    if last is not None:
        render_result(last, last_features)

    render_footer()


if __name__ == "__main__":
    main()
