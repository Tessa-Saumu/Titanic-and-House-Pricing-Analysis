"""
Prediction Lab — Streamlit frontend.

Design direction:
- Modern minimalist
- Deep olive brand rail
- Warm ivory workspace
- Insight-first prediction experience
"""

from __future__ import annotations

import html
import textwrap
from typing import Any

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def load_css(css_file_path: str) -> None:
    """Inject the product design system."""
    with open(css_file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def esc(value: Any) -> str:
    """HTML-escape dynamic content before rendering custom UI."""
    return html.escape(str(value))


def normalize_confidence(value: Any) -> float:
    """Normalize confidence to a 0–1 range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence > 1:
        confidence /= 100

    return max(0.0, min(confidence, 1.0))


def format_currency(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return esc(value)


def api_error(response: requests.Response) -> None:
    """Render a restrained API error instead of a default Streamlit alert."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    st.markdown(
        textwrap.dedent(f"""
        <div class="inline-notice error">
            <div class="notice-kicker">Request could not be completed</div>
            <div class="notice-body">{esc(detail)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def connection_error() -> None:
    st.markdown(
        textwrap.dedent("""
        <div class="inline-notice error">
            <div class="notice-kicker">Prediction service unavailable</div>
            <div class="notice-body">
                The FastAPI backend is not responding. Start the API and try again.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def model_header(title: str, description: str, model_name: str) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="page-header">
            <div class="eyebrow">PREDICTION LAB</div>
            <div class="page-title-row">
                <div>
                    <h1>{esc(title)}</h1>
                    <p>{esc(description)}</p>
                </div>
                <div class="model-chip">{esc(model_name)}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_prediction_result(
    *,
    title: str,
    headline: str,
    subline: str,
    model_name: str,
    confidence: float | None = None,
    tone: str = "olive",
    secondary_label: str | None = None,
    secondary_value: str | None = None,
) -> None:

    confidence_html = ""
    if confidence is not None:
        confidence_pct = confidence * 100
        confidence_html = f"""
<div class="confidence-block">
    <div class="confidence-row">
        <span>Model confidence</span>
        <strong>{confidence_pct:.1f}%</strong>
    </div>
    <div class="confidence-track">
        <div class="confidence-fill {tone}" style="width:{confidence_pct:.1f}%"></div>
    </div>
</div>
"""

    secondary_html = ""
    if secondary_label and secondary_value:
        secondary_html = f"""
<div class="secondary-metric">
    <span>{esc(secondary_label)}</span>
    <strong>{esc(secondary_value)}</strong>
</div>
"""

    html_block = f"""<section class="result-card {tone}">
<div class="result-kicker">{esc(title)}</div>

<div class="result-main">
    <div>
        <div class="result-headline">{headline}</div>
        <div class="result-subline">{esc(subline)}</div>
    </div>
    {secondary_html}
</div>

{confidence_html}

<div class="result-meta">
    <span>Model</span>
    <strong>{esc(model_name)}</strong>
</div>
</section>"""

    st.markdown(html_block, unsafe_allow_html=True)


def render_context_card(title: str, text: str, items: list[tuple[str, str]]) -> None:
    item_html = "".join(
        f"""
        <div class="context-item">
            <span>{esc(label)}</span>
            <strong>{esc(value)}</strong>
        </div>
        """
        for label, value in items
    )

    st.markdown(
        textwrap.dedent(f"""
        <section class="panel-card">
            <div class="panel-heading">{esc(title)}</div>
            <div class="panel-copy">{esc(text)}</div>
            <div class="context-list">{item_html}</div>
        </section>
        """),
        unsafe_allow_html=True,
    )


def render_explanation_card(title: str, body: str, bullets: list[str], tone: str = "") -> None:
    bullet_html = "".join(
        f'<li>{esc(item)}</li>' for item in bullets
    )

    st.markdown(
        textwrap.dedent(f"""
        <section class="explanation-card {tone}">
            <div class="panel-heading">{esc(title)}</div>
            <div class="panel-copy">{esc(body)}</div>
            <ul>{bullet_html}</ul>
        </section>
        """),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Prediction Lab",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css("app/ui/style.css")


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        textwrap.dedent("""
        <div class="brand-lockup">
            <div class="brand-mark">P</div>
            <div>
                <div class="brand-name">PREDICTION LAB</div>
                <div class="brand-subtitle">Applied intelligence</div>
            </div>
        </div>

        <div class="rail-divider"></div>

        <div class="rail-kicker">EXPLORE MODELS</div>
        """),
        unsafe_allow_html=True,
    )

    app_mode = st.radio(
        "Model",
        ["Housing Price", "Titanic Survival"],
        label_visibility="collapsed",
    )
    
    # RESET STATE WHEN SWITCHING PAGES
    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = app_mode

    if st.session_state["last_mode"] != app_mode:
        st.session_state.pop("housing_result", None)
        st.session_state.pop("titanic_result", None)
        st.session_state["last_mode"] = app_mode

    st.markdown(
        textwrap.dedent("""
        <div class="rail-divider"></div>

        <div class="rail-section">
            <div class="rail-kicker">ABOUT</div>
            <p>
                A small decision workspace for exploring predictions,
                confidence and model inputs.
            </p>
        </div>

        <div class="rail-footer">
            <span>Local inference</span>
            <span>FastAPI backend</span>
        </div>
        """),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# HOUSING PRICE
# -----------------------------------------------------------------------------

if app_mode == "Housing Price":
    model_header(
        "Housing Price",
        "Estimate a property's market value from its physical and location characteristics.",
        "Engineered Linear Regression",
    )

    st.markdown(
        textwrap.dedent("""
        <div class="helper-strip">
            <span class="helper-dot"></span>
            Enter the property profile, then use the estimate as a comparison point.
            This is a model estimate — not a formal appraisal.
        </div>
        """),
        unsafe_allow_html=True,
    )

    if "housing_result" in st.session_state:
        result = st.session_state["housing_result"]
        st.markdown('<div class="section-kicker">CURRENT ESTIMATE</div>', unsafe_allow_html=True)
        render_prediction_result(
            title="Estimated property value",
            headline=result.get("prediction"),  # already formatted from API
            subline="Based on the property profile submitted below.",
            model_name="Engineered Linear Regression",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            render_context_card(
                "Prediction profile",
                "These are the inputs currently associated with the estimate.",
                [
                    ("Area", f'{result["payload"]["area"]:,} sq ft'),
                    ("Bedrooms", str(result["payload"]["bedrooms"])),
                    ("Bathrooms", str(result["payload"]["bathrooms"])),
                    ("Stories", str(result["payload"]["stories"])),
                    ("Parking", str(result["payload"]["parking"])),
                ],
            )

        with col_b:
            render_explanation_card(
                "How to use the estimate",
                "Treat the output as a model-based comparison point rather than a stand-alone valuation.",
                [
                    "Change one or two property characteristics and compare the new estimate.",
                    "Use the result to explore sensitivity to different property profiles.",
                    "Keep the model estimate separate from real-world market, legal or appraisal advice.",
                ],
                tone="olive",
            )

        st.markdown('<div class="section-kicker scenario-label">NEW SCENARIO</div>', unsafe_allow_html=True)

    with st.form("housing_form", clear_on_submit=False):
        st.markdown("#### Property profile")
        st.caption("Start with the structural characteristics, then add features and location signals.")

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            area = st.number_input(
                "Area (sq ft)",
                min_value=1000,
                max_value=20000,
                value=5000,
                step=100,
            )
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
            stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)

        with col2:
            mainroad = st.selectbox("Main road access", ["yes", "no"])
            guestroom = st.selectbox("Guest room", ["yes", "no"], index=1)
            basement = st.selectbox("Basement", ["yes", "no"], index=1)
            parking = st.selectbox("Parking spaces", [0, 1, 2, 3])

        with col3:
            hotwaterheating = st.selectbox("Hot-water heating", ["yes", "no"], index=1)
            airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
            prefarea = st.selectbox("Preferred area", ["yes", "no"])
            furnishingstatus = st.selectbox(
                "Furnishing",
                ["furnished", "semi-furnished", "unfurnished"],
            )

        st.markdown(
            '<div class="form-action-row">',
            unsafe_allow_html=True,
        )
        submit_button = st.form_submit_button("Generate estimate", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if submit_button:
        payload = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }

        with st.spinner("Running prediction…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict/housing",
                    json=payload,
                    timeout=20,
                )

                if response.status_code == 200:
                    st.session_state["housing_result"] = {
                        "prediction": response.json().get("prediction"),
                        "payload": payload,
                    }
                    st.rerun()
                else:
                    api_error(response)

            except requests.exceptions.ConnectionError:
                connection_error()
            except requests.exceptions.Timeout:
                st.markdown(
                    textwrap.dedent("""
                    <div class="inline-notice error">
                        <div class="notice-kicker">Request timed out</div>
                        <div class="notice-body">
                            The prediction service took too long to respond.
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# TITANIC SURVIVAL
# -----------------------------------------------------------------------------

elif app_mode == "Titanic Survival":
    model_header(
        "Titanic Survival",
        "Estimate survival probability from a passenger profile and inspect how the model responds.",
        "Engineered XGBoost",
    )

    st.markdown(
        textwrap.dedent("""
        <div class="helper-strip">
            <span class="helper-dot"></span>
            This is a historical classification exercise. The output describes
            model behavior on the supplied profile — not historical certainty.
        </div>
        """),
        unsafe_allow_html=True,
    )

    if "titanic_result" in st.session_state:
        result = st.session_state["titanic_result"]

        pred = str(result.get("prediction", "Unknown"))
        confidence = normalize_confidence(result.get("confidence", 0))

        # Backend confidence = predicted class probability
        if pred.lower() == "survived":
            survival_probability = confidence
        else:
            survival_probability = 1 - confidence

        tone = "positive" if survival_probability >= 0.5 else "risk"

        subline = (
            "Higher likelihood of survival"
            if survival_probability >= 0.5
            else "Lower likelihood of survival"
        )

        st.markdown('<div class="section-kicker">CURRENT ASSESSMENT</div>', unsafe_allow_html=True)

        render_prediction_result(
            title="Estimated survival probability",
            headline=f"{survival_probability:.1%}",  # MUST BE TEXT ONLY
            subline=subline,
            model_name="Engineered XGBoost",
            confidence=confidence,
            tone=tone,
            secondary_label="Model prediction",
            secondary_value=pred,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            render_context_card(
                "Passenger profile",
                "The model evaluated this combination of passenger characteristics.",
                [
                    ("Age", f'{result["payload"]["Age"]:.0f}' if float(result["payload"]["Age"]).is_integer() else str(result["payload"]["Age"])),
                    ("Gender", result["payload"]["Sex"]),
                    ("Passenger class", str(result["payload"]["Pclass"])),
                    ("Ticket fare", f'${result["payload"]["Fare"]:,.2f}'),
                    ("Embarkation", result["payload"]["Embarked"]),
                ],
            )

        with col_b:
            if pred.lower() == "survived":
                render_explanation_card(
                    "Reading the result",
                    "The model assigns more probability to the survival class for this profile.",
                    [
                        "Confidence is the model's certainty in its predicted class.",
                        "A high-confidence output is still a model estimate, not a historical guarantee.",
                        "Change the passenger profile below to examine a different scenario.",
                    ],
                    tone="positive",
                )
            else:
                render_explanation_card(
                    "Reading the result",
                    "The model assigns more probability to the non-survival class for this profile.",
                    [
                        "Confidence is the model's certainty in its predicted class.",
                        "A high-confidence output is still a model estimate, not a historical guarantee.",
                        "Change the passenger profile below to examine a different scenario.",
                    ],
                    tone="risk",
                )

        st.markdown('<div class="section-kicker scenario-label">NEW SCENARIO</div>', unsafe_allow_html=True)

    with st.form("titanic_form", clear_on_submit=False):
        st.markdown("#### Passenger profile")
        st.caption("Adjust the passenger characteristics below and compare the resulting probability.")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            name = st.text_input("Full name & title", value="Smith, Mr. John")
            age = st.slider("Age", 0.0, 100.0, 30.0, step=1.0)
            sex = st.selectbox("Gender", ["male", "female"])
            pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)

        with col2:
            fare = st.slider("Ticket fare ($)", 0.0, 500.0, 15.50, step=0.50)
            sibsp = st.selectbox("Siblings / spouses aboard", [0, 1, 2, 3, 4, 5, 8])
            parch = st.selectbox("Parents / children aboard", [0, 1, 2, 3, 4, 5, 6])
            embarked = st.selectbox("Port of embarkation", ["C", "Q", "S"], index=2)

        submit_button = st.form_submit_button("Run survival estimate", type="primary")

    if submit_button:
        payload = {
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
        }

        with st.spinner("Running prediction…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict/titanic",
                    json=payload,
                    timeout=20,
                )

                if response.status_code == 200:
                    data = response.json()
                    st.session_state["titanic_result"] = {
                        "prediction": data.get("prediction"),
                        "confidence": data.get("confidence"),
                        "payload": payload,
                    }
                    st.rerun()
                else:
                    api_error(response)

            except requests.exceptions.ConnectionError:
                connection_error()
            except requests.exceptions.Timeout:
                st.markdown(
                    textwrap.dedent("""
                    <div class="inline-notice error">
                        <div class="notice-kicker">Request timed out</div>
                        <div class="notice-body">
                            The prediction service took too long to respond.
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------

st.markdown(
    textwrap.dedent("""
    <div class="product-footer">
        <span>Prediction Lab</span>
        <span>Local ML inference</span>
        <span>FastAPI + Streamlit</span>
    </div>
    """),
    unsafe_allow_html=True,
)