"""
app.py
======
Streamlit web application for the Salary Prediction model.

Features:
  - Input form with dropdowns and sliders for all 12 features
  - Predicted salary displayed in Indian lakh format with an empirical error band
   - Readable factor comparison for the current prediction
  - Profile validation alerts (e.g. experience vs age checks)
  - Loads saved joblib pipeline — does NOT retrain

Run with:
    streamlit run app/app.py
"""

import json
import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interpretation import (
    counterfactual_impacts,
    local_shap_factors,
    typical_profile,
)
from src.preprocess import (
    CITY_VALUES,
    COMPANY_SIZE_VALUES,
    DATA_PATH,
    EDUCATION_ORDER,
    GENDER_VALUES,
    INDUSTRY_VALUES,
    JOB_TITLE_VALUES,
    JOB_LEVEL_ORDER,
    WORK_MODE_VALUES,
    data_sha256,
    file_sha256,
    load_data,
)

st.set_page_config(
    page_title="Salary / Predictor",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.predict import (
    format_inr_lakhs,
    predict_with_interval,
    validate_profile,
)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ── Allowed values (shared with preprocessing and validation) ────────────────
EDUCATION_LEVELS = EDUCATION_ORDER
JOB_LEVELS = JOB_LEVEL_ORDER
JOB_TITLES = JOB_TITLE_VALUES
INDUSTRIES = INDUSTRY_VALUES
CITIES = CITY_VALUES
COMPANY_SIZES = COMPANY_SIZE_VALUES
WORK_MODES = WORK_MODE_VALUES
GENDERS = GENDER_VALUES


# ── Loaders (cached) ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "best_model.joblib")

@st.cache_resource
def load_meta():
    with open(MODELS_DIR / "training_meta.json") as f:
        return json.load(f)

@st.cache_resource
def load_metrics():
    metrics_path = MODELS_DIR / "best_model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_feature_names():
    feat_path = MODELS_DIR / "feature_names.json"
    if feat_path.exists():
        with open(feat_path) as f:
            return json.load(f)
    return []


@st.cache_data
def load_reference_profile():
    return typical_profile(load_data(DATA_PATH))

@st.cache_resource
def load_shap_explainer():
    expl_path = MODELS_DIR / "shap_explainer.joblib"
    meta_path = MODELS_DIR / "explainability_meta.json"
    if expl_path.exists():
        try:
            explainer = joblib.load(expl_path)
            if not meta_path.exists():
                return None
            with open(meta_path) as f:
                explain_meta = json.load(f)
            if explain_meta.get("data_sha256") != data_sha256(DATA_PATH):
                return None
            if explain_meta.get("model_sha256") != file_sha256(MODELS_DIR / "best_model.joblib"):
                return None
            return explainer
        except Exception as exc:
            st.warning(f"SHAP explainer could not be loaded: {exc}")
            return None
    return None


# ── SHAP helpers ─────────────────────────────────────────────────────────────
def get_shap_values_for_input(estimator, explainer, input_dict: dict, feature_names: list):
    """Compute SHAP values for a single input."""
    if explainer is None:
        return None, None

    input_df = pd.DataFrame([input_dict])
    preprocess_pipe = estimator.named_steps["preprocess"]
    X_transformed = preprocess_pipe.transform(input_df)
    if X_transformed.shape[1] != len(feature_names):
        st.warning("The explanation artifacts do not match the loaded model.")
        return None, None

    try:
        sv = explainer(X_transformed, check_additivity=False)
        vals = sv.values[0] if hasattr(sv, "values") else sv[0]
        return vals, feature_names
    except (TypeError, AttributeError, ValueError):
        try:
            vals = explainer.shap_values(X_transformed)
            if isinstance(vals, list):
                vals = vals[0]
            return vals[0], feature_names
        except Exception as exc:
            st.warning(f"SHAP values could not be computed: {exc}")
            return None, None


def plot_factor_bar(factors: pd.DataFrame, is_dark: bool = False) -> plt.Figure:
    """Render a clean INR comparison against a typical training profile."""
    factors = factors.sort_values("impact_inr")
    labels = [f"{row.label} · {row.value}" for row in factors.itertuples()]
    values = factors["impact_inr"].to_numpy() / 1e5

    if is_dark:
        bg_color = "#142420"
        text_color = "#F0F5F2"
        label_color = "#D4E4DE"
        axis_color = "#ADC2B9"
        grid_color = "#253C35"
        spine_color = "#2A433C"
        zero_line_color = "#5A776E"
        colors = ["#4ECDC4" if value >= 0 else "#F87171" for value in values]
    else:
        bg_color = "#F7F5F0"
        text_color = "#1F2A2E"
        label_color = "#34423D"
        axis_color = "#52615C"
        grid_color = "#D9DDD7"
        spine_color = "#B8C1BB"
        zero_line_color = "#8A9690"
        colors = ["#2C6E63" if value >= 0 else "#C56A54" for value in values]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.barh(labels, values, color=colors, height=0.58)
    ax.axvline(0, color=zero_line_color, linewidth=0.9)
    ax.set_xlabel("Change in estimate versus a typical profile (₹ lakhs)", color=axis_color, fontsize=9.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"₹{value:.1f}L"))
    ax.tick_params(axis="both", colors=label_color, labelsize=9)
    ax.grid(axis="x", color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(spine_color)
    ax.set_title("What changes this estimate?", loc="left", color=text_color, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Light Palette (Default) ─────────────────────────────────────────── */
    :root {
        --ink:            #1F2A2E;
        --muted:          #556861;
        --paper:          #F7F5F0;
        --line:           #D9DDD7;
        --teal:           #2C6E63;
        --teal-hover:     #22564D;
        --surface:        rgba(255, 255, 255, 0.65);
        --input-bg:       #FFFFFF;
        --input-border:   #D0D7D3;
        --metric-bg:      #FFFFFF;
        --card-bg:        #FFFFFF;
        --card-border:    #E0E6E2;
        --card-title:     #235A52;
        --card-text:      #455751;
        --card-shadow:    0 2px 10px rgba(0, 0, 0, 0.05);
        --info-bg:        #E8F4F0;
        --info-border:    #BCE0D7;
        --header-bg:      transparent;
    }

    /* ── Dark Mode Overrides (Streamlit theme & OS preference) ───────────── */
    @media (prefers-color-scheme: dark) {
        :root {
            --ink:            #F0F5F2;
            --muted:          #ADC2B9;
            --paper:          #0D1714;
            --line:           #263B34;
            --teal:           #4ECDC4;
            --teal-hover:     #3DBDB4;
            --surface:        rgba(20, 36, 32, 0.65);
            --input-bg:       #152521;
            --input-border:   #2B433C;
            --metric-bg:      #142420;
            --card-bg:        #142420;
            --card-border:    #253C35;
            --card-title:     #5EEAD4;
            --card-text:      #D5E4DE;
            --card-shadow:    0 4px 18px rgba(0, 0, 0, 0.28);
            --info-bg:        #132A24;
            --info-border:    #224D42;
            --header-bg:      #0D1714;
        }
    }

    [data-theme="dark"] {
        --ink:            #F0F5F2;
        --muted:          #ADC2B9;
        --paper:          #0D1714;
        --line:           #263B34;
        --teal:           #4ECDC4;
        --teal-hover:     #3DBDB4;
        --surface:        rgba(20, 36, 32, 0.65);
        --input-bg:       #152521;
        --input-border:   #2B433C;
        --metric-bg:      #142420;
        --card-bg:        #142420;
        --card-border:    #253C35;
        --card-title:     #5EEAD4;
        --card-text:      #D5E4DE;
        --card-shadow:    0 4px 18px rgba(0, 0, 0, 0.28);
        --info-bg:        #132A24;
        --info-border:    #224D42;
        --header-bg:      #0D1714;
    }

    /* ── Base App Layout & Typography ────────────────────────────────────── */
    .stApp {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
    }
    [data-testid="stHeader"] {
        background: var(--header-bg) !important;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.8rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--ink) !important;
    }
    .eyebrow {
        color: var(--teal) !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .intro {
        border-bottom: 1px solid var(--line);
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }
    .intro h1 {
        color: var(--ink) !important;
        font-size: 2.7rem;
        letter-spacing: -0.04em;
        margin: 0.35rem 0 0.5rem;
    }
    .intro p {
        color: var(--muted) !important;
        font-size: 1.02rem;
        max-width: 44rem;
        margin: 0;
        line-height: 1.55;
    }
    .section-note {
        color: var(--muted) !important;
        font-size: 0.88rem;
        margin-top: -0.35rem;
        margin-bottom: 0.9rem;
    }
    .fine-print {
        color: var(--muted) !important;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    /* ── Use-Case Cards Grid (Clean layout, no vertical sticking) ────────── */
    .use-cases-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.15rem;
        margin-top: 0.8rem;
        margin-bottom: 2.2rem;
    }
    .use-case-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 1.15rem 1.1rem;
        box-shadow: var(--card-shadow);
        display: flex;
        flex-direction: column;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .use-case-card:hover {
        transform: translateY(-2px);
        border-color: var(--teal);
    }
    .use-case-card .card-icon {
        font-size: 1.75rem;
        line-height: 1;
        margin-bottom: 0.55rem;
        display: block;
    }
    .use-case-card .card-title {
        color: var(--card-title) !important;
        font-weight: 700;
        font-size: 0.94rem;
        margin-bottom: 0.35rem;
        line-height: 1.35;
    }
    .use-case-card .card-body {
        color: var(--card-text) !important;
        font-size: 0.83rem;
        line-height: 1.48;
        flex-grow: 1;
    }

    /* ── Form Container ──────────────────────────────────────────────────── */
    div[data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        background: var(--surface);
        box-shadow: var(--card-shadow);
    }

    /* ── Form Labels (Fixes Invisible Black-on-Dark Labels) ──────────────── */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: var(--ink) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }

    /* ── Selectbox / Dropdowns (Fixes Stark White Boxes) ─────────────────── */
    div[data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 9px !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: var(--ink) !important;
    }
    div[data-baseweb="select"] svg {
        fill: var(--ink) !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"],
    ul[role="listbox"],
    div[data-baseweb="popover"] > div {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
    }
    ul[role="listbox"] li,
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: var(--ink) !important;
    }

    /* ── Sliders (Fixes Invisible Values & Ticks) ────────────────────────── */
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: var(--ink) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSliderTickBar"] span {
        color: var(--muted) !important;
    }

    /* ── Submit Button ───────────────────────────────────────────────────── */
    .stButton > button {
        background: var(--teal) !important;
        border: 0 !important;
        border-radius: 9px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        transition: background 0.15s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        background: var(--teal-hover) !important;
        transform: translateY(-1px);
    }

    /* ── Metric Card ─────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--metric-bg) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.25rem !important;
        box-shadow: var(--card-shadow);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--teal) !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }

    /* ── Info / Notification Banner ──────────────────────────────────────── */
    [data-testid="stNotification"] {
        background: var(--info-bg) !important;
        border: 1px solid var(--info-border) !important;
        border-left: 4px solid var(--teal) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.1rem !important;
    }
    [data-testid="stNotification"] p,
    [data-testid="stNotification"] div {
        color: var(--ink) !important;
        font-size: 0.92rem !important;
    }

    /* ── Expander ────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--ink) !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: var(--ink) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
try:
    estimator = load_model()
    meta = load_meta()
    metrics = load_metrics()
    feature_names = load_feature_names()
    explainer = load_shap_explainer()
except Exception as exc:
    st.error(
        "The model artifacts could not be loaded. Run the training and evaluation "
        f"steps first. Details: {exc}"
    )
    st.stop()

best_model_name = meta.get("best_model", "Ridge")
if meta.get("data_sha256") and meta["data_sha256"] != data_sha256(DATA_PATH):
    st.error("The dataset changed after this model was trained. Retrain the model before use.")
    st.stop()
if meta.get("model_sha256") and meta["model_sha256"] != file_sha256(MODELS_DIR / "best_model.joblib"):
    st.error("The model artifact does not match its metadata. Regenerate the model artifacts.")
    st.stop()
if (
    metrics.get("best_model") != best_model_name
    or metrics.get("data_sha256") != meta.get("data_sha256")
    or metrics.get("model_sha256") != meta.get("model_sha256")
):
    metrics = {}
best_cv_info = meta.get("cv_scores", {}).get(best_model_name, {})
test_mae = metrics.get("test_mae_inr", best_cv_info.get("test_mae_inr"))
error_band = metrics.get("error_band_inr", meta.get("error_band_inr"))
test_r2 = metrics.get("test_r2", best_cv_info.get("test_r2", None))
test_rmse = metrics.get("test_rmse_inr", best_cv_info.get("test_rmse_inr", None))
if test_mae is None:
    st.error("No matching error metric is available for the loaded model.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="intro"><div class="eyebrow">Salary / Predictor</div>', unsafe_allow_html=True)
st.title("A calmer way to estimate salary")
st.markdown(
    "Use the profile below to explore a base-salary estimate from the project’s "
    "training data. The result is a guide, not a salary benchmark."
)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("About this model"):
    model_col, data_col = st.columns(2)
    with model_col:
        st.markdown(f"**Selected model:** {best_model_name}")
        if test_r2 is not None:
            st.markdown(f"**Held-out R²:** {test_r2:.4f}")
        st.markdown(f"**Held-out MAE:** {format_inr_lakhs(test_mae)}")
    with data_col:
        st.markdown("**Training data:** 500 synthetic employee profiles")
        st.markdown("**Target:** base annual salary in INR")
        st.markdown("The model excludes bonuses, equity, and market changes.")

# ── Real-World Use Cases ───────────────────────────────────────────────────────
st.subheader("📌 Real-World Use Cases")
st.markdown(
    "<div class='section-note'>Who this tool is built for — and how each persona uses the simulation.</div>",
    unsafe_allow_html=True,
)

_USE_CASES = [
    (
        "🎓",
        "Job Seeker / Fresher",
        "Enter your profile to evaluate whether an offer is competitive for your role, city, and experience level.",
    ),
    (
        "👩‍💼",
        "HR / Hiring Manager",
        "Benchmark competitive compensation for any role and location, calibrated by company tier.",
    ),
    (
        "📚",
        "Career Counsellor",
        "Simulate education and promotion moves to quantify the expected salary increment in INR.",
    ),
    (
        "📊",
        "Compensation Analyst",
        "Compare pay trends across sectors, tiers, and work arrangements for identical job roles.",
    ),
    (
        "🏫",
        "Student / Educator",
        "Full ML pipeline simulation: feature engineering, 5-model cross-validation, and SHAP explainability.",
    ),
]

cards_html = '<div class="use-cases-grid">'
for _icon, _title, _body in _USE_CASES:
    cards_html += f"""
    <div class="use-case-card">
        <span class="card-icon">{_icon}</span>
        <div class="card-title">{_title}</div>
        <div class="card-body">{_body}</div>
    </div>"""
cards_html += "</div>"

st.markdown(cards_html, unsafe_allow_html=True)

st.subheader("Build a profile")
st.markdown("<div class='section-note'>All fields are used by the trained model.</div>", unsafe_allow_html=True)

with st.form("salary_profile"):
    personal_col, role_col = st.columns(2, gap="large")
    with personal_col:
        st.markdown("**Personal profile**")
        age = st.slider("Age", min_value=21, max_value=57, value=30, step=1)
        gender = st.selectbox("Gender", GENDERS, index=0)
        education = st.selectbox("Education level", EDUCATION_LEVELS, index=1)
        years_exp = st.slider("Years of experience", 0, 35, 7, 1)
    with role_col:
        st.markdown("**Role and workplace**")
        job_title = st.selectbox("Job title", JOB_TITLES, index=0)
        job_level = st.selectbox("Job level", JOB_LEVELS, index=1)
        industry = st.selectbox("Industry", INDUSTRIES, index=0)
        city = st.selectbox("Location", CITIES, index=0)
        company_size = st.selectbox("Company size", COMPANY_SIZES, index=1)
        work_mode = st.selectbox("Work mode", WORK_MODES, index=1)

    skills_col, action_col = st.columns([2, 1], gap="large")
    with skills_col:
        st.markdown("**Skills and credentials**")
        certs = st.slider("Certifications", 0, 5, 2, 1)
        skills = st.slider("Skills", 2, 14, 8, 1)
    with action_col:
        st.markdown("**Ready?**")
        st.markdown("<div class='fine-print'>Run the estimate when your profile is complete.</div>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button("Estimate salary", type="primary", width="stretch")

if predict_btn:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Education_Level": education,
        "Years_Experience": years_exp,
        "Job_Title": job_title,
        "Job_Level": job_level,
        "Industry": industry,
        "City": city,
        "Company_Size": company_size,
        "Work_Mode": work_mode,
        "Certifications_Count": certs,
        "Skills_Count": skills,
    }

    # Profile validation
    validation_errors = validate_profile(input_dict)
    if validation_errors:
        for error in validation_errors:
            st.error(f"⚠️ {error}")
        st.stop()

    with st.spinner("Calculating …"):
        pred, lower, upper = predict_with_interval(input_dict, error_band)
        shap_vals, feat_names = get_shap_values_for_input(
            estimator, explainer, input_dict, feature_names
        )
        factors = counterfactual_impacts(
            estimator,
            input_dict,
            load_reference_profile(),
        ).head(6)

    # ── Result and explanation ────────────────────────────────────────────────
    res_col, exp_col = st.columns([0.85, 1.35], gap="large")

    with res_col:
        st.markdown("**Estimated base salary**")
        st.metric("Annual estimate", format_inr_lakhs(pred))
        st.caption(f"Typical error band: {format_inr_lakhs(lower)} – {format_inr_lakhs(upper)}")
        st.caption("This is an empirical band, not a guaranteed prediction interval.")
        st.markdown("**Profile summary**")
        summary_data = pd.DataFrame(
            {
                "Field": ["Role", "Level", "Experience", "Location", "Education"],
                "Value": [job_title, job_level, f"{years_exp} yrs", city, education],
            }
        )
        st.dataframe(summary_data, hide_index=True, width="stretch")

    with exp_col:
        st.markdown("**What changes this estimate?**")
        is_dark_mode = False
        try:
            if hasattr(st, "context") and hasattr(st.context, "theme"):
                is_dark_mode = st.context.theme.get("type") == "dark"
        except Exception:
            is_dark_mode = False
        fig = plot_factor_bar(factors, is_dark=is_dark_mode)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(
            "Each bar compares this profile with a typical training profile while changing one "
            "field at a time. It describes the model, not a causal salary effect."
        )
        if shap_vals is not None and feat_names:
            with st.expander("Technical SHAP view"):
                technical = local_shap_factors(shap_vals, feat_names, input_dict)
                technical["value"] = technical["value"].astype(str)
                st.dataframe(
                    technical[["label", "value", "impact"]].rename(
                        columns={"label": "Factor", "value": "Selected value", "impact": "Model-score impact"}
                    ),
                    hide_index=True,
                    width="stretch",
                )

else:
    st.info("Complete the profile above and select **Estimate salary** to see the result.")

    # Show model comparison from reports if available
    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        st.markdown("---")
        st.markdown("### Model performance")
        comp_df = pd.read_csv(comp_path, index_col=0)
        display_df = comp_df.copy()
        for col in display_df.columns:
            if "INR" in col:
                display_df[col] = display_df[col].apply(
                    lambda x: format_inr_lakhs(x) if pd.notna(x) else "—"
                )
            elif "R2" in col:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        st.dataframe(display_df)
        st.caption(f"{best_model_name} was selected using 5-fold cross-validation MAE.")
