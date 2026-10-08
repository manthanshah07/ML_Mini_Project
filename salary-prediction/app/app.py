"""
app.py
======
Streamlit web application for the Salary Prediction model.

Features:
  - Input form with dropdowns and sliders for all 12 features
  - Predicted salary displayed in Indian lakh format with approx. range
  - SHAP top-factors bar chart for the current prediction
  - Loads saved joblib pipeline — does NOT retrain

Run with:
    streamlit run app/app.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ── Allowed values (from dataset spec) ───────────────────────────────────────
EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
JOB_LEVELS = ["Junior", "Mid", "Senior", "Lead", "Manager"]
JOB_TITLES = [
    "Software Engineer", "Data Scientist", "Data Analyst", "DevOps Engineer",
    "Product Manager", "Business Analyst", "UI/UX Designer", "HR Executive",
    "Marketing Manager", "Sales Executive",
]
INDUSTRIES = ["IT", "Finance", "Healthcare", "EdTech", "E-commerce", "Manufacturing"]
CITIES = ["Bangalore", "Delhi", "Chennai", "Hyderabad", "Pune", "Mumbai", "Remote"]
COMPANY_SIZES = ["Startup", "Mid-size", "Enterprise"]
WORK_MODES = ["On-site", "Hybrid", "Remote"]
GENDERS = ["Male", "Female"]


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

@st.cache_resource
def load_shap_explainer():
    expl_path = MODELS_DIR / "shap_explainer.joblib"
    if expl_path.exists():
        return joblib.load(expl_path)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_inr_lakhs(amount: float) -> str:
    """Format amount in Indian lakh notation."""
    lakhs = amount / 1e5
    if lakhs >= 100:
        return f"₹{lakhs/100:.2f} Cr"
    return f"₹{lakhs:.2f} L"


def predict_salary(estimator, meta: dict, input_dict: dict) -> float:
    """Run inference. Returns salary in INR."""
    input_df = pd.DataFrame([input_dict])
    raw = estimator.predict(input_df)
    if meta.get("log_target", False):
        return float(np.expm1(raw[0]))
    return float(raw[0])


def get_shap_values_for_input(estimator, explainer, input_dict: dict, meta: dict, feature_names: list):
    """Compute SHAP values for a single input."""
    if explainer is None:
        return None, None

    # Transform through the preprocessing pipeline
    input_df = pd.DataFrame([input_dict])
    preprocess_pipe = estimator.named_steps["preprocess"]
    X_transformed = preprocess_pipe.transform(input_df)

    try:
        sv = explainer(X_transformed, check_additivity=False)
        vals = sv.values[0] if hasattr(sv, "values") else sv[0]
        return vals, feature_names
    except Exception:
        try:
            vals = explainer.shap_values(X_transformed)
            if isinstance(vals, list):
                vals = vals[0]
            return vals[0], feature_names
        except Exception:
            return None, None


def plot_shap_bar(shap_vals: np.ndarray, feature_names: list, pred_salary: float) -> plt.Figure:
    """Top-10 SHAP feature bar chart for a single prediction."""
    n = min(10, len(shap_vals))
    abs_vals = np.abs(shap_vals)
    top_idx = np.argsort(abs_vals)[::-1][:n]

    top_names = [feature_names[i] for i in top_idx]
    top_vals = [shap_vals[i] for i in top_idx]

    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top_vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    bars = ax.barh(
        range(n),
        top_vals,
        color=colors,
        edgecolor="none",
        height=0.65,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_names, fontsize=10, color="white")
    ax.set_xlabel("SHAP value (impact on salary prediction)", fontsize=9, color="#AAAAAA")
    ax.set_title("Top Factors Influencing This Prediction", fontsize=11, color="white", pad=10)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.axvline(0, color="#555555", linewidth=0.8)

    pos_patch = mpatches.Patch(color="#C44E52", label="↑ Pushes salary higher")
    neg_patch = mpatches.Patch(color="#4C72B0", label="↓ Pushes salary lower")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, facecolor="#1A1A2E", labelcolor="white",
              loc="lower right")

    plt.tight_layout()
    return fig


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor — India",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a4a;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    .main-header p { color: #AAAACC; font-size: 0.95rem; margin: 0; }

    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .salary-number {
        font-size: 3.2rem;
        font-weight: 700;
        color: #e94560;
        line-height: 1.1;
    }
    .salary-range {
        font-size: 1rem;
        color: #8888AA;
        margin-top: 0.4rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .metric-pill {
        background: #0f3460;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #CCCCEE;
        font-size: 0.85rem;
        flex: 1;
        text-align: center;
    }
    .disclaimer {
        background: #1A1A2E;
        border-left: 3px solid #e94560;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        color: #AAAAAA;
        font-size: 0.82rem;
        margin-top: 1rem;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8888AA;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stSlider"] > div > div > div { color: #e94560; }
    .stSelectbox label, .stSlider label { color: #CCCCEE !important; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
estimator = load_model()
meta = load_meta()
metrics = load_metrics()
feature_names = load_feature_names()
explainer = load_shap_explainer()

best_model_name = meta.get("best_model", "Best Model")
test_mae = metrics.get("test_mae_inr", meta["cv_scores"][best_model_name]["cv_mae_inr"])
test_r2 = metrics.get("test_r2", None)
test_rmse = metrics.get("test_rmse_inr", None)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💼 Salary Predictor — India</h1>
    <p>Estimate your annual salary in INR based on your profile using machine learning.
    Fill in the form below and hit <strong>Predict</strong>.</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Info")
    st.markdown(f"**Best model:** `{best_model_name}`")
    if test_r2:
        st.markdown(f"**Test R²:** `{test_r2:.4f}`")
    if test_rmse:
        st.markdown(f"**Test RMSE:** `{format_inr_lakhs(test_rmse)}`")
    st.markdown(f"**Test MAE:** `{format_inr_lakhs(test_mae)}`")
    st.divider()
    st.markdown("### 📊 Dataset")
    st.markdown(
        "500 employees · 12 features · synthetic data for learning purposes."
    )
    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This is an educational ML project built on a small (500-row) synthetic dataset. "
        "Predictions should NOT be used for actual salary negotiations. "
        "The model does not account for bonuses, equity, or cost-of-living adjustments."
    )


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Your Profile")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=21, max_value=57, value=30, step=1)
    gender = st.selectbox("Gender", GENDERS, index=0)
    education = st.selectbox("Education Level", EDUCATION_LEVELS, index=1)
    years_exp = st.slider("Years of Experience", 0, 35, 7, 1)

with col2:
    job_title = st.selectbox("Job Title", JOB_TITLES, index=0)
    job_level = st.selectbox("Job Level", JOB_LEVELS, index=1)
    industry = st.selectbox("Industry", INDUSTRIES, index=0)
    city = st.selectbox("City", CITIES, index=0)

with col3:
    company_size = st.selectbox("Company Size", COMPANY_SIZES, index=1)
    work_mode = st.selectbox("Work Mode", WORK_MODES, index=1)
    certs = st.slider("Certifications Count", 0, 5, 2, 1)
    skills = st.slider("Skills Count", 1, 14, 8, 1)


# ── Predict button ─────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("🔮 Predict Salary", use_container_width=True, type="primary")

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

    with st.spinner("Predicting …"):
        pred = predict_salary(estimator, meta, input_dict)
        lower = max(0, pred - test_mae)
        upper = pred + test_mae

        shap_vals, feat_names = get_shap_values_for_input(
            estimator, explainer, input_dict, meta, feature_names
        )

    # ── Result card ───────────────────────────────────────────────────────────
    res_col, exp_col = st.columns([1, 1.3], gap="large")

    with res_col:
        st.markdown(f"""
        <div class="result-card">
            <div class="section-label">Predicted Annual Salary</div>
            <div class="salary-number">{format_inr_lakhs(pred)}</div>
            <div class="salary-range">
                Approx. range: {format_inr_lakhs(lower)} – {format_inr_lakhs(upper)}
            </div>
            <div class="metric-row">
                <div class="metric-pill">₹ {pred:,.0f} / year</div>
                <div class="metric-pill">± {format_inr_lakhs(test_mae)} (1× MAE)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            💡 Range is based on the model's test MAE. This is an educational estimate
            from 500 synthetic rows — not a formal salary benchmark.
        </div>
        """, unsafe_allow_html=True)

        # Key inputs summary
        st.markdown("**Your profile summary:**")
        summary_data = {
            "Field": ["Job Title", "Level", "Experience", "City", "Education"],
            "Value": [job_title, job_level, f"{years_exp} yrs", city, education],
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    with exp_col:
        st.markdown("### 🔍 What Drives This Prediction?")
        if shap_vals is not None and feat_names:
            fig = plot_shap_bar(shap_vals, feat_names, pred)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Red bars push the salary **higher** than the dataset average; "
                "blue bars push it **lower**. Bar length = magnitude of impact."
            )
        else:
            st.info("SHAP explainer not available. Run `python -m src.explain` first.")

else:
    st.info("👆 Fill in your profile above and click **Predict Salary** to get your estimate.")

    # Show model comparison from reports if available
    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        st.markdown("---")
        st.markdown("### 📈 Model Performance Summary")
        comp_df = pd.read_csv(comp_path, index_col=0)
        # Format for display
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
        st.dataframe(display_df, use_container_width=True)
        st.caption(
            f"✓ **{best_model_name}** selected as best model by 5-fold CV MAE (not by test score)."
        )
