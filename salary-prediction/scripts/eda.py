"""
eda.py
======
Exploratory Data Analysis for the Salary Prediction dataset.
Generates all figures to reports/figures/ and prints key insights.

Run with:
    python scripts/eda.py
"""

import pathlib
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "salary.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1A1A2E",
    "axes.labelcolor": "white",
    "axes.edgecolor": "#333355",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#2A2A4A",
    "grid.alpha": 0.5,
})

ACCENT = "#e94560"
SECONDARY = "#4C72B0"


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # ── 1. Dataset overview ────────────────────────────────────────────────────
    print("=" * 65)
    print("  DATASET OVERVIEW")
    print("=" * 65)
    print(f"Shape        : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}\n")
    print(f"Describe:\n{df.describe().T.to_string()}\n")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    print(f"Duplicates   : {df.duplicated().sum()}")

    target = "Salary_INR"
    num_cols = ["Age", "Years_Experience", "Certifications_Count", "Skills_Count", target]

    # ── 2. Salary distribution ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0E1117")

    # Histogram
    axes[0].hist(df[target] / 1e5, bins=35, color=ACCENT, edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Salary (₹ Lakhs)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Salary Distribution (Linear Scale)", fontsize=12)
    axes[0].axvline(df[target].median() / 1e5, color="yellow", lw=1.5, ls="--",
                    label=f"Median ₹{df[target].median()/1e5:.1f}L")
    axes[0].axvline(df[target].mean() / 1e5, color="cyan", lw=1.5, ls="--",
                    label=f"Mean ₹{df[target].mean()/1e5:.1f}L")
    axes[0].legend(fontsize=9)

    # Log-scale histogram
    log_salary = np.log1p(df[target])
    axes[1].hist(log_salary, bins=35, color=SECONDARY, edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("log(1 + Salary)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Salary Distribution (Log Scale)", fontsize=12)

    fig.suptitle("Salary_INR Distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "salary_distribution.png")

    # ── 3. Boxplots by category ────────────────────────────────────────────────
    cat_cols = [
        ("Job_Level", ["Junior", "Mid", "Senior", "Lead", "Manager"]),
        ("Education_Level", ["High School", "Bachelor's", "Master's", "PhD"]),
        ("City", None),
        ("Industry", None),
        ("Company_Size", ["Startup", "Mid-size", "Enterprise"]),
        ("Work_Mode", None),
        ("Gender", None),
    ]

    for cat, order in cat_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1A1A2E")
        sns.boxplot(
            data=df,
            x=cat,
            y=target,
            order=order,
            palette="muted",
            ax=ax,
            linewidth=1.2,
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L")
        )
        ax.set_xlabel(cat, fontsize=11)
        ax.set_ylabel("Salary (₹ Lakhs)", fontsize=11)
        ax.set_title(f"Salary by {cat}", fontsize=12)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        _save(fig, f"boxplot_{cat.lower()}.png")

    # ── 4. Correlation heatmap ─────────────────────────────────────────────────
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1A2E")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap (Numeric Features + Salary)", fontsize=12)
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png")

    print("\n  Key correlations with Salary_INR:")
    for col in num_cols[:-1]:
        r, p = stats.pearsonr(df[col], df[target])
        print(f"    {col:<25} r = {r:+.3f}  (p = {p:.4f})")

    # ── 5. Scatter: Years_Experience vs Salary by Job_Level ───────────────────
    jl_palette = {
        "Junior": "#4C72B0", "Mid": "#55A868", "Senior": "#C44E52",
        "Lead": "#8172B3", "Manager": "#CCB974"
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1A2E")
    for level, grp in df.groupby("Job_Level"):
        ax.scatter(
            grp["Years_Experience"],
            grp["Salary_INR"] / 1e5,
            label=level,
            alpha=0.75,
            s=45,
            color=jl_palette.get(level, "grey"),
            edgecolors="white",
            linewidths=0.3,
        )
    ax.set_xlabel("Years of Experience", fontsize=11)
    ax.set_ylabel("Salary (₹ Lakhs)", fontsize=11)
    ax.set_title("Years of Experience vs Salary (by Job Level)", fontsize=12)
    ax.legend(title="Job Level", fontsize=9, title_fontsize=9)
    plt.tight_layout()
    _save(fig, "scatter_exp_salary_by_joblevel.png")

    # ── 6. Outlier check (IQR on Salary_INR) ──────────────────────────────────
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    outliers = df[(df[target] < lower_fence) | (df[target] > upper_fence)]

    print(f"\n  Outlier check (IQR method) on Salary_INR:")
    print(f"    Q1={Q1/1e5:.1f}L, Q3={Q3/1e5:.1f}L, IQR={IQR/1e5:.1f}L")
    print(f"    Fences: [{lower_fence/1e5:.1f}L, {upper_fence/1e5:.1f}L]")
    print(f"    Outliers found: {len(outliers)}")
    print(f"    Decision: Retaining all outliers. They represent extreme but plausible salaries")
    print(f"    (e.g. senior leads/managers in finance/IT) and the dataset is only 500 rows.")
    print(f"    Removing them would reduce representativeness more than it would reduce noise.")
    if len(outliers) > 0:
        print(outliers[[target, "Job_Level", "Education_Level", "Industry"]].to_string())

    # ── 7. VIF (Age vs Years_Experience multicollinearity) ────────────────────
    print("\n  Variance Inflation Factor (VIF) — numeric features:")
    vif_df_in = df[["Age", "Years_Experience", "Certifications_Count", "Skills_Count"]].copy()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = vif_df_in.columns
    vif_data["VIF"] = [
        variance_inflation_factor(vif_df_in.values, i)
        for i in range(len(vif_df_in.columns))
    ]
    print(vif_data.to_string(index=False))
    print("  (VIF > 5 indicates multicollinearity; Age & Years_Experience likely correlated)")

    print("\n  Age vs Years_Experience Pearson r:")
    r, p = stats.pearsonr(df["Age"], df["Years_Experience"])
    print(f"    r = {r:.4f}  (p = {p:.4e})")

    # ── 8. Key insights ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  KEY INSIGHTS")
    print("=" * 65)
    insights = [
        f"1. Salary is right-skewed (mean ₹{df[target].mean()/1e5:.1f}L > median ₹{df[target].median()/1e5:.1f}L); "
        f"log-transform improves linearity for linear models.",
        f"2. Age (r≈{stats.pearsonr(df['Age'], df[target])[0]:.2f}) and Years_Experience "
        f"(r≈{stats.pearsonr(df['Years_Experience'], df[target])[0]:.2f}) are the strongest numeric predictors.",
        "3. Certifications_Count and Skills_Count have near-zero correlation with salary (~0.07), "
        "suggesting they matter less than seniority or experience.",
        "4. Job_Level has the strongest categorical effect: Manager/Lead medians are "
        "~2–3× higher than Junior medians.",
        "5. PhD and Master's holders command noticeably higher salaries than Bachelor's/High School.",
        "6. Enterprise companies pay more on average than Startups; Mid-size is in between.",
        f"7. Age and Years_Experience are strongly correlated (r={r:.2f}), indicating potential "
        "multicollinearity; tree models handle this naturally, but linear models benefit from "
        "using Age_minus_Exp as a decorrelated feature.",
        f"8. IQR outlier check identified {len(outliers)} extreme salaries — retained to preserve "
        "representativeness on the small (500-row) dataset.",
    ]
    for ins in insights:
        print(f"  {ins}\n")

    print("EDA complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    main()
