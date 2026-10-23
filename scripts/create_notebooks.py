"""
create_notebooks.py
====================
Programmatically creates Jupyter notebooks from the eda.py and
modeling pipeline scripts. Run after EDA and training are complete.
"""

import json
import pathlib
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def make_eda_notebook() -> None:
    nb = new_notebook()
    cells = [
        new_markdown_cell("# 01 — Exploratory Data Analysis\n\n"
                          "Salary Prediction using Machine Learning\n\n"
                          "This notebook explores the dataset: distributions, correlations, "
                          "outliers, and key insights."),
        new_code_cell(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path('.').resolve().parent))\n"
            "from scripts.eda import main\n"
            "main()"
        ),
        new_markdown_cell("## Figures\nAll figures are saved to `reports/figures/`."),
        new_code_cell(
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.image as mpimg\n"
            "import pathlib\n\n"
            "figs_dir = pathlib.Path('../reports/figures')\n"
            "fig_names = sorted(figs_dir.glob('*.png'))\n\n"
            "for fp in fig_names:\n"
            "    img = mpimg.imread(fp)\n"
            "    plt.figure(figsize=(10, 6))\n"
            "    plt.imshow(img)\n"
            "    plt.axis('off')\n"
            "    plt.title(fp.stem.replace('_', ' ').title())\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),
    ]
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14.0"},
    }
    path = NB_DIR / "01_eda.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Created: {path}")


def make_modeling_notebook() -> None:
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# 02 — Modeling & Evaluation\n\n"
            "This notebook trains all regression models, evaluates them, "
            "and generates explainability artefacts.\n\n"
            "> **Note**: Run cells sequentially. Training may take 5–10 minutes."
        ),
        new_markdown_cell("## 1. Training"),
        new_code_cell(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path('.').resolve().parent))\n"
            "from src.train import main as train_main\n"
            "train_main()"
        ),
        new_markdown_cell("## 2. Evaluation"),
        new_code_cell(
            "from src.evaluate import main as eval_main\n"
            "eval_main()"
        ),
        new_markdown_cell("## 3. Explainability (SHAP + Permutation Importance)"),
        new_code_cell(
            "from src.explain import main as explain_main\n"
            "explain_main()"
        ),
        new_markdown_cell("## 4. Sample Prediction"),
        new_code_cell(
            "from src.predict import predict_salary, predict_with_interval, format_inr_lakhs, get_test_mae\n\n"
            "sample = {\n"
            "    'Age': 32,\n"
            "    'Gender': 'Female',\n"
            "    'Education_Level': \"Master's\",\n"
            "    'Years_Experience': 9,\n"
            "    'Job_Title': 'Data Scientist',\n"
            "    'Job_Level': 'Senior',\n"
            "    'Industry': 'IT',\n"
            "    'City': 'Bangalore',\n"
            "    'Company_Size': 'Enterprise',\n"
            "    'Work_Mode': 'Hybrid',\n"
            "    'Certifications_Count': 3,\n"
            "    'Skills_Count': 10,\n"
            "}\n\n"
            "mae = get_test_mae()\n"
            "pred, lo, hi = predict_with_interval(sample, mae)\n"
            "print(f'Predicted: {format_inr_lakhs(pred)}')\n"
            "print(f'Range: {format_inr_lakhs(lo)} – {format_inr_lakhs(hi)}')"
        ),
        new_markdown_cell("## 5. Model Comparison Table"),
        new_code_cell(
            "import pandas as pd, pathlib\n"
            "df = pd.read_csv(pathlib.Path('../reports/model_comparison.csv'), index_col=0)\n"
            "df.style.highlight_min(subset=['CV_MAE_mean_INR'], color='lightgreen')"
        ),
    ]
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14.0"},
    }
    path = NB_DIR / "02_modeling.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Created: {path}")


if __name__ == "__main__":
    make_eda_notebook()
    make_modeling_notebook()
    print("Notebooks created.")
