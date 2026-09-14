"""
explain.py
SHAP-based explainability for the trained model.
This is the module you'll screenshot heavily for your thesis Results chapter.

Run:
    python src/explain.py
"""

import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_raw_data
from preprocess import full_preprocess_pipeline

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def explain_model():
    model = joblib.load(MODELS_DIR / "best_model.pkl")

    df = load_raw_data()
    _, X_test, _, y_test = full_preprocess_pipeline(df)

    # Use a sample for speed - SHAP on tree models is fast, but keep it light
    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Global feature importance
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved global SHAP summary to {FIGURES_DIR / 'shap_summary.png'}")

    # Explain a single flagged transaction (first predicted fraud in sample)
    preds = model.predict(sample)
    fraud_indices = [i for i, p in enumerate(preds) if p == 1]
    if fraud_indices:
        idx = fraud_indices[0]
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            sample.iloc[idx],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_single_transaction.png", dpi=150)
        plt.close()
        print(f"Saved single-transaction SHAP plot to {FIGURES_DIR / 'shap_single_transaction.png'}")
    else:
        print("No fraud predicted in this sample - try a larger sample size.")


if __name__ == "__main__":
    explain_model()
