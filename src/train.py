"""
train.py
Trains baseline and advanced models, saves the best one to models/best_model.pkl.

Run:
    python src/train.py
"""

import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_raw_data
from preprocess import full_preprocess_pipeline
from evaluate import evaluate_model

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="aucpr",
            random_state=42,
        ),
    }


def train_and_select_best():
    print("Loading data...")
    df = load_raw_data()

    print("Preprocessing (scaling + SMOTE on training set)...")
    X_train, X_test, y_train, y_test = full_preprocess_pipeline(df)

    results = {}
    best_model = None
    best_score = -1
    best_name = None

    for name, model in get_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results[name] = metrics

        # Selecting best model by AUPRC (the right metric for imbalanced fraud data)
        if metrics["auprc"] > best_score:
            best_score = metrics["auprc"]
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (AUPRC = {best_score:.4f})")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    print(f"Saved to {MODELS_DIR / 'best_model.pkl'}")

    return results, best_name


if __name__ == "__main__":
    train_and_select_best()
