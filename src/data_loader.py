"""
data_loader.py
Loads the raw credit card transaction dataset.

Expected file: data/raw/creditcard.csv
Download from Kaggle: "Credit Card Fraud Detection" (ULB dataset)
and place it at data/raw/creditcard.csv before running anything else.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and do basic sanity checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'creditcard.csv' from Kaggle (Credit Card Fraud Detection, ULB) "
            "and place it in data/raw/."
        )

    df = pd.read_csv(path)

    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Quick summary stats useful for the EDA notebook / thesis report."""
    fraud_count = int(df["Class"].sum())
    total = len(df)
    return {
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "fraud_percentage": round(fraud_count / total * 100, 4),
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    data = load_raw_data()
    print(dataset_summary(data))
