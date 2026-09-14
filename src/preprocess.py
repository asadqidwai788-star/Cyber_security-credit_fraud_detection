"""
preprocess.py
Cleans the data, scales the Amount/Time columns, splits into train/test,
and applies SMOTE to the training set only (never touch the test set with SMOTE
- that would leak synthetic data into evaluation and inflate your results).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Amount and Time; V1-V28 are already PCA-scaled in the source dataset."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def balance_with_smote(X_train, y_train, random_state: int = 42):
    """Oversample the minority (fraud) class in the TRAINING set only."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def full_preprocess_pipeline(df: pd.DataFrame):
    """Convenience wrapper used by train.py"""
    df_scaled = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    X_train_res, y_train_res = balance_with_smote(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test


if __name__ == "__main__":
    from data_loader import load_raw_data

    df = load_raw_data()
    X_train, X_test, y_train, y_test = full_preprocess_pipeline(df)
    print("Train shape after SMOTE:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train class balance after SMOTE:\n", y_train.value_counts())
