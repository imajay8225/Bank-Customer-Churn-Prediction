"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and splitting raw bank data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame, target: str = "Exited", test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
