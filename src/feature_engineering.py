"""
feature_engineering.py
-----------------------
Encodes categoricals and creates derived features.
"""
import pandas as pd


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df["BalanceSalaryRatio"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["TenureByAge"]          = df["Tenure"]  / (df["Age"] + 1)
    df["CreditScoreGivenAge"]  = df["CreditScore"] / (df["Age"] + 1)
    return df
