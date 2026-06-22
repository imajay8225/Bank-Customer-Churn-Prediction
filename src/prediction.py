"""
prediction.py
-------------
Load the best model and generate predictions on new data.
"""
import pickle
import pandas as pd
from feature_engineering import encode_categoricals, create_features


def load_model(path: str = "models/best_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    df = encode_categoricals(raw_df.copy())
    df = create_features(df)
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)
    result = raw_df.copy()
    result["Churn_Probability"] = probs
    result["Churn_Prediction"]  = preds
    return result
