"""
model_evaluation.py
-------------------
Evaluates trained models and selects the best one.
"""
import pickle
import numpy  as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)


def evaluate(model, X_test, y_test, name: str = "") -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "name":      name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    print(classification_report(y_test, y_pred))
    return metrics


def save_best(models: dict, results: list, save_path: str = "models/best_model.pkl"):
    best = max(results, key=lambda x: x["roc_auc"])
    model = models[best["name"]]
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n🏆 Best model: {best['name']} (ROC-AUC: {best['roc_auc']:.4f})")
    return best
