"""
model_training.py
-----------------
Trains Logistic Regression, Decision Tree, and Random Forest models.
"""
import pickle
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree":       DecisionTreeClassifier(random_state=42),
    "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}


def train_all(X_train, y_train, save_dir: str = "models/") -> dict:
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained[name] = model
        path = f"{save_dir}{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  ✅ Saved: {path}")
    return trained
