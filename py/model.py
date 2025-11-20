# dashboard.py (or your actual dashboard file)
# ---------------------------------------------------------
# Put these at the TOP of the file (imports and loading)
# ---------------------------------------------------------
import json
import joblib

MODEL_PATH = "models/churn_model.pkl"
THRESH_PATH = "models/churn_threshold.json"

churn_model = joblib.load(MODEL_PATH)

with open(THRESH_PATH, "r") as f:
    churn_threshold = json.load(f)["threshold"]

print("[info] Loaded churn model and threshold:", churn_threshold)

def add_churn_predictions(df_users):
    """
    df_users: DataFrame that has at least the same feature columns
    as used during training (age, plan_type, tenure_days, etc.)
    Returns a copy with churn_probability + predicted_churn columns.
    """
    feature_cols = [
        "age",
        "total_watch_hours_30d",
        "days_since_last_watch",
        "num_logins_30d",
        "plan_type",

    ]

    X = df_users[feature_cols].copy()

    proba = churn_model.predict_proba(X)[:, 1]

    df_out = df_users.copy()
    df_out["churn_probability"] = proba
    df_out["predicted_churn"] = (proba >= churn_threshold).astype(int)

    return df_out
