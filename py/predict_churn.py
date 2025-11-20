# py/predict_churn.py
# ---------------------------------------------------------
# Use trained churn_model.pkl + saved threshold
# to score all users in churn_features.csv
# and write data/final_user_risk.csv
# ---------------------------------------------------------

import os
import json
import pandas as pd
import joblib

# same data file you used for training
FEATURES_PATH = "/Users/balakrishna/Documents/bala_py_db/netflix/churn_features.csv"
MODEL_PATH    = "models/churn_model.pkl"
THRESH_PATH   = "models/churn_threshold.json"
OUT_PATH      = "data/final_user_risk.csv"

TARGET_COL = "churned"

print("[info] Loading features from:", FEATURES_PATH)
df = pd.read_csv(FEATURES_PATH)

# use same features as training: everything except target
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].copy()

print("[info] Feature columns:", feature_cols)
print("[info] Shape:", X.shape)

print("[info] Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

print("[info] Loading threshold from:", THRESH_PATH)
with open(THRESH_PATH, "r") as f:
    churn_threshold = json.load(f)["threshold"]

print(f"[info] Using threshold: {churn_threshold:.4f}")

# predict probabilities (class 1 = churn)
proba = model.predict_proba(X)[:, 1]

df_out = df.copy()
df_out["churn_probability"] = proba
df_out["predicted_churn"] = (proba >= churn_threshold).astype(int)

# sort high-risk on top
df_out = df_out.sort_values("churn_probability", ascending=False).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df_out.to_csv(OUT_PATH, index=False)

print("[info] Saved predictions to:", OUT_PATH)
print(df_out[["churn_probability", "predicted_churn"]].head(15))
