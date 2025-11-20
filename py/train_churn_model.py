
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
import joblib

# ========== CONFIG ==========
DATA_PATH = "/Users/balakrishna/Documents/bala_py_db/netflix/churn_features.csv"
TARGET_COL = "churned"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
THRESH_PATH = os.path.join(MODEL_DIR, "churn_threshold.json")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)
print("[info] Loaded data from:", DATA_PATH)
print("[info] Columns:", list(df.columns))

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

# drop rows with missing target
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Exclude target and ID columns from features
exclude_cols = [TARGET_COL, "user_id"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# separate numeric and categorical features
num_features = df[feature_cols].select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

cat_features = df[feature_cols].select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("[info] Numeric features :", num_features)
print("[info] Categorical feats:", cat_features)

if not num_features and not cat_features:
    raise ValueError("No feature columns found. Check your data.")

X = df[num_features + cat_features]
y = df[TARGET_COL]

print("[info] X shape:", X.shape)
print("[info] y distribution:\n", y.value_counts(normalize=True))

# ========== PREPROCESSING ==========
numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# ========== MODEL ==========
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",  # handle imbalance
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("rf", rf),
    ]
)

# ========== TRAIN / VALIDATION SPLIT ==========
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("[info] Train shape:", X_train.shape)
print("[info] Val   shape:", X_val.shape)

# ========== TRAIN ==========
model.fit(X_train, y_train)

# ========== PREDICT PROBABILITIES ==========
proba_val = model.predict_proba(X_val)[:, 1]

# ROC-AUC
auc = roc_auc_score(y_val, proba_val)
print(f"[info] ROC-AUC: {auc:.4f}")

# ========== FIND BEST THRESHOLD (F1) ==========
def find_best_threshold(y_true, y_proba, step=0.05):
    thresholds = np.arange(step, 1.0, step)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds_t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

best_thresh, best_f1 = find_best_threshold(y_val, proba_val, step=0.05)
print(f"[info] Best Threshold (by F1): {best_thresh:.2f}")
print(f"[info] Best F1 Score:         {best_f1:.4f}")

# ========== REPORT @ DEFAULT 0.5 ==========
y_pred_05 = (proba_val >= 0.5).astype(int)
print("\n[info] Report with threshold = 0.50")
print(classification_report(y_val, y_pred_05))

# ========== REPORT @ BEST THRESH ==========
y_pred_best = (proba_val >= best_thresh).astype(int)
print("\n[info] Report with BEST threshold")
print(classification_report(y_val, y_pred_best))

# ========== SAVE MODEL + THRESHOLD ==========
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)
print(f"[info] Saved model to {MODEL_PATH}")

with open(THRESH_PATH, "w") as f:
    json.dump({"threshold": float(best_thresh)}, f)

print(f"[info] Saved threshold to {THRESH_PATH}")
print("[info] Done.")
