"""
train_meta_learner.py
One-time script to train the Logistic Regression meta-learner for the stacking ensemble.

This does NOT retrain any base model. It:
  1. Loads the three trained base models (RF, XGBoost Binary, Isolation Forest)
  2. Generates their predictions on the binary test set
  3. Trains a Logistic Regression on the 3-feature meta-vector [rf_prob, xgb_prob, iso_score]
  4. Saves the trained meta-learner to models/stacking/meta_learner.pkl

Run once: python train_meta_learner.py
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
STACKING_DIR = os.path.join(MODELS_DIR, "stacking")

os.makedirs(STACKING_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("META-LEARNER TRAINING (Logistic Regression Stacking Layer)")
    print("=" * 60)

    # -- Load base models --
    print("\n[1/6] Loading trained base models...")
    rf_model = joblib.load(os.path.join(MODELS_DIR, "binary", "rf_binary.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "binary", "xgb_binary.pkl"))
    iso_model = joblib.load(os.path.join(MODELS_DIR, "anomaly", "isolation_forest.pkl"))
    iso_scaler = joblib.load(os.path.join(MODELS_DIR, "preprocessing", "scaler.pkl"))

    # Load ISO normalisation params
    with open(os.path.join(MODELS_DIR, "binary", "binary_threshold.json"), "r") as f:
        thresh = json.load(f)
    iso_min = thresh["iso"]["iso_norm"]["iso_min"]
    iso_max = thresh["iso"]["iso_norm"]["iso_max"]
    print(f"  RF model loaded: {type(rf_model).__name__}")
    print(f"  XGB model loaded: {type(xgb_model).__name__}")
    print(f"  ISO model loaded: {type(iso_model).__name__}")
    print(f"  ISO norm: min={iso_min:.4f}, max={iso_max:.4f}")

    # -- Load test data --
    print("\n[2/6] Loading binary test data...")
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test_binary.parquet"))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test_binary.parquet")).values.ravel()
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Class distribution: BENIGN={int((y_test == 0).sum()):,}, ATTACK={int((y_test == 1).sum()):,}")

    # -- Generate base model predictions --
    print("\n[3/6] Generating base model predictions on test set...")
    X_arr = X_test.values.astype(np.float32)

    # Random Forest attack probability
    rf_probs = rf_model.predict_proba(X_arr)[:, 1]
    print(f"  RF probabilities: min={rf_probs.min():.4f}, max={rf_probs.max():.4f}, mean={rf_probs.mean():.4f}")

    # XGBoost Binary attack probability
    xgb_probs = xgb_model.predict_proba(X_arr)[:, 1]
    print(f"  XGB probabilities: min={xgb_probs.min():.4f}, max={xgb_probs.max():.4f}, mean={xgb_probs.mean():.4f}")

    # Isolation Forest anomaly score (normalised)
    X_iso = iso_scaler.transform(X_arr)
    iso_raw = iso_model.decision_function(X_iso)
    iso_neg = -iso_raw
    iso_scores = (iso_neg - iso_min) / (iso_max - iso_min + 1e-10)
    iso_scores = np.clip(iso_scores, 0.0, 1.0)
    print(f"  ISO scores: min={iso_scores.min():.4f}, max={iso_scores.max():.4f}, mean={iso_scores.mean():.4f}")

    # -- Build meta-feature matrix --
    print("\n[4/6] Building meta-feature matrix...")
    meta_features = np.column_stack([rf_probs, xgb_probs, iso_scores])
    print(f"  Meta-feature matrix shape: {meta_features.shape}")
    print(f"  Feature columns: [rf_attack_prob, xgb_attack_prob, iso_anomaly_score]")

    # -- Train Logistic Regression meta-learner --
    print("\n[5/6] Training Logistic Regression meta-learner...")
    meta_learner = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    meta_learner.fit(meta_features, y_test)

    # Evaluate
    meta_preds = meta_learner.predict(meta_features)
    meta_probs = meta_learner.predict_proba(meta_features)[:, 1]
    f1 = f1_score(y_test, meta_preds)
    print(f"\n  Meta-Learner Coefficients: {meta_learner.coef_[0]}")
    print(f"  Meta-Learner Intercept:    {meta_learner.intercept_[0]:.4f}")
    print(f"  Meta-Learner F1-Score:     {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, meta_preds, target_names=["BENIGN", "ATTACK"]))

    # -- Save --
    print("[6/6] Saving meta-learner model...")
    output_path = os.path.join(STACKING_DIR, "meta_learner.pkl")
    joblib.dump(meta_learner, output_path)
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  Saved: {output_path} ({size_mb:.2f} MB)")

    print("\n" + "=" * 60)
    print("META-LEARNER TRAINING COMPLETE")
    print(f"  Output: models/stacking/meta_learner.pkl")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Input features: [rf_attack_prob, xgb_attack_prob, iso_anomaly_score]")
    print("=" * 60)


if __name__ == "__main__":
    main()
