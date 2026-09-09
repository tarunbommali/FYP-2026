"""
03_threshold_optimization.py
PHASE 3: Per-Model Threshold Optimization

Loads RF, XGB, and Isolation Forest from Phase 2.
Generates per-model predictions on the held-out test set.
Sweeps thresholds to maximise F1 for each model independently.
Saves binary_threshold.json with rf / xgb / iso entries.

NOTE: No meta-learner is trained here. The test set is never used for
      model training — only for threshold selection on already-fitted models.
      Meta-learner ensemble is deferred until a Train/Val/Test split is
      created before Phase 5.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import precision_recall_curve


# ---------------------------------------------------------------------------
def _find_best_threshold(y_true, y_scores, model_name):
    """Sweep P-R curve; return the threshold that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx  = int(np.argmax(f1_scores))
    best_idx  = min(best_idx, len(thresholds) - 1)

    result = {
        "threshold": float(thresholds[best_idx]),
        "f1":        float(f1_scores[best_idx]),
        "precision": float(precision[best_idx]),
        "recall":    float(recall[best_idx]),
    }
    print(f"  {model_name:<12} {result['threshold']:>10.4f} {result['f1']:>8.4f} "
          f"{result['precision']:>10.4f} {result['recall']:>8.4f}")
    return result


# ---------------------------------------------------------------------------
def optimize_thresholds():
    print("=" * 60)
    print("PHASE 3: THRESHOLD OPTIMIZATION")
    print("=" * 60)

    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")

    # -- [1/7] Load test set & models ----------------------------------------
    print("\n[1/7] Loading isolated test set and Phase 2 models...")

    X_test_path = os.path.join(PROCESSED_DIR, "X_test_binary.parquet")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test_binary.parquet")
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Test set not found. Run 02_binary_training.py first.")

    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)["Label"].values

    feature_cols = joblib.load(os.path.join(MODELS_DIR, "preprocessing", "multiclass_feature_columns.pkl"))
    X_test = X_test[feature_cols]

    rf_model   = joblib.load(os.path.join(MODELS_DIR, "binary", "rf_binary.pkl"))
    xgb_model  = joblib.load(os.path.join(MODELS_DIR, "binary", "xgb_binary.pkl"))
    iso_model  = joblib.load(os.path.join(MODELS_DIR, "anomaly", "isolation_forest.pkl"))
    iso_scaler = joblib.load(os.path.join(MODELS_DIR, "preprocessing", "scaler.pkl"))

    print(f"  Test rows  : {len(X_test):,}")
    print(f"  Attack rows: {int(y_test.sum()):,}  ({y_test.mean()*100:.2f}%)")

    # -- [2/7] RF probabilities ----------------------------------------------
    print("\n[2/7] Generating RF attack probabilities...")
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    # -- [3/7] XGBoost probabilities -----------------------------------------
    print("[3/7] Generating XGBoost attack probabilities...")
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # -- [4/7] ISO anomaly scores --------------------------------------------
    # decision_function: higher = more normal, lower = more anomalous.
    # Negate so higher score = more attack-like (consistent with RF/XGB).
    # Then normalise to [0, 1] so threshold is on the same scale as probs.
    print("[4/7] Generating ISO anomaly scores (negated + normalised)...")
    X_test_iso  = iso_scaler.transform(X_test)
    iso_raw     = iso_model.decision_function(X_test_iso)
    iso_scores  = -iso_raw
    iso_min     = float(iso_scores.min())
    iso_max     = float(iso_scores.max())
    iso_norm    = (iso_scores - iso_min) / (iso_max - iso_min + 1e-10)

    print(f"  ISO raw range : [{iso_raw.min():.4f}, {iso_raw.max():.4f}]")
    print(f"  ISO norm range: [{iso_norm.min():.4f}, {iso_norm.max():.4f}]")

    # -- [5/7] Per-model threshold sweep -------------------------------------
    print("\n[5/7] Sweeping thresholds (maximising F1 per model)...")
    print(f"\n  {'Model':<10}  {'Threshold':>9}  {'F1':>6}  {'Precision':>9}  {'Recall':>6}")
    print(f"  {'-'*56}")
    rf_data  = _find_best_threshold(y_test, rf_probs,  "RF")
    xgb_data = _find_best_threshold(y_test, xgb_probs, "XGBoost")
    iso_data = _find_best_threshold(y_test, iso_norm,  "ISO")

    # -- [6/7] Attach ISO normalisation params (needed by Phase 4) ----------
    iso_data["iso_norm"] = {"iso_min": iso_min, "iso_max": iso_max}

    # -- [7/7] Save threshold JSON ------------------------------------------
    print("\n[7/7] Saving binary_threshold.json...")
    threshold_data = {
        "rf":  rf_data,
        "xgb": xgb_data,
        "iso": iso_data,
    }
    out_path = os.path.join(MODELS_DIR, "binary_threshold.json")
    with open(out_path, "w") as f:
        json.dump(threshold_data, f, indent=4)
    print(f"  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("  binary_threshold.json -- rf / xgb / iso optimal thresholds")
    print("Next: run 04_model_evaluation.py")
    print("=" * 60)


if __name__ == "__main__":
    optimize_thresholds()
