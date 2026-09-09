"""
04_model_evaluation.py
PHASE 4: Model Evaluation

Evaluates RF, XGBoost, and Isolation Forest independently on the held-out
test set using per-model optimal thresholds from binary_threshold.json.

Outputs:
  - Classification reports  (rf, xgb, iso)
  - Confusion matrices       (rf, xgb, iso)
  - Combined ROC curve       (all 3 models)
  - Combined PR curve        (all 3 models)
  - Feature importance plots (rf, xgb)
  - feature_importance.csv
  - model_metadata.json
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    precision_score, recall_score, f1_score, average_precision_score,
)


# ---------------------------------------------------------------------------
def _plot_confusion_matrix(cm, title, labels, path, cmap):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _metrics_dict(y_true, y_pred, y_score, roc_auc_val):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_val),
        "pr_auc":    float(average_precision_score(y_true, y_score)),
    }


# ---------------------------------------------------------------------------
def evaluate_models():
    print("=" * 60)
    print("PHASE 4: MODEL EVALUATION")
    print("=" * 60)

    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")
    EVAL_DIR      = os.path.join(BASE_DIR, "training", "evaluation")
    os.makedirs(EVAL_DIR, exist_ok=True)

    # -- [1/6] Load data and models ------------------------------------------
    print("\n[1/6] Loading test set, models, and thresholds...")

    X_test_path = os.path.join(PROCESSED_DIR, "X_test_binary.parquet")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test_binary.parquet")
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Test set not found. Run 02_binary_training.py first.")

    thresh_path = os.path.join(MODELS_DIR, "binary", "binary_threshold.json")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError("binary_threshold.json not found. Run 03_threshold_optimization.py first.")

    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)["Label"].values

    feature_cols = joblib.load(os.path.join(MODELS_DIR, "preprocessing", "multiclass_feature_columns.pkl"))
    X_test = X_test[feature_cols]

    rf_model   = joblib.load(os.path.join(MODELS_DIR, "binary", "rf_binary.pkl"))
    xgb_model  = joblib.load(os.path.join(MODELS_DIR, "binary", "xgb_binary.pkl"))
    iso_model  = joblib.load(os.path.join(MODELS_DIR, "anomaly", "isolation_forest.pkl"))
    iso_scaler = joblib.load(os.path.join(MODELS_DIR, "preprocessing", "scaler.pkl"))

    with open(thresh_path, "r") as f:
        thresh = json.load(f)

    rf_threshold  = thresh["rf"]["threshold"]
    xgb_threshold = thresh["xgb"]["threshold"]
    iso_threshold = thresh["iso"]["threshold"]
    iso_min       = thresh["iso"]["iso_norm"]["iso_min"]
    iso_max       = thresh["iso"]["iso_norm"]["iso_max"]

    print(f"  RF  threshold : {rf_threshold:.4f}")
    print(f"  XGB threshold : {xgb_threshold:.4f}")
    print(f"  ISO threshold : {iso_threshold:.4f}")

    # -- [2/6] Generate predictions ------------------------------------------
    print("\n[2/6] Generating predictions on held-out test set...")

    # RF
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_pred  = (rf_probs >= rf_threshold).astype(int)

    # XGBoost
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred  = (xgb_probs >= xgb_threshold).astype(int)

    # Isolation Forest (negate + normalise using Phase 3 params)
    X_test_iso = iso_scaler.transform(X_test)
    iso_raw    = iso_model.decision_function(X_test_iso)
    iso_scores = -iso_raw
    iso_norm   = (iso_scores - iso_min) / (iso_max - iso_min + 1e-10)
    iso_pred   = (iso_norm >= iso_threshold).astype(int)

    # ROC-AUC per model
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_norm)
    auc_rf  = auc(fpr_rf,  tpr_rf)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    auc_iso = auc(fpr_iso, tpr_iso)

    # -- [3/6] Print comparison table ----------------------------------------
    print("\n[3/6] Model Comparison Table")
    print(f"\n  {'Model':<12} {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8}")
    print(f"  {'-'*72}")
    for name, y_pred, y_score, roc_a, thr in [
        ("RF",       rf_pred,  rf_probs,  auc_rf,  rf_threshold),
        ("XGBoost",  xgb_pred, xgb_probs, auc_xgb, xgb_threshold),
        ("ISO",      iso_pred, iso_norm,  auc_iso, iso_threshold),
    ]:
        p  = precision_score(y_test, y_pred, zero_division=0)
        r  = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        pr = average_precision_score(y_test, y_score)
        print(f"  {name:<12} {thr:>10.4f} {p:>10.4f} {r:>8.4f} {f1:>8.4f} {roc_a:>9.4f} {pr:>8.4f}")

    # -- [4/6] Classification reports & confusion matrices -------------------
    print("\n[4/6] Saving classification reports and confusion matrices...")
    labels = ["BENIGN", "ATTACK"]

    for name, y_pred, cmap, cm_fname, rep_fname in [
        ("RF",      rf_pred,  "Blues",   "confusion_matrix_rf.png",  "classification_report_rf.txt"),
        ("XGBoost", xgb_pred, "Oranges", "confusion_matrix_xgb.png", "classification_report_xgb.txt"),
        ("ISO",     iso_pred, "Purples", "confusion_matrix_iso.png", "classification_report_iso.txt"),
    ]:
        report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
        with open(os.path.join(EVAL_DIR, rep_fname), "w") as f:
            f.write(f"Model: {name}\n\n{report}")

        cm = confusion_matrix(y_test, y_pred)
        _plot_confusion_matrix(
            cm, f"Confusion Matrix — {name}", labels,
            os.path.join(EVAL_DIR, cm_fname), cmap
        )
        print(f"  Saved {cm_fname} and {rep_fname}")

    # -- [5/6] Combined ROC and PR curves ------------------------------------
    print("\n[5/6] Saving combined ROC and PR curves...")

    # ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf,  tpr_rf,  color="green",      lw=2, label=f"RF      (AUC={auc_rf:.4f})")
    plt.plot(fpr_xgb, tpr_xgb, color="darkorange",  lw=2, label=f"XGBoost (AUC={auc_xgb:.4f})")
    plt.plot(fpr_iso, tpr_iso, color="mediumpurple", lw=2, label=f"ISO     (AUC={auc_iso:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Binary IDS (RF vs XGBoost vs ISO)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved roc_curve.png")

    # PR
    prec_rf,  rec_rf,  _ = precision_recall_curve(y_test, rf_probs)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_probs)
    prec_iso, rec_iso, _ = precision_recall_curve(y_test, iso_norm)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_rf,  prec_rf,  color="green",       lw=2, label="RF")
    plt.plot(rec_xgb, prec_xgb, color="darkorange",  lw=2, label="XGBoost")
    plt.plot(rec_iso, prec_iso, color="mediumpurple", lw=2, label="ISO")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Binary IDS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved pr_curve.png")

    # -- [6/6] Feature importances & metadata --------------------------------
    print("\n[6/6] Saving feature importances and model_metadata.json...")

    for name, model, color, fname in [
        ("XGBoost",      xgb_model, "darkorange", "xgb_feature_importance.png"),
        ("Random Forest", rf_model,  "green",      "rf_feature_importance.png"),
    ]:
        imp     = model.feature_importances_
        indices = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(10, 8))
        plt.title(f"Top 20 Feature Importances ({name})")
        plt.bar(range(20), imp[indices], color=color, align="center")
        plt.xticks(range(20), [feature_cols[i] for i in indices], rotation=90)
        plt.xlim([-1, 20])
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, fname), dpi=150)
        plt.close()
        print(f"  Saved {fname}")

    # Feature importance CSV
    pd.DataFrame({
        "Feature":          feature_cols,
        "XGBoost_Importance": xgb_model.feature_importances_,
        "RF_Importance":      rf_model.feature_importances_,
    }).sort_values("XGBoost_Importance", ascending=False).to_csv(
        os.path.join(EVAL_DIR, "feature_importance.csv"), index=False
    )
    print("  Saved feature_importance.csv")

    # Metadata JSON
    metadata = {
        "dataset":       "CICIDS2017",
        "feature_count": len(feature_cols),
        "test_records":  int(len(y_test)),
        "rf": {
            **_metrics_dict(y_test, rf_pred, rf_probs, auc_rf),
            "threshold": rf_threshold,
        },
        "xgboost": {
            **_metrics_dict(y_test, xgb_pred, xgb_probs, auc_xgb),
            "threshold": xgb_threshold,
        },
        "isolation_forest": {
            **_metrics_dict(y_test, iso_pred, iso_norm, auc_iso),
            "threshold": iso_threshold,
        },
    }
    with open(os.path.join(EVAL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print("  Saved model_metadata.json")

    print(f"\nAll evaluation artifacts saved to: {EVAL_DIR}")
    print("=" * 60)
    print("PHASE 4 COMPLETE")
    print("Next: review model_metadata.json, then plan Phase 5 (multiclass).")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_models()
