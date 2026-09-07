# pyrefly: ignore [missing-import]
"""
06_multiclass_evaluation.py
PHASE 6: Multiclass Attack Classification Evaluation

Evaluates the 15-class XGBoost model on the held-out test set from Phase 5.
Reports macro and weighted metrics — essential for imbalanced datasets
where rare classes (Heartbleed=11, SQL Injection=21, Infiltration=36) exist.

Outputs (written to training/evaluation/):
  classification_report_multiclass.txt
  confusion_matrix_multiclass.png
  feature_importance_multiclass.png
  multiclass_metrics.json
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_multiclass():
    print("=" * 60)
    print("PHASE 6: MULTICLASS EVALUATION")
    print("=" * 60)

    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")
    EVAL_DIR      = os.path.join(BASE_DIR, "training", "evaluation")
    os.makedirs(EVAL_DIR, exist_ok=True)

    # -- [1/6] Load model, test set, metadata --------------------------------
    print("\n[1/6] Loading model, test set, and metadata...")

    for fname in ["xgb_multiclass.pkl", "label_encoder.pkl", "multiclass_metadata.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            raise FileNotFoundError(f"{fname} not found. Run 05_multiclass_training.py first.")

    X_test_path = os.path.join(PROCESSED_DIR, "X_test_multiclass.parquet")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test_multiclass.parquet")
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Multiclass test set not found. Run 05_multiclass_training.py first.")

    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_multiclass.pkl"))
    le        = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    metadata  = joblib.load(os.path.join(MODELS_DIR, "multiclass_metadata.pkl"))

    # Use dedicated multiclass feature list — separate from binary feature_columns.pkl
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "multiclass_feature_columns.pkl"))
    class_names  = metadata["class_names"]
    num_classes  = metadata["num_classes"]

    X_test = pd.read_parquet(X_test_path)[feature_cols]
    y_test = pd.read_parquet(y_test_path)["Label"].values   # integer encoded

    print(f"  Test rows    : {len(X_test):,}")
    print(f"  Classes      : {num_classes}")

    # -- [2/6] Predict -------------------------------------------------------
    print("\n[2/6] Generating predictions...")
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)   # shape: (n, num_classes)

    # Decode integer labels to class names for reports
    y_test_names = le.inverse_transform(y_test)
    y_pred_names = le.inverse_transform(y_pred)

    # -- [3/6] Print comparison table ----------------------------------------
    accuracy = accuracy_score(y_test_names, y_pred_names)
    print(f"\n[3/6] Model Performance Table")
    print(f"  Overall Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"\n  {'Class':<35} {'Count':>7} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*72}")

    for cls in class_names:
        mask  = y_test_names == cls
        count = int(mask.sum())
        if count == 0:
            print(f"  {cls:<35} {count:>7}       N/A      N/A     N/A")
            continue
        p = precision_score(y_test_names == cls,
                            y_pred_names == cls, zero_division=0)
        r = recall_score(y_test_names == cls,
                         y_pred_names == cls, zero_division=0)
        f = f1_score(y_test_names == cls,
                     y_pred_names == cls, zero_division=0)
        rare = "  [RARE]" if count < 50 else ""
        print(f"  {cls:<35} {count:>7} {p:>10.4f} {r:>8.4f} {f:>8.4f}{rare}")

    macro_f1    = f1_score(y_test_names, y_pred_names, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test_names, y_pred_names, average="weighted", zero_division=0)
    macro_p     = precision_score(y_test_names, y_pred_names, average="macro",    zero_division=0)
    macro_r     = recall_score(y_test_names, y_pred_names, average="macro",    zero_division=0)
    weighted_p  = precision_score(y_test_names, y_pred_names, average="weighted", zero_division=0)
    weighted_r  = recall_score(y_test_names, y_pred_names, average="weighted",    zero_division=0)

    print(f"\n  {'Macro Average':<35} {'':>7} {macro_p:>10.4f} {macro_r:>8.4f} {macro_f1:>8.4f}")
    print(f"  {'Weighted Average':<35} {'':>7} {weighted_p:>10.4f} {weighted_r:>8.4f} {weighted_f1:>8.4f}")

    # -- [4/6] Classification report & confusion matrix ----------------------
    print("\n[4/6] Saving classification report and confusion matrix...")

    report = classification_report(
        y_test_names, y_pred_names,
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )
    report_path = os.path.join(EVAL_DIR, "classification_report_multiclass.txt")
    with open(report_path, "w") as f:
        f.write("PHASE 6: Multiclass Classification Report\n")
        f.write("Dataset: CICIDS2017\n\n")
        f.write(report)
    print(f"  Saved classification_report_multiclass.txt")

    # Confusion matrix — ordered by class index
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4
    )
    ax.set_title("Confusion Matrix — XGBoost Multiclass (CICIDS2017)", fontsize=13)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix_multiclass.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrix_multiclass.png")

    # -- [5/6] Feature importance --------------------------------------------
    print("\n[5/6] Saving feature importance...")
    imp     = xgb_model.feature_importances_
    indices = np.argsort(imp)[::-1][:20]

    plt.figure(figsize=(11, 8))
    plt.title("Top 20 Feature Importances — XGBoost Multiclass", fontsize=13)
    plt.bar(range(20), imp[indices], color="steelblue", align="center")
    plt.xticks(range(20), [feature_cols[i] for i in indices], rotation=45, ha="right", fontsize=8)
    plt.xlim([-1, 20])
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "feature_importance_multiclass.png"), dpi=150)
    plt.close()
    print("  Saved feature_importance_multiclass.png")

    # -- [6/6] Save metrics JSON ---------------------------------------------
    print("\n[6/6] Saving multiclass_metrics.json...")

    per_class = {}
    per_class_rows = []
    for cls in class_names:
        mask  = y_test_names == cls
        count = int(mask.sum())
        p = float(precision_score(mask, y_pred_names == cls, zero_division=0))
        r = float(recall_score(mask, y_pred_names == cls, zero_division=0))
        f = float(f1_score(mask, y_pred_names == cls, zero_division=0))
        per_class[cls] = {"test_count": count, "precision": p, "recall": r, "f1": f}
        per_class_rows.append({
            "Class":     cls,
            "Support":   count,
            "Precision": round(p, 4),
            "Recall":    round(r, 4),
            "F1":        round(f, 4),
            "Rare":      count < 50,
        })

    # per_class_metrics.csv — useful for dissertation appendix
    pd.DataFrame(per_class_rows).sort_values("Support", ascending=False).to_csv(
        os.path.join(EVAL_DIR, "per_class_metrics.csv"), index=False
    )
    print("  Saved per_class_metrics.csv")

    metrics_out = {
        "dataset":           "CICIDS2017",
        "model":             "XGBoost multiclass (multi:softprob)",
        "num_classes":       num_classes,
        "test_records":      int(len(y_test)),
        "test_attack_count": int((y_test_names != "BENIGN").sum()),
        "test_benign_count": int((y_test_names == "BENIGN").sum()),
        "overall_accuracy":  float(accuracy),
        "macro": {
            "precision": float(macro_p),
            "recall":    float(macro_r),
            "f1":        float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall":    float(weighted_r),
            "f1":        float(weighted_f1),
        },
        "per_class": per_class,
    }

    with open(os.path.join(EVAL_DIR, "multiclass_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=4)
    print("  Saved multiclass_metrics.json")

    print(f"\nAll multiclass evaluation artifacts saved to: {EVAL_DIR}")
    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETE")
    print(f"  Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_multiclass()
