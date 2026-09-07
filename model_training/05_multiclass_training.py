# pyrefly: ignore [missing-import]
"""
05_multiclass_training.py
PHASE 5: Multiclass Attack Classification Training

Classifies all 15 CICIDS2017 traffic types.
Uses XGBoost only (binary pipeline showed XGB = 1.5 MB vs RF = 63.5 MB).

Split strategy: 70 / 10 / 20  (train / val / test)
  - Validation set enables future threshold tuning and meta-learner training
  - Test set is never seen during training or threshold selection

Artifacts produced:
  models/xgb_multiclass.pkl
  models/label_encoder.pkl
  models/multiclass_metadata.pkl
  data/processed/X_val_multiclass.parquet
  data/processed/y_val_multiclass.parquet
  data/processed/X_test_multiclass.parquet
  data/processed/y_test_multiclass.parquet
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier


def train_multiclass():
    print("=" * 60)
    print("PHASE 5: MULTICLASS ATTACK CLASSIFICATION")
    print("=" * 60)

    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLEAN_FILE    = os.path.join(BASE_DIR, "data", "processed", "cicids2017_clean.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -- [1/8] Load clean dataset -------------------------------------------
    print("\n[1/8] Loading clean dataset...")
    if not os.path.exists(CLEAN_FILE):
        raise FileNotFoundError(f"Clean dataset not found: {CLEAN_FILE}\n"
                                "Run 01_data_cleaning.py first.")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    print(f"  Shape: {df.shape}")

    # -- [2/8] Label encoding -----------------------------------------------
    print("\n[2/8] Encoding class labels...")
    le = LabelEncoder()
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label_enc"] = le.fit_transform(df["Label"])

    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}):")
    for idx, cls in enumerate(le.classes_):
        count = (df["Label_enc"] == idx).sum()
        print(f"    [{idx:>2}] {cls:<35} : {count:>9,}")

    # Rare-class warning — affects metric stability, note in report
    class_counts = pd.Series(df["Label_enc"]).value_counts()
    rare = class_counts[class_counts < 50]
    if len(rare) > 0:
        print("\n  WARNING: Extremely rare classes detected (< 50 samples):")
        for enc_idx, cnt in rare.items():
            print(f"    [{enc_idx:>2}] {le.classes_[enc_idx]:<35} : {cnt} samples")
        print("  Per-class metrics for these will be unstable. Note in report.")

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"  Saved label_encoder.pkl")

    # -- [3/8] Feature / label split ----------------------------------------
    print("\n[3/8] Splitting features and labels...")
    feature_cols = [c for c in df.columns if c not in ("Label", "Label_enc")]
    X = df[feature_cols].copy()
    y = df["Label_enc"].copy()

    # Save under a dedicated name — do NOT overwrite binary feature_columns.pkl
    fc_path = os.path.join(MODELS_DIR, "multiclass_feature_columns.pkl")
    joblib.dump(feature_cols, fc_path)
    print(f"  Saved multiclass_feature_columns.pkl")
    print(f"  Feature count : {len(feature_cols)}")
    print(f"  Total samples : {len(X):,}")

    del df
    gc.collect()

    # -- [4/8] 70 / 10 / 20 stratified split --------------------------------
    print("\n[4/8] Stratified 70/10/20 split (train/val/test)...")

    # Step 1: carve out 20% test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Step 2: from remaining 80%, take 12.5% as val → 80% * 12.5% = 10% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.125, random_state=42, stratify=y_trainval
    )

    print(f"  Train : {len(X_train):,}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val   : {len(X_val):,}  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test  : {len(X_test):,}  ({len(X_test)/len(X)*100:.1f}%)")

    del X_trainval, y_trainval
    gc.collect()

    # -- [5/8] Save val and test parquets -----------------------------------
    print("\n[5/8] Saving validation and test sets...")
    X_val.to_parquet(os.path.join(PROCESSED_DIR, "X_val_multiclass.parquet"), index=False)
    pd.Series(y_val, name="Label").to_frame().to_parquet(
        os.path.join(PROCESSED_DIR, "y_val_multiclass.parquet"), index=False)
    X_test.to_parquet(os.path.join(PROCESSED_DIR, "X_test_multiclass.parquet"), index=False)
    pd.Series(y_test, name="Label").to_frame().to_parquet(
        os.path.join(PROCESSED_DIR, "y_test_multiclass.parquet"), index=False)
    print("  Saved X_val_multiclass.parquet + y_val_multiclass.parquet")
    print("  Saved X_test_multiclass.parquet + y_test_multiclass.parquet")

    del X_val, X_test, y_val, y_test
    gc.collect()

    # -- [6/8] Compute sample weights (handles severe class imbalance) -------
    print("\n[6/8] Computing sample weights...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"  Weight range : [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # -- [7/8] Train XGBoost multiclass -------------------------------------
    print("\n[7/8] Training XGBoost multiclass classifier...")
    print("  objective    : multi:softprob")
    print(f"  num_class    : {num_classes}")
    print("  n_estimators : 300")
    print("  max_depth    : 6")
    print("  tree_method  : hist  (GPU-optional, memory-efficient)")

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,      # reduced from 400 — near-identical accuracy, lower RAM
        max_depth=6,           # reduced from 8 — less overfitting on rare classes
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,         # L1 regularisation — consistent with binary model
        reg_lambda=1.0,        # L2 regularisation
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    xgb_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_multiclass.pkl"))
    print("  Saved xgb_multiclass.pkl")

    del X_train, y_train, sample_weights
    gc.collect()

    # -- [8/8] Save metadata ------------------------------------------------
    print("\n[8/8] Saving multiclass_metadata.pkl...")
    metadata = {
        "num_classes":    num_classes,
        "class_names":    list(le.classes_),
        "feature_count":  len(feature_cols),
        "feature_names":  feature_cols,
        "split": {
            "train_pct": 70,
            "val_pct":   10,
            "test_pct":  20,
        },
        "model_params": {
            "n_estimators": 300,
            "max_depth":    6,
            "learning_rate": 0.05,
            "reg_alpha":    0.5,
            "reg_lambda":   1.0,
        },
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "multiclass_metadata.pkl"))
    print("  Saved multiclass_metadata.pkl")

    model_size = os.path.getsize(
        os.path.join(MODELS_DIR, "xgb_multiclass.pkl")) / 1024 ** 2
    print(f"\n  xgb_multiclass.pkl : {model_size:.1f} MB")

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("  xgb_multiclass.pkl     -- 15-class XGBoost classifier")
    print("  label_encoder.pkl      -- LabelEncoder (int -> class name)")
    print("  multiclass_metadata.pkl")
    print("  X/y_val_multiclass.parquet   -- validation set (for tuning)")
    print("  X/y_test_multiclass.parquet  -- test set (for Phase 6)")
    print("Next: run 06_multiclass_evaluation.py")
    print("=" * 60)


if __name__ == "__main__":
    train_multiclass()
