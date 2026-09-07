import pandas as pd
import numpy as np
import os
import gc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-import]
from imblearn.combine import SMOTETomek
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

def train_binary():
    print("="*50)
    print("PHASE 2: BINARY TRAINING")
    print("="*50)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cicids2017_clean.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\n[1/7] Loading clean dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Clean dataset not found at {DATA_PATH}. Run 01_data_cleaning.py first!")
            
    # EXPERIMENTATION LIMIT: For trial runs on 8GB RAM, uncomment nrows=1000000 
    # to avoid SMOTE/training crashes. Use full dataset only for final training.
    df = pd.read_csv(DATA_PATH, low_memory=False) #, nrows=1000000)
    print(f"  Shape: {df.shape}")
    
    print("\n[2/7] Converting Labels (BENIGN -> 0, ATTACK -> 1)...")
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)
    
    X = df.drop("Label", axis=1)
    y = df["Label"]
    
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    del df
    gc.collect()
    
    print("\n[3/7] Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    del X, y
    gc.collect()
    
    print("\n[4/7] Saving Isolated Test Set...")
    # Save the test set so optimization and evaluation can be strictly separated from training data.
    X_test.to_parquet(os.path.join(PROCESSED_DIR, "X_test_binary.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(PROCESSED_DIR, "y_test_binary.parquet"), index=False)
    print("  Saved X_test_binary.parquet and y_test_binary.parquet")
    
    del X_test, y_test
    gc.collect()
    
    USE_SMOTE = False

    if USE_SMOTE:
        print("\n[5/7] Applying SMOTETomek (Train Data Only)...")
        print("  Class distribution BEFORE SMOTE:")
        print(y_train.value_counts())
        
        try:
            smt = SMOTETomek(random_state=42, n_jobs=-1)
            X_train, y_train = smt.fit_resample(X_train, y_train)
            print("\n  Class distribution AFTER SMOTE:")
            print(y_train.value_counts())
        except MemoryError:
            print("\n  [WARNING] MemoryError hit! Bypassing SMOTETomek to protect system RAM.")
            gc.collect()
    else:
        print("\n[5/7] Skipping SMOTETomek (USE_SMOTE=False)...")
        print("  Class distribution:")
        print(y_train.value_counts())
        
    print("\n[6/8] Training Random Forest (class_weight=balanced)...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",   # handles 5.04:1 imbalance
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_binary.pkl"))
    print("  Saved rf_binary.pkl")
    
    print("\n[7/8] Training XGBoost (scale_pos_weight + regularization)...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight = {scale_pos_weight:.4f}  (BENIGN:ATTACK ratio from train split)")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # dynamic from actual split
        reg_alpha=0.5,                       # L1 — reduces overconfident probabilities
        reg_lambda=1.0,                      # L2 — standard stabiliser
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_binary.pkl"))
    print("  Saved xgb_binary.pkl")

    print("\n[8/8] Training Isolation Forest (BENIGN-only, dedicated scaler)...")
    iso_scaler = StandardScaler()
    X_benign_scaled = iso_scaler.fit_transform(X_train[y_train == 0])
    print(f"  Fitting on {(y_train == 0).sum():,} BENIGN samples only.")

    iso_model = IsolationForest(
        n_estimators=200,
        contamination="auto",   # evaluated via score_samples in Phase 3
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_benign_scaled)
    joblib.dump(iso_model,  os.path.join(MODELS_DIR, "iso_model.pkl"))
    joblib.dump(iso_scaler, os.path.join(MODELS_DIR, "iso_scaler.pkl"))  # separate from any global scaler
    print("  Saved iso_model.pkl")
    print("  Saved iso_scaler.pkl  (BENIGN-fitted — do NOT use as global inference scaler)")

    # Save train-time metadata — Phase 3 uses this for threshold replacement
    metadata = {
        "feature_count":    len(feature_cols),
        "feature_names":    feature_cols,
        "rf_threshold":     0.5,            # Phase 3 will override
        "xgb_threshold":    0.5,            # Phase 3 will override
        "scale_pos_weight": float(scale_pos_weight),
        "train_rows":       int(len(y_train)),
        "benign_train":     int((y_train == 0).sum()),
        "attack_train":     int((y_train == 1).sum()),
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))
    print("  Saved metadata.pkl")

    print("\n" + "="*50)
    print("ARTIFACTS SAVED:")
    for fname in ["rf_binary.pkl", "xgb_binary.pkl", "iso_model.pkl",
                  "iso_scaler.pkl", "feature_columns.pkl", "metadata.pkl"]:
        fpath = os.path.join(MODELS_DIR, fname)
        size  = os.path.getsize(fpath) / 1024**2 if os.path.exists(fpath) else 0
        print(f"  {fname:<25} {size:.1f} MB")
    print("="*50)
    print("Next: run 03_threshold_optimization.py")

if __name__ == "__main__":
    train_binary()
