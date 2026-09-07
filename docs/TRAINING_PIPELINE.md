# IDS Training Pipeline
### Offline Model Development  CICIDS2017 Dataset
**Final Year Project 2026 | Network Security & Machine Learning**

---

> [!IMPORTANT]
> This pipeline is executed **once during model development** and is **NOT part of the deployed IDS runtime application.**
> The runtime IDS only loads the pretrained `.pkl` artifacts produced here. It never retrains models.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Folders Belonging to Training](#3-folders-belonging-to-training)
4. [Prerequisites & Setup](#4-prerequisites--setup)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Phase-by-Phase Execution](#6-phase-by-phase-execution)
7. [Model Artifacts Generated](#7-model-artifacts-generated)
8. [Evaluation Results](#8-evaluation-results)
9. [Validation & Benchmarking](#9-validation--benchmarking)
10. [Model Selection Rationale](#10-model-selection-rationale)

---

## 1. Overview

The training pipeline transforms raw CICIDS2017 network traffic CSVs into production-ready ML model artifacts. It is a **sequential, one-time offline process** spanning 8 phases:

| Phase | Script | Output |
|---|---|---|
| 0 | `00_dataset_analysis.py` | EDA report (console) |
| 1 | `01_data_cleaning.py` | `data/processed/*.parquet` |
| 2 | `02_binary_training.py` | `xgb_binary.pkl`, `rf_binary.pkl` |
| 3 | `03_threshold_optimization.py` | `binary_threshold.json`, `isolation_forest.pkl`, `scaler.pkl` |
| 4 | `04_model_evaluation.py` | `docs/figures/*.png`, evaluation reports |
| 5 | `05_multiclass_training.py` | `xgb_multiclass.pkl`, `label_encoder.pkl` |
| 6 | `06_multiclass_evaluation.py` | per-class metrics, confusion matrix |
| 7 | `compute_medians.py` | `feature_medians.pkl` |

---

## 2. Pipeline Architecture

```
CICIDS2017 Dataset (8 CSV files, ~2.83M flows)
        |
        v
Phase 0 -- Exploratory Data Analysis
        |  Class distribution, missing values, feature statistics
        |
        v
Phase 1 -- Data Cleaning
        |  Drop Inf/NaN, rename columns, label encode, 80/20 stratified split
        |  Output: data/processed/train.parquet  test.parquet
        |
        v
Phase 2 -- Binary Training
        |  XGBoost Binary Classifier (n_estimators=500, max_depth=8)
        |  Random Forest Classifier  (n_estimators=200, balanced weights)
        |  Output: models/binary/xgb_binary.pkl  rf_binary.pkl
        |
        v
Phase 3 -- Threshold Optimisation + Anomaly Detector
        |  Sweep thresholds 0.01 to 0.99, maximise F1-score
        |  Train Isolation Forest on BENIGN-only flows
        |  Fit StandardScaler for anomaly input
        |  Output: binary_threshold.json  isolation_forest.pkl  scaler.pkl
        |
        v
Phase 4 -- Binary Model Evaluation
        |  ROC, PR curve, Confusion Matrix, Feature Importance plots
        |  Output: docs/figures/*.png
        |
        v
Phase 5 -- Multiclass Training (15-class XGBoost)
        |  objective=multi:softprob, n_estimators=1000, max_depth=10
        |  Output: models/multiclass/xgb_multiclass.pkl  label_encoder.pkl
        |
        v
Phase 6 -- Multiclass Evaluation
        |  Per-class precision/recall/F1, 15x15 confusion matrix
        |  Output: model_training/evaluation/*  docs/figures/confusion_matrix_multiclass.png
        |
        v
Phase 7 -- Feature Median Computation
           Compute per-feature median from training split (prevents data leakage)
           Output: models/preprocessing/feature_medians.pkl

              +----------------------------------+
              |     EXPORTED MODEL ARTIFACTS     |
              |   (consumed by runtime IDS only) |
              +----------------------------------+
              | xgb_binary.pkl                   |
              | rf_binary.pkl                    |
              | xgb_multiclass.pkl               |
              | isolation_forest.pkl             |
              | binary_threshold.json            |
              | scaler.pkl                       |
              | feature_medians.pkl              |
              | label_encoder.pkl                |
              | multiclass_feature_columns.pkl   |
              +----------------------------------+
```

---

## 3. Folders Belonging to Training

```
IDS-Codebase/
+-- model_training/           <- All training scripts
|   +-- 00_dataset_analysis.py
|   +-- 01_data_cleaning.py
|   +-- 02_binary_training.py
|   +-- 03_threshold_optimization.py
|   +-- 04_model_evaluation.py
|   +-- 05_multiclass_training.py
|   +-- 06_multiclass_evaluation.py
|   +-- compute_medians.py
|   +-- verify_clean.py
|   +-- test_inference.py
|   +-- benchmark_inference.py
|   +-- evaluation/     <- All evaluation outputs (txt, json, csv)
+-- models/             <- Generated model artifacts (consumed by runtime)
|   +-- binary/
|   +-- multiclass/
|   +-- preprocessing/
|   +-- anomaly/
+-- data/
|   +-- raw/            <- Place CICIDS2017 CSVs here
|   +-- processed/      <- Cleaned Parquet files (generated by Phase 1)
+-- docs/
    +-- figures/        <- Evaluation plots (PNG) generated by Phases 4 & 6
```

---

## 4. Prerequisites & Setup

| Requirement | Notes |
|---|---|
| **Python 3.10+** | [python.org](https://www.python.org/downloads/) |
| **CICIDS2017 Dataset** | [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **8+ GB RAM** | Random Forest training peak: ~6-8 GB |
| **CPU** | All training runs on CPU â€” no GPU required |

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
```

---

## 5. Dataset Preparation

Download the **CICIDS2017** dataset from the Canadian Institute for Cybersecurity and place all CSV files in `data/raw/`:

```
data/raw/
+-- Monday-WorkingHours.pcap_ISCX.csv
+-- Tuesday-WorkingHours.pcap_ISCX.csv
+-- Wednesday-workingHours.pcap_ISCX.csv
+-- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
+-- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
+-- Friday-WorkingHours-Morning.pcap_ISCX.csv
+-- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
+-- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

**Dataset statistics:**

| Statistic | Value |
|---|---|
| Total raw flows | ~2,830,743 |
| Features (raw) | 79 (including label column) |
| Features used for ML | **78** |
| Classes | **15** (1 BENIGN + 14 attack types) |
| Class imbalance | BENIGN = 83.4% of dataset |
| Infinite values | Present in `Flow Bytes/s`, `Flow Packets/s` |

**Full class distribution:**

| Class | Count | % of Total |
|---|---|---|
| BENIGN | 2,359,787 | 83.40% |
| DoS Hulk | 231,073 | 8.17% |
| PortScan | 158,930 | 5.62% |
| DDoS | 128,027 | 4.52% |
| DoS GoldenEye | 10,293 | 0.36% |
| FTP-Patator | 7,938 | 0.28% |
| SSH-Patator | 5,897 | 0.21% |
| DoS slowloris | 5,796 | 0.20% |
| DoS Slowhttptest | 5,499 | 0.19% |
| Bot | 1,966 | 0.069% |
| Web Attack - Brute Force | 1,507 | 0.053% |
| Web Attack - XSS | 652 | 0.023% |
| Infiltration | 36 | 0.001% |
| Web Attack - SQL Injection | 21 | 0.0007% |
| Heartbleed | 11 | 0.0004% |

> **Extreme class imbalance:** Heartbleed has only 11 samples vs. 2.36M BENIGN flows. A naive classifier that always predicts BENIGN achieves 83.4% accuracy yet catches zero attacks. This is corrected using `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Random Forest.

---

## 6. Phase-by-Phase Execution

Run all scripts from the **project root** (`IDS-Codebase/`) with the virtual environment active.

### Phase 0 â€” Exploratory Data Analysis (`00_dataset_analysis.py`)

```powershell
venv\Scripts\python.exe training\00_dataset_analysis.py
```

**Purpose:** Understand the raw data before any transformation.

**What it does:**
- Loads all CICIDS2017 CSV files from `data/raw/`
- Profiles the dataset: shape, dtypes, null counts, infinite values
- Prints class distribution across all 15 labels
- Computes per-feature statistics (mean, std, min, max, percentiles)

**Output:** Console report â€” no files written.

---

### Phase 1 â€” Data Cleaning (`01_data_cleaning.py`)

```powershell
venv\Scripts\python.exe training\01_data_cleaning.py
```

**Purpose:** Convert raw, dirty CSVs into clean, ML-ready Parquet files.

**Steps performed:**
1. **Column normalisation** â€” strip whitespace from column headers (CICIDS2017 CSVs have leading spaces)
2. **Infinite value replacement** â€” replace `+/-Inf` in `Flow Bytes/s` and `Flow Packets/s` with `NaN`
3. **NaN handling** â€” fill `NaN` with per-column median (training split only, to prevent data leakage)
4. **Outlier clipping** â€” clip extreme values at 99th percentile
5. **Label standardisation** â€” normalise label strings to consistent casing
6. **Binary label creation** â€” `Label_binary = 0` (BENIGN) / `1` (any attack)
7. **Train/test split** â€” stratified 80/20 split preserving class proportions
8. **Parquet serialisation** â€” compressed Parquet for 10x faster I/O in subsequent phases

**Output:**
```
data/processed/
+-- train.parquet    (~2.26M flows, 78 features + labels)
+-- test.parquet     (~0.57M flows, 78 features + labels)
```

**Verify data quality:**
```powershell
venv\Scripts\python.exe training\verify_clean.py
```
Confirms zero `NaN` and `Inf` values; validates all 78 feature columns are present.

---

### Phase 2 â€” Binary Classifier Training (`02_binary_training.py`)

```powershell
venv\Scripts\python.exe training\02_binary_training.py
```

**Purpose:** Train XGBoost and Random Forest binary classifiers to distinguish BENIGN from ATTACK.

#### XGBoost Binary Configuration

```python
XGBClassifier(
    n_estimators      = 500,
    max_depth         = 8,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = benign_count / attack_count,  # ~3.4x weight on attack class
    eval_metric       = 'logloss',
    early_stopping_rounds = 20,
    tree_method       = 'hist'   # histogram-based: 4-10x faster on large datasets
)
```

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 500 | Sufficient depth; early stopping prevents overfitting |
| `max_depth` | 8 | Captures feature interactions without memorisation |
| `scale_pos_weight` | ~3.4 | Compensates 83.4% / 16.6% class imbalance |
| `subsample` | 0.8 | Row sampling reduces variance |
| `colsample_bytree` | 0.8 | Feature sampling adds implicit regularisation |
| `tree_method` | `hist` | 4-10x faster than exact split at CICIDS2017 scale |

#### Random Forest Binary Configuration

```python
RandomForestClassifier(
    n_estimators     = 200,
    max_depth        = None,        # full depth
    min_samples_leaf = 2,
    class_weight     = 'balanced',  # auto-adjusts for class imbalance
    n_jobs           = -1           # all CPU cores
)
```

**Output artifacts:**
- `models/binary/xgb_binary.pkl` (~1.5 MB)
- `models/binary/rf_binary.pkl` (~80 MB)

---

### Phase 3 â€” Threshold Optimisation (`03_threshold_optimization.py`)

```powershell
venv\Scripts\python.exe training\03_threshold_optimization.py
```

**Purpose:** Find the F1-optimal probability threshold for each binary classifier; fit the Isolation Forest anomaly detector.

#### Why Threshold Optimisation Matters

XGBoost outputs `P(attack)` in [0.0, 1.0]. The default threshold of 0.5 is **not optimal** for imbalanced datasets. Too low = more false positives; too high = missed attacks.

**Method â€” threshold sweep:**
```python
best_f1, best_t = 0, 0.5
for t in np.arange(0.01, 1.0, 0.01):
    preds = (probabilities >= t).astype(int)
    score = f1_score(y_true, preds)
    if score > best_f1:
        best_f1, best_t = score, t
```

**Optimal thresholds found:**

| Model | Optimal Threshold | F1 at Threshold | Interpretation |
|---|---|---|---|
| **XGBoost** | **0.8624** | **99.74%** | Must be 86% confident before classifying as attack |
| **Random Forest** | **0.5991** | **99.65%** | RF probabilities are less concentrated near 0 and 1 |
| **Isolation Forest** | **0.3690** | **58.19%** | Anomaly score cutoff (unsupervised) |

#### Isolation Forest Training

```python
IsolationForest(
    n_estimators  = 200,
    contamination = 0.05,    # expect ~5% outliers in training data
    max_samples   = 'auto',
    random_state  = 42,
    n_jobs        = -1
)
```

- Trained on **BENIGN-only flows** â€” learns the normal traffic manifold in 78-dimensional feature space
- At inference: flows deviating from normal receive a high anomaly score (0 = normal, 1 = anomalous)

**Output:**
- `models/binary/binary_threshold.json`
- `models/anomaly/isolation_forest.pkl`
- `models/preprocessing/scaler.pkl` (StandardScaler fit on training features)

---

### Phase 4 â€” Binary Model Evaluation (`04_model_evaluation.py`)

```powershell
venv\Scripts\python.exe training\04_model_evaluation.py
```

**Purpose:** Generate publication-quality evaluation plots and numeric metrics.

**Outputs generated:**

| Output File | Description |
|---|---|
| `docs/figures/roc_curve.png` | ROC AUC comparison â€” XGBoost vs. RF vs. Isolation Forest |
| `docs/figures/pr_curve.png` | Precision-Recall at all thresholds |
| `docs/figures/confusion_matrix_xgb.png` | XGBoost TP/FP/TN/FN heatmap |
| `docs/figures/confusion_matrix_rf.png` | Random Forest confusion matrix |
| `docs/figures/confusion_matrix_iso.png` | Isolation Forest binary performance |
| `docs/figures/xgb_feature_importance.png` | Top-20 XGBoost features by gain |
| `docs/figures/rf_feature_importance.png` | Top-20 RF features by Gini impurity |
| `model_training/evaluation/classification_report_xgb.txt` | Full precision/recall/F1 report |
| `model_training/evaluation/classification_report_rf.txt` | RF classification report |
| `model_training/evaluation/model_metadata.json` | All numeric metrics (precision/recall/F1/AUC) |
| `model_training/evaluation/feature_importance.csv` | XGBoost + RF importances for all 78 features |

---

### Phase 5 â€” Multiclass Training (`05_multiclass_training.py`)

```powershell
venv\Scripts\python.exe training\05_multiclass_training.py
```

**Purpose:** Train a 15-class XGBoost classifier to identify the specific attack type for flows already flagged by Stage 1.

#### Why a Separate Multiclass Model?

1. **Speed:** ~84% of flows exit at Stage 1 as BENIGN â€” the multiclass model is never invoked for them.
2. **Focused boundary:** Trains only on confirmed attack flows, providing sharper class separation.
3. **Rare class benefit:** Extremely rare classes (Heartbleed: 11 samples) benefit from the attack-only training distribution.

#### Multiclass XGBoost Configuration

```python
XGBClassifier(
    objective         = 'multi:softprob',  # returns per-class probability vector
    num_class         = 15,
    n_estimators      = 1000,
    max_depth         = 10,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    gamma             = 0.1,              # minimum loss reduction for split
    early_stopping_rounds = 30,
    eval_metric       = 'mlogloss',
    tree_method       = 'hist'
)
```

**Handling extreme class imbalance:**
- `sample_weight` proportional to inverse class frequency passed to `model.fit()`
- Heartbleed (11 training samples) receives ~200,000x more weight per sample than BENIGN

**Output artifacts:**
- `models/multiclass/xgb_multiclass.pkl` (~9 MB)
- `models/multiclass/label_encoder.pkl`
- `models/preprocessing/multiclass_feature_columns.pkl`

---

### Phase 6 â€” Multiclass Evaluation (`06_multiclass_evaluation.py`)

```powershell
venv\Scripts\python.exe training\06_multiclass_evaluation.py
```

**Purpose:** Evaluate the 15-class model with per-class breakdown, overall metrics, and visual outputs.

**Outputs:**

| Output | Description |
|---|---|
| `docs/figures/confusion_matrix_multiclass.png` | 15x15 confusion matrix heatmap |
| `docs/figures/feature_importance_multiclass.png` | Top-20 features for multiclass problem |
| `model_training/evaluation/classification_report_multiclass.txt` | sklearn full classification report |
| `model_training/evaluation/multiclass_metrics.json` | Overall + per-class metrics |
| `model_training/evaluation/per_class_metrics.csv` | Per-class support/precision/recall/F1 |

---

### Phase 7 â€” Feature Median Computation (`compute_medians.py`)

```powershell
venv\Scripts\python.exe training\compute_medians.py
```

**Purpose:** Compute per-feature median values from the training set for robust NaN imputation during live inference.

**Why medians rather than means?**
- Network traffic features are **heavily right-skewed** (`Flow Bytes/s` spans 0 to 10 Gbps)
- The mean is sensitive to outliers; the median (50th percentile) is robust regardless of distribution shape
- During live capture, some features are undefined for short-lived flows â€” these are filled with the training median
- Computed **exclusively on the training split** to prevent data leakage into evaluation

**Output:** `models/preprocessing/feature_medians.pkl`
- Python dictionary: `{feature_name: median_value}` for all 78 features

---

## 7. Model Artifacts Generated

> All artifacts below are produced during training and consumed exclusively by the **runtime IDS application**. See [REALTIME_IDS_APPLICATION.md](REALTIME_IDS_APPLICATION.md).

### Directory Structure

```text
models/
│
├── anomaly/
│   └── isolation_forest.pkl             <- Unsupervised Isolation Forest model
│
├── binary/
│   ├── binary_threshold.json            <- F1-optimal probability thresholds
│   ├── rf_binary.pkl                    <- Random Forest binary classifier (~80 MB)
│   └── xgb_binary.pkl                   <- XGBoost binary classifier (~1.5 MB)
│
├── multiclass/
│   ├── label_encoder.pkl                <- Target class string <-> int mapping
│   └── xgb_multiclass.pkl               <- 15-class XGBoost multiclass classifier (~9 MB)
│
└── preprocessing/
    ├── feature_medians.pkl              <- Median values across all 78 features
    ├── multiclass_feature_columns.pkl   <- Exact 78-feature column ordering
    └── scaler.pkl                       <- Fitted StandardScaler instance
```

---

### Detailed Breakdown of Artifact Subdirectories

#### 1. `models/anomaly/`
* **`isolation_forest.pkl`**:
  * **Algorithm**: Isolation Forest (scikit-learn)
  * **Type**: Unsupervised Anomaly Detection
  * **Purpose**: Evaluates whether network traffic deviates significantly from normal benign behavior.
  * **Output**: Anomaly decision (`1` = Benign, `-1` = Anomaly) and continuous anomaly score $s_{\text{iso}} \in [0, 1]$. Used as a secondary zero-day detector and severity modifier, rather than the primary blocking classifier.

#### 2. `models/binary/`
* **`xgb_binary.pkl`**:
  * **Algorithm**: XGBoost (`XGBClassifier`)
  * **Type**: Supervised Binary Classification
  * **Purpose**: Primary high-speed screening for `BENIGN` (0) vs. `ATTACK` (1) traffic. Best performing model ($99.92\%$ accuracy, $99.74\%$ F1).
* **`rf_binary.pkl`**:
  * **Algorithm**: Random Forest (`RandomForestClassifier`)
  * **Type**: Supervised Binary Classification
  * **Purpose**: Secondary binary classifier available for ensemble or fallback comparison ($99.89\%$ accuracy, $99.66\%$ F1).
* **`binary_threshold.json`**:
  * **Type**: Configuration JSON
  * **Purpose**: Stores F1-optimal classification decision thresholds determined during validation (e.g., `xgb: 0.8624`, `rf: 0.5991`), replacing rigid default `0.5` probability cutoffs to minimize false positives.

#### 3. `models/multiclass/`
* **`xgb_multiclass.pkl`**:
  * **Algorithm**: XGBoost Multiclass (`XGBClassifier`, `multi:softprob`)
  * **Type**: Supervised Multiclass Classification (15 classes)
  * **Purpose**: Identifies the exact attack category (e.g., `DDoS`, `DoS Hulk`, `PortScan`, `Bot`, `FTP-Patator`, `SSH-Patator`, `Web Attack - Brute Force`, `XSS`, `SQL Injection`, `Heartbleed`, `Infiltration`).
* **`label_encoder.pkl`**:
  * **Algorithm**: `LabelEncoder` (scikit-learn)
  * **Purpose**: Maps numeric class indices back to human-readable attack names (`0` $\rightarrow$ `BENIGN`, `1` $\rightarrow$ `Bot`, `2` $\rightarrow$ `DDoS`, `10` $\rightarrow$ `PortScan`, etc.).

#### 4. `models/preprocessing/`
* **`feature_medians.pkl`**:
  * **Type**: Dictionary of median values per feature
  * **Purpose**: Imputes missing values (`NaN` / `Inf`) during inference using training-set medians, maintaining pipeline stability.
* **`multiclass_feature_columns.pkl`**:
  * **Type**: Serialized list of feature names
  * **Purpose**: Enforces strict 78-feature column ordering during runtime prediction, preventing feature mismatch issues.
* **`scaler.pkl`**:
  * **Algorithm**: `StandardScaler` (scikit-learn)
  * **Purpose**: Normalizes feature vectors ($\mu=0, \sigma=1$) prior to model ingestion.

---

### End-to-End Runtime Inference Pipeline Workflow

```text
Incoming Network Flow
       │
       ▼
78 Feature Extraction (CICFlowMeter compatibility)
       │
       ▼
Preprocessing
│  ├── Impute NaNs with feature_medians.pkl
│  ├── Align columns with multiclass_feature_columns.pkl
│  └── Standardize features with scaler.pkl
       │
       ▼
Inference Execution (ModelManager Singleton)
│
├── 1. Binary Classification (xgb_binary.pkl + binary_threshold.json)
│      │
│      ├── P(Attack) < Threshold (0.8624) ──> Return BENIGN
│      │
│      └── P(Attack) >= Threshold ──> Flag as ATTACK
│                                          │
│                                          ▼
├── 2. Multiclass Classification (xgb_multiclass.pkl + label_encoder.pkl)
│      │
│      └── Identify exact attack type (e.g., DDoS, PortScan, DoS Hulk)
│
└── 3. Anomaly Scoring (isolation_forest.pkl)
       │
       └── Compute zero-day anomaly score (s_iso) for composite severity index
```

---

### Consolidated Artifact Summary Table

| Artifact File | Subdirectory | Purpose | Type / Algorithm | Output / Function |
|---|---|---|---|---|
| `isolation_forest.pkl` | `anomaly/` | Unsupervised Anomaly Detection | Isolation Forest | Anomaly score $s_{\text{iso}} \in [0, 1]$ |
| `xgb_binary.pkl` | `binary/` | Primary Binary Screening | XGBoost Classifier | $P(\text{ATTACK}) \in [0, 1]$ |
| `rf_binary.pkl` | `binary/` | Secondary Binary Classifier | Random Forest | $P(\text{ATTACK}) \in [0, 1]$ |
| `binary_threshold.json` | `binary/` | Optimal Decision Thresholds | JSON Metadata | e.g. XGB threshold `0.8624` |
| `xgb_multiclass.pkl` | `multiclass/` | 15-Class Attack Identification | XGBoost Multiclass | 15-class probability vector |
| `label_encoder.pkl` | `multiclass/` | Category Name Decoding | LabelEncoder | Numeric index $\rightarrow$ Attack string |
| `feature_medians.pkl` | `preprocessing/` | Missing Value Imputation | Dict of Float Medians | Replaces `NaN`/`Inf` with training medians |
| `multiclass_feature_columns.pkl` | `preprocessing/` | Feature Column Alignment | List of Feature Strings | Enforces exact 78-feature order |
| `scaler.pkl` | `preprocessing/` | Feature Standardization | StandardScaler | Zero-mean, unit-variance scaling |

---

**Verify all artifacts are present in the directory:**

```powershell
Get-ChildItem -Recurse models\ | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,2)}}
```

---

## 8. Evaluation Results

### Binary Classification (514,812 test samples)

The binary classification models answer the core security question: **"Is this network flow BENIGN (0) or an ATTACK (1)?"**

| Model | Accuracy | Precision | Recall | F1-Score | False Positives (FP) | False Negatives (FN) | FPR | FNR | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 **XGBoost** | **99.92%** | **99.68%** | **99.81%** | **99.74%** | **277** | **159** | **0.064%** | **0.19%** | 0.8624 |
| 🥈 **Random Forest** | 99.89% | 99.53% | 99.78% | 99.66% | 398 | 191 | 0.093% | 0.22% | 0.5991 |
| 🥉 **Isolation Forest** | 86.12% | 58.00% | 58.39% | 58.20% | 36,057 | 35,428 | 8.39% | 41.61% | 0.3690 |

---

#### Confusion Matrix Breakdown by Model

##### 1. XGBoost Classifier (Best Performing Supervised Model)
* **True Negative (TN)**: 429,362 (Benign traffic correctly classified)
* **False Positive (FP)**: 277 (Benign traffic incorrectly flagged as attack - false alarms)
* **False Negative (FN)**: 159 (Attack traffic missed and classified as benign)
* **True Positive (TP)**: 85,014 (Attack traffic correctly detected)
* **Key Observations**: Extremely low FPR (0.064%) and FNR (0.19%), providing superior detection with minimal alert fatigue.

```
                   Predicted
                BENIGN   ATTACK
Actual BENIGN   429,362     277
       ATTACK       159  85,014
```

##### 2. Random Forest Classifier
* **True Negative (TN)**: 429,241
* **False Positive (FP)**: 398
* **False Negative (FN)**: 191
* **True Positive (TP)**: 84,982
* **Key Observations**: Strong performance near XGBoost; misclassifies 121 more benign flows and misses 32 more attack flows than XGBoost.

```
                   Predicted
                BENIGN   ATTACK
Actual BENIGN   429,241     398
       ATTACK       191  84,982
```

##### 3. Isolation Forest (Unsupervised Anomaly Detector)
* **True Negative (TN)**: 393,582
* **False Positive (FP)**: 36,057
* **False Negative (FN)**: 35,428
* **True Positive (TP)**: 49,745
* **Key Observations**: High False Negative Rate (41.61%) and FPR (8.39%) as a standalone binary classifier. It struggles when attack flows resemble benign traffic. However, as an unsupervised anomaly detector, it serves as a valuable complementary layer for zero-day attack detection in the dual-stage pipeline.

```
                   Predicted
                BENIGN   ATTACK
Actual BENIGN   393,582   36,057
       ATTACK    35,428   49,745
```

#### Why XGBoost Outperforms Random Forest
1. **Sequential correction** — XGBoost builds trees sequentially, each correcting the previous tree's errors. RF builds trees independently. This focuses learning on hard minority-class attack samples.
2. **Explicit regularisation** — L1/L2 penalty terms directly penalise model complexity and prevent overfitting on imbalanced classes.
3. **Sharper probability calibration** — Probabilities concentrated near 0 and 1 allow a high-confidence threshold (0.8624), minimising false positives.
4. **Higher-order feature interactions** — Depth-8 trees capture interactions across up to 8 features simultaneously.

#### Why Isolation Forest's Lower Score is Expected
Isolation Forest is **unsupervised** — trained with zero attack labels. Its purpose is:
- Catching **zero-day attacks** outside the training distribution
- Providing a secondary anomaly score that contributes to composite severity scoring
- A ~58% F1 from an entirely unsupervised model on a 14-class attack dataset is meaningful performance (random chance is ~7%), confirming its utility alongside supervised models.

### Multiclass Classification (15 classes)

| Metric | Value | Interpretation |
|---|---|---|
| **Overall Accuracy** | **99.84%** | 513,998 of 514,812 flows correctly classified |
| Macro Precision | 82.18% | Unweighted average across all 15 classes |
| Macro Recall | 91.10% | System catches 91% of every attack type on average |
| Macro F1 | 85.07% | Harmonic mean of macro precision and recall |
| **Weighted F1** | **99.85%** | F1 weighted by class support â€” the correct operational metric |

#### Per-Class Performance

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| BENIGN | 429,639 | 100.00% | 99.86% | **99.93%** |
| DoS Hulk | 34,570 | 99.79% | 99.94% | **99.87%** |
| DDoS | 25,603 | 99.89% | 100.00% | **99.94%** |
| PortScan | 18,164 | 98.75% | 99.93% | **99.34%** |
| DoS GoldenEye | 2,056 | 98.66% | 99.95% | **99.30%** |
| FTP-Patator | 1,187 | 99.83% | 100.00% | **99.92%** |
| DoS slowloris | 1,075 | 97.80% | 99.16% | **98.48%** |
| DoS Slowhttptest | 1,046 | 97.19% | 99.04% | **98.11%** |
| SSH-Patator | 644 | 100.00% | 100.00% | **100.00%** |
| Bot | 391 | 65.33% | 99.74% | **78.95%** |
| Web Attack - Brute Force | 294 | 75.00% | 75.51% | **75.25%** |
| Web Attack - XSS | 130 | 39.61% | 46.92% | **42.96%** |
| Infiltration *(7 test samples)* | 7 | 83.33% | 71.43% | **76.92%** |
| Web Attack - SQL Injection *(4 test samples)* | 4 | 37.50% | 75.00% | **50.00%** |
| Heartbleed *(2 test samples)* | 2 | 40.00% | 100.00% | **57.14%** |

> **Weighted F1 (99.85%) is the correct operational metric.** Macro F1 (85.07%) gives equal weight to Heartbleed (2 samples) and BENIGN (429,639 samples). Classes representing <0.001% of real traffic drag the macro average down. Weighted F1 reflects actual day-to-day performance.

#### Why Web Attack Performance is Lower — A Feature-Space Limitation

Web attacks (XSS, SQL Injection, Brute Force) are inherently difficult for flow-based IDS because:
1. **Payload-based attacks** — the attack string lives inside the HTTP request body. Our feature set is metadata-only by design, to remain effective on encrypted (TLS/HTTPS) traffic.
2. **Low class support** — XSS has 130 test samples. Two misclassified flows drop precision by ~1.5%.
3. **Feature ambiguity** — a legitimate user with a slow browser generates very similar flow statistics to a web brute-force attacker.

> This is a **known fundamental limitation of flow-level IDS** for application-layer attacks. Our system still correctly flags these flows as ATTACK at Stage 1 (binary) — only the specific subtype at Stage 2 is uncertain. A complementary Web Application Firewall (WAF) handles payload-level detection.

#### Multiclass Confusion Matrix Breakdown (XGBoost 15-Class)

The 15-class confusion matrix evaluates fine-grained classification across all traffic categories. 

##### Reading the Matrix Axes
* **Y-axis (True Label)**: Actual ground-truth class.
* **X-axis (Predicted Label)**: Class assigned by the XGBoost multiclass model.
* **Main Diagonal ($X=Y$)**: Correct predictions. Values off the diagonal indicate misclassifications.

```
               Predicted Class
             BENIGN   DDoS   PortScan   ...
Actual BENIGN   ✓       ✗       ✗
       DDoS     ✗       ✓       ✗
     PortScan   ✗       ✗       ✓
```

##### Detailed Row Analysis Examples
* **BENIGN Traffic**: Out of 429,639 total test flows, **429,031** were correctly classified as BENIGN ($99.86\%$ recall). False alarms were minimal (207 predicted as Bot, 28 as DDoS, 226 as PortScan).
* **DDoS**: Out of 25,603 actual DDoS flows, **25,602** were correctly identified ($100.00\%$ recall), with only 1 flow misclassified as BENIGN.
* **DoS Hulk**: **34,549** correctly identified out of 34,570 flows ($99.94\%$ recall), with minor confusions (17 as DoS GoldenEye, 3 as PortScan, 1 as BENIGN).
* **PortScan**: **18,152** correctly identified out of 18,164 flows ($99.93\%$ recall), with 8 predicted as DoS Hulk and 1 as BENIGN.
* **Web Attack - Brute Force**: **222** correctly predicted out of 294 flows ($75.51\%$ recall), with 71 misclassified as XSS and 1 as SQL Injection due to shared HTTP session characteristics.

##### Confusion Analysis & Behavior Patterns
Misclassifications are heavily concentrated among attack categories with overlapping network behaviors:
* **Brute Force $\leftrightarrow$ XSS**: Both involve repeated HTTP POST requests over TCP sessions.
* **Slowloris $\leftrightarrow$ SlowHTTPTest**: Both are low-and-slow DoS attacks relying on uncompleted HTTP headers/requests.
* **DoS GoldenEye $\leftrightarrow$ DoS Hulk**: Both generate high-frequency HTTP GET floods causing application-level server stress.

##### Binary vs. Multiclass Comparison
* **Stage 1 (Binary Model)**: Answers **"Is this network flow an attack?"** (`BENIGN` vs `ATTACK`) for high-throughput initial filtering.
* **Stage 2 (Multiclass Model)**: Answers **"What specific type of attack is it?"** across 15 distinct categories to guide automated response actions.

##### Thesis Interpretation Summary
> *The multiclass confusion matrix demonstrates that the XGBoost classifier effectively distinguishes among different network attack categories in the CICIDS2017 dataset. Most predictions lie heavily on the main diagonal, indicating high classification precision and recall across benign traffic and major attack types (DDoS, DoS Hulk, PortScan, SSH-Patator, FTP-Patator). Minor misclassifications are concentrated exclusively among attack classes with similar network behavior (such as Web Attack–Brute Force vs. XSS, or SlowHTTPTest vs. Slowloris). Overall, the multiclass model achieves exceptional discrimination capability (99.84% overall accuracy and 99.85% weighted F1-score), making it highly effective for fine-grained threat identification in real-time network intrusion detection.*

### Top Feature Importances â€” What the Model Learned

| Rank | Feature | XGBoost Importance | Why It Matters |
|---|---|---|---|
| 1 | **Average Packet Size** | 0.2693 | DoS floods use abnormally large packets; port scans send tiny SYN-only packets |
| 2 | **Bwd Packet Length Std** | 0.2239 | High variance in server-to-client packets indicates server stress under attack |
| 3 | **Avg Bwd Segment Size** | 0.0911 | Server response size differs significantly under attack-generated traffic |
| 4 | **Bwd Header Length** | 0.0865 | Backward header structure differs between attack traffic types |
| 5 | **Max Packet Length** | 0.0514 | DDoS floods saturate with max-size packets; scans use min-size |

**Key insight:** The top two features are both **backward (server response) statistics**. The model learned to monitor the server side of the conversation as the primary attack signal â€” when a server is under attack, its responses become highly variable (DDoS), disappear (resource exhaustion), or become extremely uniform (bot).

---

## 9. Validation & Benchmarking

### Training Output Validation

```powershell
# Verify data quality (zero NaN/Inf, all 78 features present)
venv\Scripts\python.exe training\verify_clean.py

# Run inference unit tests against saved model artifacts
venv\Scripts\python.exe training\test_inference.py
```

### Inference Benchmarks

```powershell
venv\Scripts\python.exe training\benchmark_inference.py
```

| Metric | Single Thread | 4 Workers | Design Target |
|---|---|---|---|
| **Average Latency** | ~3-5 ms/flow | ~1-2 ms/flow | < 20 ms PASS |
| **Throughput** | ~300 flows/sec | **~2,700 flows/sec** | > 200/sec PASS |
| **CPU Usage (avg)** | ~15-20% | ~60-70% | < 80% PASS |
| **Memory (total)** | ~500 MB | ~550 MB | -- |

---

## 10. Model Selection Rationale

### Why XGBoost over Deep Learning / SVM / Naive Bayes?

| Criterion | XGBoost | Deep Learning | SVM | Naive Bayes |
|---|---|---|---|---|
| **Training speed** | Very fast | Hours (GPU needed) | Slow on large data | Fast |
| **Inference latency** | < 5 ms | 10-50 ms (GPU) | Slow | Fast |
| **Handles imbalance** | `scale_pos_weight` | Needs careful tuning | Poor | Very poor |
| **Tabular data** | State-of-the-art | Designed for images/sequences | Good | Assumes independence |
| **Interpretability** | Feature importance | Black box | Limited | Transparent |
| **No GPU required** | CPU-native | GPU required | CPU | CPU |

### Why Isolation Forest for Anomaly Detection?

- Trained exclusively on **BENIGN traffic** â€” no attack labels needed
- Detects **zero-day attacks** that were never seen during training
- O(n log n) complexity â€” scales to millions of samples
- Complementary to XGBoost: catches what supervised models miss

### Why Random Forest as the Benchmark?

- Serves as the **validation baseline** â€” if XGBoost did not significantly outperform RF, model selection would be re-evaluated
- Both models trained on identical data with identical features, ensuring a fair comparison

### Comparison with Published Literature

| System | Dataset | Overall Accuracy | Attack Classes | Real-Time |
|---|---|---|---|---|
| **Our IDS (XGBoost Dual-Stage)** | **CICIDS2017** | **99.84%** | **15** | Yes, 2,700 flows/sec |
| Unified ML IDS (Khraisat et al., 2019) | UNSW-NB15 | 98.20% | 9 | No - Offline only |
| Deep Learning IDS (Yin et al., 2017) | NSL-KDD | 99.10% | 5 | No - Requires GPU |
| Random Forest IDS (Farnaaz & Jabbar, 2016) | NSL-KDD | 99.67% | 5 | No - Offline only |
| Ensemble IDS (Moustafa et al., 2019) | CICIDS2017 | 97.30% | 7 | No - Offline only |

Our system achieves **state-of-the-art accuracy on the hardest comparable problem** (15 classes vs. 5-9 in all cited literature) while being **operational in real-time without GPU hardware** â€” a combination no prior compared work achieves simultaneously.

---

*IDS-Codebase FYP-2026 | CICIDS2017 Dataset | Canadian Institute for Cybersecurity*
*See also: [REALTIME_IDS_APPLICATION.md](REALTIME_IDS_APPLICATION.md)*
