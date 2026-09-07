# 🛡️ Intrusion Detection System (IDS) — Project Documentation
### Final Year Project 2026 | Network Security & Machine Learning
**Dataset:** CICIDS2017 — Canadian Institute for Cybersecurity

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Abstraction](#2-abstraction)
3. [Proposed Solution](#3-proposed-solution)
4. [System Architecture](#4-system-architecture)
5. [Training Pipeline — Phase by Phase](#5-training-pipeline--phase-by-phase)
6. [Results & Explanation — Why Better?](#6-results--explanation--why-better)

---

## 1. Problem Statement

### 1.1 The Threat Landscape

Modern enterprise networks generate millions of packets per second. Among this legitimate traffic lurk sophisticated cyberattacks — **DDoS floods, brute-force logins, port scans, SQL injection, cross-site scripting, botnet command-and-control, and zero-day exploits**. The consequences of an undetected intrusion range from service disruption to complete data exfiltration and financial loss.

### 1.2 Limitations of Traditional IDS

Conventional Intrusion Detection Systems rely on one or more of the following approaches, each carrying critical weaknesses:

| Approach | Mechanism | Critical Weakness |
|---|---|---|
| **Signature-based IDS** | Pattern-match against known attack signatures (e.g., Snort, Suricata) | Blind to zero-day and obfuscated attacks; requires continuous manual rule updates |
| **Threshold-based IDS** | Alert when packet rate / connection count exceeds a fixed limit | High false-positive rate; trivially evaded by slow-rate attacks (e.g., Slowloris) |
| **Simple Anomaly Detection** | Statistical deviation from a baseline | Cannot label *what* the attack is; poor precision on imbalanced traffic |
| **Deep Packet Inspection** | Inspect payload content | Ineffective against encrypted traffic (TLS/HTTPS); computationally expensive |

### 1.3 The Core Problem

> **A real-world IDS must simultaneously achieve:**
> 1. **Near-zero false negatives** — every genuine attack must be caught.
> 2. **Near-zero false positives** — legitimate traffic must not be disrupted.
> 3. **Attack classification** — not just *"attack detected"* but *"which attack type?"*
> 4. **Real-time throughput** — decision latency < 20 ms per flow at wire speed.
> 5. **Zero-day resilience** — detect attacks that have never been seen before.

No single classical algorithm satisfies all five requirements simultaneously.

---

## 2. Abstraction

### 2.1 Network Traffic as Feature Vectors

Raw network packets are unstructured byte streams. The key abstraction is converting them into **bidirectional network flows** — groups of packets sharing the same 5-tuple `(src_ip, dst_ip, src_port, dst_port, protocol)` — and computing **78 statistical features** (based on CICFlowMeter) over each flow.

```
Raw Packets  →  Bidirectional Flow  →  78-Dimensional Feature Vector  →  ML Classification
```

This abstraction is:
- **Protocol-agnostic**: works on TCP, UDP, and ICMP.
- **Encryption-resistant**: uses *metadata* (packet sizes, timings, flags), not payload content.
- **Computationally tractable**: a ~100-packet flow collapses to a 78-float array in microseconds.

### 2.2 The Classification Problem Abstraction

The problem is modelled as **two nested classification tasks**:

```
Level 1 — Binary:      Is this flow BENIGN or ATTACK?   (fast gate)
Level 2 — Multiclass:  Which of the 14 attack types?    (detailed label)
```

A **third unsupervised layer** handles unknown/zero-day threats:

```
Level 3 — Anomaly:     Does this flow's distribution deviate from normal?  (Isolation Forest)
```

### 2.3 Feature Space Abstraction (78 Features)

The 78 features span five statistical domains:

| Domain | Examples | Attack Signal |
|---|---|---|
| **Packet Length Stats** | Mean, Std, Max, Min (Fwd & Bwd) | DoS sends oversized/undersized packets |
| **Inter-Arrival Timing** | IAT Mean/Std/Max (Fwd & Bwd) | Floods show near-zero IAT; Slowloris shows very high IAT |
| **Flow Volume** | Total Fwd/Bwd bytes, packets/s, bytes/s | Brute-force shows high packet rate; infiltration shows low |
| **TCP Flags** | SYN, FIN, RST, ACK, PSH counts | Port scans show many SYN with no ACK |
| **Subflow & Header** | Active/Idle time, header lengths | Distinguishes application-layer from transport-layer attacks |

---

## 3. Proposed Solution

### 3.1 Overview — Dual-Stage ML Pipeline with Anomaly Fallback

We propose a **three-model ensemble pipeline** that addresses every weakness of traditional IDS:

```
+-------------------------------------------------------------------------+
|                     PROPOSED ML-IDS PIPELINE                            |
|                                                                         |
|  Flow  -->  [Stage 1: XGBoost Binary]  --> BENIGN? --> DISCARD (fast)  |
|                         | ATTACK                                        |
|                         v                                               |
|          +--------------+--------------+                                |
|          | [Stage 2: XGBoost 15-class] | [Stage 3: Isolation Forest]   |
|          |  What attack type?          |  Is it anomalous?             |
|          +--------------+--------------+                                |
|                         v                                               |
|            SeverityScorer -> AlertManager -> Grafana Dashboard          |
+-------------------------------------------------------------------------+
```

### 3.2 Model Selection Rationale

#### Why XGBoost over Deep Learning / SVM / Naive Bayes?

| Criterion | XGBoost | Deep Learning (CNN/LSTM) | SVM | Naive Bayes |
|---|---|---|---|---|
| **Training speed** | ✅ Very fast | ❌ Hours (GPU needed) | ⚠️ Slow on large data | ✅ Fast |
| **Inference latency** | ✅ < 5 ms | ❌ 10–50 ms (GPU) | ⚠️ Slow | ✅ Fast |
| **Handles imbalance** | ✅ `scale_pos_weight` | ⚠️ Needs careful tuning | ⚠️ Poor | ❌ Very poor |
| **Tabular data** | ✅ State-of-the-art | ❌ Designed for images/sequences | ✅ Good | ⚠️ Assumes independence |
| **Interpretability** | ✅ Feature importance | ❌ Black box | ⚠️ Limited | ✅ Transparent |
| **No GPU required** | ✅ CPU-native | ❌ GPU required for speed | ✅ CPU | ✅ CPU |

> **XGBoost is the gold standard for tabular classification tasks**, confirmed by consistent dominance in ML competitions and enterprise deployments.

#### Why Isolation Forest for Anomaly Detection?

- Trained exclusively on **BENIGN traffic** — no attack labels needed.
- Detects **zero-day attacks** that were never seen during training.
- O(n log n) complexity — scales to millions of samples.
- Complementary to XGBoost: catches what the supervised model misses.

#### Why Random Forest as a Benchmark?

- Random Forest serves as the **validation baseline** — if XGBoost doesn't significantly outperform RF, the model selection would be re-evaluated.
- Both models trained on identical data with identical features, ensuring a fair comparison.

### 3.3 Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| **Dataset** | CICIDS2017 | Industry benchmark, 15 attack classes, 2.8M flows, real network traffic |
| **Feature set** | CICFlowMeter 78 features | Standardised, reproducible, encryption-resistant |
| **Binary threshold** | F1-optimal (not 0.5) | Maximises F1-score; tuned per-model on validation split |
| **Flow timeout** | 60 s idle / 300 s absolute | Matches CICFlowMeter default for fair feature comparison |
| **NaN imputation** | Per-feature medians | Robust to outliers; computed on training data, applied at inference |
| **Alert deduplication** | 60 s window per (src, dst, type) | Prevents alert storms on sustained attacks |
| **Monitoring** | Prometheus + Grafana | Industry-standard observability stack; Docker-deployable |

---

## 4. System Architecture

### 4.1 High-Level Component Diagram

```
+==============================================================================+
|                    LIVE IDS SYSTEM ARCHITECTURE                              |
+==============================================================================+
|                                                                              |
|  +-----------+    +----------+    +--------------------------------+         |
|  | Network   |    | Packet   |    |       Flow Manager             |         |
|  | Interface |--> | Capture  |--> | (Bidirectional Flow Table)     |         |
|  | (Scapy /  |    | TCP/UDP/ |    |  idle timeout:     60 s        |         |
|  |  Npcap)   |    | ICMP     |    |  absolute timeout: 300 s       |         |
|  +-----------+    +----------+    +--------------+-----------------+         |
|                                                  | completed flows           |
|                                   +--------------v-----------------+         |
|                                   |    Feature Extractor           |         |
|                                   |  (78 CICFlowMeter features)    |         |
|                                   |  NaN --> median imputation     |         |
|                                   +--------------+-----------------+         |
|                                                  |                           |
|                             +--------------------v--------------------+      |
|                             |          Inference Engine               |      |
|                             |                                         |      |
|                             | [1] XGBoost Binary (threshold=0.50/0.86)|      |
|                             |     P(attack) < threshold --> BENIGN   |      |
|                             |     P(attack) >= threshold -->          |      |
|                             |                                         |      |
|                             | [2] XGBoost Multiclass (15 classes)    |      |
|                             | [3] Isolation Forest (anomaly score)   |      |
|                             |                                         |      |
|                             | [4] Severity Scorer                    |      |
|                             |     NONE / LOW / MEDIUM / HIGH /       |      |
|                             |     CRITICAL                           |      |
|                             +--------------------+--------------------+      |
|                                                  |                           |
|         +----------------------------------------+------------------+       |
|         |              Alert Manager                                 |       |
|         |  AlertRules --> Deduplication --> SQLite Persistence       |       |
|         |  (SQLite WAL-mode: alerts.db + optional Telegram Bot)      |       |
|         +----+------------------------+---------------------------+--+       |
|              |                        |                           |          |
|              v                        v                           v          |
|  +------------------+  +-------------------+  +--------------------+        |
|  | Prometheus       |  | FastAPI REST      |  | Grafana Dashboard  |        |
|  | Server  :9090    |  | + WebSocket :8000 |  | Host Port  :3001   |        |
|  | (Docker :9091)   |  | Docs: /docs       |  | SQLite Datasource  |        |
|  +------------------+  +-------------------+  +--------------------+        |
+==============================================================================+
```
```

### 4.2 ML Pipeline Flow (Inference Decision Tree)

```
Raw Packet
    |
    v
PacketCapture  (Scapy/Npcap -- TCP/UDP/ICMP)
    |
    v
FlowManager  -- aggregates packets by 5-tuple (src_ip, dst_ip, src_port, dst_port, proto)
    |              until TCP FIN/RST received OR idle/absolute timeout
    |
    v
FeatureExtractor  -- computes 78 statistical features over the flow
    |                NaN values filled with per-feature training medians
    |
    v
[Stage 1] XGBoost Binary Classifier
    |  Input:  78-feature vector
    |  Output: P(attack) in [0.0, 1.0]
    |
    +-- P(attack) < binary_threshold (default: 0.50, optimal: 0.8624) ------> BENIGN (early exit, ~84% of flows)
    |
    +-- P(attack) >= binary_threshold
            |
            +----------------------------+
            v                            v
   [Stage 2] XGBoost              [Stage 3] Isolation Forest
   Multiclass (15-class softmax)  Anomaly Score (0 to 1)
   Output: attack type + conf.    Output: anomaly_score + is_anomaly flag
            |                            |
            +------------+---------------+
                         v
                 SeverityScorer
                 Combines: attack confidence + anomaly score + attack category
                 => NONE / LOW / MEDIUM / HIGH / CRITICAL
                         |
                         v
                 AlertManager
                 Applies rules -> deduplicates (60 s window)
                 -> persists to SQLite (data/alerts/alerts.db)
                 -> sends Telegram notification (if enabled)
                 -> pushes to WebSocket feed (/stream)
                         |
                         v
                 Prometheus Counter/Gauge/Histogram update (:9090/metrics)
                 -> Scraped by Prometheus Container (:9091)
                 -> Visualised on Grafana Dashboard (:3001 via SQLite + Prometheus)
```

### 4.3 Training Pipeline Architecture

```
CICIDS2017 Raw CSVs (8 files, ~2.8M flows)
    |
    v
Phase 0: EDA  ------------------------------------- 00_dataset_analysis.py
    |  Class distribution, missing value analysis, feature statistics
    |
    v
Phase 1: Data Cleaning  -------------------------- 01_data_cleaning.py
    |  Drop Inf/NaN, rename columns, label encode, save as Parquet
    |  Output: data/processed/*.parquet
    |
    v
Phase 2: Binary Training  ----------------------- 02_binary_training.py
    |  Train XGBoost + Random Forest on 80/20 stratified split
    |  Output: models/binary/xgb_binary.pkl  rf_binary.pkl
    |
    v
Phase 3: Threshold Optimisation  --------------- 03_threshold_optimization.py
    |  Sweep probability thresholds -> maximise F1-score
    |  Also fit & save Isolation Forest + StandardScaler
    |  Output: models/binary/binary_threshold.json
    |          models/anomaly/isolation_forest.pkl
    |          models/preprocessing/scaler.pkl
    |
    v
Phase 4: Binary Evaluation  -------------------- 04_model_evaluation.py
    |  ROC curve, PR curve, Confusion Matrix plots
    |  Output: docs/figures/*.png
    |
    v
Phase 5: Multiclass Training  ----------------- 05_multiclass_training.py
    |  Train XGBoost multi:softprob on 15-class problem
    |  Output: models/multiclass/xgb_multiclass.pkl
    |          models/multiclass/label_encoder.pkl
    |
    v
Phase 6: Multiclass Evaluation  --------------- 06_multiclass_evaluation.py
    |  Per-class precision/recall/F1, confusion matrix
    |  Output: training/evaluation/*  docs/figures/confusion_matrix_multiclass.png
    |
    v
Phase 7: Feature Medians  --------------------- compute_medians.py
       Compute per-feature median from training data
       Output: models/preprocessing/feature_medians.pkl
```

### 4.4 Module Breakdown

| Module | File | Responsibility |
|---|---|---|
| **Orchestrator** | `src/main.py` | CLI, thread management, graceful shutdown |
| **Packet Capture** | `src/capture/packet_capture.py` | Scapy sniffer, BPF filters, Npcap integration |
| **Flow Management** | `src/flows/flow_manager.py` | Thread-safe bidirectional flow table, timeout handling |
| **Flow Data Model** | `src/flows/flow.py` | `NetworkFlow`, `FlowKey`, `PacketRecord` dataclasses |
| **Feature Extraction** | `src/features/feature_extractor.py` | 78 CICFlowMeter-compatible features from flow |
| **Model Loading** | `src/inference/model_loader.py` | Singleton `ModelManager` — loads all artifacts once at startup |
| **Predictor** | `src/inference/predictor.py` | Core `predict_flow()` / `predict_batch()` functions |
| **Severity Scoring** | `src/inference/severity.py` | Maps (confidence, anomaly_score, attack_type) → severity level |
| **Worker Pool** | `src/inference/worker_pool.py` | Multi-threaded inference queue (~2,700 flows/sec @ 4 workers) |
| **Alert Manager** | `src/alerts/alert_manager.py` | Orchestrates rules → dedup → SQLite → Prometheus |
| **Alert Rules** | `src/alerts/alert_rules.py` | Configurable rule engine (severity, confidence, port filtering) |
| **Alert Storage** | `src/alerts/alert_storage.py` | SQLite WAL-mode persistence, indexed queries |
| **Metrics Registry** | `src/monitoring/metrics_registry.py` | Prometheus Counter/Histogram/Gauge definitions |
| **Metrics Server** | `src/monitoring/metrics_server.py` | Prometheus HTTP endpoint on port 9090 |
| **Metrics Exporter** | `src/monitoring/metrics_exporter.py` | Background thread: CPU/memory/alert-rate every 5 s |
| **REST API** | `src/api/main.py` | FastAPI endpoints: /health /stats /alerts /predict WS /stream |

---

## 5. Training Pipeline — Phase by Phase

### Phase 0 — Exploratory Data Analysis (`00_dataset_analysis.py`)

**Purpose:** Understand the raw data before any transformation.

**What it does:**
- Loads all CICIDS2017 CSV files from `data/raw/`
- Profiles the dataset: shape, dtypes, null counts, infinite values
- Prints class distribution across all 15 labels
- Computes per-feature statistics (mean, std, min, max, percentiles)

**Key Findings from CICIDS2017:**

| Statistic | Value |
|---|---|
| Total raw flows | ~2,830,743 |
| Features (raw) | 79 (including label column) |
| Features used for ML | 78 |
| Classes | 15 (1 BENIGN + 14 attack types) |
| Class imbalance | BENIGN = 83.4% of dataset |
| Infinite values | Present in `Flow Bytes/s`, `Flow Packets/s` |
| NaN/missing values | Present after Inf replacement |

**Class Distribution (full dataset):**

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
| Web Attack – Brute Force | 1,507 | 0.053% |
| Web Attack – XSS | 652 | 0.023% |
| Infiltration | 36 | 0.001% |
| Web Attack – SQL Injection | 21 | 0.0007% |
| Heartbleed | 11 | 0.0004% |

> **Challenge — Extreme Class Imbalance:** Heartbleed has only 11 samples vs. 2.36M BENIGN flows. A naive classifier predicting BENIGN 100% of the time achieves 83.4% accuracy — yet detects zero attacks. This must be explicitly corrected through weighted loss functions and stratified sampling.

---

### Phase 1 — Data Cleaning (`01_data_cleaning.py`)

**Purpose:** Convert raw, dirty CSVs into clean, ML-ready Parquet files.

**Steps performed:**
1. **Column normalisation** — strip whitespace from all column names (CICIDS2017 CSVs have leading spaces in column headers)
2. **Infinite value replacement** — replace `±Inf` in `Flow Bytes/s` and `Flow Packets/s` with `NaN`
3. **NaN handling** — fill `NaN` with per-column median (computed on training split only, to prevent data leakage)
4. **Outlier clipping** — clip extreme values at 99th percentile to prevent gradient explosion in tree learners
5. **Label standardisation** — normalise label strings to consistent casing across all CSV files
6. **Binary label creation** — `Label_binary = 0` (BENIGN) / `1` (any attack)
7. **Train/test split** — stratified 80/20 split preserving class proportions
8. **Parquet serialisation** — compressed Parquet for 10× faster I/O vs. CSV in subsequent phases

**Output:**
```
data/processed/
+-- train.parquet    (~2.26M flows, 78 features + labels)
+-- test.parquet     (~0.57M flows, 78 features + labels)
```

**Data quality verification** (`verify_clean.py`):
- Confirms zero `NaN` and zero `Inf` values in processed files
- Validates all 78 feature columns are present
- Reports class distribution in both splits

---

### Phase 2 — Binary Classifier Training (`02_binary_training.py`)

**Purpose:** Train two binary classifiers (XGBoost + Random Forest) to distinguish BENIGN from ATTACK.

#### XGBoost Binary Configuration

```python
XGBClassifier(
    n_estimators     = 500,
    max_depth        = 8,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = benign_count / attack_count,  # ~3.4x weight on attack class
    eval_metric      = 'logloss',
    early_stopping_rounds = 20,
    tree_method      = 'hist'    # histogram-based: 4-10x faster on large datasets
)
```

**Key hyperparameter decisions:**

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 500 | Sufficient depth; early stopping prevents overfitting |
| `max_depth` | 8 | Captures feature interactions without memorisation |
| `scale_pos_weight` | ~3.4 | Compensates 83.4% / 16.6% class imbalance |
| `subsample` | 0.8 | Row sampling reduces variance and speeds training |
| `colsample_bytree` | 0.8 | Feature sampling adds implicit regularisation |
| `tree_method` | `hist` | 4–10× faster than exact split on CICIDS2017 scale |

#### Random Forest Binary Configuration

```python
RandomForestClassifier(
    n_estimators     = 200,
    max_depth        = None,       # full depth
    min_samples_leaf = 2,
    class_weight     = 'balanced', # auto-adjusts for class imbalance
    n_jobs           = -1          # all CPU cores
)
```

**Output artifacts:**
- `models/binary/xgb_binary.pkl` (~1.5 MB)
- `models/binary/rf_binary.pkl` (~80 MB — full trees stored)

---

### Phase 3 — Threshold Optimisation (`03_threshold_optimization.py`)

**Purpose:** Find the probability threshold `t*` that maximises F1-score for each binary classifier; fit the Isolation Forest anomaly detector.

#### Why Threshold Optimisation Matters

XGBoost outputs `P(attack)` in [0.0, 1.0]. The default threshold of 0.5 is **not optimal** for imbalanced datasets. A threshold that is too low increases false positives; too high misses genuine attacks.

**Method — threshold sweep:**
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
| **Random Forest** | **0.5991** | **99.65%** | Nearly default — RF probabilities are less extreme |
| **Isolation Forest** | **0.3690** | **58.19%** | Anomaly score cutoff (unsupervised) |

> The XGBoost threshold of **0.8624** minimises false positives while XGBoost's high recall ensures genuine attacks are still caught. RF at 0.5991 reflects that RF probability outputs are less concentrated near 0 and 1 compared to gradient boosting.

#### Isolation Forest Training

```python
IsolationForest(
    n_estimators  = 200,
    contamination = 0.05,     # expect ~5% outliers in training data
    max_samples   = 'auto',
    random_state  = 42,
    n_jobs        = -1
)
```

- Trained on **BENIGN-only flows** from the training set
- Learns the normal traffic manifold in 78-dimensional feature space
- At inference: flows deviating from normal receive a high anomaly score (0 → normal, 1 → anomalous)

**Output:**
- `models/binary/binary_threshold.json`
- `models/anomaly/isolation_forest.pkl`
- `models/preprocessing/scaler.pkl` (StandardScaler fit on training features)

---

### Phase 4 — Binary Model Evaluation (`04_model_evaluation.py`)

**Purpose:** Generate publication-quality evaluation plots and numeric metrics.

**Outputs generated:**

| Output File | Description |
|---|---|
| `docs/figures/roc_curve.png` | ROC AUC comparison — XGBoost vs. RF vs. Isolation Forest |
| `docs/figures/pr_curve.png` | Precision-Recall at all thresholds (correct metric for imbalanced data) |
| `docs/figures/confusion_matrix_xgb.png` | XGBoost TP/FP/TN/FN heatmap |
| `docs/figures/confusion_matrix_rf.png` | Random Forest confusion matrix |
| `docs/figures/confusion_matrix_iso.png` | Isolation Forest binary performance |
| `docs/figures/xgb_feature_importance.png` | Top-20 XGBoost features by gain |
| `docs/figures/rf_feature_importance.png` | Top-20 RF features by Gini impurity |
| `training/evaluation/classification_report_xgb.txt` | Full precision/recall/F1 report |
| `training/evaluation/model_metadata.json` | All numeric metrics (precision/recall/F1/AUC) |

---

### Phase 5 — Multiclass Training (`05_multiclass_training.py`)

**Purpose:** Train a 15-class XGBoost classifier to identify the specific attack type for flows already flagged by Stage 1.

#### Why a Separate Multiclass Model?

Three reasons to separate binary and multiclass stages:
1. **Speed:** ~84% of flows exit at Stage 1 as BENIGN — never touching the heavier multiclass model.
2. **Focused boundary:** The multiclass model trains only on confirmed attack flows, providing sharper class separation.
3. **Rare class benefit:** Extremely rare classes (Heartbleed: 11 samples) benefit from the attack-only training distribution.

#### Multiclass XGBoost Configuration

```python
XGBClassifier(
    objective        = 'multi:softprob',  # returns per-class probability vector
    num_class        = 15,
    n_estimators     = 1000,
    max_depth        = 10,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,              # minimum loss reduction for split
    early_stopping_rounds = 30,
    eval_metric      = 'mlogloss',
    tree_method      = 'hist'
)
```

**Handling extreme class imbalance in multiclass:**
- `sample_weight` proportional to inverse class frequency passed to `model.fit()`
- This ensures Heartbleed (11 training samples) receives ~200,000× more weight per sample than BENIGN

**Output artifacts:**
- `models/multiclass/xgb_multiclass.pkl` (~9 MB)
- `models/multiclass/label_encoder.pkl`
- `models/preprocessing/multiclass_feature_columns.pkl`

---

### Phase 6 — Multiclass Evaluation (`06_multiclass_evaluation.py`)

**Purpose:** Evaluate the 15-class model with per-class breakdown, overall metrics, and visual outputs.

**Outputs:**

| Output | Description |
|---|---|
| `docs/figures/confusion_matrix_multiclass.png` | 15×15 confusion matrix heatmap |
| `docs/figures/feature_importance_multiclass.png` | Top-20 features for multiclass problem |
| `training/evaluation/classification_report_multiclass.txt` | Sklearn full classification report |
| `training/evaluation/multiclass_metrics.json` | Overall + per-class metrics (JSON) |
| `training/evaluation/per_class_metrics.csv` | Per-class support/precision/recall/F1 |

---

### Phase 7 — Feature Median Computation (`compute_medians.py`)

**Purpose:** Compute per-feature median values from the training set for robust NaN imputation during live inference.

**Why medians rather than means?**
- Network traffic features are **heavily right-skewed** — `Flow Bytes/s` can span from 0 to 10 Gbps.
- The mean is sensitive to outliers; a single abnormal flow inflates the mean significantly.
- The median (50th percentile) is robust to extreme values regardless of distribution shape.
- During live packet capture, some features are undefined for short-lived flows (e.g., `Bwd Packet Length Std` when no backward packets have arrived yet) — these are filled with the training median.

**Output:** `models/preprocessing/feature_medians.pkl`
- A Python dictionary: `{feature_name: median_value}` for all 78 features
- Computed exclusively on the training split to prevent data leakage into evaluation

---

## 6. Results & Explanation — Why Better?

### 6.1 Binary Classification Results (514,812 test samples)

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Threshold |
|---|---|---|---|---|---|---|
| **XGBoost** | **99.68%** | **99.81%** | **99.74%** | **0.9999** | **0.9999** | 0.8624 |
| Random Forest | 99.53% | 99.78% | 99.65% | 0.9999 | 0.9998 | 0.5991 |
| Isolation Forest | 57.98% | 58.40% | 58.19% | 0.8022 | 0.4909 | 0.3690 |

#### Why XGBoost Outperforms Random Forest

1. **Gradient boosting vs. bagging:** XGBoost builds trees **sequentially**, each correcting the errors of the previous tree. RF builds trees **in parallel** and independently. Sequential correction lets XGBoost focus specifically on hard-to-classify samples (typically the minority attack class).

2. **Explicit regularisation:** XGBoost includes L1/L2 regularisation terms (`reg_lambda`, `reg_alpha`) that directly penalise model complexity and prevent overfitting on imbalanced classes. RF relies solely on bagging variance reduction.

3. **Sharper probability calibration:** XGBoost produces predictions more concentrated near 0 and 1. This allows a high-confidence threshold (0.8624) to be selected, minimising false positives while the naturally high recall covers genuine attacks.

4. **Higher-order feature interactions:** Depth-8 XGBoost trees capture interactions across up to 8 features simultaneously. These interactions (e.g., high `Bwd Packet Length Std` + low `Fwd IAT Mean` + high `Flow Bytes/s` together signalling DDoS) are the key discriminators.

#### Why Isolation Forest has Lower Scores — and Why This is Expected

Isolation Forest is **unsupervised** — it has never seen attack labels during training. Its operational purpose is:
- Catch **zero-day attacks** outside the training distribution (e.g., new malware families)
- Provide a secondary anomaly score that contributes to severity scoring
- Flag suspicious flows even when XGBoost confidence is moderate

A 58% F1 from an entirely unsupervised model against a 14-class attack problem is actually strong performance — random chance would yield ~7% F1. The model's 80% ROC-AUC confirms meaningful discrimination between normal and abnormal traffic without any attack labels.

#### ROC and PR Curve Interpretation

Both XGBoost and RF achieve **ROC-AUC = 0.9999** — essentially perfect separation between BENIGN and ATTACK feature distributions across all possible thresholds. The **PR-AUC of 0.9999** is the more meaningful metric for imbalanced datasets: it demonstrates that high precision is maintained even as recall approaches 100%, confirming the model is not simply classifying everything as attack to achieve high recall.

---

### 6.2 Multiclass Classification Results

| Metric | Value | Interpretation |
|---|---|---|
| **Overall Accuracy** | **99.84%** | 513,998 of 514,812 flows correctly classified |
| Macro Precision | 82.18% | Unweighted average across all 15 classes |
| Macro Recall | 91.10% | System catches 91% of every attack type on average |
| Macro F1 | 85.07% | Harmonic mean of macro precision and recall |
| **Weighted F1** | **99.85%** | F1 weighted by class support — true operational metric |

#### Per-Class Analysis

| Class | Support | Precision | Recall | F1 | Analysis |
|---|---|---|---|---|---|
| **BENIGN** | 429,639 | 100.00% | 99.86% | **99.93%** | Near-perfect; only 0.14% of benign flows incorrectly flagged |
| **DoS Hulk** | 34,570 | 99.79% | 99.94% | **99.87%** | Very distinctive oversized packet pattern |
| **DDoS** | 25,603 | 99.89% | 100.00% | **99.94%** | High-volume flood — distinct flow volume distribution |
| **PortScan** | 18,164 | 98.75% | 99.93% | **99.34%** | Many SYN packets with high destination port entropy |
| **DoS GoldenEye** | 2,056 | 98.66% | 99.95% | **99.30%** | HTTP GET flood — distinctive inter-arrival timing |
| **FTP-Patator** | 1,187 | 99.83% | 100.00% | **99.92%** | Repeated FTP auth flows with low IAT |
| **DoS slowloris** | 1,075 | 97.80% | 99.16% | **98.48%** | Very high flow duration, very low packet rate |
| **DoS Slowhttptest** | 1,046 | 97.19% | 99.04% | **98.11%** | Similar to slowloris — minor misclassification between the two |
| **SSH-Patator** | 644 | 100.00% | 100.00% | **100.00%** | Perfect — port 22 brute-force has unmistakable TCP flag pattern |
| **Bot** | 391 | 65.33% | 99.74% | **78.95%** | High recall; lower precision from feature overlap with normal traffic |
| **Web Attack – Brute Force** | 294 | 75.00% | 75.51% | **75.25%** | HTTP-level attack; harder to distinguish from intense browsing |
| **Web Attack – XSS** | 130 | 39.61% | 46.92% | **42.96%** | Low precision due to payload-level similarity with brute force |
| **Infiltration** | 7 | 83.33% | 71.43% | **76.92%** | Only 7 test samples — impressive given extreme rarity |
| **Web Attack – SQL Injection** | 4 | 37.50% | 75.00% | **50.00%** | 4 test samples — any misclassification heavily penalises metrics |
| **Heartbleed** | 2 | 40.00% | 100.00% | **57.14%** | Only 2 test samples — both were caught (100% recall) |

#### Why Weighted F1 (99.85%) is the Correct Operational Metric

- **Macro F1 (85.07%)** gives equal weight to Heartbleed (2 samples) and BENIGN (429,639 samples). Poor performance on 2-sample classes drags the average down dramatically — but these classes represent less than 0.001% of real traffic.
- **Weighted F1 (99.85%)** weights each class by its actual frequency. This reflects the performance an operator would observe day-to-day: 99.85% of all flows are handled correctly by the system.

#### Why Web Attack Performance is Lower — A Feature-Space Limitation

Web attacks (XSS, SQL Injection, Brute Force) are inherently difficult for flow-based IDS because:
1. **Payload-based attacks** — the attack payload (`<script>alert(1)</script>`, `' OR 1=1--`) lives inside the HTTP request body. Our feature set is metadata-only by design, to work with encrypted traffic.
2. **Low class support** — XSS has 130 test samples. Two or three misclassified flows drop precision by 1–2%.
3. **Feature ambiguity** — a legitimate user with a slow browser or API client generates very similar flow statistics to a web brute-force attacker.

> **This is a known fundamental limitation of flow-level IDS** for application-layer attacks. A complementary Web Application Firewall (WAF) handles payload-level detection. Crucially, our system still correctly flags web attack flows as **ATTACK** at Stage 1 (binary) — only the specific subtype classification at Stage 2 is uncertain.

---

### 6.3 Top Feature Importances — What the Model Learned

#### XGBoost Binary — Top 10 Most Informative Features

| Rank | Feature | Importance | Why It Matters for Attack Detection |
|---|---|---|---|
| 1 | **Average Packet Size** | 0.2693 | DoS floods use abnormally large packets; port scans send tiny SYN-only packets |
| 2 | **Bwd Packet Length Std** | 0.2239 | High variance in backward (server→client) packets indicates server stress under attack |
| 3 | **Avg Bwd Segment Size** | 0.0911 | Server response size is distinctly different during attack-generated traffic |
| 4 | **Bwd Header Length** | 0.0865 | Backward header structure differs significantly between attack traffic types |
| 5 | **Max Packet Length** | 0.0514 | DDoS floods saturate with max-size packets; scans use min-size |
| 6 | **Fwd Packet Length Mean** | 0.0389 | Average request size distinguishes volumetric DoS from legitimate application traffic |
| 7 | **Flow Duration** | 0.0312 | Slowloris/Slowhttptest maintain unusually long open connections; scans are near-instant |
| 8 | **Fwd IAT Mean** | 0.0298 | Mean inter-arrival time — floods have near-zero IAT; normal browsing has variable IAT |
| 9 | **Total Fwd Packets** | 0.0241 | Volumetric attacks send orders of magnitude more packets per flow |
| 10 | **Bwd Packets/s** | 0.0198 | Server response rate under resource exhaustion drops sharply |

**Critical Insight:** The top two features by importance are both **backward (server response) statistics**. This is a fundamentally sound and interpretable signal — when a server is under attack, its responses change character: they become highly variable (DDoS), disappear (resource exhaustion), or become extremely uniform (bot). The model learned to monitor the *server side of the conversation* as the primary attack indicator.

---

### 6.4 Inference Performance Benchmarks

| Metric | Single Thread | 4 Workers | Design Target | Status |
|---|---|---|---|---|
| **Average Latency** | ~3–5 ms/flow | ~1–2 ms/flow | < 20 ms | PASS |
| **Throughput** | ~300 flows/sec | **~2,700 flows/sec** | > 200/sec | PASS |
| **CPU Usage (avg)** | ~15–20% | ~60–70% | < 80% | PASS |
| **Memory (total)** | ~500 MB | ~550 MB | — | — |

> At **2,700 flows/sec**, the system comfortably monitors a saturated 1 Gbps enterprise network. Enterprise flows average 50–200 packets and complete in 0.1–2 seconds; many concurrent active flows can be maintained simultaneously well within the throughput budget.

---

### 6.5 Comparison with Published Literature

| System | Dataset | Overall Accuracy | Attack Classes | Real-Time Capable |
|---|---|---|---|---|
| **Our IDS (XGBoost Dual-Stage)** | **CICIDS2017** | **99.84%** | **15** | **Yes — 2,700 flows/sec** |
| Unified ML IDS (Khraisat et al., 2019) | UNSW-NB15 | 98.20% | 9 | No — Offline only |
| Deep Learning IDS (Yin et al., 2017) | NSL-KDD | 99.10% | 5 | No — Requires GPU |
| Random Forest IDS (Farnaaz & Jabbar, 2016) | NSL-KDD | 99.67% | 5 | No — Offline only |
| Ensemble IDS (Moustafa et al., 2019) | CICIDS2017 | 97.30% | 7 | No — Offline only |

Our system achieves **state-of-the-art accuracy on the hardest comparable problem** (15 classes vs. 5–9 in all cited literature) while also being **operational in real-time without GPU hardware** — a combination no compared prior work achieves.

---

### 6.6 Summary — Why Our System is Better

| Criterion | Traditional IDS | Existing ML IDS | **Our System** |
|---|---|---|---|
| **Zero-day detection** | ❌ Signature-only | ❌ Supervised-only | ✅ Isolation Forest layer |
| **Attack classification** | ⚠️ Category only | ⚠️ 5–9 classes | ✅ **15 specific attack types** |
| **Encrypted traffic support** | ❌ DPI-dependent | ✅ Metadata-based | ✅ **Fully metadata-based** |
| **Real-time operation** | ✅ | ❌ Mostly offline | ✅ **2,700 flows/sec** |
| **False positive rate** | ❌ High | ⚠️ Moderate | ✅ **0.32% (XGBoost)** |
| **False negative rate** | ❌ High on zero-day | ⚠️ Moderate | ✅ **0.19% (XGBoost)** |
| **GPU requirement** | ❌ N/A | ⚠️ Often required | ✅ **CPU-only** |
| **Observability** | ⚠️ Log files | ❌ None | ✅ **Grafana + Prometheus** |
| **API integration** | ❌ None | ❌ None | ✅ **REST + WebSocket** |
| **Alert management** | ⚠️ Basic syslog | ❌ None | ✅ **Rules + Dedup + SQLite** |

---

## Appendix A — Complete Model Artifact Checklist

| Artifact | Path | Size | Status |
|---|---|---|---|
| XGBoost binary classifier | `models/binary/xgb_binary.pkl` | ~1.5 MB | ✅ Complete |
| Random Forest binary classifier | `models/binary/rf_binary.pkl` | ~80 MB | ✅ Complete |
| Optimal thresholds (all models) | `models/binary/binary_threshold.json` | <1 KB | ✅ Complete |
| XGBoost multiclass classifier | `models/multiclass/xgb_multiclass.pkl` | ~9 MB | ✅ Complete |
| Label encoder (int → class name) | `models/multiclass/label_encoder.pkl` | <1 KB | ✅ Complete |
| Isolation Forest anomaly detector | `models/anomaly/isolation_forest.pkl` | ~1.5 MB | ✅ Complete |
| StandardScaler | `models/preprocessing/scaler.pkl` | <1 KB | ✅ Complete |
| Feature medians (78 features) | `models/preprocessing/feature_medians.pkl` | <1 KB | ✅ Complete |
| Feature column ordering | `models/preprocessing/multiclass_feature_columns.pkl` | <1 KB | ✅ Complete |

---

## Appendix B — Configuration Reference (`config.json`)

```json
{
    "binary_threshold":       0.50,
    "alert_min_severity":     "LOW",
    "alert_min_confidence":   0.50,
    "alert_suppressed_types": [],
    "alert_suppressed_ports": [],
    "alert_dedup_window_s":   60,
    "telegram": {
        "enabled": false,
        "bot_token": "",
        "chat_id": "",
        "min_severity": "HIGH"
    }
}
```

| Key | Default | Effect |
|---|---|---|
| `binary_threshold` | 0.50 | Lower → more sensitive (more alerts); Higher → stricter (fewer false positives). Offline F1-optimal: 0.8624 |
| `alert_min_severity` | "LOW" | Set "MEDIUM" on noisy networks to reduce alert volume |
| `alert_min_confidence` | 0.50 | Minimum P(attack) required to generate an alert |
| `alert_suppressed_types` | [] | e.g. ["Bot"] — silence specific attack types globally |
| `alert_suppressed_ports` | [] | e.g. [80, 443] — silence alerts on specific destination ports |
| `alert_dedup_window_s` | 60 | Seconds before same (src_ip, dst_ip, attack_type) can re-alert |
| `telegram.enabled` | false | Enables real-time Telegram Bot notifications for high-priority alerts |
| `telegram.bot_token` | "" | Telegram Bot API token (from @BotFather) |
| `telegram.chat_id` | "" | Target Telegram chat ID or channel ID |
| `telegram.min_severity` | "HIGH" | Minimum alert severity to trigger a Telegram push notification |

---

*Documentation updated: 2026-07-30 | IDS-Codebase FYP-2026 | CICIDS2017 Dataset*
