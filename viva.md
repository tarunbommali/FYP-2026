This is a good stopping point for the project lifecycle.

From a software engineering perspective, your IDS has now moved through the typical stages:

```text
Research
    ↓
Dataset Preparation
    ↓
Model Training
    ↓
Inference Pipeline
    ↓
System Integration
    ↓
Testing
    ↓
Deployment Tooling
    ↓
Documentation
    ↓
Viva Readiness
```

The cleanup you performed improves maintainability and makes the repository easier to explain during evaluation.

## Current State

| Area                 | Status     |
| -------------------- | ---------- |
| Training Pipeline    | ✅ Complete |
| Runtime IDS Pipeline | ✅ Complete |
| Model Artifacts      | ✅ Complete |
| FastAPI Integration  | ✅ Complete |
| Alert Engine         | ✅ Complete |
| SQLite Storage       | ✅ Complete |
| Prometheus Alertmanager | ✅ Complete |
| Prometheus Metrics   | ✅ Complete |
| Grafana Dashboard    | ✅ Complete |
| Integration Tests    | ✅ Passing  |
| Documentation        | ✅ Complete |
| Repository Structure | ✅ Clean    |
| Demo Scripts         | ✅ Ready    |
| Viva Preparation     | ✅ Ready    |

---

# Hierarchical Decision Architecture (3 Stages)

```text
Live Network Traffic
        │
        ▼
Packet Capture (Npcap) → Flow Generation (5-Tuple) → Feature Extraction (78 Features)
        │
        ▼
─────────────────────────────────────────────────────────────────────────────
Stage 1: Evidence Generation (Base Learners)
  - Random Forest      ──► P_Attack(RF)
  - XGBoost Binary     ──► P_Attack(XGB)
  - Isolation Forest   ──► Anomaly Score
─────────────────────────────────────────────────────────────────────────────
        │
        ▼
Stage 2: Decision Fusion (Binary Decision Engine)
  - Logistic Regression Meta Learner ──► Final Binary Decision (BENIGN / ATTACK)
        │
        ├──────────────────────────┐
        ▼                          ▼
     BENIGN                      ATTACK
  (Stop Processing)                │
                                   ▼
─────────────────────────────────────────────────────────────────────────────
Stage 3: Decision Routing & Novelty Detection
  - Isolation Forest Novelty Filter (iso_score >= UNKNOWN_THRESHOLD)
        │
        ├──────────────────────────┐
        ▼                          ▼
  UNKNOWN_ATTACK             Known Attack
(Skip Multiclass)                  │
                                   ▼
                       XGBoost Multiclass (15 Classes)
─────────────────────────────────────────────────────────────────────────────
        │
        ▼
Confidence + Severity Engine ──► SQLite → Prometheus → Grafana → Alerts
```

---

# Model Responsibilities & Academic Terminology

| Stage | Component | Academic Terminology | Technical Responsibility |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Random Forest | Evidence Generation | Supervised probability estimation $P_{\text{RF}}$ |
| **Stage 1** | XGBoost Binary | Evidence Generation | Supervised probability estimation $P_{\text{XGB}}$ |
| **Stage 1** | Isolation Forest | Evidence Generation | Novelty / Anomaly score estimation |
| **Stage 2** | Logistic Regression | Decision Fusion Engine | Stacking combiner $\rightarrow$ Binary decision |
| **Stage 3** | Isolation Forest Filter | Novelty Detection Filter | Zero-day / Unseen threat routing |
| **Stage 3** | XGBoost Multiclass | Attack Classification | Categorizes 15 known attack classes |
| **Post** | Severity Engine | Risk Assessment Engine | Operational impact scoring (LOW–CRITICAL) |

---

# Confidence Mapping Strategy

| Detection Case | Deciding Model | Confidence Source |
| :--- | :--- | :--- |
| **BENIGN** | Binary Decision Engine | `1.0` (Definitive non-attack) |
| **UNKNOWN_ATTACK** | Meta Learner | `meta_probability` (LR decision boundary) |
| **Known Attack** | XGBoost Multiclass | $\max(\text{multiclass\_probabilities})$ |

---

# Suggested Viva Narrative (30 Seconds)

> "The system uses a 3-stage hierarchical decision architecture. First, **Stage 1 (Evidence Generation)** extracts 78 network features and passes them through Random Forest, XGBoost Binary, and Isolation Forest. Second, **Stage 2 (Decision Fusion)** uses a Logistic Regression meta-learner to combine this evidence into a final binary decision. BENIGN traffic stops immediately. For detected attacks, **Stage 3 (Decision Routing & Novelty Detection)** evaluates the Isolation Forest novelty filter. Novel or zero-day threats are flagged as `UNKNOWN_ATTACK` and skip multiclass inference. Known attacks are categorized by a 15-class XGBoost multiclass model. Finally, confidence and operational severity are calculated, stored in SQLite, exported to Prometheus, visualized in Grafana, and dispatched via alerts."

---

# Expected Viva Questions

## Why flow-based detection instead of packet-based detection?

Because attacks are behavioral patterns that emerge across multiple packets rather than individual packets. Flow features such as packet counts, duration, and inter-arrival times provide richer context for machine learning models. 

---

## Why XGBoost?

* High accuracy on tabular cybersecurity datasets.
* Fast inference suitable for real-time deployment.
* Handles feature interactions effectively.

---

## Why Isolation Forest?

To detect anomalous behavior and previously unseen attacks that supervised models may miss.

---

## Why Grafana?

Grafana provides operational visibility into:

* Attack count
* Flow rate
* Packet rate
* Resource utilization
* Model latency

---

## Why CICIDS2017?

It contains modern attack scenarios and realistic traffic patterns compared with older datasets such as KDD99 and NSL-KDD, which are now considered outdated for evaluating modern IDS systems. 

---

## Why was Prometheus Alertmanager chosen for real-time notifications?

Prometheus Alertmanager was chosen because:

* It decouples alert detection from notification delivery — the Python IDS only exports metrics and evaluates local alert rules, avoiding external network I/O in the detection loop.
* Prometheus evaluates declarative alert rules (`alert_rules.yml`) on scraped metrics and fires alerts directly to Alertmanager.
* Alertmanager provides native deduplication, grouping, rate-limiting, and routing to standard email (SMTP) without requiring proprietary bot APIs.
* Separation of concerns: the ML runtime focuses strictly on packet capture, feature extraction, and inference.

---

## What research gap did you address?

The project attempts to bridge the gap between:

* offline academic model training
* and deployable real-time IDS systems with monitoring and alerting capabilities.

---

## What are the known evaluation methodology constraints in the current pipeline?

* **Threshold Optimization on Test Data**: In the initial binary prototype pipeline (`03_threshold_optimization.py`), the threshold was evaluated on the 20% test slice. The multiclass pipeline (`05_multiclass_training.py`) correctly adopted a 70/10/20 train/validation/test split, which is the recommended architectural target for future binary re-training.
* **Meta-Learner Training Split**: The initial logistic regression meta-learner (`train_meta_learner.py`) was fitted on predictions from the 20% test slice. In rigorous production re-training, an isolated validation split or K-fold out-of-fold cross-validation should be used to avoid data leakage.
* **Data Imbalance**: Several minor attack classes (e.g. Infiltration, Heartbleed) have very low support in CICIDS2017, leading to wider confidence variance for rare classes.

---

## Architectural Distinctions: Live Capture vs. Offline Flow Inference vs. Model Evaluation

| Paradigm | Input Pipeline | Processing Chain | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Live Runtime** | Live NIC via Npcap | Packet $\rightarrow$ FlowManager $\rightarrow$ FeatureExtractor (78 features) $\rightarrow$ `predict_flow()` | Real-time network threat detection |
| **Offline Flow Inference** | Held-Out Test Parquet (`X_test_binary.parquet`) | Row dict (78 features) $\rightarrow$ `predict_flow()` | Validates complete deployed runtime inference pipeline against unseen flows |
| **Formal Model Evaluation** | Held-Out Test Parquet (`X_test_binary.parquet`) | Batch DataFrame $\rightarrow$ `model.predict_proba()` directly | Evaluates standalone algorithmic model performance (`04_model_evaluation.py`) |

> **Key Terminology Defense:**
> * Do not refer to Parquet test rows as "test packets" — they are **test flow feature vectors**.
> * Do not refer to Parquet evaluation as "PCAP replay" — it is **offline flow-level runtime inference**.
> * "PCAP replay" is strictly reserved for packet ingestion: `raw packets -> FlowManager -> FeatureExtractor -> Predictor`.

> **Recommended Panel Defense Narrative:**
> *"The CICIDS2017 dataset is divided into training and test subsets. The training subset is used to fit the machine-learning models, while the isolated test subset is passed through the same inference function used by the real-time IDS (`predict_flow`). This enables evaluation of the deployed inference logic without requiring packet replay. For real-time operation, the same inference function receives flow features generated from Npcap-captured traffic."*

---

## Future Work

* Federated learning
* Distributed sensors
* Online learning
* Adaptive thresholding
* Quantum-safe communication mechanisms

---

For an MCA final-year project, this now resembles a small production security platform rather than a standalone machine learning experiment, which is a strong position to be in during evaluation.
