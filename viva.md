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
| Telegram Notifications | ✅ Complete |
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

## Why was Telegram chosen for real-time notifications?

Telegram was chosen because:

* It requires no third-party Python library — only Python's built-in `urllib` is used.
* The Telegram Bot API is straightforward and free.
* Notifications are dispatched in a daemon background thread so they never block the detection pipeline.
* Severity filtering (`min_severity`) prevents notification fatigue on high-traffic networks.
* It can be enabled/disabled at runtime by editing `config.json` — no code changes or restarts of the full stack required.

---

## What research gap did you address?

The project attempts to bridge the gap between:

* offline academic model training
* and deployable real-time IDS systems with monitoring and alerting capabilities.

---

## Future Work

* Federated learning
* Distributed sensors
* Online learning
* Adaptive thresholding
* Quantum-safe communication mechanisms

---

For an MCA final-year project, this now resembles a small production security platform rather than a standalone machine learning experiment, which is a strong position to be in during evaluation.
