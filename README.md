# 🛡️ AI-Driven Real-Time Intrusion Detection System (AI-RTIDS)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.103.0-green.svg)](https://fastapi.tiangolo.com/)
[![ML Framework](https://img.shields.io/badge/XGBoost-1.7.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-lightgrey.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Monitoring](https://img.shields.io/badge/Prometheus-Grafana-red.svg)](https://grafana.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A production-grade, low-latency **AI-Driven Real-Time Intrusion Detection System (AI-RTIDS)** built on the **CICIDS2017** dataset. The system captures live network traffic, reconstructs bidirectional flows, extracts 78 statistical features in real time, classifies flows via a dual-stage machine learning pipeline, detects zero-day anomalies, generates deduplicated alerts, and exposes live monitoring via Prometheus & Grafana.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Performance](#-model-performance-cicids2017)
- [Project Documentation](#-project-documentation)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Offline Training Pipeline](#-offline-training-pipeline)
- [Compiling LaTeX Documentation](#-compiling-latex-documentation)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## ⚡ Key Features

- **Real-Time Packet Capture & Flow Reconstruction**: Captures raw network packets using Scapy (Npcap/libpcap) and aggregates them into bidirectional network flows based on 5-tuple keys `(src_ip, dst_ip, src_port, dst_port, protocol)`.
- **78-Feature Real-Time Extractor**: Extracts statistical features matching the CICFlowMeter standard (packet lengths, inter-arrival times, TCP flag counts, active/idle periods).
- **Dual-Stage ML Classification Pipeline**:
  - **Stage 1 (Binary XGBoost)**: High-throughput screening (`BENIGN` vs `ATTACK`). Achieves **99.68% Precision**, **99.81% Recall**, and **99.74% F1-Score**.
  - **Stage 2 (Multiclass XGBoost)**: Detailed 15-class attack categorization for identified threats. Achieves **99.84% Overall Accuracy** and **99.85% Weighted F1**.
- **Zero-Day Anomaly Detection**: Unsupervised **Isolation Forest** computes anomaly scores for novel threats with composite risk scoring:
  $$S = 0.65 \times p_{\text{attack}} + 0.35 \times s_{\text{iso}}$$
- **Intelligent Alert Engine & Alertmanager**: Rule-based filtering with a 60-second sliding window deduplication mechanism and persistent storage in **SQLite** (`alerts.db`). Alerts are exposed via Prometheus metrics to **Prometheus Alertmanager** for automated HTML email notifications via Gmail SMTP.
- **Prometheus & Grafana Integration**: Real-time operational metrics export (flow counts, packet rates, latency, alert counts by severity) rendered on pre-configured Grafana dashboards.
- **REST & WebSocket API**: Exposes FastAPI endpoints for alert queries, system status, live metric streaming, and interactive Swagger UI.

---

## 🏗️ System Architecture

```
                                 ┌──────────────────────────────────────────────┐
                                 │          LIVE NETWORK INTERFACE              │
                                 └──────────────────────┬───────────────────────┘
                                                        │ Raw Packets (Scapy/Npcap)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │     Bidirectional Flow Manager (5-Tuple)     │
                                 └──────────────────────┬───────────────────────┘
                                                        │ Completed Flows
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │     78-Feature Extractor (CICFlowMeter)     │
                                 └──────────────────────┬───────────────────────┘
                                                        │ Feature Vector (78 floats)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │    STAGE 1: XGBoost Binary Classifier        │
                                 └──────────────┬────────────────┼──────────────┘
                                                │                │
                             BENIGN (p < 0.862) │                │ ATTACK (p ≥ 0.862)
                                                ▼                ▼
                                 ┌──────────────────┐  ┌────────────────────────┐
                                 │ Isolation Forest │  │ STAGE 2: Multiclass    │
                                 │ Anomaly Check    │  │ XGBoost (15 Classes)   │
                                 └────────┬─────────┘  └───────────┬────────────┘
                                          │                        │
                                          └───────────┬────────────┘
                                                      │ Composite Score & Severity
                                                      ▼
                                 ┌──────────────────────────────────────────────┐
                                 │ Alert Manager (Rules, Dedup, SQLite Storage) │
                                 └──────────────┬────────────────┬──────────────┘
                                                │                │
                                                ▼                ▼
                                 ┌──────────────────┐  ┌────────────────────────┐
                                 │ Prometheus Metrs │  │ FastAPI REST & WS API  │
                                 └────────┬─────────┘  └────────────────────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │ Grafana Dashbrd  │
                                 └──────────────────┘
```

---

## 📊 Model Performance (CICIDS2017)

Evaluated on **514,812 test flow samples** from the CICIDS2017 dataset.

### 1. Binary Classification Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | F1-Threshold | Model Size |
|---|---|---|---|---|---|---|---|
| **XGBoost (Binary)** | **99.92%** | **99.68%** | **99.81%** | **99.74%** | **0.9999** | 0.8624 | **1.47 MB** |
| Random Forest (Binary) | 99.89% | 99.53% | 99.78% | 99.66% | 0.9999 | 0.5991 | 76.10 MB |
| Isolation Forest *(Anomaly)* | 86.12% | 58.00% | 58.39% | 58.20% | 0.8022 | 0.3690 | 1.42 MB |

### 2. Multiclass XGBoost Performance (15 Classes)

- **Overall Accuracy**: **99.84%**
- **Weighted F1-Score**: **99.85%**
- **Macro F1-Score**: **85.07%**
- **Total Model Suite Size**: **11.58 MB** (XGBoost Binary + Multiclass + Isolation Forest)
- **Inference Latency**: **5.08 ms** average per flow (single-thread), scaling to **~2,700 flows/sec** with 14 worker threads.

#### Per-Class Classification Breakdown:

| Traffic Class | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **BENIGN** | 429,639 | 0.99997 | 0.9986 | 0.9993 |
| **DoS Hulk** | 34,570 | 0.9979 | 0.9994 | 0.9987 |
| **DDoS** | 25,603 | 0.9989 | 1.0000 | 0.9994 |
| **PortScan** | 18,164 | 0.9875 | 0.9993 | 0.9934 |
| **DoS GoldenEye** | 2,056 | 0.9866 | 0.9995 | 0.9930 |
| **FTP-Patator** | 1,187 | 0.9983 | 1.0000 | 0.9992 |
| **DoS slowloris** | 1,075 | 0.9780 | 0.9916 | 0.9848 |
| **DoS Slowhttptest** | 1,046 | 0.9719 | 0.9904 | 0.9811 |
| **SSH-Patator** | 644 | 1.0000 | 1.0000 | 1.0000 |
| **Bot** | 391 | 0.6533 | 0.9974 | 0.7895 |
| **Web Attack - Brute Force** | 294 | 0.7500 | 0.7551 | 0.7525 |
| **Web Attack - XSS** | 130 | 0.3961 | 0.4692 | 0.4296 |
| **Infiltration** | 7 | 0.8333 | 0.7143 | 0.7692 |
| **Web Attack - SQL Injection** | 4 | 0.3750 | 0.7500 | 0.5000 |
| **Heartbleed** | 2 | 0.4000 | 1.0000 | 0.5714 |

---

## 📚 Project Documentation

Detailed documentation files are available in the repository:

| Document | Description |
|---|---|
| [`docs/documentation.tex`](docs/documentation.tex) | Complete academic LaTeX report detailing research design, equations, architecture, algorithms, and experimental analysis |
| [`docs/REALTIME_IDS_APPLICATION.md`](docs/REALTIME_IDS_APPLICATION.md) | Runtime architecture, flow handling, inference engine, alerting, Prometheus, and API reference |
| [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md) | Offline model development guide — dataset preparation, cleaning, training scripts, and evaluation |
| [`START-GUIDE.md`](START-GUIDE.md) | Quick command reference for setup, monitoring stack, running IDS, and testing |
| [`TrainingPhase-PanelExpelnation.md`](TrainingPhase-PanelExpelnation.md) | Deep-dive explanation for project viva/presentation covering problem, architecture, per-phase evaluation, and results |
| [`viva.md`](viva.md) | Comprehensive viva voce preparation guide with expected questions, answers, and system design breakdowns |

---

## 📁 Repository Structure

```
IDS-Codebase/
├── src/                             # Live application source code
│   ├── main.py                      # CLI entry point
│   ├── api/                         # FastAPI REST & WebSocket endpoints
│   ├── capture/                     # Scapy live packet capture engine
│   ├── flows/                       # Flow manager (5-tuple grouping, timeouts)
│   ├── features/                    # 78 CICFlowMeter feature extraction module
│   ├── inference/                   # Model loader, predictor, worker pool
│   ├── alerts/                      # Alert rules, deduplication, SQLite persistence
│   └── monitoring/                  # Prometheus metrics exporter
├── alertmanager/                    # Prometheus Alertmanager configuration & templates
├── alert_rules.yml                  # Prometheus alert rules (HighSeverityAttack, etc.)
├── model_training/                  # Offline model training pipeline
├── models/                          # Trained ML artifacts (.pkl, .json)
│   ├── binary/                      # XGBoost & Random Forest binary classifiers
│   ├── multiclass/                  # XGBoost multiclass classifier & LabelEncoder
│   ├── preprocessing/               # StandardScaler & feature medians
│   ├── stacking/                    # Logistic Regression meta-learner
│   └── anomaly/                     # Isolation Forest model
├── docs/                            # LaTeX report, Markdown docs, figures, results
│   ├── documentation.tex            # Full academic LaTeX documentation
│   ├── figures/                     # Architecture diagrams
│   └── results/                     # Confusion matrices & evaluation plots
├── grafana/                         # Grafana dashboard definitions
├── config.json                      # Runtime application configuration
├── docker-compose.yml               # Prometheus + Alertmanager + Grafana stack definition
├── requirements.txt                 # Python dependencies
└── README.md                        # Project landing document
```

---

## ⚡ Quick Start Guide

### Prerequisites

- **Python**: 3.10+
- **Packet Capture Driver**: 
  - **Windows**: Install [Npcap](https://npcap.com/#download) (enable *"WinPcap API-compatible mode"* during setup).
  - **Linux**: Install `libpcap-dev` (`sudo apt install libpcap-dev`).
- **Docker & Docker Compose**: For running Prometheus and Grafana dashboards.

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/your-repo/IDS-Codebase.git
cd IDS-Codebase

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Linux/macOS: source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

### 2. Identify Network Interface

```powershell
python src/main.py --list-interfaces
```

*Example Output (Windows):*
```
[0] \Device\NPF_{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX} (Intel(R) Wi-Fi 6 AX201)
[1] \Device\NPF_{YYYYYYYY-YYYY-YYYY-YYYY-YYYYYYYYYYYY} (Realtek PCIe GbE Family Controller)
```

### 3. Launch Monitoring Stack (Prometheus, Alertmanager & Grafana)

```powershell
docker compose up -d
```

- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) *(Default login: `admin` / `admin`)*
- **Prometheus UI**: [http://localhost:9091](http://localhost:9091)
- **Alertmanager UI**: [http://localhost:9093](http://localhost:9093)

### 4. Start the IDS Engine

```powershell
# Replace with your interface GUID from Step 2
python src/main.py --interface "\Device\NPF_{YOUR-INTERFACE-GUID}" --workers 4
```

### 5. Access Interactive API & Documentation

- **Swagger UI (FastAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API View**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Prometheus Metrics Endpoint**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 🔬 Offline Training Pipeline

To retrain the machine learning models from raw CICIDS2017 CSV files:

```powershell
# 1. Place raw CICIDS2017 CSV files under data/raw/
# 2. Execute training scripts sequentially:
python model_training/00_verify_environment.py
python model_training/01_load_and_merge.py
python model_training/02_preprocess.py
python model_training/03_train_binary.py
python model_training/04_train_multiclass.py
python model_training/05_train_anomaly.py
python model_training/06_evaluate_all.py
```

*Trained artifacts will automatically be saved into the `models/` directory for runtime loading.*

---

## 📄 Compiling LaTeX Documentation

The complete academic paper/report is provided in [`docs/documentation.tex`](docs/documentation.tex).

To compile the LaTeX file into a PDF:

```powershell
cd docs
pdflatex documentation.tex
pdflatex documentation.tex   # Run twice to resolve section/figure cross-references
```

*Or open `docs/documentation.tex` directly in Overleaf or TeXStudio.*

---

## 👤 Author & Acknowledgements

- **Author**: Bommali Tarun
- **Department**: Department of Information Technology, JNTU GV (CEV)
- **Project**: Final Year Project 2026 — AI-Driven Real-Time Intrusion Detection System
- **Dataset**: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) provided by the Canadian Institute for Cybersecurity (CIC), University of New Brunswick.

---
