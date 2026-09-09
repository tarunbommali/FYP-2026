```markdown
# 🚀 IDS Quick-Start Guide

> **One file, everything you need to run the system.**  
> Three sections: [Training](#1-training-pipeline) · [Real-Time IDS + FastAPI](#2-real-time-ids--fastapi) · [Grafana Monitoring](#3-grafana-monitoring-stack)

---

## ⚠️ Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | [python.org](https://www.python.org/downloads/) |
| **Npcap** | [npcap.com](https://npcap.com/) — **required for live packet capture on Windows** |
| **Docker Desktop** | [docker.com](https://www.docker.com/products/docker-desktop/) — for Grafana + Prometheus |

---

## 🔧 One-Time Setup

```powershell
# Create virtual environment
python -m venv venv

# Install all dependencies
venv\Scripts\pip.exe install -r requirements.txt

# Verify models load correctly
venv\Scripts\python.exe -c "from inference.model_loader import MODELS; print(f'OK — {len(MODELS.feature_columns)} features loaded')"
```

> ✅ Expected output: `OK — 78 features loaded`

---

## 1. Training Pipeline

> **Skip if models are already trained** — all `.pkl` and `.json` artifacts already exist in `models/`.

### 1.1 Place the Dataset

Download [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and copy the CSV files into `data/raw/`.

### 1.2 Run Scripts Sequentially

```powershell
# Phase 0 — Exploratory Data Analysis
venv\Scripts\python.exe model_training\00_dataset_analysis.py

# Phase 1 — Data Cleaning (CSV → Parquet)
venv\Scripts\python.exe model_training\01_data_cleaning.py

# Phase 2 — Binary Classifier Training (XGBoost + Random Forest)
venv\Scripts\python.exe model_training\02_binary_training.py

# Phase 3 — Threshold Optimisation (F1-optimal + ISO normalisation params)
venv\Scripts\python.exe model_training\03_threshold_optimization.py

# Phase 4 — Binary Model Evaluation (ROC, PR, Confusion Matrix → docs/figures/)
venv\Scripts\python.exe model_training\04_model_evaluation.py

# Phase 5 — Multiclass Classifier Training (15-class XGBoost)
venv\Scripts\python.exe model_training\05_multiclass_training.py

# Phase 6 — Multiclass Evaluation + Per-Class Metrics
venv\Scripts\python.exe model_training\06_multiclass_evaluation.py

# Phase 7 — Compute Feature Medians (required for live NaN imputation)
venv\Scripts\python.exe model_training\compute_medians.py
```

### 1.3 Verify Training Output

```powershell
Get-ChildItem -Recurse models\ | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,2)}}
```

Expected key artifacts:

| File | Location | Size |
|---|---|---|
| `xgb_binary.pkl` | `models/binary/` | ~1.5 MB |
| `xgb_multiclass.pkl` | `models/multiclass/` | ~9 MB |
| `isolation_forest.pkl` | `models/anomaly/` | ~1.5 MB |
| `rf_binary.pkl` | `models/binary/` | ~80 MB |
| `binary_threshold.json` | `models/binary/` | <1 KB |
| `feature_medians.pkl` | `models/preprocessing/` | <1 KB |
| `scaler.pkl` | `models/preprocessing/` | <1 KB |
| `multiclass_feature_columns.pkl` | `models/preprocessing/` | <1 KB |

### 1.4 Run Tests & Benchmarks

```powershell
# Inference unit tests
venv\Scripts\python.exe model_training\test_inference.py

# Latency + throughput benchmark (target: ~5 ms avg, ~200 pred/sec)
venv\Scripts\python.exe model_training\benchmark_inference.py
```

---

## 2. Real-Time IDS + FastAPI

> The live application. All commands run from the **project root** (`IDS-Codebase/`).

### 2.1 Find Your Network Interface

```powershell
venv\Scripts\python.exe src\main.py --list-interfaces
```

**Interface → Adapter mapping on this machine:**

| NPF GUID | Adapter | Status | Use? |
|---|---|---|---|
| `{E2D51F95-...}` | **MediaTek MT7921 Wi-Fi 6** (10.157.52.29) | ✅ Up | **← Use this** |
| `{3459A679-...}` | ExpressVPN TAP Adapter | Disconnected | ❌ |
| `{93C893C6-...}` | ExpressVPN TUN Driver | Disconnected | ❌ |
| `{54E2D776-...}` / `{20B181E0-...}` | Wi‑Fi Direct Virtual | N/A | ❌ |
| `{DB30984C-...}` / `{CC158697-...}` / `{F95A5E74-...}` | WAN Miniports | N/A | ❌ |
| `NPF_Loopback` | Loopback (127.0.0.1) | — | Test only |

> 🎯 **Your active interface:** `\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}`

---

### 2.2 Launch the IDS (FastAPI REST + WebSocket Enabled)

Run the IDS using the main runtime script. By default, the FastAPI server will start automatically alongside the packet capture loop.

```powershell
# Standard run (default: API active on port 8000, 1 inference worker)
venv\Scripts\python.exe src\main.py --interface "\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}"

# High-throughput option (e.g., 4 inference workers for ~2,700 flows/sec)
venv\Scripts\python.exe src\main.py --interface "\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}" --workers 4

# Custom capture filter (e.g., TCP-only traffic)
venv\Scripts\python.exe src\main.py --interface "\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}" --filter "tcp"

# Custom port and debug logging
venv\Scripts\python.exe src\main.py --interface "\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}" --api-port 8080 --log-level DEBUG
```

---

### 2.3 All CLI Options

```
--interface  / -i    Network interface name     (default: Ethernet)
--workers    / -w    Inference worker threads   (default: 1 | use 4 for ~2700 flows/s)
--log-level  / -l    DEBUG | INFO | WARNING | ERROR  (default: INFO)
--metrics-port / -m  Prometheus endpoint port   (default: 9090)
--api                Start FastAPI REST/WebSocket server (enabled by default)
--api-port           FastAPI port               (default: 8000)
--list-interfaces    Print available interfaces and exit
--filter             BPF capture filter string  (default: "ip")
```

---

### 2.4 FastAPI Endpoints

> Interactive docs (Swagger UI): **http://localhost:8000/docs**

| Method | URL | Description |
|---|---|---|
| `GET` | `http://localhost:8000/health` | System health + uptime + model status |
| `GET` | `http://localhost:8000/stats` | Alert counters (flows, attacks, severities) |
| `GET` | `http://localhost:8000/alerts?limit=50` | Recent alerts from SQLite |
| `GET` | `http://localhost:8000/models` | Loaded model metadata |
| `POST` | `http://localhost:8000/predict` | Single-flow inference (JSON feature dict) |
| `WS` | `ws://localhost:8000/stream` | Real-time WebSocket alert feed |

**Test the API with PowerShell:**

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json

# Live stats snapshot
Invoke-RestMethod http://localhost:8000/stats | ConvertTo-Json

# Last 10 alerts
Invoke-RestMethod "http://localhost:8000/alerts?limit=10" | ConvertTo-Json

# Model info
Invoke-RestMethod http://localhost:8000/models | ConvertTo-Json

# Single-flow prediction (POST /predict)
$body = @{ "Average Packet Size" = 500; "Bwd Packet Length Std" = 100 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -Body $body -ContentType "application/json"
```

**Open Swagger UI in browser:**
```powershell
Start-Process http://localhost:8000/docs
```

---

### 2.5 Configuration Tuning — `config.json`

Edit `config.json` to adjust detection sensitivity without retraining:

```json
{
    "binary_threshold": 0.50,
    "alert_min_severity": "LOW",
    "alert_min_confidence": 0.50,
    "alert_suppressed_types": [],
    "alert_suppressed_ports": [],
    "alert_dedup_window_s": 60
}
```

| Key | Default | Description |
|---|---|---|
| `binary_threshold` | `0.50` | XGBoost attack probability cutoff. ↓ = more sensitive / ↑ = stricter. |
| `alert_min_severity` | `"LOW"` | Set `"MEDIUM"` on noisy networks to reduce alert volume. |
| `alert_min_confidence` | `0.50` | Min probability to generate an alert. |
| `alert_suppressed_types` | `[]` | e.g. `["Bot"]` — silence specific attack types. |
| `alert_suppressed_ports` | `[]` | e.g. `[80, 443]` — silence alerts on specific ports. |
| `alert_dedup_window_s` | `60` | Seconds before the same `(src_ip, dst_ip, attack_type)` can re‑alert. |

---

### 2.6 Alertmanager Email Notifications

The IDS exposes alert counters via Prometheus. Prometheus evaluates `alert_rules.yml` and forwards firing alerts to Prometheus Alertmanager, which dispatches HTML email notifications via Gmail SMTP.

**Setup steps:**
1. Generate a Google Account App Password for Gmail.
2. Edit `alertmanager/alertmanager.yml`:
   * Set `auth_username` to your Gmail address.
   * Set `auth_password` to your 16-character App Password.
   * Set `to: 'your-email@gmail.com'` under `email_configs`.
3. Start the Docker monitoring stack:
   ```bash
   docker compose up -d
   ```
4. Verify Alertmanager status at `http://localhost:9093`.


---

### 2.7 Stop the IDS

```
Press  Ctrl+C
```

> The system flushes all in-progress flows and prints a session summary before exiting cleanly.

---

## 3. Grafana Monitoring Stack

### 3.1 Start Prometheus + Grafana

```powershell
docker-compose up -d
```

This starts:
- **Prometheus** → `http://localhost:9091` (scrapes IDS metrics every 5 s & evaluates alert rules)
- **Alertmanager** → `http://localhost:9093` (deduplicates alerts & sends Gmail email notifications)
- **Grafana** → `http://localhost:3000` (auto-provisioned IDS dashboard)


### 3.2 Open the Dashboard

```powershell
Start-Process http://localhost:3000
```

| Setting | Value |
|---|---|
| Username | `admin` |
| Password | `admin` |

> **"Real-Time IDS Dashboard"** auto-loads — no manual datasource or dashboard configuration needed.

---

### 3.3 Full Stack — Two Terminals

**Terminal 1 — Monitoring stack:**
```powershell
docker-compose up -d
```

**Terminal 2 — IDS application:**
```powershell
venv\Scripts\python.exe src\main.py --interface "\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}" --workers 4
```

> Grafana shows live data within **10–15 seconds** of the IDS starting.

---

### 3.4 Verify Prometheus

```powershell
# Check Prometheus targets (ids_pipeline should show UP)
Start-Process http://localhost:9091/targets

# View raw metrics
Start-Process http://localhost:9090/metrics
```

Key metrics:

| Metric | Type | Description |
|---|---|---|
| `ids_packets_total` | Counter | Total packets captured |
| `ids_packets_processed_total` | Counter | Total packets captured |
| `ids_packets_per_second` | Gauge | Current packet rate calculated by the IDS |
| `ids_flows_total` | Counter | Total flows completed |
| `ids_flows_completed_total` | Counter | Total flows completed |
| `ids_flows_per_second` | Gauge | Current completed-flow rate calculated by the IDS |
| `ids_predictions_total` | Counter | Total inference calls |
| `ids_attack_total` | Counter | Total attack predictions |
| `ids_attacks_detected_total` | Counter | Total flows classified as attacks |
| `ids_benign_total` | Counter | Total benign predictions |
| `ids_benign_detected_total` | Counter | Total flows classified as benign |
| `ids_attack_rate` | Gauge | Ratio of attack predictions to total predictions |
| `ids_attacks_by_type` | Counter (label: `type`) | Attack predictions per attack type |
| `ids_attack_type_total` | Counter (label: `type`) | Predictions per attack type |
| `ids_src_ip_flows` | Counter (label: `src_ip`) | Completed flows per source IP |
| `ids_dst_ip_flows` | Counter (label: `dst_ip`) | Completed flows per destination IP |
| `ids_protocol_flows` | Counter (label: `protocol`) | Completed flows per IP protocol number |
| `ids_model_confidence` | Gauge | Latest model confidence |
| `ids_xgb_probability` | Gauge | Latest XGBoost attack probability |
| `ids_rf_probability` | Gauge | Placeholder for RF probability if runtime predictor supplies it |
| `ids_iso_score` | Gauge | Latest Isolation Forest anomaly score |
| `ids_anomalies_total` | Counter | Flows with a positive anomaly score |
| `ids_high_confidence_attacks` | Counter | Attacks with confidence >= 0.85 |
| `ids_alerts_total` | Counter | Total alerts stored in SQLite |
| `ids_alerts_deduped_total` | Counter | Alerts suppressed by deduplication |
| `ids_alerts_severity_total` | Counter (label: `severity`) | Alert counts per severity level |
| `ids_inference_latency_ms` | Histogram | End-to-end predict_flow() latency |
| `ids_feature_extraction_latency_ms` | Histogram | Feature extraction latency |
| `ids_processing_time_ms` | Histogram | Feature extraction plus inference processing time |
| `ids_active_flows` | Gauge | Active flows in the flow table |
| `ids_alert_rate` | Gauge | Ratio of alerts to total flows (rolling) |
| `ids_cpu_usage_percent` | Gauge | IDS process CPU usage |
| `ids_memory_usage_percent` | Gauge | IDS process memory usage |

---

### 3.5 Stop the Monitoring Stack

```powershell
# Stop containers (keep data)
docker-compose down

# Stop containers and delete stored Prometheus data
docker-compose down -v
```

---

### 3.6 Troubleshooting

| Problem | Fix |
|---|---|
| Prometheus target shows **DOWN** | Ensure IDS is running with `--metrics-port 9090` |
| `host.docker.internal` not resolving | Restart Docker Desktop |
| Grafana shows **No Data** | Wait 10–15 s after IDS starts; check Prometheus target |
| SQLite panel is empty | Alerts only appear after the first attack is detected and stored |
| Port 3000 already in use | Edit `docker-compose.yml`: change `"3000:3000"` → `"3001:3000"` |
| Port 9091 already in use | Edit `docker-compose.yml`: change `"9091:9090"` → `"9092:9090"` |

---

## ⚡ Quick Reference

```powershell
# ----- SETUP -----
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt

# ----- FIND INTERFACE -----
venv\Scripts\python.exe src\main.py --list-interfaces

# ----- START FULL STACK -----
docker-compose up -d
venv\Scripts\python.exe src\main.py --interface "\Device\NPF_{E2D51F95-981B-4D72-889D-15319DF94403}" --workers 4

# ----- OPEN DASHBOARDS -----
Start-Process http://localhost:8000/docs      # FastAPI Swagger UI
Start-Process http://localhost:3000           # Grafana  (admin / admin)
Start-Process http://localhost:9091/targets   # Prometheus targets
Start-Process http://localhost:9090/metrics   # Raw metrics

# ----- STOP -----
# Ctrl+C                → stops the IDS
docker-compose down     → stops Grafana + Prometheus
```

---

## 📁 Logs & Alert Data

| Path | Contents |
|---|---|
| `data/logs/ids.log` | Rotating IDS log (10 MB × 5 files) |
| `data/alerts/alerts.db` | SQLite alert database |

```powershell
# Tail the live IDS log
Get-Content data\logs\ids.log -Wait -Tail 50

# Query the 10 most recent alerts from SQLite
venv\Scripts\python.exe -c "
import sqlite3
conn = sqlite3.connect('data/alerts/alerts.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('SELECT * FROM alerts ORDER BY id DESC LIMIT 10').fetchall()
for r in rows: print(dict(r))
conn.close()
"
```
```
