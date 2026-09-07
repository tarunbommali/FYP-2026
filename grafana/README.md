# Phase 10: Prometheus Monitoring & Grafana

This directory contains the necessary configuration to spin up the observability stack for the Real-Time IDS.

## Architecture

```text
IDS Pipeline (Python)
    ↓ exposes /metrics on port 9090
Prometheus (Docker)
    ↓ scrapes port 9090 every 5 seconds
Grafana (Docker)
    ↓ queries Prometheus
    ↓ queries SQLite (alerts.db) directly
Real-Time Dashboard
```

## Prerequisites
- Docker Desktop installed and running.
- Python virtual environment with `prometheus-client` installed.

## Getting Started

1. **Start the Docker Stack**
   From the `dataset` directory:
   ```powershell
   docker-compose up -d
   ```

2. **Start the IDS with Metrics**
   In your Python code, ensure you start the metrics server before your capture loop begins:
   ```python
   from monitoring.metrics_server import start_metrics_server
   start_metrics_server(port=9090)
   ```

3. **View the Dashboard**
   Open your browser and navigate to:
   http://localhost:3000
   
   The "IDS Dashboards" folder will automatically be provisioned with the "Real-Time IDS Dashboard". No manual datasource or dashboard setup is required.

## Troubleshooting

- **Prometheus cannot reach host**: If Prometheus shows target DOWN, ensure `host.docker.internal` is resolving. This works automatically on Docker Desktop for Windows/Mac.
- **SQLite errors**: The `data/alerts` folder is volume-mounted into the Grafana container. Ensure your Python pipeline is writing to `data/alerts/alerts.db` and the path exists.
- **Empty Panels**: Trigger the benchmark script (`python benchmark_inference.py`) while the stack is running to pump test data into the Prometheus endpoint.
