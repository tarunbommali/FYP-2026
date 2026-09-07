"""
monitoring/metrics_server.py
Starts the Prometheus HTTP server.
"""

import logging
# pyrefly: ignore [missing-import]
from prometheus_client import start_http_server

logger = logging.getLogger(__name__)

def start_metrics_server(port: int = 9090) -> None:
    """
    Start the Prometheus HTTP endpoint on the given port.
    Grafana scrapes: http://localhost:{port}/metrics
    """
    start_http_server(port)
    logger.info("Prometheus metrics server started on port %d", port)
    print(f"[Metrics] Prometheus endpoint: http://localhost:{port}/metrics")
