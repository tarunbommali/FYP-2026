"""
alerts/alert_manager.py
Orchestrator — connects predict_flow() output to the alert pipeline.

Responsibilities
----------------
  - Receive a PredictionResult dict + flow metadata (IPs, ports, protocol)
  - Apply AlertRules
  - Deduplicate repeated alerts within a configurable time window
  - Build an Alert dataclass
  - Persist via AlertStorage
  - Expose per-severity counters for Prometheus metrics (Phase 10)

Deduplication
-------------
  Cache key : (src_ip, dst_ip, attack_type)
  Window    : alert_dedup_window_s seconds (default: 60, from config.json)

  Within the window the first alert is stored. Subsequent identical alerts
  increment a counter but are not re-stored. When the window expires the
  next occurrence opens a new record.

  This prevents a 1000 flow/sec DDoS from creating 1000 identical alerts.

Usage
-----
    from alerts.alert_manager import AlertManager

    def on_flow_complete(flow):
        features = extract_features(flow)
        result   = predict_flow(features)
        alert_mgr.process(result, flow)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple

from alerts.alert import Alert
from alerts.alert_rules import AlertRules
from alerts.alert_storage import AlertStorage
from flows.flow import NetworkFlow
from monitoring import metrics_registry as reg

logger = logging.getLogger(__name__)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DB_PATH = os.path.join(BASE_DIR, "data", "alerts", "alerts.db")

# Dedup cache key: (src_ip, dst_ip, attack_type)
_DedupKey = Tuple[str, str, str]


@dataclass
class _DedupEntry:
    first_seen: float
    last_seen: float
    count: int = 1


class AlertManager:
    """
    Orchestrates alert rule validation, deduplication, and SQLite persistence.
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        config_path: str = CONFIG_PATH,
    ) -> None:
        self._rules = AlertRules(config_path)
        self._storage = AlertStorage(db_path)
        self._lock = Lock()

        self._dedup_window: float = self._load_dedup_window(config_path)
        self._dedup_cache: Dict[_DedupKey, _DedupEntry] = {}

        self._total_flows: int = 0
        self._total_alerts: int = 0
        self._total_suppressed: int = 0
        self._total_deduped: int = 0
        self._severity_counts: Dict[str, int] = {
            "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0,
        }

        logger.info("AlertManager ready | db=%s | dedup_window=%.0fs", db_path, self._dedup_window)

    def process(self, prediction: dict, flow: NetworkFlow) -> Optional[Alert]:
        with self._lock:
            self._total_flows += 1

        dst_port = flow.key.dst_port
        should_alert, reason = self._rules.should_alert(prediction, dst_port=dst_port)
        if not should_alert:
            with self._lock:
                self._total_suppressed += 1
            logger.debug("Alert suppressed | reason=%s", reason)
            return None

        dedup_key: _DedupKey = (
            flow.key.src_ip,
            flow.key.dst_ip,
            prediction.get("attack_type", "Unknown"),
        )
        now = time.time()

        with self._lock:
            entry = self._dedup_cache.get(dedup_key)
            if entry is not None:
                age = now - entry.first_seen
                if age < self._dedup_window:
                    entry.last_seen = now
                    entry.count    += 1
                    self._total_deduped += 1
                    reg.alerts_deduped_total.inc()
                    logger.debug(
                        "Dedup suppressed | key=%s | count=%d | age=%.1fs",
                        dedup_key, entry.count, age,
                    )
                    return None
                else:
                    self._dedup_cache[dedup_key] = _DedupEntry(
                        first_seen=now, last_seen=now
                    )
            else:
                self._dedup_cache[dedup_key] = _DedupEntry(
                    first_seen=now, last_seen=now
                )

        alert = Alert(
            timestamp          = Alert.now_utc(),
            src_ip             = flow.key.src_ip,
            dst_ip             = flow.key.dst_ip,
            src_port           = flow.key.src_port,
            dst_port           = flow.key.dst_port,
            protocol           = flow.key.protocol,
            attack_type        = prediction.get("attack_type", "Unknown"),
            attack_probability = prediction.get("attack_probability", 0.0),
            rf_probability     = prediction.get("rf_probability", 0.0),
            xgb_probability    = prediction.get("xgb_probability", 0.0),
            meta_probability   = prediction.get("meta_probability", 0.0),
            attack_confidence  = prediction.get("attack_confidence", 0.0),
            iso_score          = prediction.get("iso_score", 0.0),
            severity           = prediction.get("severity", "LOW"),
            latency_ms         = prediction.get("latency_ms", 0.0),
        )

        try:
            self._storage.insert(alert)
        except Exception as exc:
            reg.alerts_failed_total.inc()
            logger.error("Failed to write alert to SQLite database: %s", exc)
            raise

        with self._lock:
            self._total_alerts += 1
            sev = alert.severity
            if sev in self._severity_counts:
                self._severity_counts[sev] += 1

        reg.alerts_total.inc()
        reg.alerts_severity_total.labels(severity=alert.severity).inc()

        logger.warning(
            "ALERT #%d | %s | %s -> %s:%d | sev=%s | prob=%.4f",
            alert.id, alert.attack_type,
            alert.src_ip, alert.dst_ip, alert.dst_port,
            alert.severity, alert.attack_probability,
        )

        return alert

    @property
    def total_flows(self) -> int:
        with self._lock:
            return self._total_flows

    @property
    def total_alerts(self) -> int:
        with self._lock:
            return self._total_alerts

    @property
    def total_suppressed(self) -> int:
        with self._lock:
            return self._total_suppressed

    @property
    def total_deduped(self) -> int:
        with self._lock:
            return self._total_deduped

    @property
    def critical_alerts(self) -> int:
        with self._lock:
            return self._severity_counts["CRITICAL"]

    @property
    def high_alerts(self) -> int:
        with self._lock:
            return self._severity_counts["HIGH"]

    @property
    def medium_alerts(self) -> int:
        with self._lock:
            return self._severity_counts["MEDIUM"]

    @property
    def low_alerts(self) -> int:
        with self._lock:
            return self._severity_counts["LOW"]

    @property
    def alert_rate(self) -> float:
        with self._lock:
            return self._total_alerts / max(self._total_flows, 1)

    def stats(self) -> dict:
        """Snapshot of all counters for health check and Prometheus."""
        with self._lock:
            alert_rate = self._total_alerts / max(self._total_flows, 1)
            return {
                "total_flows": self._total_flows,
                "total_alerts": self._total_alerts,
                "total_suppressed": self._total_suppressed,
                "total_deduped": self._total_deduped,
                "alert_rate": round(alert_rate, 4),
                "severity": {
                    "CRITICAL": self._severity_counts["CRITICAL"],
                    "HIGH": self._severity_counts["HIGH"],
                    "MEDIUM": self._severity_counts["MEDIUM"],
                    "LOW": self._severity_counts["LOW"],
                },
                "by_type": self._storage.count_by_type(),
                "by_severity": self._storage.count_by_severity(),
            }

    def recent_alerts(self, limit: int = 50):
        return self._storage.recent(limit)

    def purge_dedup_cache(self) -> int:
        """Remove expired dedup entries to prevent unbounded memory growth."""
        now = time.time()
        with self._lock:
            expired = [
                k for k, v in self._dedup_cache.items()
                if (now - v.first_seen) >= self._dedup_window
            ]
            for k in expired:
                del self._dedup_cache[k]
        if expired:
            logger.debug("Dedup cache purged: %d expired entries", len(expired))
        return len(expired)

    @staticmethod
    def _load_dedup_window(config_path: str) -> float:
        default = 60.0
        if not os.path.exists(config_path):
            return default
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {config_path}: {exc}") from exc
        except OSError as exc:
            raise OSError(f"Could not read config file {config_path}: {exc}") from exc

        val = cfg.get("alert_dedup_window_s", default)
        try:
            val_float = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"alert_dedup_window_s must be a number in {config_path}: {exc}") from exc
        return max(1.0, val_float)


