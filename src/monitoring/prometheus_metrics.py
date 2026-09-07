"""
monitoring/prometheus_metrics.py
Exposes the METRICS singleton which acts as an orchestrator for updating 
the Prometheus counters defined in metrics_registry.py.
"""

from monitoring import metrics_registry as reg

class IDSMetrics:
    def record_prediction(self, result: dict) -> None:
        reg.predictions_total.inc()
        reg.inference_latency_ms.observe(result.get("latency_ms", 0))

        attack_type = result.get("attack_type", "BENIGN")
        if result.get("is_attack"):
            reg.attacks_detected_total.inc()
        else:
            reg.benign_detected_total.inc()

        reg.attack_type_total.labels(type=attack_type).inc()

    def record_alert(self, alert) -> None:
        reg.alerts_total.inc()
        reg.alerts_severity_total.labels(severity=alert.severity).inc()

    def record_dedup(self) -> None:
        reg.alerts_deduped_total.inc()

METRICS = IDSMetrics()
