"""
monitoring/metrics_exporter.py
Background thread that periodically pushes system-level and application-level
gauge values to the Prometheus registry.

Metrics exported every `interval` seconds
------------------------------------------
    ids_active_flows          ← FlowManager.active_flow_count
    ids_alert_rate            ← AlertManager.alert_rate (alerts / total_flows)
    ids_cpu_usage_percent     ← psutil.cpu_percent()
    ids_memory_usage_percent  ← psutil.virtual_memory().percent

Usage
-----
    from alerts.alert_manager import AlertManager
    from monitoring.metrics_exporter import start_metrics_export_thread

    alert_mgr = AlertManager()
    start_metrics_export_thread(alert_mgr=alert_mgr, interval=5.0)
"""

import logging
import threading
import time

from monitoring import metrics_registry as reg

logger = logging.getLogger(__name__)

_last_rate_sample = {
    "timestamp": None,
    "packets": 0.0,
    "flows": 0.0,
}

_start_time = time.time()


def start_metrics_export_thread(
    alert_mgr,
    flow_mgr=None,
    interval: float = 5.0,
) -> threading.Thread:
    """
    Start a daemon thread that exports gauge metrics to Prometheus.

    Parameters
    ----------
    alert_mgr : AlertManager
        Live AlertManager instance — provides alert rate and severity counts.
    flow_mgr : FlowManager | None
        Optional FlowManager — provides active_flow_count.
        If None, `ids_active_flows` is sourced from PacketCapture.active_flows
        via alert_mgr stats (fallback: 0).
    interval : float
        Seconds between export cycles (default: 5.0).

    Returns
    -------
    threading.Thread
        The running exporter thread (daemon=True).
    """
    t = threading.Thread(
        target=_export_loop,
        args=(alert_mgr, flow_mgr, interval),
        daemon=True,
        name="metrics-exporter",
    )
    t.start()
    logger.info("Metrics exporter started (interval=%.1fs)", interval)
    return t


def _export_loop(alert_mgr, flow_mgr, interval: float) -> None:
    """Main loop — runs forever, exiting when the process ends (daemon thread)."""
    # Lazy import psutil so the module loads even if psutil is absent
    try:
        import psutil  # pyrefly: ignore [missing-import]
        _has_psutil = True
    except ImportError:
        logger.warning(
            "psutil not installed — CPU/memory metrics will not be exported. "
            "Install with: pip install psutil"
        )
        _has_psutil = False

    while True:
        try:
            _export_tick(alert_mgr, flow_mgr, _has_psutil)
        except Exception as exc:
            logger.warning("Metrics export tick failed: %s", exc)
        time.sleep(interval)


def _export_tick(alert_mgr, flow_mgr, has_psutil: bool) -> None:
    reg.uptime_seconds.set(time.time() - _start_time)

    now = time.time()
    packets = float(reg.packets_total._value.get())
    flows = float(reg.flows_total._value.get())
    last_ts = _last_rate_sample["timestamp"]

    if last_ts is not None:
        elapsed = max(now - last_ts, 0.001)
        reg.packets_per_second.set(
            max((packets - _last_rate_sample["packets"]) / elapsed, 0.0)
        )
        reg.flows_per_second.set(
            max((flows - _last_rate_sample["flows"]) / elapsed, 0.0)
        )

    _last_rate_sample["timestamp"] = now
    _last_rate_sample["packets"] = packets
    _last_rate_sample["flows"] = flows

    if flow_mgr is not None:
        reg.active_flows.set(flow_mgr.active_flow_count)

    reg.alert_rate_gauge.set(alert_mgr.alert_rate)

    attacks = float(reg.attack_total._value.get())
    predictions = float(reg.predictions_total._value.get())
    reg.attack_rate.set(attacks / max(predictions, 1.0))

    if has_psutil:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        reg.cpu_usage_percent.set(cpu)
        reg.memory_usage_percent.set(mem)
        logger.debug(
            "Metrics tick | active_flows=%s | alert_rate=%.4f | cpu=%.1f%% | mem=%.1f%%",
            flow_mgr.active_flow_count if flow_mgr else "n/a",
            alert_mgr.alert_rate,
            cpu,
            mem,
        )

