"""
main.py
IDS Runtime Orchestrator — the single entry point for the live IDS.

Wires together:
    PacketCapture  →  FlowManager  →  FeatureExtractor
        →  Predictor  →  AlertManager  →  Prometheus Metrics  →  Grafana

Usage
-----
    # Default interface (Ethernet) with API server:
    venv\\Scripts\\python.exe main.py

    # Specify interface:
    venv\\Scripts\\python.exe main.py --interface "Wi-Fi"

    # Custom API port:
    venv\\Scripts\\python.exe main.py --interface "Ethernet" --api-port 8080

    # Verbose logging:
    venv\\Scripts\\python.exe main.py --log-level DEBUG

    # List available network interfaces:
    venv\\Scripts\\python.exe main.py --list-interfaces

Shutdown
--------
    Press Ctrl+C — the system will flush all in-progress flows and exit cleanly.
"""

import argparse
import logging
import logging.handlers
import os
import signal
import socket
import sys
import threading
import time
import urllib.request

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _configure_logging(level: str) -> None:
    log_dir = os.path.join(BASE_DIR, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
        if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "ids.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


logger = logging.getLogger(__name__)


def _import_components():
    from capture.packet_capture import PacketCapture
    from features.feature_extractor import extract_features
    from inference.predictor import predict_flow
    from alerts.alert_manager import AlertManager
    from monitoring.metrics_server import start_metrics_server
    from monitoring.metrics_exporter import start_metrics_export_thread
    from monitoring import metrics_registry as reg
    from flows.flow_manager import FlowManager

    return (
        PacketCapture, extract_features, predict_flow,
        AlertManager, start_metrics_server, start_metrics_export_thread,
        reg, FlowManager,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ids-main",
        description="IDS Runtime Orchestrator — captures live traffic and detects intrusions.",
    )
    parser.add_argument(
        "--interface", "-i",
        default="Ethernet",
        help="Network interface name to capture on (default: 'Ethernet'). "
             "Use --list-interfaces to see available adapters.",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--metrics-port", "-m",
        type=int,
        default=9090,
        help="Prometheus HTTP metrics endpoint port (default: 9090).",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        default=True,
        help="Start the FastAPI REST/WebSocket server alongside the capture loop (enabled by default).",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="FastAPI server port (default: 8000). Only used when --api is set.",
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        default=False,
        help="Print available network interfaces and exit.",
    )
    parser.add_argument(
        "--filter",
        default="ip",
        help="BPF filter string for packet capture (default: 'ip').",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel inference workers (default: 1). "
             "Set to 4 for ~2700 flows/sec throughput.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=15.0,
        dest="idle_timeout",
        help="Seconds of flow inactivity before export (default: 15 for dev; "
             "use 120 for production).",
    )
    parser.add_argument(
        "--absolute-timeout",
        type=float,
        default=60.0,
        dest="absolute_timeout",
        help="Maximum flow lifetime in seconds (default: 60 for dev; "
             "use 600 for production).",
    )
    return parser.parse_args()


def _list_interfaces() -> None:
    try:
        from scapy.arch import get_if_list
        ifaces = get_if_list()
        print("\nAvailable network interfaces:")
        for iface in ifaces:
            print(f"  {iface}")
        print()
    except ImportError:
        print("Scapy not installed. Install with: pip install scapy")
    except Exception as exc:
        print(f"Could not list interfaces: {exc}")


def _start_api_server(port: int, alert_mgr) -> None:
    try:
        import uvicorn
        from api.main import build_app
        app = build_app(alert_mgr)
        logger.info("Starting FastAPI server on http://0.0.0.0:%d", port)
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True, name="api-server")
        t.start()
    except ImportError as exc:
        logger.warning("FastAPI/uvicorn not available — API server skipped: %s", exc)
    except Exception as exc:
        logger.error("Failed to start API server: %s", exc)


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False



def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _tick(ok: bool) -> str:
    return "✅ Running   " if ok else "❌ Not Running"


def _tick_conn(ok: bool) -> str:
    return "✅ Connected " if ok else "❌ Unreachable"


def _print_health_dashboard(args, models_ok: bool, alert_mgr_ok: bool,
                             worker_pool_ok: bool, capture_ok: bool) -> None:

    """Print the full startup health dashboard to stdout."""
    W = 62  # box width

    # -- external service checks (non-blocking, 2-second timeout each) -------
    api_ok        = _port_open("127.0.0.1", args.api_port) if args.api else None
    prom_exp_ok   = _port_open("127.0.0.1", args.metrics_port)
    prom_srv_ok   = _http_ok(f"http://localhost:9091/-/healthy")
    grafana_ok    = _http_ok("http://localhost:3001/api/health") or \
                    _http_ok("http://localhost:3000/api/health")
    grafana_port  = 3001 if _http_ok("http://localhost:3001/api/health") else 3000
    sqlite_ok     = alert_mgr_ok   # alert manager initialised → SQLite is open

    sep   = "─" * W
    thick = "═" * W

    lines = [
        "",
        thick,
        " AI-Driven Real-Time Intrusion Detection System".center(W),
        " MCA Final Year Project  |  CICIDS2017  |  Stacking Ensemble".center(W),
        thick,
        "",
        f"  {'IDS Status':<22} : {'✅ LIVE' if capture_ok else '⚠  Starting...'}",
        f"  {'Interface':<22} : {args.interface}",
        f"  {'Inference Workers':<22} : {args.workers}",
        f"  {'Models':<22} : {'✅ Loaded (78 Features)' if models_ok else '❌ Load Failed'}",
        "",
        sep,
        "  Subsystems",
        sep,
        f"  {'Packet Capture':<22} : {'✅ Active' if capture_ok else '⚠  Starting...'}",
        f"  {'Flow Manager':<22} : ✅ Active",
        f"  {'Feature Extractor':<22} : ✅ Active  (78 CICFlowMeter features)",
        f"  {'Random Forest':<22} : {'✅ Loaded' if models_ok else '❌ Failed'}",
        f"  {'XGBoost Binary':<22} : {'✅ Loaded' if models_ok else '❌ Failed'}",
        f"  {'Isolation Forest':<22} : {'✅ Loaded' if models_ok else '❌ Failed'}",
        f"  {'Meta-Learner (LR)':<22} : {'✅ Loaded' if models_ok else '❌ Failed'}",
        f"  {'XGBoost Multiclass':<22} : {'✅ Loaded' if models_ok else '❌ Failed'}",
        f"  {'Alert Manager':<22} : {'✅ Active' if alert_mgr_ok else '❌ Failed'}",
        f"  {'Inference Worker Pool':<22} : {'✅ Active (%d workers)' % args.workers if worker_pool_ok else ('─ Single-thread mode' if not worker_pool_ok and args.workers == 1 else '❌ Failed')}",
        "",
        sep,
        "  Services",
        sep,
    ]

    # FastAPI
    if args.api:
        lines += [
            f"  {'FastAPI':<22} : {_tick(api_ok)}",
            f"  {'  API':<22} : http://localhost:{args.api_port}",
            f"  {'  Swagger UI':<22} : http://localhost:{args.api_port}/docs",
            f"  {'  WebSocket':<22} : ws://localhost:{args.api_port}/stream",
        ]
    else:
        lines.append(f"  {'FastAPI':<22} : ─ Disabled  (use --api to enable)")

    lines += [
        "",
        f"  {'Prometheus Exporter':<22} : {_tick(prom_exp_ok)}",
        f"  {'  Metrics Endpoint':<22} : http://localhost:{args.metrics_port}/metrics",
        "",
        f"  {'Prometheus Server':<22} : {_tick_conn(prom_srv_ok)}",
        f"  {'  URL':<22} : http://localhost:9091",
        f"  {'  Targets':<22} : http://localhost:9091/targets",
        "",
        f"  {'Grafana':<22} : {_tick_conn(grafana_ok)}",
        f"  {'  Dashboard':<22} : http://localhost:{grafana_port}",
        f"  {'  Login':<22} : admin / admin",
        "",
        f"  {'SQLite Alerts DB':<22} : {_tick_conn(sqlite_ok)}",
        "",
        sep,
        "  Live Counters  (updated every 10 s — watch below)",
        sep,
        "",
        f"  System Ready ✔   —   Press Ctrl+C to stop.",
        "",
        thick,
        "",
    ]

    print("\n".join(lines), flush=True)


def _start_status_ticker(reg, alert_mgr, shutdown_event: threading.Event,
                          interval: float = 10.0) -> None:
    try:
        import psutil
        _psutil = True
    except ImportError:
        _psutil = False

    def _ticker():
        while not shutdown_event.wait(timeout=interval):
            try:
                packets = int(reg.packets_processed_total._value.get())
                flows   = int(reg.flows_completed_total._value.get())
                preds   = int(reg.predictions_total._value.get())
                attacks = int(reg.attacks_detected_total._value.get())
                benign  = int(reg.benign_detected_total._value.get())
                alerts  = alert_mgr.stats().get("total_alerts", 0)
                cpu     = f"{psutil.cpu_percent():.0f}%" if _psutil else "n/a"
                mem     = f"{psutil.virtual_memory().percent:.0f}%" if _psutil else "n/a"
                print(
                    f"\r  📡 Packets:{packets:>7}  Flows:{flows:>6}  "
                    f"Pred:{preds:>6}  Attacks:{attacks:>5}  "
                    f"Benign:{benign:>6}  Alerts:{alerts:>4}  "
                    f"CPU:{cpu}  Mem:{mem}",
                    end="", flush=True
                )
            except OSError:
                pass

    t = threading.Thread(target=_ticker, daemon=True, name="status-ticker")
    t.start()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    if args.list_interfaces:
        _list_interfaces()
        return

    (
        PacketCapture, extract_features, predict_flow,
        AlertManager, start_metrics_server, start_metrics_export_thread,
        reg, FlowManager,
    ) = _import_components()

    # Set system status to running
    reg.system_status.set(1)

    # -- Load models (MODELS singleton already loaded on import) --------------
    from inference.model_loader import MODELS
    models_ok = MODELS is not None and len(MODELS.feature_columns) == 78

    # -- Initialise AlertManager ---------------------------------------------
    alert_mgr_ok = False
    try:
        alert_mgr = AlertManager()
        alert_mgr_ok = True
    except Exception as exc:
        logger.error("AlertManager init failed: %s", exc)
        sys.exit(1)

    # -- Prometheus metrics server -------------------------------------------
    start_metrics_server(port=args.metrics_port)

    # -- Inference worker pool (multi-threaded if --workers > 1) -------------
    worker_pool = None
    worker_pool_ok = False
    if args.workers > 1:
        from inference.worker_pool import InferenceWorkerPool
        def _on_result(result, flow):
            alert_mgr.process(result, flow)
        worker_pool = InferenceWorkerPool(
            n_workers = args.workers,
            on_result = _on_result,
            maxsize   = 512,
        )
        worker_pool.start()
        worker_pool_ok = True
    else:
        worker_pool_ok = False   # single-thread mode — not a failure

    # -- Optional FastAPI REST/WebSocket server -------------------------------
    if args.api:
        _start_api_server(args.api_port, alert_mgr)
        time.sleep(1.5)   # give uvicorn a moment to bind before health-check

    # -- Shutdown flag -------------------------------------------------------
    _shutdown = threading.Event()

    def _handle_signal(signum, frame):
        print("\n", flush=True)   # end the rolling status line cleanly
        logger.info("Shutdown signal received — stopping capture...")
        _shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # -- Flow completion callback --------------------------------------------
    capture = None

    def on_flow_complete(flow) -> None:
        """
        Called by FlowManager when a flow is complete (TCP FIN/RST or timeout).
        Runs in the capture thread — must be fast and non-blocking.
        """
        try:
            logger.info(
                "[Pipeline] Flow received | %s→%s:%d | proto=%d | pkts=%d | bytes=%d",
                flow.key.src_ip, flow.key.dst_ip, flow.key.dst_port,
                flow.key.protocol, flow.total_packets, flow.total_bytes,
            )

            reg.src_ip_flows.labels(src_ip=flow.key.src_ip).inc()
            reg.dst_ip_flows.labels(dst_ip=flow.key.dst_ip).inc()
            reg.protocol_flows.labels(protocol=str(flow.key.protocol)).inc()

            if worker_pool is not None:
                worker_pool.submit(flow)
                logger.info("[Pipeline] Flow submitted to worker pool")
            else:
                pipeline_start = time.perf_counter()
                reg.flows_completed_total.inc()
                reg.flows_total.inc()

                # --- Feature Extraction ---
                feature_start = time.perf_counter()
                features = extract_features(flow)
                feat_ms = (time.perf_counter() - feature_start) * 1000
                reg.feature_extraction_latency_ms.observe(feat_ms)

                # --- ML Prediction ---
                result = predict_flow(features)
                reg.processing_time_ms.observe(
                    (time.perf_counter() - pipeline_start) * 1000
                )

                reg.predictions_total.inc()
                reg.inference_latency_ms.observe(result.get("latency_ms", 0))

                attack_type        = result.get("attack_type", "BENIGN")
                attack_probability = float(result.get("attack_probability", 0.0) or 0.0)
                attack_confidence  = float(result.get("attack_confidence", 0.0) or 0.0)
                iso_score          = float(result.get("iso_score", 0.0) or 0.0)
                is_attack          = bool(result.get("is_attack", False))

                logger.info(
                    "[Pipeline] Prediction | is_attack=%s | type=%s | prob=%.4f "
                    "| conf=%.4f | iso=%.4f | latency=%.2fms",
                    is_attack, attack_type, attack_probability,
                    attack_confidence, iso_score, result.get("latency_ms", 0),
                )

                reg.xgb_probability.set(float(result.get("xgb_probability", 0.0) or 0.0))
                reg.rf_probability.set(float(result.get("rf_probability", 0.0) or 0.0))
                reg.model_confidence.set(attack_confidence)
                reg.iso_score.set(iso_score)
                reg.meta_probability.set(float(result.get("meta_probability", 0.0) or 0.0))

                if is_attack:
                    reg.attacks_detected_total.inc()
                    reg.attack_total.inc()
                    reg.attacks_by_type.labels(type=attack_type).inc()
                    if attack_confidence >= 0.85:
                        reg.high_confidence_attacks.inc()
                else:
                    reg.benign_detected_total.inc()
                    reg.benign_total.inc()
                if iso_score > 0:
                    reg.anomalies_total.inc()
                reg.attack_type_total.labels(type=attack_type).inc()

                alert_mgr.process(result, flow)

        except Exception as exc:
            logger.exception("on_flow_complete error: %s", exc)

    # -- Start capture in background thread ----------------------------------
    logger.info(
        "Starting packet capture on interface '%s' | idle_timeout=%.0fs | absolute_timeout=%.0fs",
        args.interface, args.idle_timeout, args.absolute_timeout,
    )
    capture_ok = False
    try:
        capture = PacketCapture(
            interface        = args.interface,
            on_flow_complete = on_flow_complete,
            filter_bpf       = args.filter,
            idle_timeout     = args.idle_timeout,
            absolute_timeout = args.absolute_timeout,
        )
        # -- Metrics exporter (pushes system gauges every 5 s) -------------------
        start_metrics_export_thread(alert_mgr=alert_mgr, flow_mgr=capture._flow_manager, interval=5.0)

        cap_thread = threading.Thread(
            target=capture.start,
            kwargs={"packet_count": 0},
            daemon=True,
            name="packet-capture",
        )
        cap_thread.start()
        time.sleep(0.5)   # brief pause so capture reports started before dashboard
        capture_ok = cap_thread.is_alive()

        # -- Print startup health dashboard ----------------------------------
        _print_health_dashboard(
            args, models_ok, alert_mgr_ok,
            worker_pool_ok=(args.workers > 1),
            capture_ok=capture_ok,
        )

        # -- Start live rolling status line ----------------------------------
        _start_status_ticker(reg, alert_mgr, _shutdown, interval=10.0)

        # Block main thread until shutdown signal
        while not _shutdown.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n", flush=True)
        logger.info("KeyboardInterrupt received.")
    except Exception as exc:
        logger.exception("Fatal error in capture: %s", exc)
    finally:
        logger.info("Shutting down IDS...")
        reg.system_status.set(0)
        if capture:
            capture.stop()
        time.sleep(1.0)
        if worker_pool is not None:
            worker_pool.shutdown(wait=True)
        stats = alert_mgr.stats()
        print("\n")
        logger.info(
            "Session summary | flows=%d | alerts=%d | suppressed=%d | deduped=%d",
            stats["total_flows"],
            stats["total_alerts"],
            stats["total_suppressed"],
            stats["total_deduped"],
        )
        logger.info("IDS stopped cleanly.")


if __name__ == "__main__":
    main()
