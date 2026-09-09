"""
main.py
IDS Runtime Orchestrator — supporting Offline Test Data replay and Live Npcap capture.

Usage:
    # Default: Run offline 20% CICIDS2017 test dataset
    venv\\Scripts\\python.exe src\\main.py

    # Optional worker count:
    venv\\Scripts\\python.exe src\\main.py --workers 4

    # Live Mode: Run live Npcap packet capture
    venv\\Scripts\\python.exe src\\main.py --live --interface "Ethernet" --workers 4

    # List network interfaces:
    venv\\Scripts\\python.exe src\\main.py --list-interfaces
"""

import argparse
import logging
import logging.handlers
import os
import queue
import signal
import sys
import threading
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _configure_logging(level: str) -> None:
    log_dir = os.path.join(BASE_DIR, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler records all system logs (INFO, WARNING, ERROR, DEBUG)
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "ids.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)

    # Console handler: only show ERRORs (or DEBUG if explicitly requested) to keep the terminal status ticker clean
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    if level.upper() == "DEBUG":
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.ERROR)
    root.addHandler(ch)


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ids-main",
        description="AI-Driven Real-Time Intrusion Detection System",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run in LIVE mode with Npcap packet capture (default: offline 20%% test dataset).",
    )
    parser.add_argument(
        "--interface", "-i",
        default="Ethernet",
        help="Network interface name for live capture (default: 'Ethernet').",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of inference workers (default: 4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of flows to process in TEST DATA mode (default: all).",
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
        help="Prometheus metrics endpoint port (default: 9090).",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        default=True,
        help="Start the FastAPI REST/WebSocket server (default: True).",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="FastAPI server port (default: 8000).",
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
        help="BPF filter string for live capture (default: 'ip').",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=15.0,
        help="Seconds of flow inactivity before export in live mode (default: 15.0).",
    )
    parser.add_argument(
        "--absolute-timeout",
        type=float,
        default=60.0,
        help="Maximum flow lifetime in seconds in live mode (default: 60.0).",
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
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                logger.warning("Port %d is already in use — FastAPI server skipped", port)
                return

        import uvicorn
        from api.main import build_app
        app = build_app(alert_mgr)
        logger.info("Starting FastAPI server on port %d", port)
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True, name="api-server")
        t.start()
    except ImportError as exc:
        logger.warning("FastAPI/uvicorn not available — API server skipped: %s", exc)
    except Exception as exc:
        logger.error("Failed to start API server: %s", exc)


def _handle_prediction(result: dict, alert_mgr, reg, flow=None) -> None:
    attack_type = result.get("attack_type", "BENIGN")
    attack_confidence = float(result.get("attack_confidence", 0.0) or 0.0)
    iso_score = float(result.get("iso_score", 0.0) or 0.0)
    is_attack = bool(result.get("is_attack", False))

    reg.flows_completed_total.inc()
    reg.flows_total.inc()
    reg.predictions_total.inc()
    reg.inference_latency_ms.observe(result.get("latency_ms", 0))

    reg.xgb_probability.set(float(result.get("xgb_probability", 0.0) or 0.0))
    reg.rf_probability.set(float(result.get("rf_probability", 0.0) or 0.0))
    reg.model_confidence.set(attack_confidence)
    reg.iso_score.set(iso_score)
    reg.meta_probability.set(float(result.get("meta_probability", 0.0) or 0.0))

    if is_attack:
        reg.attacks_detected_total.inc()
        reg.attack_total.inc()
        reg.attacks_by_type.labels(type=attack_type).inc()
        reg.attack_type_total.labels(type=attack_type).inc()
        if attack_confidence >= 0.85:
            reg.high_confidence_attacks.inc()
    else:
        reg.benign_detected_total.inc()
        reg.benign_total.inc()

    if iso_score > 0:
        reg.anomalies_total.inc()

    alert_mgr.process(result, flow=flow)


def _run_test_mode(args, alert_mgr, reg, shutdown_event: threading.Event) -> None:
    X_test_path = os.path.join(BASE_DIR, "data", "processed", "X_test_binary.parquet")
    if not os.path.exists(X_test_path):
        print("\n Status     : ERROR")
        print(f" Reason     : Test dataset not found at {X_test_path}. Run Phase 2 (02_binary_training.py) first.\n")
        sys.exit(1)

    import pandas as pd
    from inference.predictor import predict_flow

    df = pd.read_parquet(X_test_path)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit]
    total_flows = len(df)
    cols = list(df.columns)

    header = (
        "============================================================\n"
        " AI-Driven Real-Time Intrusion Detection System\n"
        "============================================================\n\n"
        " Mode       : TEST DATA\n"
        " Dataset    : CICIDS2017 20% Test\n"
        f" Workers    : {args.workers}\n"
        " Features   : 78\n\n"
        " Models     : RF + XGBoost + Isolation Forest + Meta Learner\n"
        " Monitoring : Prometheus + Grafana\n"
        " Alerting   : Alertmanager -> Email\n\n"
        "------------------------------------------------------------\n"
        " Test Progress\n"
        "------------------------------------------------------------"
    )
    print(header, flush=True)

    num_workers = max(1, args.workers)
    work_queue: queue.Queue = queue.Queue(maxsize=num_workers * 128)
    _SENTINEL = object()

    processed_count = 0
    attack_count = 0
    benign_count = 0
    count_lock = threading.Lock()

    def _worker():
        nonlocal processed_count, attack_count, benign_count
        while not shutdown_event.is_set():
            try:
                item = work_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                work_queue.task_done()
                break

            try:
                result = predict_flow(item)
                _handle_prediction(result, alert_mgr, reg, flow=None)
                with count_lock:
                    processed_count += 1
                    if result.get("is_attack", False):
                        attack_count += 1
                    else:
                        benign_count += 1
            except Exception as exc:
                logger.error("Error predicting flow: %s", exc)
            finally:
                work_queue.task_done()

    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=_worker, daemon=True, name=f"test-worker-{i}")
        t.start()
        threads.append(t)

    status_running = True

    def _status_ticker():
        while not shutdown_event.is_set() and status_running:
            alerts = alert_mgr.stats().get("total_alerts", 0)
            with count_lock:
                p = processed_count
                att = attack_count
                ben = benign_count
            print(
                f"\r Flows: {p:>7,} / {total_flows:>7,} | "
                f"Pred: {p:>7,} | Attacks: {att:>5,} | "
                f"Benign: {ben:>7,} | Alerts: {alerts:>4,} | "
                f"Status: Running  ",
                end="", flush=True,
            )
            time.sleep(2.0)

    ticker = threading.Thread(target=_status_ticker, daemon=True, name="status-ticker")
    ticker.start()

    try:
        for row in df.itertuples(index=False):
            if shutdown_event.is_set():
                break
            feat_dict = dict(zip(cols, row))
            while not shutdown_event.is_set():
                try:
                    work_queue.put(feat_dict, timeout=0.2)
                    break
                except queue.Full:
                    pass

        while not shutdown_event.is_set():
            if work_queue.empty():
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        status_running = False
        for _ in range(num_workers):
            work_queue.put(_SENTINEL)
        for t in threads:
            t.join(timeout=1.0)

        alerts = alert_mgr.stats().get("total_alerts", 0)
        with count_lock:
            p = processed_count
            att = attack_count
            ben = benign_count
        final_status = "Stopped" if shutdown_event.is_set() else "Completed"
        print(
            f"\r Flows: {p:>7,} / {total_flows:>7,} | "
            f"Pred: {p:>7,} | Attacks: {att:>5,} | "
            f"Benign: {ben:>7,} | Alerts: {alerts:>4,} | "
            f"Status: {final_status} ",
            flush=True,
        )
        print(f"\n\nTest processing {final_status.lower()}. Metrics and alert services active. Press Ctrl+C to exit.")
        while not shutdown_event.is_set():
            time.sleep(0.5)


def _run_live_mode(args, alert_mgr, reg, shutdown_event: threading.Event) -> None:
    from capture.packet_capture import PacketCapture
    from features.feature_extractor import extract_features
    from inference.predictor import predict_flow

    header = (
        "============================================================\n"
        " AI-Driven Real-Time Intrusion Detection System\n"
        "============================================================\n\n"
        " Mode       : LIVE\n"
        f" Interface  : {args.interface}\n"
        f" Workers    : {args.workers}\n"
        " Features   : 78\n\n"
        " Models     : RF + XGBoost + Isolation Forest + Meta Learner\n"
        " Monitoring : Prometheus + Grafana\n"
        " Alerting   : Alertmanager -> Email\n\n"
        "------------------------------------------------------------\n"
        " Live Traffic\n"
        "------------------------------------------------------------"
    )
    print(header, flush=True)

    worker_pool = None
    if args.workers > 1:
        from inference.worker_pool import InferenceWorkerPool
        def _on_result(result, flow):
            _handle_prediction(result, alert_mgr, reg, flow=flow)
        worker_pool = InferenceWorkerPool(
            n_workers=args.workers,
            on_result=_on_result,
            maxsize=512,
        )
        worker_pool.start()

    def on_flow_complete(flow) -> None:
        try:
            reg.src_ip_flows.labels(src_ip=flow.key.src_ip).inc()
            reg.dst_ip_flows.labels(dst_ip=flow.key.dst_ip).inc()
            reg.protocol_flows.labels(protocol=str(flow.key.protocol)).inc()

            if worker_pool is not None:
                worker_pool.submit(flow)
            else:
                features = extract_features(flow)
                result = predict_flow(features)
                _handle_prediction(result, alert_mgr, reg, flow=flow)
        except Exception as exc:
            logger.exception("on_flow_complete error: %s", exc)

    capture_error = None
    capture_failed_event = threading.Event()

    try:
        capture = PacketCapture(
            interface=args.interface,
            on_flow_complete=on_flow_complete,
            filter_bpf=args.filter,
            idle_timeout=args.idle_timeout,
            absolute_timeout=args.absolute_timeout,
        )
    except Exception as exc:
        print("\n Status     : ERROR")
        print(f" Reason     : {exc}\n")
        sys.exit(1)

    def _run_sniff():
        nonlocal capture_error
        try:
            capture.start(packet_count=0)
        except Exception as exc:
            capture_error = exc
            capture_failed_event.set()
            shutdown_event.set()

    cap_thread = threading.Thread(target=_run_sniff, daemon=True, name="packet-capture")
    cap_thread.start()

    # Wait up to 1.5s to see if capture failed immediately
    if capture_failed_event.wait(timeout=1.5):
        print("\n Status     : ERROR")
        print(f" Reason     : {capture_error}\n")
        sys.exit(1)

    if not cap_thread.is_alive():
        print("\n Status     : ERROR")
        print(f" Reason     : {capture_error or 'Packet capture thread exited unexpectedly'}\n")
        sys.exit(1)

    from monitoring.metrics_exporter import start_metrics_export_thread
    start_metrics_export_thread(alert_mgr=alert_mgr, flow_mgr=capture._flow_manager, interval=5.0)

    last_pkts = 0
    last_flows = 0
    last_time = time.time()

    def _live_ticker():
        nonlocal last_pkts, last_flows, last_time
        while not shutdown_event.wait(timeout=2.0):
            try:
                now = time.time()
                dt = max(now - last_time, 1e-3)
                curr_pkts = int(reg.packets_processed_total._value.get())
                curr_flows = int(reg.flows_completed_total._value.get())
                preds = int(reg.predictions_total._value.get())
                attacks = int(reg.attacks_detected_total._value.get())
                benign = int(reg.benign_detected_total._value.get())
                alerts = alert_mgr.stats().get("total_alerts", 0)

                pkt_rate = max(0.0, (curr_pkts - last_pkts) / dt)
                flow_rate = max(0.0, (curr_flows - last_flows) / dt)

                last_pkts = curr_pkts
                last_flows = curr_flows
                last_time = now

                status = "Capturing" if cap_thread.is_alive() else "Stopped"
                print(
                    f"\r Packets/s: {pkt_rate:>6,.0f} | Flows/s: {flow_rate:>4,.0f} | "
                    f"Pred: {preds:>6,} | Attacks: {attacks:>4,} | "
                    f"Benign: {benign:>6,} | Alerts: {alerts:>4,} | "
                    f"Status: {status}  ",
                    end="", flush=True,
                )
            except Exception:
                pass

    ticker = threading.Thread(target=_live_ticker, daemon=True, name="live-ticker")
    ticker.start()

    while not shutdown_event.is_set():
        if not cap_thread.is_alive():
            print("\n Status     : ERROR")
            print(f" Reason     : {capture_error or 'Capture thread crashed'}\n")
            break
        time.sleep(0.5)

    capture.stop()
    if worker_pool is not None:
        worker_pool.shutdown(wait=True)
    print("\n\nLive capture stopped cleanly.")


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    if args.list_interfaces:
        _list_interfaces()
        return

    from alerts.alert_manager import AlertManager
    from monitoring.metrics_server import start_metrics_server
    from monitoring import metrics_registry as reg

    # Set system status to running
    reg.system_status.set(1)

    # Initialize AlertManager
    try:
        alert_mgr = AlertManager()
    except Exception as exc:
        print("\n Status     : ERROR")
        print(f" Reason     : AlertManager init failed: {exc}\n")
        sys.exit(1)

    # Prometheus metrics server
    try:
        start_metrics_server(port=args.metrics_port)
    except OSError as exc:
        logger.warning("Prometheus metrics server port %d already in use: %s", args.metrics_port, exc)

    # Optional FastAPI server
    if args.api:
        _start_api_server(args.api_port, alert_mgr)

    shutdown_event = threading.Event()

    def _handle_signal(signum, frame):
        print("\n\n[IDS] Shutdown signal received (Ctrl+C). Exiting...", flush=True)
        try:
            reg.system_status.set(0)
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)

    try:
        if args.live:
            _run_live_mode(args, alert_mgr, reg, shutdown_event)
        else:
            _run_test_mode(args, alert_mgr, reg, shutdown_event)
    finally:
        reg.system_status.set(0)
        os._exit(0)


if __name__ == "__main__":
    main()
