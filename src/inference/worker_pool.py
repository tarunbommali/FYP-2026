"""
inference/worker_pool.py
Queue-based multi-worker inference pool for high-throughput IDS operation.

Architecture
------------
    PacketCapture → FlowManager → InferenceWorkerPool.submit(flow)
                                           │
                              ┌────────────┴────────────┐
                              ▼            ▼            ▼
                           Worker-1    Worker-2    Worker-N
                              │            │            │
                         extract_features()  predict_flow()
                              │            │            │
                              └────────────┴────────────┘
                                           │
                                    AlertManager.process()

Expected throughput
-------------------
    Single-threaded baseline : ~679 pred/sec (benchmark_inference.py result)
    4-worker pool            : ~679 × 4 ≈ 2,700 flows/sec
    8-worker pool            : ~679 × 8 ≈ 5,400 flows/sec

Usage
-----
    from inference.worker_pool import InferenceWorkerPool
    from alerts.alert_manager import AlertManager
    from features.feature_extractor import extract_features
    from inference.predictor import predict_flow

    alert_mgr = AlertManager()

    def on_result(result, flow):
        alert_mgr.process(result, flow)

    pool = InferenceWorkerPool(n_workers=4, on_result=on_result)
    pool.start()

    # In the capture callback:
    def on_flow_complete(flow):
        pool.submit(flow)          # non-blocking

    # On shutdown:
    pool.shutdown(wait=True)
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

from features.feature_extractor import extract_features
from inference.predictor        import predict_flow
from monitoring                 import metrics_registry as reg

logger = logging.getLogger(__name__)

# Sentinel value — signals a worker to exit
_STOP = object()


class InferenceWorkerPool:
    """
    Thread pool that processes completed NetworkFlow objects in parallel.

    Parameters
    ----------
    n_workers : int
        Number of parallel inference workers (default: 4).
    on_result  : Callable[[dict, NetworkFlow], None]
        Callback invoked with (prediction_result, flow) after each prediction.
        Called from worker threads — must be thread-safe.
    maxsize    : int
        Maximum number of flows queued before submit() blocks (default: 512).
        Prevents unbounded RAM growth under burst traffic.
    """

    def __init__(
        self,
        n_workers:  int      = 4,
        on_result:  Callable = None,
        maxsize:    int      = 512,
    ) -> None:
        self._n_workers = n_workers
        self._on_result = on_result
        self._queue:   queue.Queue     = queue.Queue(maxsize=maxsize)
        self._workers: list[threading.Thread] = []
        self._running  = False

        # Counters
        self._total_processed: int = 0
        self._total_errors:    int = 0
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    def start(self) -> None:
        """Spawn all worker threads and begin processing."""
        if self._running:
            logger.warning("InferenceWorkerPool.start() called while already running.")
            return

        self._running = True
        for i in range(self._n_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"inference-worker-{i}",
            )
            t.start()
            self._workers.append(t)

        logger.info(
            "InferenceWorkerPool started | workers=%d | queue_maxsize=%d",
            self._n_workers, self._queue.maxsize,
        )

    # -----------------------------------------------------------------------
    def submit(self, flow, block: bool = True, timeout: float = 1.0) -> bool:
        """
        Enqueue a completed flow for inference.

        Parameters
        ----------
        flow    : NetworkFlow
            Completed flow from FlowManager.
        block   : bool
            If True, block until there is space in the queue.
        timeout : float
            Maximum seconds to wait when block=True (default: 1.0).

        Returns
        -------
        bool
            True if the flow was enqueued, False if the queue was full.
        """
        try:
            self._queue.put(flow, block=block, timeout=timeout)
            return True
        except queue.Full:
            logger.warning("Inference queue full — flow dropped (queue_size=%d)",
                           self._queue.qsize())
            return False

    # -----------------------------------------------------------------------
    def shutdown(self, wait: bool = True, timeout_per_worker: float = 5.0) -> None:
        """
        Signal all workers to stop and optionally wait for them to finish.

        Parameters
        ----------
        wait                : bool
            If True, block until all workers have exited (default: True).
        timeout_per_worker  : float
            Maximum seconds to wait per worker thread (default: 5.0).
        """
        logger.info("InferenceWorkerPool shutting down...")
        self._running = False

        # Drain remaining items and push stop sentinels
        for _ in range(self._n_workers):
            try:
                self._queue.put(_STOP, block=False)
            except queue.Full:
                pass

        if wait:
            for t in self._workers:
                t.join(timeout=timeout_per_worker)

        logger.info(
            "InferenceWorkerPool stopped | processed=%d | errors=%d",
            self._total_processed, self._total_errors,
        )

    # -----------------------------------------------------------------------
    @property
    def queue_size(self) -> int:
        """Number of flows currently waiting in the queue."""
        return self._queue.qsize()

    @property
    def total_processed(self) -> int:
        with self._lock:
            return self._total_processed

    @property
    def total_errors(self) -> int:
        with self._lock:
            return self._total_errors

    # -----------------------------------------------------------------------
    def _worker_loop(self, worker_id: int) -> None:
        """Main loop for each worker thread."""
        logger.debug("Worker-%d started", worker_id)

        while True:
            try:
                flow = self._queue.get(block=True, timeout=0.5)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if flow is _STOP:
                self._queue.task_done()
                break

            self._process_flow(flow, worker_id)
            self._queue.task_done()

        logger.debug("Worker-%d stopped", worker_id)

    def _process_flow(self, flow, worker_id: int) -> None:
        """
        Run the full inference pipeline for one flow.
        Errors are caught and counted — a single bad flow must not crash the worker.
        """
        try:
            t0 = time.perf_counter()

            feature_start = time.perf_counter()
            features = extract_features(flow)
            reg.feature_extraction_latency_ms.observe(
                (time.perf_counter() - feature_start) * 1000
            )
            result   = predict_flow(features)

            # Update Prometheus counters
            reg.predictions_total.inc()
            reg.inference_latency_ms.observe(result.get("latency_ms", 0))
            reg.flows_completed_total.inc()
            reg.flows_total.inc()
            reg.processing_time_ms.observe((time.perf_counter() - t0) * 1000)

            attack_type = result.get("attack_type", "BENIGN")
            attack_probability = float(result.get("attack_probability", 0.0) or 0.0)
            attack_confidence = float(result.get("attack_confidence", 0.0) or 0.0)
            iso_score = float(result.get("iso_score", 0.0) or 0.0)
            rf_probability = float(result.get("rf_probability", 0.0) or 0.0)
            meta_probability = float(result.get("meta_probability", 0.0) or 0.0)
            xgb_probability_val = float(result.get("xgb_probability", 0.0) or 0.0)

            reg.xgb_probability.set(xgb_probability_val)
            reg.rf_probability.set(rf_probability)
            reg.model_confidence.set(attack_confidence)
            reg.iso_score.set(iso_score)
            reg.meta_probability.set(meta_probability)

            if result.get("is_attack"):
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

            # Invoke caller's result handler (e.g. AlertManager.process)
            if self._on_result is not None:
                self._on_result(result, flow)

            with self._lock:
                self._total_processed += 1

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "Worker-%d | type=%s | latency=%.3fms | pipeline=%.3fms",
                worker_id, attack_type, result.get("latency_ms", 0), elapsed,
            )

        except Exception as exc:
            with self._lock:
                self._total_errors += 1
            logger.error("Worker-%d pipeline error: %s", worker_id, exc, exc_info=True)
