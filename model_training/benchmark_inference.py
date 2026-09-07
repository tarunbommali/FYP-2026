"""
benchmark_inference.py
Measures predict_flow() latency, throughput, and system resource usage
over N iterations.

Distinguishes first-call warmup from steady-state performance and captures
CPU and memory utilisation during the benchmark run.

Usage:
    venv\\Scripts\\python.exe benchmark_inference.py

Targets (final-year project IDS):
    Steady-state latency : < 20 ms avg         (Excellent: < 5 ms)
    Throughput           : > 50 predictions/sec (Excellent: > 100 pred/sec)
    CPU usage (avg)      : < 80 %
    Memory headroom      : > 10 % free
"""

import os
import sys
import time
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore [missing-import]
from inference.predictor import predict_flow
import pandas as pd

N_WARMUP = 5
N_BENCH  = 1000


# ---------------------------------------------------------------------------
# System metrics sampler
# ---------------------------------------------------------------------------

class _SystemSampler:
    """Samples CPU and memory in a background thread during the benchmark."""

    def __init__(self, interval: float = 0.1):
        try:
            import psutil
            self._psutil    = psutil
            self._available = True
        except ImportError:
            self._psutil    = None
            self._available = False

        self._interval  = interval
        self._cpu_samples: list = []
        self._mem_samples: list = []
        self._running   = False
        self._thread    = None

    def start(self) -> None:
        if not self._available:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            self._cpu_samples.append(self._psutil.cpu_percent(interval=None))
            self._mem_samples.append(self._psutil.virtual_memory().percent)
            time.sleep(self._interval)

    @property
    def cpu_avg(self):
        return sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else None

    @property
    def cpu_max(self):
        return max(self._cpu_samples) if self._cpu_samples else None

    @property
    def mem_avg(self):
        return sum(self._mem_samples) / len(self._mem_samples) if self._mem_samples else None

    @property
    def mem_max(self):
        return max(self._mem_samples) if self._mem_samples else None


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _build_features():
    """Load one real BENIGN and one real ATTACK sample from the test set."""
    print("Loading test data to find real samples...")
    X_path = os.path.join(PROJECT_ROOT, "data", "processed", "X_test_binary.parquet")
    y_path = os.path.join(PROJECT_ROOT, "data", "processed", "y_test_binary.parquet")

    if not os.path.exists(X_path):
        print("Test data not found. Cannot run benchmark.")
        return {}, {}

    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)

    label_col  = y.columns[0]
    attack_idx = y[y[label_col] == 1].index[0]
    benign_idx = y[y[label_col] == 0].index[0]

    return X.loc[benign_idx].to_dict(), X.loc[attack_idx].to_dict()


# ---------------------------------------------------------------------------
# Single benchmark pass
# ---------------------------------------------------------------------------

def _run_pass(label: str, features: dict) -> dict:
    """Run one benchmark pass and return a results dict."""
    sampler = _SystemSampler(interval=0.1)

    # -- Warmup --------------------------------------------------------------
    for _ in range(N_WARMUP):
        predict_flow(features)

    # -- Benchmark -----------------------------------------------------------
    latencies = []
    sampler.start()
    t_start = time.perf_counter()

    for _ in range(N_BENCH):
        result = predict_flow(features)
        latencies.append(result["latency_ms"])

    t_elapsed = time.perf_counter() - t_start
    sampler.stop()

    latencies.sort()
    avg_ms     = sum(latencies) / len(latencies)
    min_ms     = latencies[0]
    max_ms     = latencies[-1]
    p50_ms     = latencies[int(0.50 * N_BENCH)]
    p95_ms     = latencies[int(0.95 * N_BENCH)]
    p99_ms     = latencies[int(0.99 * N_BENCH)]
    throughput = N_BENCH / t_elapsed

    print(f"\n[{label} Path — {N_BENCH} iterations]")
    print(f"  {'Metric':<28} {'Value':>12}")
    print(f"  {'-'*42}")
    print(f"  {'Avg latency':<28} {avg_ms:>11.3f} ms")
    print(f"  {'Min latency':<28} {min_ms:>11.3f} ms")
    print(f"  {'P50 latency':<28} {p50_ms:>11.3f} ms")
    print(f"  {'P95 latency':<28} {p95_ms:>11.3f} ms")
    print(f"  {'P99 latency':<28} {p99_ms:>11.3f} ms")
    print(f"  {'Max latency':<28} {max_ms:>11.3f} ms")
    print(f"  {'Throughput':<28} {throughput:>10.2f} pred/sec")

    if sampler.cpu_avg is not None:
        print(f"  {'CPU avg %':<28} {sampler.cpu_avg:>10.1f} %")
        print(f"  {'CPU max %':<28} {sampler.cpu_max:>10.1f} %")
        print(f"  {'Memory avg %':<28} {sampler.mem_avg:>10.1f} %")
        print(f"  {'Memory max %':<28} {sampler.mem_max:>10.1f} %")
    else:
        print("  (install psutil for CPU/memory metrics: pip install psutil)")

    return {
        "label":      label,
        "avg_ms":     avg_ms,
        "min_ms":     min_ms,
        "p50_ms":     p50_ms,
        "p95_ms":     p95_ms,
        "p99_ms":     p99_ms,
        "max_ms":     max_ms,
        "throughput": throughput,
        "cpu_avg":    sampler.cpu_avg,
        "cpu_max":    sampler.cpu_max,
        "mem_avg":    sampler.mem_avg,
        "mem_max":    sampler.mem_max,
    }


# ---------------------------------------------------------------------------
# Rating helpers
# ---------------------------------------------------------------------------

def _rate_latency(avg_ms: float) -> str:
    if avg_ms < 5:
        return "Excellent  (< 5 ms)  — production-ready for high-throughput IDS"
    if avg_ms < 20:
        return "Good       (< 20 ms) — suitable for final-year IDS deployment"
    if avg_ms < 50:
        return "Acceptable (< 50 ms) — consider model quantisation or batch mode"
    return "Needs investigation — check sklearn warnings or RAM pressure"


def _rate_throughput(thr: float) -> str:
    if thr >= 100:
        return f"Excellent — {thr:.0f} pred/sec exceeds 100 target"
    if thr >= 50:
        return f"Good      — {thr:.0f} pred/sec exceeds 50 target"
    return f"Low       — {thr:.0f} pred/sec, target is 50+"


def _rate_cpu(cpu_avg) -> str:
    if cpu_avg is None:
        return "n/a (psutil not installed)"
    if cpu_avg < 50:
        return f"Low      ({cpu_avg:.1f}%) — ample headroom"
    if cpu_avg < 80:
        return f"Moderate ({cpu_avg:.1f}%) — acceptable"
    return f"High     ({cpu_avg:.1f}%) — consider optimisation"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> dict:
    print("=" * 60)
    print("INFERENCE BENCHMARK WITH SYSTEM METRICS")
    print(f"Warmup calls : {N_WARMUP}")
    print(f"Benchmark    : {N_BENCH} predictions per path")
    print("=" * 60)

    benign_features, attack_features = _build_features()
    if not benign_features:
        return {}

    r_b = _run_pass("BENIGN", benign_features)
    r_a = _run_pass("ATTACK", attack_features)

    # Weighted average — CICIDS2017 benign ratio ~83.5%
    w_b, w_a    = 0.835, 0.165
    overall_avg = r_b["avg_ms"] * w_b + r_a["avg_ms"] * w_a
    overall_thr = 1000 / overall_avg

    cpu_avg_overall = None
    if r_b["cpu_avg"] is not None and r_a["cpu_avg"] is not None:
        cpu_avg_overall = r_b["cpu_avg"] * w_b + r_a["cpu_avg"] * w_a

    print("\n" + "=" * 60)
    print("[OVERALL RATING — Weighted (83.5% Benign / 16.5% Attack)]")
    print(f"  Weighted Avg Latency  : {overall_avg:.2f} ms")
    print(f"  Weighted Throughput   : {overall_thr:.0f} pred/sec")
    print(f"  Latency   : {_rate_latency(overall_avg)}")
    print(f"  Throughput: {_rate_throughput(overall_thr)}")
    print(f"  CPU       : {_rate_cpu(cpu_avg_overall)}")
    print("=" * 60)

    return {
        "benign": r_b,
        "attack": r_a,
        "overall": {
            "weighted_avg_latency_ms": round(overall_avg, 3),
            "weighted_throughput":     round(overall_thr, 2),
            "cpu_avg_percent":         round(cpu_avg_overall, 1) if cpu_avg_overall else None,
        },
    }


if __name__ == "__main__":
    run_benchmark()
