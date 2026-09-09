"""
model_training/test_runtime_inference.py
Offline Flow-Level Runtime Inference Evaluation

Validates the deployed IDS runtime inference engine (src/inference/predictor.py)
against held-out CICIDS2017 test flow feature vectors (X_test_binary.parquet).

Key Characteristics:
- Offline flow-level simulation (does NOT require packet capture or PCAP replay).
- Passes feature dictionaries directly to predict_flow(), validating the exact
  stacking ensemble (RF + XGBoost + Isolation Forest -> LR Meta-Learner).
- Supports fast sampled validation (e.g. 100 flows) or complete 20% test-set evaluation.
- Outputs precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and latency statistics.

Usage:
    # Quick 100-flow smoke test
    python model_training/test_runtime_inference.py --sample 100

    # Stratified 1,000-flow evaluation
    python model_training/test_runtime_inference.py --sample 1000

    # Complete 20% test-set evaluation (~514k flows)
    python model_training/test_runtime_inference.py --full
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# pyrefly: ignore [missing-import]
from inference.predictor import predict_flow  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_runtime_inference")


# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deployed IDS runtime inference on held-out flow feature vectors."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of test flow feature vectors to evaluate (default: 100). Ignored if --full is set.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate the full 20%% test set (~514k flows). May take several minutes.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(BASE_DIR, "model_training", "evaluation", "runtime_inference_report.json"),
        help="Path to save the JSON evaluation report.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
def load_test_data(sample_size: int, full: bool, random_state: int):
    x_path = os.path.join(BASE_DIR, "data", "processed", "X_test_binary.parquet")
    y_path = os.path.join(BASE_DIR, "data", "processed", "y_test_binary.parquet")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Test data parquets not found.\n  Expected: {x_path}\n  Expected: {y_path}\n"
            "Run 02_binary_training.py to generate isolated test parquets."
        )

    logger.info("Loading test flow feature vectors from parquet...")
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["Label"].astype(int)

    total_flows = len(X)
    logger.info("Loaded full test set: %d total flow feature vectors", total_flows)

    if full or sample_size >= total_flows:
        return X, y, total_flows

    logger.info("Extracting stratified sample of %d flow feature vectors...", sample_size)
    # Stratified sample to preserve benign/attack ratio
    attack_ratio = y.mean()
    n_attack = max(1, int(round(sample_size * attack_ratio)))
    n_benign = sample_size - n_attack

    attack_indices = y[y == 1].sample(n=n_attack, random_state=random_state).index
    benign_indices = y[y == 0].sample(n=n_benign, random_state=random_state).index
    sample_indices = attack_indices.union(benign_indices)

    return X.loc[sample_indices], y.loc[sample_indices], sample_size


# ---------------------------------------------------------------------------
def run_runtime_inference(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    n_flows = len(X)
    logger.info("Running predict_flow() sequentially on %d test flows...", n_flows)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    latencies: List[float] = []
    attack_types: Dict[str, int] = {}
    severities: Dict[str, int] = {}
    errors: int = 0

    progress_interval = max(1, n_flows // 10)
    t_start = time.perf_counter()

    for idx, (flow_idx, row) in enumerate(X.iterrows()):
        features = row.to_dict()
        true_label = int(y.loc[flow_idx])

        t0 = time.perf_counter()
        result = predict_flow(features)
        t_flow = (time.perf_counter() - t0) * 1000.0  # ms

        latencies.append(t_flow)

        if result.get("error"):
            errors += 1
            pred_label = 0
            prob = 0.0
        else:
            pred_label = 1 if result.get("is_attack") else 0
            prob = float(result.get("meta_probability", result.get("attack_probability", 0.0)))

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_prob.append(prob)

        atk_type = result.get("attack_type", "BENIGN")
        attack_types[atk_type] = attack_types.get(atk_type, 0) + 1

        sev = result.get("severity", "NONE")
        severities[sev] = severities.get(sev, 0) + 1

        if (idx + 1) % progress_interval == 0 or (idx + 1) == n_flows:
            logger.info("Progress: %d/%d flows evaluated (%.1f%%)", idx + 1, n_flows, ((idx + 1) / n_flows) * 100)

    total_time = time.perf_counter() - t_start

    # Compute classification metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    lat_arr = np.array(latencies)
    lat_stats = {
        "mean_ms": float(np.mean(lat_arr)),
        "median_ms": float(np.median(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "min_ms": float(np.min(lat_arr)),
        "max_ms": float(np.max(lat_arr)),
        "total_elapsed_sec": float(total_time),
        "flows_per_second": float(n_flows / total_time) if total_time > 0 else 0.0,
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_type": "Offline Flow-Level Runtime Inference",
        "dataset": "CICIDS2017 Held-Out Test Set (X_test_binary.parquet)",
        "total_flows_evaluated": n_flows,
        "errors": errors,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "latency_stats": lat_stats,
        "attack_type_distribution": attack_types,
        "severity_distribution": severities,
    }

    return report


# ---------------------------------------------------------------------------
def print_summary(report: Dict[str, Any]) -> None:
    m = report["metrics"]
    cm = report["confusion_matrix"]
    lat = report["latency_stats"]

    print("\n" + "=" * 65)
    print("      OFFLINE FLOW-LEVEL RUNTIME INFERENCE EVALUATION")
    print("=" * 65)
    print(f" Flows Evaluated    : {report['total_flows_evaluated']:,}")
    print(f" Execution Time     : {lat['total_elapsed_sec']:.2f} s ({lat['flows_per_second']:.1f} flows/sec)")
    print(f" Mean Latency       : {lat['mean_ms']:.2f} ms (p95: {lat['p95_ms']:.2f} ms)")
    print("-" * 65)
    print(" CLASSIFICATION METRICS (Runtime predict_flow vs Ground Truth y_test)")
    print("-" * 65)
    print(f" Accuracy           : {m['accuracy'] * 100:.2f} %")
    print(f" Precision          : {m['precision'] * 100:.2f} %")
    print(f" Recall             : {m['recall'] * 100:.2f} %")
    print(f" F1-Score           : {m['f1_score'] * 100:.2f} %")
    print(f" ROC-AUC            : {m['roc_auc']:.4f}")
    print(f" PR-AUC             : {m['pr_auc']:.4f}")
    print(f" False Positive Rate: {m['false_positive_rate'] * 100:.2f} %")
    print(f" False Negative Rate: {m['false_negative_rate'] * 100:.2f} %")
    print("-" * 65)
    print(" CONFUSION MATRIX")
    print("-" * 65)
    print(f"  True Negatives  (TN) [Benign -> Benign] : {cm['true_negatives']:,}")
    print(f"  False Positives (FP) [Benign -> Attack] : {cm['false_positives']:,}")
    print(f"  False Negatives (FN) [Attack -> Benign] : {cm['false_negatives']:,}")
    print(f"  True Positives  (TP) [Attack -> Attack] : {cm['true_positives']:,}")
    print("-" * 65)
    print(" DETECTED ATTACK TYPES")
    for atk, count in sorted(report["attack_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {atk:<25}: {count:,}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()
    X, y, sample_size = load_test_data(args.sample, args.full, args.random_state)
    report = run_runtime_inference(X, y)
    print_summary(report)

    # Save output report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written to %s", args.output)


if __name__ == "__main__":
    main()
