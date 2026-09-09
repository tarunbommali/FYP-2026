"""
Inference engine executing the stacking ensemble pipeline for NetFlow records.
"""

import logging
import math
import threading
import time
import warnings
import numpy as np

# Suppress feature names warnings and XGBoost serialization warnings during high-throughput inference
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from inference.model_loader import MODELS
from inference.schemas import PredictionResult
from inference.severity import calculate_severity
from monitoring import metrics_registry as reg

logger = logging.getLogger(__name__)

_last_attack_lock = threading.Lock()
_current_attack_type = None


def _update_prediction_metrics(
    attack_type: str,
    confidence: float,
    rf_prob: float,
    xgb_prob: float,
    iso_score: float,
    meta_prob: float,
) -> None:
    global _current_attack_type
    with _last_attack_lock:
        if attack_type != "BENIGN":
            if _current_attack_type and _current_attack_type != attack_type:
                reg.last_attack_type.labels(type=_current_attack_type).set(0)
            _current_attack_type = attack_type
            reg.last_attack_type.labels(type=attack_type).set(1)

    reg.attack_confidence.set(confidence)
    reg.rf_probability.set(rf_prob)
    reg.xgb_probability.set(xgb_prob)
    reg.iso_score.set(iso_score)
    reg.meta_probability.set(meta_prob)


def predict_flow(features: dict) -> dict:
    """Run the full stacking ensemble IDS inference pipeline on one flow."""
    try:
        return _predict(features)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        reg.prediction_errors_total.inc()
        return PredictionResult(error=str(exc)).to_dict()


def _predict(features: dict) -> dict:
    start = time.perf_counter()
    logger.debug("[Predictor] _predict() called | feature_count=%d", len(features))

    missing = [col for col in MODELS.feature_columns if col not in features]
    if missing:
        raise ValueError(
            f"{len(missing)} missing feature(s): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    # Preprocessing: single-pass array construction with NaN/inf median imputation
    X_arr = np.zeros((1, len(MODELS.feature_columns)), dtype=np.float32)
    for i, col in enumerate(MODELS.feature_columns):
        val = features.get(col, 0.0)
        if math.isinf(val) or math.isnan(val):
            val = MODELS.feature_medians.get(col, 0.0) if MODELS.feature_medians else 0.0
        X_arr[0, i] = val

    rf_prob = float(MODELS.rf_binary.predict_proba(X_arr)[0, 1])
    xgb_prob = float(MODELS.xgb_binary.predict_proba(X_arr)[0, 1])

    # Negate because IsolationForest decision_function produces higher values for normal samples
    X_iso = MODELS.iso_scaler.transform(X_arr)
    iso_raw = float(MODELS.iso_model.decision_function(X_iso)[0])
    iso_neg = -iso_raw
    iso_norm = (iso_neg - MODELS.iso_min) / (MODELS.iso_max - MODELS.iso_min + 1e-10)
    iso_score = float(max(0.0, min(1.0, iso_norm)))

    # Logistic Regression meta-learner decision
    meta_features = np.array([[rf_prob, xgb_prob, iso_score]], dtype=np.float64)
    meta_prob = float(MODELS.meta_learner.predict_proba(meta_features)[0, 1])
    is_attack = meta_prob >= MODELS.meta_learner_threshold

    # -- 7. Early Exit for BENIGN --------------------------------------------
    if not is_attack:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        _update_prediction_metrics("BENIGN", 1.0, rf_prob, xgb_prob, iso_score, meta_prob)
        return PredictionResult(
            is_attack          = False,
            attack_probability = round(meta_prob, 6),
            rf_probability     = round(rf_prob, 6),
            xgb_probability    = round(xgb_prob, 6),
            meta_probability   = round(meta_prob, 6),
            attack_type        = "BENIGN",
            attack_confidence  = 1.0,
            iso_score          = round(iso_score, 6),
            severity           = "NONE",
            latency_ms         = latency_ms,
        ).to_dict()

    # Unknown attack check based on isolation forest threshold
    if iso_score >= MODELS.unknown_attack_iso_threshold:
        attack_type = "UNKNOWN_ATTACK"
        attack_confidence = meta_prob
        severity = "CRITICAL"
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        logger.warning(
            "UNKNOWN_ATTACK | iso=%.4f >= %.2f | meta=%.4f | sev=%s | latency=%.3fms",
            iso_score, MODELS.unknown_attack_iso_threshold, meta_prob, severity, latency_ms,
        )
        reg.unknown_attacks_total.inc()
        _update_prediction_metrics(attack_type, attack_confidence, rf_prob, xgb_prob, iso_score, meta_prob)

        return PredictionResult(
            is_attack          = True,
            attack_probability = round(meta_prob, 6),
            rf_probability     = round(rf_prob, 6),
            xgb_probability    = round(xgb_prob, 6),
            meta_probability   = round(meta_prob, 6),
            attack_type        = attack_type,
            attack_confidence  = round(attack_confidence, 6),
            iso_score          = round(iso_score, 6),
            severity           = severity,
            latency_ms         = latency_ms,
        ).to_dict()

    # Known attack classification via multiclass XGBoost
    multi_probs = MODELS.xgb_multiclass.predict_proba(X_arr)[0]
    class_idx = int(np.argmax(multi_probs))
    attack_type = str(MODELS.label_encoder.inverse_transform([class_idx])[0])
    attack_confidence = float(np.max(multi_probs))

    severity = calculate_severity(attack_type, attack_confidence, iso_score)
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    logger.info(
        "ATTACK | type=%s | meta=%.4f | conf=%.4f | sev=%s | latency=%.3fms",
        attack_type, meta_prob, attack_confidence, severity, latency_ms,
    )
    _update_prediction_metrics(attack_type, attack_confidence, rf_prob, xgb_prob, iso_score, meta_prob)

    return PredictionResult(
        is_attack          = True,
        attack_probability = round(meta_prob, 6),
        rf_probability     = round(rf_prob, 6),
        xgb_probability    = round(xgb_prob, 6),
        meta_probability   = round(meta_prob, 6),
        attack_type        = attack_type,
        attack_confidence  = round(attack_confidence, 6),
        iso_score          = round(iso_score, 6),
        severity           = severity,
        latency_ms         = latency_ms,
    ).to_dict()


def predict_batch(flows: list[dict]) -> list[dict]:
    """Run predict_flow() on a list of flow feature dictionaries."""
    return [predict_flow(f) for f in flows]
