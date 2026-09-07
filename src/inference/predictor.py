"""
inference/predictor.py
Core inference engine — the single entry point for all flow predictions.

Stacking Ensemble Pipeline
--------------------------
    Features → RF → XGBoost Binary → ISO Forest → Meta-Learner (LR)
      ├─ BENIGN → return immediately (store benign stats)
      └─ ATTACK → Unknown Check (ISO score only)
                    ├─ UNKNOWN_ATTACK → return (skip multiclass, severity=CRITICAL)
                    └─ Known attack → Multiclass XGBoost → Severity

Public API
----------
    predict_flow(features: dict) -> dict

    features : dict mapping feature name (str) -> value (float)
               Must contain all 78 columns in MODELS.feature_columns.

    Returns a dict matching PredictionResult schema:
    {
        "is_attack":          bool,
        "attack_probability": float,   # Meta-learner combined prob [0.0 – 1.0]
        "rf_probability":     float,   # Random Forest attack prob [0.0 – 1.0]
        "xgb_probability":    float,   # XGBoost Binary attack prob [0.0 – 1.0]
        "meta_probability":   float,   # Meta-learner prob (same as attack_probability)
        "attack_type":        str,     # e.g. "DDoS", "BENIGN", "UNKNOWN_ATTACK"
        "attack_confidence":  float,   # Confidence from deciding model [0.0 – 1.0]
        "iso_score":          float,   # normalised anomaly score [0.0 – 1.0]
        "severity":           str      # NONE | LOW | MEDIUM | HIGH | CRITICAL
    }

Confidence Semantics
--------------------
    BENIGN          : confidence = 1.0 (by definition)
    UNKNOWN_ATTACK  : confidence = meta_probability (meta-learner decided)
    Known attack    : confidence = multiclass softmax max (multiclass decided)

Model Roles
-----------
    Random Forest       → Stable supervised prediction (base learner)
    XGBoost Binary      → High-accuracy attack detection (base learner)
    Isolation Forest    → Detects anomalous / unknown attacks (base learner)
    Logistic Regression → Combines all three into a single binary decision (meta-learner)
    XGBoost Multiclass  → Identifies attack category (runs only for known attacks)
"""

import logging
import time
import math
import numpy as np
import pandas as pd

from inference.model_loader import MODELS
from inference.severity import calculate_severity
from inference.schemas import PredictionResult

import threading

logger = logging.getLogger(__name__)

# State lock and tracking for current attack type (to set previous to 0 and current to 1 cleanly)
_last_attack_lock = threading.Lock()
_current_attack_type = "BENIGN"


# ---------------------------------------------------------------------------
def predict_flow(features: dict) -> dict:
    """
    Run the full stacking ensemble IDS inference pipeline on one flow.

    Parameters
    ----------
    features : dict
        Mapping of feature name -> numeric value for all 78 NetFlow features.

    Returns
    -------
    dict
        Serialisable PredictionResult. Always returns a dict — exceptions are
        caught and returned as {"error": "<message>", "is_attack": False, ...}.
    """
    try:
        return _predict(features)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        from monitoring import metrics_registry as reg
        try:
            reg.prediction_errors_total.inc()
        except Exception:
            pass
        return PredictionResult(error=str(exc)).to_dict()


# ---------------------------------------------------------------------------
def _predict(features: dict) -> dict:
    global _current_attack_type
    start = time.perf_counter()
    logger.debug("[Predictor] _predict() called | feature_count=%d", len(features))

    # -- 1. Feature validation -----------------------------------------------
    missing = [col for col in MODELS.feature_columns if col not in features]
    if missing:
        raise ValueError(
            f"{len(missing)} missing feature(s): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    # -- 2. Preprocessing (NumPy-based) — single pass for all models ----------
    X_arr = np.zeros((1, len(MODELS.feature_columns)), dtype=np.float32)

    for i, col in enumerate(MODELS.feature_columns):
        val = features.get(col, 0.0)
        if math.isinf(val) or math.isnan(val):
            val = MODELS.feature_medians.get(col, 0.0) if MODELS.feature_medians else 0.0
        X_arr[0, i] = val

    # -- 3. Random Forest attack probability (Base Learner 1) ----------------
    rf_prob = float(MODELS.rf_binary.predict_proba(X_arr)[0, 1])

    # -- 4. XGBoost Binary attack probability (Base Learner 2) ---------------
    xgb_prob = float(MODELS.xgb_binary.predict_proba(X_arr)[0, 1])

    # -- 5. Isolation Forest anomaly score (Base Learner 3) ------------------
    X_iso    = MODELS.iso_scaler.transform(X_arr)
    iso_raw  = float(MODELS.iso_model.decision_function(X_iso)[0])
    iso_neg  = -iso_raw
    iso_norm = (iso_neg - MODELS.iso_min) / (MODELS.iso_max - MODELS.iso_min + 1e-10)
    iso_score = float(max(0.0, min(1.0, iso_norm)))   # clip to [0, 1]

    # -- 6. Meta-Learner (Logistic Regression) — stacking combiner -----------
    meta_features = np.array([[rf_prob, xgb_prob, iso_score]], dtype=np.float64)
    meta_prob = float(MODELS.meta_learner.predict_proba(meta_features)[0, 1])
    is_attack = meta_prob >= MODELS.meta_learner_threshold

    # -- 7. Early Exit for BENIGN --------------------------------------------
    if not is_attack:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        from monitoring import metrics_registry as reg
        try:
            with _last_attack_lock:
                if _current_attack_type != "BENIGN":
                    reg.last_attack_type.labels(type=_current_attack_type).set(0)
                    _current_attack_type = "BENIGN"
                reg.last_attack_type.labels(type="BENIGN").set(1)
            reg.attack_confidence.set(1.0)
            reg.rf_probability.set(rf_prob)
            reg.xgb_probability.set(xgb_prob)
            reg.iso_score.set(iso_score)
            reg.meta_probability.set(meta_prob)
        except Exception:
            pass
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

    # -- 8. Unknown Attack Check (ISO score only) ----------------------------
    #   Isolation Forest answers: "Does this look unlike anything seen before?"
    #   If iso_score >= threshold → UNKNOWN_ATTACK. Skip multiclass entirely.
    is_unknown = iso_score >= MODELS.unknown_attack_iso_threshold

    if is_unknown:
        # Unknown / zero-day attack — multiclass model is NOT executed.
        # Confidence = meta-learner probability (the model that made the decision).
        attack_type       = "UNKNOWN_ATTACK"
        attack_confidence = meta_prob          # meta-learner is the deciding model
        severity          = "CRITICAL"         # unknown threats are always critical

        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        logger.warning(
            "UNKNOWN_ATTACK | iso=%.4f >= %.2f | meta=%.4f | "
            "rf=%.4f | xgb=%.4f | sev=%s | latency=%.3fms",
            iso_score, MODELS.unknown_attack_iso_threshold,
            meta_prob, rf_prob, xgb_prob, severity, latency_ms,
        )

        from monitoring import metrics_registry as reg
        try:
            reg.unknown_attacks_total.inc()
            with _last_attack_lock:
                if _current_attack_type != attack_type:
                    reg.last_attack_type.labels(type=_current_attack_type).set(0)
                    _current_attack_type = attack_type
                reg.last_attack_type.labels(type=attack_type).set(1)
            reg.attack_confidence.set(attack_confidence)
            reg.rf_probability.set(rf_prob)
            reg.xgb_probability.set(xgb_prob)
            reg.iso_score.set(iso_score)
            reg.meta_probability.set(meta_prob)
        except Exception:
            pass

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

    # -- 9. Multiclass XGBoost (known attacks only) --------------------------
    #   Only runs when the attack is NOT flagged as unknown.
    multi_probs      = MODELS.xgb_multiclass.predict_proba(X_arr)[0]
    class_idx        = int(np.argmax(multi_probs))
    attack_type      = str(MODELS.label_encoder.inverse_transform([class_idx])[0])
    attack_confidence = float(np.max(multi_probs))  # multiclass is the deciding model

    # -- 10. Severity scoring (attack-type-aware) ----------------------------
    severity = calculate_severity(attack_type, attack_confidence, iso_score)

    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    logger.info(
        "ATTACK | type=%s | meta=%.4f | rf=%.4f | xgb=%.4f | "
        "iso=%.4f | conf=%.4f | sev=%s | latency=%.3fms",
        attack_type, meta_prob, rf_prob, xgb_prob,
        iso_score, attack_confidence, severity, latency_ms,
    )

    from monitoring import metrics_registry as reg
    try:
        with _last_attack_lock:
            if _current_attack_type != attack_type:
                reg.last_attack_type.labels(type=_current_attack_type).set(0)
                _current_attack_type = attack_type
            reg.last_attack_type.labels(type=attack_type).set(1)
        reg.attack_confidence.set(attack_confidence)
        reg.rf_probability.set(rf_prob)
        reg.xgb_probability.set(xgb_prob)
        reg.iso_score.set(iso_score)
        reg.meta_probability.set(meta_prob)
    except Exception:
        pass

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


# ---------------------------------------------------------------------------
# Batch helper — convenience wrapper for processing multiple flows at once
# ---------------------------------------------------------------------------
def predict_batch(flows: list[dict]) -> list[dict]:
    """
    Run predict_flow() on a list of feature dicts.
    Each flow is processed independently; errors are isolated per-flow.

    Parameters
    ----------
    flows : list[dict]
        List of feature dicts, each matching the predict_flow() spec.

    Returns
    -------
    list[dict]
        List of PredictionResult dicts, same order as input.
    """
    return [predict_flow(f) for f in flows]
