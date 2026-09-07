"""
inference/schemas.py
Input / output schemas for the inference engine.

Uses Python dataclasses for zero-dependency validation.
No external schema libraries (pydantic, marshmallow) required.

Updated for the stacking ensemble pipeline — includes RF probability
and meta-learner probability fields alongside the original metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

@dataclass
class FlowFeatures:
    """
    Represents the 78 NetFlow features extracted from a network flow.
    Pass as a plain dict to predict_flow(); this class is available
    for type-checking and documentation purposes.
    """
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return self.features


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Structured result returned by predict_flow().

    Fields
    ------
    is_attack           : True if the meta-learner classifies the flow as an attack.
    attack_probability  : Meta-learner combined attack probability [0.0 – 1.0].
    rf_probability      : Random Forest attack probability [0.0 – 1.0].
    xgb_probability     : XGBoost Binary attack probability [0.0 – 1.0].
    meta_probability    : Logistic Regression meta-learner probability [0.0 – 1.0].
    attack_type         : Class label (e.g. "DDoS", "BENIGN", "PortScan", "UNKNOWN_ATTACK").
    attack_confidence   : Multiclass softmax confidence [0.0 – 1.0].
                          Set to 1.0 for BENIGN flows.
    iso_score           : Normalised Isolation Forest anomaly score [0.0 – 1.0].
                          Higher = more anomalous.
    severity            : NONE | LOW | MEDIUM | HIGH | CRITICAL
    error               : Populated only if an exception occurred.

    Pipeline
    --------
    Features → RF → XGBoost Binary → ISO Forest → Meta-Learner (LR)
      ├─ BENIGN → return
      └─ ATTACK → Multiclass XGBoost → Unknown Attack Check → Severity → Result
    """
    is_attack:          bool    = False
    attack_probability: float   = 0.0
    rf_probability:     float   = 0.0
    xgb_probability:    float   = 0.0
    meta_probability:   float   = 0.0
    attack_type:        str     = "BENIGN"
    attack_confidence:  float   = 1.0
    iso_score:          float   = 0.0
    severity:           str     = "NONE"
    latency_ms:         float   = 0.0   # end-to-end inference time in milliseconds
    error:              Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Severity levels (ordered)
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
