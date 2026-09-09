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


@dataclass
class FlowFeatures:
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return self.features


@dataclass
class PredictionResult:
    is_attack:          bool    = False
    attack_probability: float   = 0.0
    rf_probability:     float   = 0.0
    xgb_probability:    float   = 0.0
    meta_probability:   float   = 0.0
    attack_type:        str     = "BENIGN"
    attack_confidence:  float   = 1.0
    iso_score:          float   = 0.0
    severity:           str     = "NONE"
    latency_ms:         float   = 0.0
    error:              Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

