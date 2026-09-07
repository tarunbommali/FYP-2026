"""
alerts/alert.py
Alert dataclass — the canonical event record produced by the alert engine.

An Alert is created whenever the meta-learner classifies a flow as an attack
AND the flow passes alert_rules.py filtering.

Includes stacking ensemble scores: RF probability, XGBoost probability,
ISO anomaly score, and meta-learner combined probability.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Alert:
    """
    A single IDS alert event.

    Fields
    ------
    timestamp           : ISO-8601 UTC string at time of detection
    src_ip              : Source IP address
    dst_ip              : Destination IP address
    src_port            : Source port (0 for ICMP)
    dst_port            : Destination port (0 for ICMP)
    protocol            : IP protocol number (6=TCP, 17=UDP, 1=ICMP)
    attack_type         : Multiclass label (e.g. "DDoS", "PortScan", "UNKNOWN_ATTACK")
    attack_probability  : Meta-learner combined attack probability [0.0 – 1.0]
    rf_probability      : Random Forest attack probability [0.0 – 1.0]
    xgb_probability     : XGBoost Binary attack probability [0.0 – 1.0]
    meta_probability    : Logistic Regression meta-learner probability [0.0 – 1.0]
    attack_confidence   : Multiclass softmax max [0.0 – 1.0]
    iso_score           : Isolation Forest anomaly score [0.0 – 1.0]
    severity            : LOW | MEDIUM | HIGH | CRITICAL
    latency_ms          : End-to-end predict_flow() latency
    id                  : Auto-assigned by SQLite on insert (None before insert)
    """
    timestamp:          str
    src_ip:             str
    dst_ip:             str
    src_port:           int
    dst_port:           int
    protocol:           int
    attack_type:        str
    attack_probability: float
    rf_probability:     float
    xgb_probability:    float
    meta_probability:   float
    attack_confidence:  float
    iso_score:          float
    severity:           str
    latency_ms:         float
    id:                 Optional[int] = field(default=None, compare=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def now_utc() -> str:
        """Return current UTC time as ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Protocol name helper
# ---------------------------------------------------------------------------
PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}


def protocol_name(proto: int) -> str:
    return PROTOCOL_NAMES.get(proto, str(proto))
