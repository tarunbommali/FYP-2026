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
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}


def protocol_name(proto: int) -> str:
    return PROTOCOL_NAMES.get(proto, str(proto))

