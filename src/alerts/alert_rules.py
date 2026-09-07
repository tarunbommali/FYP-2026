"""
alerts/alert_rules.py
Configurable filtering rules applied before an alert is stored.

Rules prevent alert fatigue and suppress known-safe traffic.
All rules are read from config.json at startup; defaults are conservative.

Rule evaluation order:
    1. is_attack check (never alert on BENIGN)
    2. minimum severity gate
    3. minimum confidence gate
    4. suppressed attack types list
    5. suppressed destination ports list

If ALL rules pass, the flow becomes an Alert.
"""

import json
import logging
import os
from typing import Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering (used for >= comparison)
# ---------------------------------------------------------------------------
_SEVERITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class AlertRules:
    """
    Stateless rule evaluator loaded from config.json.

    config.json keys (all optional — safe defaults shown):
    -------------------------------------------------------
    alert_min_severity      : "LOW"     — minimum severity to alert
    alert_min_confidence    : 0.5       — minimum attack_probability to alert
    alert_suppressed_types  : []        — attack types to suppress (e.g. ["Bot"])
    alert_suppressed_ports  : []        — dst ports to suppress (e.g. [80, 443])
    """

    def __init__(self, config_path: str) -> None:
        cfg = self._load_config(config_path)

        self.min_severity:      str       = cfg.get("alert_min_severity",   "LOW")
        self.min_confidence:    float     = float(cfg.get("alert_min_confidence", 0.5))
        self.suppressed_types:  Set[str]  = set(cfg.get("alert_suppressed_types", []))
        self.suppressed_ports:  Set[int]  = set(cfg.get("alert_suppressed_ports", []))

        logger.info(
            "AlertRules loaded | min_severity=%s | min_confidence=%.2f | "
            "suppressed_types=%s | suppressed_ports=%s",
            self.min_severity, self.min_confidence,
            self.suppressed_types, self.suppressed_ports,
        )

    def should_alert(self, prediction: dict, dst_port: int = 0) -> tuple[bool, str]:
        """
        Evaluate whether a prediction warrants an alert.

        Parameters
        ----------
        prediction : dict
            Return value from predict_flow().
        dst_port : int
            Destination port of the flow being evaluated.

        Returns
        -------
        (should_alert: bool, reason: str)
            reason is populated when should_alert is False.
        """
        # Rule 1 — must be classified as an attack
        if not prediction.get("is_attack", False):
            return False, "BENIGN flow"

        severity      = prediction.get("severity", "NONE")
        attack_type   = prediction.get("attack_type", "")
        # Use meta-learner probability as primary confidence signal for the
        # stacking ensemble; fall back to attack_probability for compatibility.
        attack_prob   = prediction.get("meta_probability",
                                        prediction.get("attack_probability", 0.0))

        # Rule 2 — minimum severity
        if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK.get(self.min_severity, 1):
            return False, f"severity {severity} < min {self.min_severity}"

        # Rule 3 — minimum confidence (meta-learner probability)
        if attack_prob < self.min_confidence:
            return False, f"meta_probability {attack_prob:.3f} < min {self.min_confidence:.3f}"

        # Rule 4 — suppressed attack types
        if attack_type in self.suppressed_types:
            return False, f"attack_type '{attack_type}' suppressed"

        # Rule 5 — suppressed destination ports
        if dst_port in self.suppressed_ports:
            return False, f"dst_port {dst_port} suppressed"

        return True, ""

    # -----------------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str) -> dict:
        if not os.path.exists(config_path):
            logger.info("config.json not found — using default alert rules")
            return {}
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load config.json: %s — using defaults", exc)
            return {}
