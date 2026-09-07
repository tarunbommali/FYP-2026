"""
alerts/severity.py
Alert severity scoring for the IDS pipeline.

Severity is a weighted combination of:
  - Meta-learner attack probability  (weight 0.8) — primary signal
  - ISO anomaly score                (weight 0.2) — auxiliary signal

Both inputs must be in [0.0, 1.0].

The meta-learner probability replaces the raw XGBoost probability as the
primary signal. It already incorporates RF, XGBoost Binary, and ISO Forest
outputs through the stacking ensemble.

Severity Thresholds
-------------------
  CRITICAL  : combined_score >= 0.95
  HIGH      : combined_score >= 0.85
  MEDIUM    : combined_score >= 0.70
  LOW       : combined_score >= 0.0   (attack confirmed but low confidence)
  NONE      : flow classified as BENIGN (severity not computed)
"""


def calculate_severity(attack_probability: float, iso_score: float) -> str:
    """
    Compute alert severity from meta-learner attack probability and ISO anomaly score.

    Parameters
    ----------
    attack_probability : float
        Meta-learner predicted probability that the flow is an attack [0.0 – 1.0].
        This is the combined output of the stacking ensemble.
    iso_score : float
        Normalised Isolation Forest anomaly score [0.0 – 1.0].
        Higher = more anomalous.

    Returns
    -------
    str
        One of: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        (caller is responsible for returning "NONE" for BENIGN flows)
    """
    attack_probability = float(max(0.0, min(1.0, attack_probability)))
    iso_score          = float(max(0.0, min(1.0, iso_score)))

    combined = attack_probability * 0.8 + iso_score * 0.2

    if combined >= 0.95:
        return "CRITICAL"
    if combined >= 0.85:
        return "HIGH"
    if combined >= 0.70:
        return "MEDIUM"
    return "LOW"


def severity_to_int(severity: str) -> int:
    """Convert severity label to integer for comparison / sorting."""
    return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(
        severity, 0
    )
