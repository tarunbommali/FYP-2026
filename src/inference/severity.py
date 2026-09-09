"""
inference/severity.py
Attack-type-aware severity scoring for detected attacks.

Severity answers: "How dangerous is this attack?"
This is separate from unknown detection (which answers: "Have we seen this before?")

Severity is determined by three factors:
  1. Attack type — intrinsic danger level (DDoS > PortScan)
  2. Confidence  — model's certainty in the classification
  3. Anomaly     — ISO Forest score can escalate severity

Severity tiers
--------------
    NONE     — flow classified as benign
    LOW      — marginal attack signal or low-risk attack type
    MEDIUM   — moderate confidence or medium-risk attack
    HIGH     — high confidence or high-risk attack
    CRITICAL — near-certain high-risk attack, or UNKNOWN_ATTACK
"""

from typing import Dict

_ATTACK_BASE_SEVERITY: Dict[str, str] = {
    # Denial of Service — service disruption → CRITICAL
    "DDoS":              "CRITICAL",
    "DoS Hulk":          "CRITICAL",
    "DoS GoldenEye":     "CRITICAL",
    "DoS slowloris":     "CRITICAL",
    "DoS Slowhttptest":  "CRITICAL",

    # Exploitation — data breach / system compromise → CRITICAL
    "Heartbleed":        "CRITICAL",

    # Brute Force — credential theft → HIGH
    "SSH-Patator":       "HIGH",
    "FTP-Patator":       "HIGH",

    # Web Attacks — application exploitation → HIGH
    "Web Attack \u2013 Brute Force": "HIGH",
    "Web Attack \u2013 XSS":         "HIGH",
    "Web Attack \u2013 Sql Injection": "HIGH",
    "Web Attack - Brute Force": "HIGH",
    "Web Attack - XSS":         "HIGH",
    "Web Attack - Sql Injection": "HIGH",

    # Infiltration — lateral movement → HIGH
    "Infiltration":      "HIGH",

    # Bot — command & control → MEDIUM
    "Bot":               "MEDIUM",

    # Reconnaissance — information gathering → MEDIUM
    "PortScan":          "MEDIUM",

    # Unknown — novel/zero-day → CRITICAL (always)
    "UNKNOWN_ATTACK":    "CRITICAL",
}

# Severity rank for ordering and adjustment
_SEVERITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
_RANK_TO_SEVERITY = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}


def calculate_severity(
    attack_type: str,
    confidence: float,
    iso_score: float,
) -> str:
    """
    Compute severity from attack type, classification confidence, and anomaly.

    Parameters
    ----------
    attack_type : str
        Predicted attack class (e.g. "DDoS", "PortScan", "UNKNOWN_ATTACK").
    confidence : float
        Classification confidence [0.0, 1.0].
        For known attacks: multiclass softmax max.
        For unknown attacks: meta-learner probability.
    iso_score : float
        Normalised Isolation Forest anomaly score [0.0, 1.0].
        Higher = more anomalous. Can escalate severity.

    Returns
    -------
    str
        One of: "LOW", "MEDIUM", "HIGH", "CRITICAL"
    """
    # UNKNOWN_ATTACK is always CRITICAL regardless of confidence
    if attack_type == "UNKNOWN_ATTACK":
        return "CRITICAL"

    # Look up the base severity for this attack type
    base = _ATTACK_BASE_SEVERITY.get(attack_type, "MEDIUM")
    base_rank = _SEVERITY_RANK.get(base, 2)

    # Adjust by confidence:
    #   Low confidence   (< 0.50) → demote by 2 levels
    #   Medium confidence (0.50 – 0.75) → demote by 1 level
    #   High confidence  (0.75 – 0.90) → keep base
    #   Very high conf   (>= 0.90) → keep base (no promotion past base)
    if confidence < 0.50:
        adjusted_rank = max(base_rank - 2, 1)   # floor at LOW
    elif confidence < 0.75:
        adjusted_rank = max(base_rank - 1, 1)   # floor at LOW
    else:
        adjusted_rank = base_rank

    # High anomaly score can escalate by 1 level (but not past CRITICAL)
    if iso_score >= 0.80:
        adjusted_rank = min(adjusted_rank + 1, 4)

    return _RANK_TO_SEVERITY.get(adjusted_rank, "MEDIUM")
