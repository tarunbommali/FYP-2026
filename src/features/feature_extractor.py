"""
features/feature_extractor.py
Converts a completed NetworkFlow into the 78 CICIDS2017 NetFlow features
expected by the inference engine.

Feature names exactly match those used during model training.
Any deviation in name or ordering will cause silent prediction errors.

Reference: CICIDS2017 dataset feature set
          (Canadian Institute for Cybersecurity)
"""

import math
import statistics
import logging
from typing import Dict, List

from flows.flow import NetworkFlow, PacketRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name constants — must exactly match training column names
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(flow: NetworkFlow) -> Dict[str, float]:
    """
    Convert a completed NetworkFlow into a CICIDS2017-compatible feature dict.

    Parameters
    ----------
    flow : NetworkFlow
        A completed or timed-out flow from FlowManager.

    Returns
    -------
    dict[str, float]
        78 named features ready for predict_flow().
        All values are Python floats. Missing/undefined features default to 0.0.
    """
    fwd = flow.fwd_packets
    bwd = flow.bwd_packets
    all_pkts = fwd + bwd

    duration_s  = max(flow.duration_us / 1_000_000, 1e-9)   # seconds, avoid /0
    duration_us = flow.duration_us

    fwd_lengths = [p.length for p in fwd]
    bwd_lengths = [p.length for p in bwd]
    all_lengths = fwd_lengths + bwd_lengths

    fwd_ts = [p.timestamp for p in fwd]
    bwd_ts = [p.timestamp for p in bwd]
    all_ts = sorted(p.timestamp for p in all_pkts)

    # TCP flag counts across all packets
    tcp_flags_all = [p.tcp_flags for p in all_pkts]
    fin_count = sum(1 for f in tcp_flags_all if f & 0x01)
    syn_count = sum(1 for f in tcp_flags_all if f & 0x02)
    rst_count = sum(1 for f in tcp_flags_all if f & 0x04)
    psh_count = sum(1 for f in tcp_flags_all if f & 0x08)
    ack_count = sum(1 for f in tcp_flags_all if f & 0x10)
    urg_count = sum(1 for f in tcp_flags_all if f & 0x20)
    cwe_count = sum(1 for f in tcp_flags_all if f & 0x40)
    ece_count = sum(1 for f in tcp_flags_all if f & 0x80)

    fwd_psh = sum(1 for p in fwd if p.push_flag)
    bwd_psh = sum(1 for p in bwd if p.push_flag)
    fwd_urg = sum(1 for p in fwd if p.urg_flag)
    bwd_urg = sum(1 for p in bwd if p.urg_flag)

    features: Dict[str, float] = {
        "Destination Port":           float(flow.key.dst_port),
        "Flow Duration":              float(duration_us),

        # Packet counts
        "Total Fwd Packets":          float(len(fwd)),
        "Total Backward Packets":     float(len(bwd)),

        # Byte counts
        "Total Length of Fwd Packets": float(sum(fwd_lengths)),
        "Total Length of Bwd Packets": float(sum(bwd_lengths)),

        # Fwd packet length stats
        "Fwd Packet Length Max":  float(max(fwd_lengths, default=0)),
        "Fwd Packet Length Min":  float(min(fwd_lengths, default=0)),
        "Fwd Packet Length Mean": _mean(fwd_lengths),
        "Fwd Packet Length Std":  _std(fwd_lengths),

        # Bwd packet length stats
        "Bwd Packet Length Max":  float(max(bwd_lengths, default=0)),
        "Bwd Packet Length Min":  float(min(bwd_lengths, default=0)),
        "Bwd Packet Length Mean": _mean(bwd_lengths),
        "Bwd Packet Length Std":  _std(bwd_lengths),

        # Flow rates
        "Flow Bytes/s":   float(sum(all_lengths)) / duration_s,
        "Flow Packets/s": float(len(all_pkts))    / duration_s,

        # Flow inter-arrival times (all packets, sorted)
        **_iat_features("Flow IAT", all_ts),

        # Forward IAT
        **_iat_features("Fwd IAT", fwd_ts, total_key="Fwd IAT Total"),

        # Backward IAT
        **_iat_features("Bwd IAT", bwd_ts, total_key="Bwd IAT Total"),

        # TCP flags
        "Fwd PSH Flags": float(fwd_psh),
        "Bwd PSH Flags": float(bwd_psh),
        "Fwd URG Flags": float(fwd_urg),
        "Bwd URG Flags": float(bwd_urg),

        # Header lengths
        "Fwd Header Length":   float(sum(p.header_len for p in fwd)),
        "Bwd Header Length":   float(sum(p.header_len for p in bwd)),
        "Fwd Header Length.1": float(sum(p.header_len for p in fwd)),   # duplicate col in CICIDS2017

        # Per-second rates
        "Fwd Packets/s": float(len(fwd)) / duration_s,
        "Bwd Packets/s": float(len(bwd)) / duration_s,

        # Global packet length stats
        "Min Packet Length":      float(min(all_lengths, default=0)),
        "Max Packet Length":      float(max(all_lengths, default=0)),
        "Packet Length Mean":     _mean(all_lengths),
        "Packet Length Std":      _std(all_lengths),
        "Packet Length Variance": _var(all_lengths),

        # TCP flag counts
        "FIN Flag Count": float(fin_count),
        "SYN Flag Count": float(syn_count),
        "RST Flag Count": float(rst_count),
        "PSH Flag Count": float(psh_count),
        "ACK Flag Count": float(ack_count),
        "URG Flag Count": float(urg_count),
        "CWE Flag Count": float(cwe_count),
        "ECE Flag Count": float(ece_count),

        # Ratios
        "Down/Up Ratio":       float(len(bwd)) / max(len(fwd), 1),
        "Average Packet Size": _mean(all_lengths),

        # Segment sizes (CICIDS uses payload length here)
        "Avg Fwd Segment Size": _mean(fwd_lengths),
        "Avg Bwd Segment Size": _mean(bwd_lengths),

        # Bulk features — CICIDS2017 treats these as 0 for most flows
        "Fwd Avg Bytes/Bulk":    0.0,
        "Fwd Avg Packets/Bulk":  0.0,
        "Fwd Avg Bulk Rate":     0.0,
        "Bwd Avg Bytes/Bulk":    0.0,
        "Bwd Avg Packets/Bulk":  0.0,
        "Bwd Avg Bulk Rate":     0.0,

        # Subflow features (CICFlowMeter treats each flow as one subflow)
        "Subflow Fwd Packets": float(len(fwd)),
        "Subflow Fwd Bytes":   float(sum(fwd_lengths)),
        "Subflow Bwd Packets": float(len(bwd)),
        "Subflow Bwd Bytes":   float(sum(bwd_lengths)),

        # TCP window sizes (first packet)
        "Init_Win_bytes_forward":  float(flow.init_win_fwd),
        "Init_Win_bytes_backward": float(flow.init_win_bwd),

        # Active data packets fwd (packets with payload)
        "act_data_pkt_fwd":      float(sum(1 for p in fwd if p.length > p.header_len)),
        "min_seg_size_forward":  float(min((p.header_len for p in fwd), default=0)),

        # Active/Idle times — require activity detection; use 0.0 for single-subflow
        "Active Mean": 0.0, "Active Std": 0.0,
        "Active Max":  0.0, "Active Min":  0.0,
        "Idle Mean":   0.0, "Idle Std":   0.0,
        "Idle Max":    0.0, "Idle Min":   0.0,
    }

    # Return in exact training column order — do not reorder
    from monitoring import metrics_registry as reg
    reg.features_extracted_total.inc()
    return {name: features.get(name, 0.0) for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _var(values: List[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def _iat_features(prefix: str, timestamps: List[float], total_key: str = "") -> Dict[str, float]:
    """Compute inter-arrival time statistics from a sorted timestamp list."""
    if len(timestamps) < 2:
        iats = [0.0]
    else:
        sorted_ts = sorted(timestamps)
        iats = [(sorted_ts[i+1] - sorted_ts[i]) * 1_000_000   # convert to microseconds
                for i in range(len(sorted_ts) - 1)]

    result = {
        f"{prefix} Mean": _mean(iats),
        f"{prefix} Std":  _std(iats),
        f"{prefix} Max":  float(max(iats, default=0.0)),
        f"{prefix} Min":  float(min(iats, default=0.0)),
    }
    if total_key:
        result[total_key] = float(sum(iats))
    return result
