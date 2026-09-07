"""
test_inference.py
Phase 8 Validation Harness

Runs 4 checks before live traffic testing:

  CHECK 1 — Feature count (must be exactly 78)
  CHECK 2 — Feature name alignment (must match multiclass_feature_columns.pkl)
  CHECK 3 — Bidirectional flow merging (A->B + B->A = 1 flow, not 2)
  CHECK 4 — End-to-end smoke test (synthetic flow -> predict_flow -> result)

Usage:
    venv\\Scripts\\python.exe test_inference.py

All 4 checks must print PASS before live capture is enabled.
Any FAIL exits with code 1.
"""

import os
import sys
import time
import joblib

# Ensure project root is on the path regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
WARN = "[WARN]"

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
        _failures += 1


# ===========================================================================
# CHECK 1 — Feature count
# ===========================================================================

def check_feature_count():
    print("\n[CHECK 1] Feature count")
    # pyrefly: ignore [missing-import]
    from features.feature_extractor import extract_features, FEATURE_NAMES
    # pyrefly: ignore [missing-import]
    from flows.flow import NetworkFlow, FlowKey, PacketRecord

    # Build a minimal synthetic flow (3 fwd packets, 2 bwd)
    key  = FlowKey("192.168.1.1", "8.8.8.8", 54321, 443, 6)
    flow = NetworkFlow(key=key, start_time=time.time() - 1.5)

    for i, (ts_offset, direction, length, flags) in enumerate([
        (0.0,  "fwd", 74,  0x02),   # SYN
        (0.1,  "bwd", 74,  0x12),   # SYN-ACK
        (0.2,  "fwd", 66,  0x10),   # ACK
        (0.3,  "fwd", 200, 0x18),   # PSH+ACK (request)
        (0.5,  "bwd", 800, 0x18),   # PSH+ACK (response)
    ]):
        pkt = PacketRecord(
            timestamp   = flow.start_time + ts_offset,
            length      = length,
            direction   = direction,
            tcp_flags   = flags,
            header_len  = 40,
            push_flag   = bool(flags & 0x08),
            urg_flag    = bool(flags & 0x20),
            window_size = 65535,
        )
        flow.add_packet(pkt)

    features = extract_features(flow)
    n_features = len(features)

    check("FEATURE_NAMES list length == 78", len(FEATURE_NAMES) == 78,
          f"got {len(FEATURE_NAMES)}")
    check("extract_features() returns 78 keys", n_features == 78,
          f"got {n_features}")

    # Check no None/NaN values
    import math
    bad = {k: v for k, v in features.items() if v is None or (isinstance(v, float) and math.isnan(v))}
    check("No NaN/None values in features", len(bad) == 0,
          f"bad features: {list(bad.keys())[:5]}")

    return features   # pass to later checks


# ===========================================================================
# CHECK 2 — Feature name alignment with training columns
# ===========================================================================

def check_feature_alignment(features: dict):
    print("\n[CHECK 2] Feature name alignment vs multiclass_feature_columns.pkl")

    models_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "models")
    fc_path    = os.path.join(models_dir, "preprocessing", "multiclass_feature_columns.pkl")

    if not os.path.exists(fc_path):
        print(f"  {WARN}  multiclass_feature_columns.pkl not found at {fc_path} — skipping")
        return

    training_cols = joblib.load(fc_path)
    extracted_set = set(features.keys())
    training_set  = set(training_cols)

    missing_from_extractor = training_set - extracted_set
    extra_in_extractor     = extracted_set - training_set

    check("No features missing from extractor",  len(missing_from_extractor) == 0,
          f"missing: {sorted(missing_from_extractor)}")
    check("No extra features in extractor",       len(extra_in_extractor) == 0,
          f"extra: {sorted(extra_in_extractor)}")

    # Check column ORDER matches training order
    extractor_order = list(features.keys())
    order_ok = all(
        extractor_order[i] == training_cols[i]
        for i in range(min(len(extractor_order), len(training_cols)))
    )
    check("Feature order matches training order", order_ok,
          "first mismatch at index "
          + str(next((i for i in range(min(len(extractor_order), len(training_cols)))
                      if extractor_order[i] != training_cols[i]), "N/A")))

    print(f"  {INFO}  Training columns: {len(training_cols)}  |  "
          f"Extracted: {len(extracted_set)}")


# ===========================================================================
# CHECK 3 — Bidirectional flow merging
# ===========================================================================

def check_bidirectional_merging():
    print("\n[CHECK 3] Bidirectional flow merging (A->B + B->A = 1 flow)")
    # pyrefly: ignore [missing-import]
    from flows.flow import FlowKey, PacketRecord
    
    # pyrefly: ignore [missing-import]
    from flows.flow_manager import FlowManager

    completed_flows = []

    def on_complete(flow):
        completed_flows.append(flow)

    mgr = FlowManager(on_flow_complete=on_complete, idle_timeout=1.0)

    base_ts = time.time()

    # Forward: client → server (SYN)
    fwd_key = FlowKey("10.0.0.1", "10.0.0.2", 12345, 80, 6)
    mgr.process_packet(PacketRecord(
        timestamp=base_ts, length=74, direction="fwd",
        tcp_flags=0x02, header_len=40,
        push_flag=False, urg_flag=False, window_size=65535,
    ), fwd_key)

    # Reverse: server → client (SYN-ACK)
    rev_key = FlowKey("10.0.0.2", "10.0.0.1", 80, 12345, 6)
    mgr.process_packet(PacketRecord(
        timestamp=base_ts + 0.001, length=74, direction="fwd",
        tcp_flags=0x12, header_len=40,
        push_flag=False, urg_flag=False, window_size=65535,
    ), rev_key)

    # RST to close flow
    mgr.process_packet(PacketRecord(
        timestamp=base_ts + 0.002, length=54, direction="fwd",
        tcp_flags=0x04, header_len=40,
        push_flag=False, urg_flag=False, window_size=0,
    ), rev_key)

    check("A->B and B->A merged into 1 flow",
          len(completed_flows) == 1,
          f"got {len(completed_flows)} flow(s)")

    if completed_flows:
        flow = completed_flows[0]
        check("Flow has forward packets",  len(flow.fwd_packets) > 0,
              f"fwd={len(flow.fwd_packets)}")
        check("Flow has backward packets", len(flow.bwd_packets) > 0,
              f"bwd={len(flow.bwd_packets)}")
        print(f"  {INFO}  fwd_pkts={len(flow.fwd_packets)}  "
              f"bwd_pkts={len(flow.bwd_packets)}  "
              f"total_bytes={flow.total_bytes}")


# ===========================================================================
# CHECK 4 — End-to-end smoke test
# ===========================================================================

def check_end_to_end(features: dict):
    print("\n[CHECK 4] End-to-end smoke test (synthetic features -> predict_flow)")

    try:
        # pyrefly: ignore [missing-import]
        from inference.predictor import predict_flow
    except Exception as exc:
        print(f"  {FAIL}  Failed to import inference.predictor: {exc}")
        global _failures
        _failures += 1
        return

    # Predict on synthetic features from Check 1
    try:
        result = predict_flow(features)
    except Exception as exc:
        check("predict_flow() does not raise", False, str(exc))
        return

    # Validate result schema
    required_keys = {
        "is_attack", "attack_probability", "attack_type",
        "attack_confidence", "iso_score", "severity", "latency_ms",
    }
    missing_keys = required_keys - set(result.keys())
    check("Result contains all required keys",   len(missing_keys) == 0,
          f"missing: {missing_keys}")
    check("attack_probability in [0, 1]",
          0.0 <= result.get("attack_probability", -1) <= 1.0)
    check("latency_ms > 0",
          result.get("latency_ms", 0) > 0,
          f"got {result.get('latency_ms')}")
    check("severity is valid string",
          result.get("severity") in {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"},
          f"got '{result.get('severity')}'")

    print(f"\n  {INFO}  Prediction result:")
    for k, v in result.items():
        print(f"         {k:<22} : {v}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 8 VALIDATION HARNESS")
    print("=" * 60)

    features = check_feature_count()
    check_feature_alignment(features)
    check_bidirectional_merging()
    check_end_to_end(features)

    print("\n" + "=" * 60)
    if _failures == 0:
        print(f"{PASS}  ALL CHECKS PASSED — safe to proceed with live capture")
    else:
        print(f"{FAIL}  {_failures} CHECK(S) FAILED — fix before live capture")
        sys.exit(1)
    print("=" * 60)
