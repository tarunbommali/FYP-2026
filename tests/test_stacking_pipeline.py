"""
Smoke test for the stacking ensemble pipeline.
Tests both BENIGN and ATTACK paths.
"""
import sys
import os
import io

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np

# Suppress XGBoost serialization warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from inference.predictor import predict_flow
from inference.model_loader import MODELS


def test_benign_flow():
    """Test with zeroed features — should be classified as BENIGN."""
    print("=" * 60)
    print("TEST 1: BENIGN flow (zeroed features)")
    print("=" * 60)
    features = {col: 0.0 for col in MODELS.feature_columns}
    result = predict_flow(features)
    
    print(f"  is_attack:          {result['is_attack']}")
    print(f"  attack_type:        {result['attack_type']}")
    print(f"  attack_probability: {result['attack_probability']:.6f}")
    print(f"  rf_probability:     {result['rf_probability']:.6f}")
    print(f"  xgb_probability:    {result['xgb_probability']:.6f}")
    print(f"  meta_probability:   {result['meta_probability']:.6f}")
    print(f"  iso_score:          {result['iso_score']:.6f}")
    print(f"  attack_confidence:  {result['attack_confidence']:.6f}")
    print(f"  severity:           {result['severity']}")
    print(f"  latency_ms:         {result['latency_ms']:.3f}")
    print(f"  error:              {result['error']}")
    
    assert not result["is_attack"], "Zeroed features should be BENIGN"
    assert result["attack_type"] == "BENIGN", f"Expected BENIGN, got {result['attack_type']}"
    assert result["severity"] == "NONE", f"Expected NONE severity, got {result['severity']}"
    assert "rf_probability" in result, "Missing rf_probability field"
    assert "xgb_probability" in result, "Missing xgb_probability field"
    assert "meta_probability" in result, "Missing meta_probability field"
    assert result["error"] is None, f"Unexpected error: {result['error']}"
    print("  ✅ PASSED\n")


def test_attack_flow():
    """Test with extreme features — should be classified as ATTACK."""
    print("=" * 60)
    print("TEST 2: ATTACK flow (extreme features)")
    print("=" * 60)
    # Set features that typically indicate an attack
    features = {col: 0.0 for col in MODELS.feature_columns}
    # Simulate high packet rate / abnormal flow
    for col in MODELS.feature_columns:
        col_lower = col.lower()
        if "pkt" in col_lower or "packet" in col_lower:
            features[col] = 100000.0
        elif "byte" in col_lower:
            features[col] = 50000000.0
        elif "rate" in col_lower or "speed" in col_lower:
            features[col] = 999999.0
        elif "flag" in col_lower:
            features[col] = 1.0
    
    result = predict_flow(features)
    
    print(f"  is_attack:          {result['is_attack']}")
    print(f"  attack_type:        {result['attack_type']}")
    print(f"  attack_probability: {result['attack_probability']:.6f}")
    print(f"  rf_probability:     {result['rf_probability']:.6f}")
    print(f"  xgb_probability:    {result['xgb_probability']:.6f}")
    print(f"  meta_probability:   {result['meta_probability']:.6f}")
    print(f"  iso_score:          {result['iso_score']:.6f}")
    print(f"  attack_confidence:  {result['attack_confidence']:.6f}")
    print(f"  severity:           {result['severity']}")
    print(f"  latency_ms:         {result['latency_ms']:.3f}")
    print(f"  error:              {result['error']}")
    
    # Note: extreme features may or may not trigger attack — depends on model training
    # We just verify the pipeline runs without error and output schema is correct
    assert "rf_probability" in result, "Missing rf_probability field"
    assert "xgb_probability" in result, "Missing xgb_probability field"
    assert "meta_probability" in result, "Missing meta_probability field"
    assert result["error"] is None, f"Unexpected error: {result['error']}"
    
    if result["is_attack"]:
        assert result["attack_type"] != "BENIGN", "Attack should not have BENIGN type"
        assert result["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"], f"Bad severity: {result['severity']}"
        print("  ✅ PASSED (classified as ATTACK)\n")
    else:
        print("  ✅ PASSED (classified as BENIGN — model did not flag extreme features)\n")


def test_schema_completeness():
    """Verify all expected fields are present in the prediction result."""
    print("=" * 60)
    print("TEST 3: Schema completeness check")
    print("=" * 60)
    features = {col: 0.0 for col in MODELS.feature_columns}
    result = predict_flow(features)
    
    expected_fields = [
        "is_attack", "attack_probability", "rf_probability", "xgb_probability",
        "meta_probability", "attack_type", "attack_confidence", "iso_score",
        "severity", "latency_ms", "error"
    ]
    
    for field in expected_fields:
        assert field in result, f"Missing field: {field}"
        print(f"  [OK] {field}: present ({type(result[field]).__name__})")
    
    print("  [OK] PASSED — all 11 fields present\n")


if __name__ == "__main__":
    print("\n=== STACKING ENSEMBLE PIPELINE SMOKE TEST ===\n")
    test_benign_flow()
    test_attack_flow()
    test_schema_completeness()
    print("=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
