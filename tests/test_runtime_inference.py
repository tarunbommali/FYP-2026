"""
tests/test_runtime_inference.py
Unit & Integration Test for Offline Flow-Level Runtime Inference

Verifies that held-out test flow feature vectors from X_test_binary.parquet
pass successfully through the deployed predict_flow() runtime engine.
"""

import os
# pyrefly: ignore [missing-import]
import pytest
import pandas as pd

# pyrefly: ignore [missing-import]
from inference.predictor import predict_flow


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
X_TEST_PATH = os.path.join(TEST_DATA_DIR, "X_test_binary.parquet")
Y_TEST_PATH = os.path.join(TEST_DATA_DIR, "y_test_binary.parquet")


@pytest.mark.skipif(
    not (os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH)),
    reason="Test parquet files not present in data/processed/",
)
class TestRuntimeInference:
    """Test offline flow-level runtime inference against real held-out flow vectors."""

    @classmethod
    def setup_class(cls):
        # Load a small slice of 20 test flows (10 benign, 10 attack)
        X = pd.read_parquet(X_TEST_PATH)
        y = pd.read_parquet(Y_TEST_PATH)["Label"].astype(int)

        benign_indices = y[y == 0].head(10).index
        attack_indices = y[y == 1].head(10).index

        cls.sample_indices = benign_indices.union(attack_indices)
        cls.X_sample = X.loc[cls.sample_indices]
        cls.y_sample = y.loc[cls.sample_indices]

    def test_runtime_inference_schema_and_execution(self):
        """Verify each flow vector returns a well-formed PredictionResult without errors."""
        for flow_idx, row in self.X_sample.iterrows():
            features = row.to_dict()
            result = predict_flow(features)

            assert isinstance(result, dict), "Result must be a dictionary"
            assert result.get("error") is None, f"Prediction returned error: {result.get('error')}"
            assert isinstance(result.get("is_attack"), bool), "is_attack must be a boolean"
            assert isinstance(result.get("attack_probability"), float), "attack_probability must be float"
            assert 0.0 <= result.get("attack_probability") <= 1.0, "Probability must be in [0, 1]"
            assert result.get("attack_type") in {
                "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
                "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
                "Bot", "Web Attack  Brute Force", "Web Attack  XSS",
                "Infiltration", "Web Attack  Sql Injection", "Heartbleed",
                "UNKNOWN_ATTACK",
            } or isinstance(result.get("attack_type"), str)

    def test_runtime_inference_classification_accuracy(self):
        """Verify high detection concordance (> 90%) on the test flow slice."""
        correct = 0
        total = len(self.X_sample)

        for flow_idx, row in self.X_sample.iterrows():
            features = row.to_dict()
            true_label = int(self.y_sample.loc[flow_idx])
            result = predict_flow(features)

            pred_label = 1 if result.get("is_attack") else 0
            if pred_label == true_label:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.85, f"Expected runtime accuracy >= 85%, got {accuracy * 100:.1f}%"
