"""
inference/__init__.py
Exposes the top-level predict_flow() function for external consumers
(API server, packet processor, test harness).
"""

# pyrefly: ignore [missing-import]
from inference.predictor import predict_flow

__all__ = ["predict_flow"]
