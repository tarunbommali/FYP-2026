"""
api/main.py
FastAPI REST + WebSocket interface for the IDS runtime.

Endpoints
---------
    GET  /health        System health check + uptime
    GET  /stats         AlertManager counters snapshot
    GET  /alerts        Recent alerts from SQLite (default: 50)
    GET  /models        Loaded model metadata
    POST /predict       Single-flow inference (JSON body: feature dict)
    WS   /stream        Real-time alert WebSocket feed

Usage
-----
    # Start standalone:
    venv\\Scripts\\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000

    # Or via main.py (API starts automatically):
    venv\\Scripts\\python.exe main.py --interface Ethernet

    # Test:
    curl http://localhost:8000/health
    curl http://localhost:8000/stats
    curl http://localhost:8000/alerts?limit=10
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# FastAPI imports (lazy-checked at import time)
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as e:
    raise ImportError(
        "FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn"
    ) from e

logger = logging.getLogger(__name__)

_alert_mgr = None
_start_time = time.time()
_ws_clients: List[WebSocket] = []
_ws_lock = asyncio.Lock()


def build_app(alert_mgr=None) -> FastAPI:
    global _alert_mgr
    _alert_mgr = alert_mgr

    app = FastAPI(
        title       = "IDS API",
        description = "Real-Time Intrusion Detection System — REST & WebSocket Interface",
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    app.include_router(_router)
    return app


from fastapi import APIRouter
_router = APIRouter()



# ── GET /health ─────────────────────────────────────────────────────────────

@_router.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """
    System health check.

    Returns uptime, model load status, and a simple 'ok' status flag.
    """
    uptime_s = round(time.time() - _start_time, 1)
    return {
        "status":     "ok",
        "uptime_s":   uptime_s,
        "model_ready": _models_loaded(),
        "alert_manager": _alert_mgr is not None,
    }


# ── GET /stats ───────────────────────────────────────────────────────────────

@_router.get("/stats", tags=["Metrics"])
async def stats() -> Dict[str, Any]:
    """
    AlertManager statistics snapshot.

    Returns total flows, alerts, severity breakdown, and alert rate.
    """
    if _alert_mgr is None:
        return {"error": "AlertManager not initialised — start IDS with main.py --api"}
    return _alert_mgr.stats()


# ── GET /alerts ───────────────────────────────────────────────────────────────

@_router.get("/alerts", tags=["Alerts"])
async def alerts(limit: int = Query(default=50, ge=1, le=1000)) -> List[Dict]:
    """
    Recent alerts from SQLite, newest first.

    Parameters
    ----------
    limit : int
        Number of alerts to return (1–1000, default 50).
    """
    if _alert_mgr is None:
        return []
    raw = _alert_mgr.recent_alerts(limit=limit)
    return [_alert_to_dict(a) for a in raw]


# ── GET /models ───────────────────────────────────────────────────────────────

@_router.get("/models", tags=["Models"])
async def models() -> Dict[str, Any]:
    """
    Metadata about loaded ML models.

    Returns feature count, threshold values, and ISO normalisation parameters.
    """
    if not _models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        from inference.model_loader import MODELS
        return {
            "feature_count":     len(MODELS.feature_columns),
            "binary_threshold":  MODELS.binary_threshold,
            "iso_min":           MODELS.iso_min,
            "iso_max":           MODELS.iso_max,
            "feature_medians_n": len(MODELS.feature_medians),
            "models_dir":        os.path.join(BASE_DIR, "models"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /predict ─────────────────────────────────────────────────────────────

@_router.post("/predict", tags=["Inference"])
async def predict(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run single-flow inference.

    Body: JSON object mapping feature name (str) → value (float).
    Must contain all 78 CICIDS2017 feature columns.

    Returns a PredictionResult dict.
    """
    try:
        from inference.predictor import predict_flow
        features = {k: float(v) for k, v in body.items()}
        result   = predict_flow(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Push to WebSocket clients if attack detected
    if result.get("is_attack") and _ws_clients:
        asyncio.create_task(_broadcast(result))

    return result


# ── WS /stream ────────────────────────────────────────────────────────────────

@_router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """
    Real-time alert WebSocket feed.

    Connect with any WebSocket client:
        wscat -c ws://localhost:8000/stream

    Each message is a JSON-encoded PredictionResult for attack flows only.
    A heartbeat {"type":"ping"} is sent every 30 seconds.
    """
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    logger.info("WebSocket client connected | total=%d", len(_ws_clients))

    try:
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except (WebSocketDisconnect, RuntimeError):
                break
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected | total=%d", len(_ws_clients))


async def _broadcast(message: dict) -> None:
    if not _ws_clients:
        return
    payload = json.dumps(message)
    dead: List[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(payload)
        except (WebSocketDisconnect, RuntimeError):
            dead.append(ws)
    if dead:
        async with _ws_lock:
            for ws in dead:
                if ws in _ws_clients:
                    _ws_clients.remove(ws)


def _models_loaded() -> bool:
    try:
        from inference.model_loader import MODELS
        return MODELS is not None
    except (ImportError, FileNotFoundError, AttributeError):
        return False



def _alert_to_dict(alert) -> dict:
    return alert.to_dict() if hasattr(alert, "to_dict") else dict(alert.__dict__)


app = build_app(alert_mgr=None)

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
