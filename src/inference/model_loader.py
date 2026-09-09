"""
inference/model_loader.py
Singleton model manager — loads all artifacts once at startup.

Models loaded
-------------
  rf_binary.pkl                  Random Forest binary classifier (stacking base learner)
  xgb_binary.pkl                 Binary XGBoost classifier (stacking base learner)
  isolation_forest.pkl           Isolation Forest anomaly detector (stacking base learner)
  meta_learner.pkl               Logistic Regression meta-learner (stacking combiner)
  xgb_multiclass.pkl             15-class XGBoost classifier (attack categorisation)
  label_encoder.pkl              LabelEncoder (int -> class name)
  scaler.pkl                     StandardScaler for ISO Forest input
  binary_threshold.json          Per-model optimal thresholds + ISO norm params
  multiclass_feature_columns.pkl Ordered feature list (78 features)
  feature_medians.pkl            Training-set medians for NaN imputation

Stacking Ensemble Architecture
------------------------------
  Features → RF → XGBoost Binary → ISO Forest → Meta-Learner (LR)
    ├─ BENIGN → return
    └─ ATTACK → Multiclass XGBoost → Unknown Attack Check → Severity → Result

The module-level MODELS singleton is imported by predictor.py.
Never reload models inside prediction hot-paths.
"""

import os
import json
import joblib
import logging

from monitoring import metrics_registry as reg

logger = logging.getLogger(__name__)

# Project root is three levels above this file:
#   src/inference/model_loader.py -> inference/ -> src/ -> project root
_THIS_FILE  = os.path.abspath(__file__)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Sub-folder layout produced by the training pipeline
_DIR_BINARY      = os.path.join(MODELS_DIR, "binary")
_DIR_MULTICLASS  = os.path.join(MODELS_DIR, "multiclass")
_DIR_PREPROC     = os.path.join(MODELS_DIR, "preprocessing")
_DIR_ANOMALY     = os.path.join(MODELS_DIR, "anomaly")
_DIR_STACKING    = os.path.join(MODELS_DIR, "stacking")

# Safe production default — Phase 3 threshold (0.8624) was dataset-specific.
# Override by setting "binary_threshold" in config.json.
DEFAULT_BINARY_THRESHOLD: float = 0.50

# Meta-learner decision boundary (standard LR threshold)
DEFAULT_META_LEARNER_THRESHOLD: float = 0.50

# Unknown attack detection — ISO score only (novelty detection)
DEFAULT_UNKNOWN_ATTACK_ISO_THRESHOLD: float = 0.70

EXPECTED_FEATURE_COUNT: int = 78


class ModelManager:
    """
    Loads and validates all inference artifacts at construction time.
    Raises FileNotFoundError immediately if any required artifact is missing.

    Stacking Ensemble Models
    ------------------------
    Base learners (produce the 3-feature meta-vector):
      - rf_binary        : Random Forest binary classifier
      - xgb_binary       : XGBoost binary classifier
      - iso_model        : Isolation Forest anomaly detector

    Meta-learner (combines base learner outputs):
      - meta_learner     : Logistic Regression on [rf_prob, xgb_prob, iso_score]

    Attack categorisation (runs only when meta-learner says ATTACK):
      - xgb_multiclass   : 15-class XGBoost classifier
    """

    # Map of artifact filename -> sub-directory
    _REQUIRED_FILES = {
        "rf_binary.pkl":                   _DIR_BINARY,
        "xgb_binary.pkl":                  _DIR_BINARY,
        "xgb_multiclass.pkl":              _DIR_MULTICLASS,
        "label_encoder.pkl":               _DIR_MULTICLASS,
        "isolation_forest.pkl":            _DIR_ANOMALY,
        "meta_learner.pkl":                _DIR_STACKING,
        "scaler.pkl":                      _DIR_PREPROC,
        "binary_threshold.json":           _DIR_BINARY,
        "multiclass_feature_columns.pkl":  _DIR_PREPROC,
        "feature_medians.pkl":             _DIR_PREPROC,
    }



    def __init__(self) -> None:
        try:
            self._validate_artifacts()

            logger.info("Loading inference models from: %s", MODELS_DIR)

            # -- Base learners (stacking) --
            self.rf_binary      = joblib.load(_path("rf_binary.pkl"))
            self.xgb_binary     = joblib.load(_path("xgb_binary.pkl"))
            self.iso_model      = joblib.load(_path("isolation_forest.pkl"))
            self.iso_scaler     = joblib.load(_path("scaler.pkl"))

            # -- Meta-learner (stacking combiner) --
            self.meta_learner   = joblib.load(_path("meta_learner.pkl"))
            if not hasattr(self.meta_learner, "multi_class"):
                self.meta_learner.multi_class = "auto"

            # -- Attack categorisation --
            self.xgb_multiclass = joblib.load(_path("xgb_multiclass.pkl"))
            self.label_encoder  = joblib.load(_path("label_encoder.pkl"))

            with open(_path("binary_threshold.json"), "r") as f:
                thresh = json.load(f)

            # Binary threshold: prefer config.json override, then fall back to safe default.
            # Do NOT blindly use Phase 3's threshold (0.8624) in production —
            # it was optimised on the evaluation set and may be too aggressive on live traffic.
            config_threshold = self._load_config_threshold()
            self.binary_threshold: float = config_threshold if config_threshold is not None \
                else DEFAULT_BINARY_THRESHOLD

            logger.info(
                "Binary threshold: %.4f (%s)",
                self.binary_threshold,
                "config.json" if config_threshold is not None else "default",
            )

            # Meta-learner threshold from config
            self.meta_learner_threshold: float = self._load_config_float(
                "meta_learner_threshold", DEFAULT_META_LEARNER_THRESHOLD
            )
            logger.info("Meta-learner threshold: %.4f", self.meta_learner_threshold)

            # Unknown attack detection — ISO score only
            self.unknown_attack_iso_threshold: float = self._load_config_float(
                "unknown_attack_iso_threshold", DEFAULT_UNKNOWN_ATTACK_ISO_THRESHOLD
            )
            logger.info(
                "Unknown attack ISO threshold: %.2f",
                self.unknown_attack_iso_threshold,
            )

            # ISO normalisation params — must match Phase 3 training values
            self.iso_min: float = thresh["iso"]["iso_norm"]["iso_min"]
            self.iso_max: float = thresh["iso"]["iso_norm"]["iso_max"]

            if self.iso_max <= self.iso_min:
                raise ValueError(
                    f"Invalid ISO normalisation parameters: "
                    f"iso_min={self.iso_min:.4f} iso_max={self.iso_max:.4f}. "
                    "Re-run 03_threshold_optimization.py to regenerate binary_threshold.json."
                )

            # Feature list — defines column order expected by all models
            self.feature_columns: list = joblib.load(
                _path("multiclass_feature_columns.pkl")
            )

            if len(self.feature_columns) != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"Expected {EXPECTED_FEATURE_COUNT} features, "
                    f"got {len(self.feature_columns)}. "
                    "Regenerate multiclass_feature_columns.pkl from 05_multiclass_training.py."
                )

            # Feature medians — required for training-consistent NaN imputation.
            self.feature_medians: dict = joblib.load(_path("feature_medians.pkl"))
            logger.info("Loaded feature_medians.pkl (%d features)", len(self.feature_medians))

            # Set model loaded metrics
            reg.model_loaded.set(1)
            reg.binary_model_loaded.set(1)
            reg.multiclass_model_loaded.set(1)
            reg.isolation_model_loaded.set(1)
            reg.rf_model_loaded.set(1)
            reg.meta_learner_loaded.set(1)

            logger.info(
                "ModelManager ready | features=%d | binary_threshold=%.4f | "
                "meta_learner_threshold=%.4f | models=RF+XGB+ISO+LR+Multiclass",
                len(self.feature_columns),
                self.binary_threshold,
                self.meta_learner_threshold,
            )
        except Exception:
            reg.model_loaded.set(0)
            reg.binary_model_loaded.set(0)
            reg.multiclass_model_loaded.set(0)
            reg.isolation_model_loaded.set(0)
            reg.rf_model_loaded.set(0)
            reg.meta_learner_loaded.set(0)
            raise

    def _validate_artifacts(self) -> None:
        """Validate all required model artifacts exist in their sub-directories."""
        missing = [
            os.path.join(subdir, fname)
            for fname, subdir in self._REQUIRED_FILES.items()
            if not os.path.exists(os.path.join(subdir, fname))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts:\n"
                + "\n".join(f"  - {f}" for f in missing)
            )

    @staticmethod
    def _load_config_threshold() -> "float | None":
        if not os.path.exists(CONFIG_PATH):
            return None
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {CONFIG_PATH}: {exc}") from exc
        except OSError as exc:
            raise OSError(f"Could not read config file {CONFIG_PATH}: {exc}") from exc

        val = cfg.get("binary_threshold")
        if val is not None:
            try:
                val = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"binary_threshold must be a float, got {val}") from exc
            if not (0.0 < val < 1.0):
                raise ValueError(f"binary_threshold must be in (0, 1), got {val}")
            return val
        return None

    @staticmethod
    def _load_config_float(key: str, default: float) -> float:
        if not os.path.exists(CONFIG_PATH):
            return default
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {CONFIG_PATH}: {exc}") from exc
        except OSError as exc:
            raise OSError(f"Could not read config file {CONFIG_PATH}: {exc}") from exc

        val = cfg.get(key)
        if val is not None:
            try:
                val = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key} must be a float, got {val}") from exc
            if not (0.0 < val < 1.0):
                raise ValueError(f"{key} must be in (0, 1), got {val}")
            return val
        return default



def _path(filename: str) -> str:
    subdir = ModelManager._REQUIRED_FILES.get(filename, MODELS_DIR)
    return os.path.join(subdir, filename)


MODELS = ModelManager()
