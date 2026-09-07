"""
monitoring/metrics_registry.py
Contains all Prometheus metric definitions (Counters, Histograms, Gauges).
Decoupled from the update logic so they can be imported anywhere.
"""

# pyrefly: ignore [missing-import]
from prometheus_client import Counter, Histogram, Gauge

# -- Flow / Packet counters --------------------------------------------------
packets_processed_total = Counter(
    "ids_packets_processed_total",
    "Total IP packets processed by the capture layer",
)

flows_completed_total = Counter(
    "ids_flows_completed_total",
    "Total flows exported from the flow manager",
)

# MCA/demo-friendly aliases. Kept alongside the original metric names so the
# existing dashboard continues to work while the project report can use the
# cleaner names from the final architecture.
packets_total = Counter(
    "ids_packets_total",
    "Total IP packets processed by the capture layer",
)

flows_total = Counter(
    "ids_flows_total",
    "Total flows exported from the flow manager",
)

packets_per_second = Gauge(
    "ids_packets_per_second",
    "Current packet processing rate calculated by the IDS exporter",
)

flows_per_second = Gauge(
    "ids_flows_per_second",
    "Current completed-flow rate calculated by the IDS exporter",
)

# -- Prediction counters -----------------------------------------------------
predictions_total = Counter(
    "ids_predictions_total",
    "Total predictions made by the inference engine",
)

attacks_detected_total = Counter(
    "ids_attacks_detected_total",
    "Total flows classified as attacks",
)

benign_detected_total = Counter(
    "ids_benign_detected_total",
    "Total flows classified as benign",
)

attack_total = Counter(
    "ids_attack_total",
    "Total flows classified as attacks",
)

benign_total = Counter(
    "ids_benign_total",
    "Total flows classified as benign",
)

attack_rate = Gauge(
    "ids_attack_rate",
    "Ratio of attack predictions to total predictions",
)

# -- Attack type counters ----------------------------------------------------
attack_type_total = Counter(
    "ids_attack_type_total",
    "Predictions per attack type",
    ["type"],
)

attacks_by_type = Counter(
    "ids_attacks_by_type",
    "Attack predictions grouped by attack type",
    ["type"],
)

anomalies_total = Counter(
    "ids_anomalies_total",
    "Total flows with a positive Isolation Forest anomaly score",
)

high_confidence_attacks = Counter(
    "ids_high_confidence_attacks",
    "Total attacks with high model confidence",
)

# -- Top talkers / protocol counters ----------------------------------------
src_ip_flows = Counter(
    "ids_src_ip_flows",
    "Completed flows grouped by source IP",
    ["src_ip"],
)

dst_ip_flows = Counter(
    "ids_dst_ip_flows",
    "Completed flows grouped by destination IP",
    ["dst_ip"],
)

protocol_flows = Counter(
    "ids_protocol_flows",
    "Completed flows grouped by IP protocol number",
    ["protocol"],
)

# -- Latest model score gauges ----------------------------------------------
model_confidence = Gauge(
    "ids_model_confidence",
    "Latest model confidence value emitted by the inference engine",
)

rf_probability = Gauge(
    "ids_rf_probability",
    "Latest Random Forest probability if supplied by the runtime predictor",
)

xgb_probability = Gauge(
    "ids_xgb_probability",
    "Latest XGBoost binary attack probability",
)

iso_score = Gauge(
    "ids_iso_score",
    "Latest normalised Isolation Forest anomaly score",
)

meta_probability = Gauge(
    "ids_meta_probability",
    "Latest Logistic Regression meta-learner attack probability",
)

# -- Latency histogram -------------------------------------------------------
inference_latency_ms = Histogram(
    "ids_inference_latency_ms",
    "End-to-end predict_flow() latency in milliseconds",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
)

feature_extraction_latency_ms = Histogram(
    "ids_feature_extraction_latency_ms",
    "Feature extraction latency in milliseconds",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
)

processing_time_ms = Histogram(
    "ids_processing_time_ms",
    "Feature extraction plus inference processing time in milliseconds",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
)

# -- Alert counters ----------------------------------------------------------
alerts_total = Counter(
    "ids_alerts_total",
    "Total alerts stored in SQLite",
)

alerts_deduped_total = Counter(
    "ids_alerts_deduped_total",
    "Total alerts suppressed by deduplication",
)

alerts_severity_total = Counter(
    "ids_alerts_severity_total",
    "Alerts per severity level",
    ["severity"],
)

# -- Active state gauges -----------------------------------------------------
active_flows = Gauge(
    "ids_active_flows",
    "Number of flows currently in the flow table",
)

# -- Alert rate gauge --------------------------------------------------------
alert_rate_gauge = Gauge(
    "ids_alert_rate",
    "Ratio of alerts generated to total flows processed (rolling)",
)

# -- System resource gauges --------------------------------------------------
cpu_usage_percent = Gauge(
    "ids_cpu_usage_percent",
    "System CPU usage percentage (sampled every export interval)",
)

memory_usage_percent = Gauge(
    "ids_memory_usage_percent",
    "System memory usage percentage (sampled every export interval)",
)


# ---------------------------------------------------------------------------
# System Status Metrics
# ---------------------------------------------------------------------------

system_status = Gauge(
    "ids_system_status",
    "IDS running status (1=running, 0=stopped)",
)

model_loaded = Gauge(
    "ids_model_loaded",
    "Model loading status (1=loaded, 0=failed)",
)

capture_running = Gauge(
    "ids_capture_running",
    "Packet capture status (1=running, 0=stopped)",
)

uptime_seconds = Gauge(
    "ids_uptime_seconds",
    "IDS uptime in seconds",
)

# ---------------------------------------------------------------------------
# Pipeline Monitoring
# ---------------------------------------------------------------------------

features_extracted_total = Counter(
    "ids_features_extracted_total",
    "Total feature vectors extracted",
)

# ---------------------------------------------------------------------------
# Multiclass Prediction
# ---------------------------------------------------------------------------

last_attack_type = Gauge(
    "ids_last_attack_type_info",
    "Latest detected attack type",
    ["type"],
)

attack_confidence = Gauge(
    "ids_attack_confidence",
    "Confidence of latest multiclass attack prediction",
)

# ---------------------------------------------------------------------------
# Model Status
# ---------------------------------------------------------------------------

binary_model_loaded = Gauge(
    "ids_binary_model_loaded",
    "Binary XGBoost model loaded",
)

multiclass_model_loaded = Gauge(
    "ids_multiclass_model_loaded",
    "Multiclass XGBoost model loaded",
)

isolation_model_loaded = Gauge(
    "ids_isolation_model_loaded",
    "Isolation Forest model loaded",
)

rf_model_loaded = Gauge(
    "ids_rf_model_loaded",
    "Random Forest binary model loaded",
)

meta_learner_loaded = Gauge(
    "ids_meta_learner_loaded",
    "Logistic Regression meta-learner model loaded",
)

# ---------------------------------------------------------------------------
# Offline Model Evaluation Metrics (CICIDS2017 Test Set Verification)
# ---------------------------------------------------------------------------

binary_xgb_f1 = Gauge(
    "ids_binary_xgb_f1",
    "Binary XGBoost model F1-Score on test set",
)
binary_xgb_f1.set(0.9974)

binary_xgb_roc_auc = Gauge(
    "ids_binary_xgb_roc_auc",
    "Binary XGBoost model ROC-AUC on test set",
)
binary_xgb_roc_auc.set(0.99997)

multiclass_xgb_accuracy = Gauge(
    "ids_multiclass_xgb_accuracy",
    "Multiclass XGBoost model Accuracy percentage on test set",
)
multiclass_xgb_accuracy.set(99.84)

multiclass_xgb_weighted_f1 = Gauge(
    "ids_multiclass_xgb_weighted_f1",
    "Multiclass XGBoost model Weighted F1-Score on test set",
)
multiclass_xgb_weighted_f1.set(0.9985)

multiclass_xgb_macro_f1 = Gauge(
    "ids_multiclass_xgb_macro_f1",
    "Multiclass XGBoost model Macro F1-Score on test set",
)
multiclass_xgb_macro_f1.set(0.8507)


# ---------------------------------------------------------------------------
# Operational Errors / Failure Monitoring
# ---------------------------------------------------------------------------

prediction_errors_total = Counter(
    "ids_prediction_errors_total",
    "Total prediction failures/errors in the inference engine",
)

alerts_failed_total = Counter(
    "ids_alerts_failed_total",
    "Total database/SQLite insert failures in AlertManager",
)

unknown_attacks_total = Counter(
    "ids_unknown_attacks_total",
    "Total flows classified as UNKNOWN_ATTACK (high anomaly, low multiclass confidence)",
)
