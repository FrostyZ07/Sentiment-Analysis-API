"""Prometheus metrics — isolated module to avoid circular imports."""
try:
    from prometheus_client import Counter, Histogram

    sentiment_counter = Counter(
        "sentiment_prediction_total",
        "Total predictions by sentiment label",
        ["label", "model_version"],
    )
    confidence_histogram = Histogram(
        "sentiment_confidence",
        "Distribution of prediction confidence scores",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    )
    inference_duration = Histogram(
        "inference_duration_seconds",
        "Model inference duration in seconds",
        buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    sentiment_counter = None
    confidence_histogram = None
    inference_duration = None
