"""
Prometheus metrics for monitoring fraud detection model.
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY


# Counters
prediction_counter = Counter(
    "fraud_detection_predictions_total",
    "Total number of predictions made",
    ["prediction_label"],
)

fraud_detected_counter = Counter(
    "fraud_detected_total", "Total number of fraudulent transactions detected"
)

legitimate_counter = Counter(
    "legitimate_transactions_total", "Total number of legitimate transactions"
)

# Gauges
last_prediction_probability = Gauge(
    "last_prediction_probability", "Probability score of the last prediction"
)

last_prediction_label = Gauge(
    "last_prediction_label", "Label of the last prediction (0=legitimate, 1=fraud)"
)

# Drift Gauges (Simulated)
data_drift_score = Gauge(
    "data_drift_score", "Simulated data drift score (0-1)"
)

concept_drift_score = Gauge(
    "concept_drift_score", "Simulated concept drift score (0-1)"
)

# Histogram for latency
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction request",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


class MetricsCollector:
    """Collector for fraud detection metrics."""

    @staticmethod
    def record_prediction(prediction: int, probability: float, latency: float):
        """
        Record a prediction event.

        Args:
            prediction: Prediction label (0 or 1)
            probability: Fraud probability score
            latency: Prediction latency in seconds
        """
        # Update counters
        label = "fraud" if prediction == 1 else "legitimate"
        prediction_counter.labels(prediction_label=label).inc()

        if prediction == 1:
            fraud_detected_counter.inc()
        else:
            legitimate_counter.inc()

        # Update gauges
        last_prediction_probability.set(probability)
        last_prediction_label.set(prediction)

        # Record latency
        prediction_latency.observe(latency)

    @staticmethod
    def record_drift(data_drift: float, concept_drift: float):
        """
        Record simulated drift metrics.

        Args:
            data_drift: Data drift score
            concept_drift: Concept drift score
        """
        data_drift_score.set(data_drift)
        concept_drift_score.set(concept_drift)

    @staticmethod
    def get_metrics() -> bytes:
        """
        Get current metrics in Prometheus format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(REGISTRY)


# Singleton instance
metrics_collector = MetricsCollector()


if __name__ == "__main__":
    # Example usage
    import random

    for i in range(10):
        prediction = random.choice([0, 1])
        probability = random.random()
        latency = random.uniform(0.01, 0.1)

        metrics_collector.record_prediction(prediction, probability, latency)

    print(metrics_collector.get_metrics().decode("utf-8"))
