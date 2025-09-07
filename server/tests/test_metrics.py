import sys
import os
import json

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.metrics import compute_all_metrics

def test_metrics_computation_and_json_serialization():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95]

    metrics = compute_all_metrics(y_true, y_prob)
    assert 0.9 <= metrics["roc_auc"] <= 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[4, 0], [0, 4]]

    # Ensure JSON serialization succeeds without TypeError on NumPy floats
    serialized = json.dumps(metrics)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["roc_auc"] == metrics["roc_auc"]

def test_single_class_metrics_safety():
    y_true = [1, 1, 1, 1]
    y_prob = [0.8, 0.9, 0.85, 0.95]

    # Should not raise ValueError even though only one class is present
    metrics = compute_all_metrics(y_true, y_prob)
    assert "roc_auc" in metrics
    serialized = json.dumps(metrics)
    assert isinstance(serialized, str)
