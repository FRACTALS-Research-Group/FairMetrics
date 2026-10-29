"""
    Fairness test utilities for evaluating models in fairness experiments.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def evaluate_fairness_metrics(model, X, y, sensitive_features):
    """
        Evaluate basic fairness metrics for a given model and dataset.

        Parameters:
        - model: The model to evaluate, expected to have a predict() method.
        - X: Feature matrix for evaluation.
        - y: True target vector.
        - sensitive_features: Sensitive attributes for fairness evaluation.

        Returns:
        - Dictionary containing basic fairness metrics and sample information.
    """
    y_pred = model.predict(X)
    return {
        'n_samples': len(X),
        'predicted_positive_rate': float(np.mean(y_pred == 1)) if hasattr(y_pred, '__iter__') else 0.0,
        'y_pred_shape': np.asarray(y_pred).shape,
    }


def fairness_test_suite(models, X, y, sensitive_features):
    """
        Run the fairness test loop for multiple model candidates.

        Parameters:
        - models: Dictionary of model names and model instances.
        - X: Feature matrix for evaluation.
        - y: True target vector.
        - sensitive_features: Sensitive attributes for fairness evaluation.

        Returns:
        - List of dictionaries containing model names and their corresponding fairness metrics.
    """
    results = []
    for model_name, model in models.items():
        results.append({
            'model': model_name,
            'metrics': evaluate_fairness_metrics(model, X, y, sensitive_features),
        })
    return results
