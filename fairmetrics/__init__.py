"""
    FairMetrics package initialization.
    This module exposes the main functions and classes from the fairmetrics package.
"""
from fairmetrics.metrics import (
    conditional_demographic_parity,
    demographic_parity,
    equal_accuracy,
    equal_opportunity,
    equalized_odds,
    predictive_equality,
    binary_predictive_parity,
    diagonal_equality,
    off_diagonal_equality,
    multiclass_predictive_parity,
)

from fairmetrics.tests import (
    find_distribution,
    load_adult,
    load_students_success,
)

__all__ = [
    "demographic_parity",
    "conditional_demographic_parity",
    "equal_opportunity",
    "diagonal_equality",
    "predictive_equality",
    "off_diagonal_equality",
    "equalized_odds",
    "binary_predictive_parity",
    "multiclass_predictive_parity",
    "equal_accuracy",
    "find_distribution",
    "load_adult",
    "load_students_success",
    "hyperparams_tuning",
]