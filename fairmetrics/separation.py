"""
    Separation-related metrics for the FairMetrics package.
"""
from fairmetrics.metrics import (
    equalized_odds,
    equal_opportunity,
    predictive_equality,
    diagonal_equality,
    off_diagonal_equality,
)

__all__ = [
    "equalized_odds",
    "equal_opportunity",
    "predictive_equality",
    "diagonal_equality",
    "off_diagonal_equality",
]
