"""
    Independence-related metrics for the FairMetrics package.
"""
from fairmetrics.metrics import (
    demographic_parity,
    conditional_demographic_parity,
)

__all__ = [
    "demographic_parity",
    "conditional_demographic_parity",
]
