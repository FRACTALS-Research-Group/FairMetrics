"""
Experiment utilities for FairMetrics paper runs.

This package provides:
- target binarization helpers (one-vs-rest: bin0/bin1/bin2)
- the HPO runner for Adult and Students datasets
- fairness-audit experiment entry points used in reproducible paper runs

Typical usage:
    python -m fairmetrics_experiments.run_hpo --dataset adult --mode bin0
"""