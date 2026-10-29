import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fairmetrics import (
    conditional_demographic_parity,
    demographic_parity,
    equal_accuracy,
    equalized_odds,
    binary_predictive_parity,
    multiclass_predictive_parity,
)
from fairmetrics.tests import load_adult, load_students_success
from fairmetrics_experiments.hpo import hyperparams_tuning

# Configuration for datasets, including sensitive and strata columns.
DATASET_CONFIG = {
    'adult': {
        'sensitive_cols': ['Sex', 'Race_White'],
        'strata_cols': ['Country_United-States'],
    },
    'students': {
        'sensitive_cols': ['Gender', 'International'],
        'strata_cols': ['Displaced'],
    },
}


def _project_feature_block(frame, cols):
    """
        Project the specified columns from the DataFrame into a numpy array.

        Parameters:
        - frame: Input DataFrame
        - cols: List of column names to project

        Returns:
        - Numpy array containing the projected features.
            - If no columns are provided, returns an array of shape [n, 0]
            - If columns are provided, returns an array of shape [n, k] where k is the number of columns
    """
    existing = [c for c in cols if c in frame.columns]
    if not existing:
        return np.zeros((len(frame), 0), dtype=float)
    return frame[existing].to_numpy(dtype=float)


def _build_intersectional_group(frame, cols):
    """
        Build an intersectional group representation for the given columns in the DataFrame.

        Parameters:
        - frame: Input DataFrame
        - cols: List of column names to include in the intersectional group

        Returns:
        - Numpy array representing the intersectional group.
            - If no columns are provided, returns an array of shape [0, n]
            - If one column is provided, returns an array of shape [n]
            - If multiple columns are provided, returns an array of shape [k, n] where k is the number of columns
    """
    if not cols:
        return np.zeros((0, len(frame)), dtype=float)   # <- shape [0, n]
    if len(cols) == 1:
        return frame[cols[0]].to_numpy(dtype=float)     # <- shape [n]
    arr = frame[cols].to_numpy(dtype=float)             # <- shape [n, k]
    return arr.T                                        # <- FIX: shape [k, n]


def _metric_snapshot(dataset_name, mode, X, y, y_pred):
    """
        Compute a snapshot of fairness metrics for the given dataset, mode, and predictions.

        Parameters:
        - dataset_name: Name of the dataset ('adult' or 'students')
        - mode: One of 'mul', 'bin0', 'bin1', 'bin2'
        - X: Feature matrix
        - y: True labels
        - y_pred: Predicted labels

        Returns:
        - Dictionary containing fairness metrics.
    """
    cfg = DATASET_CONFIG[dataset_name]
    sens = _build_intersectional_group(X, cfg['sensitive_cols'])
    strata = _build_intersectional_group(X, cfg['strata_cols'])

    y_true_t = torch.tensor(y.to_numpy(dtype=np.int64), dtype=torch.int64)
    y_pred_t = torch.tensor(np.asarray(y_pred, dtype=np.int64), dtype=torch.int64)
    sens_t = torch.tensor(np.asarray(sens, dtype=np.float32), dtype=torch.float32)
    strata_t = torch.tensor(np.asarray(strata, dtype=np.float32), dtype=torch.float32)

    # default behavior: for binary mode use the binary metric on the one-vs-rest label;
    # for multiclass use the multiclass metric.
    if mode == 'mul':
        fairness = {
            'demographic_parity': float(demographic_parity(None, y_pred_t, sens_t).item()),
            'equalized_odds': float(equalized_odds(y_true_t, y_pred_t, sens_t).item()),
            'predictive_parity': float(multiclass_predictive_parity(y_true_t, y_pred_t, sens_t).item()),
            'equal_accuracy': float(equal_accuracy(y_true_t, y_pred_t, sens_t).item()),
        }
        if len(cfg['strata_cols']) > 0:
            fairness['conditional_demographic_parity'] = float(conditional_demographic_parity(None, y_pred_t, sens_t, strata_t).item())
    else:
        fairness = {
            'demographic_parity': float(demographic_parity(None, y_pred_t, sens_t).item()),
            'equalized_odds': float(equalized_odds(y_true_t, y_pred_t, sens_t).item()),
            'predictive_parity': float(binary_predictive_parity(y_true_t, y_pred_t, sens_t).item()),
            'equal_accuracy': float(equal_accuracy(y_true_t, y_pred_t, sens_t).item()),
        }
        if len(cfg['strata_cols']) > 0:
            fairness['conditional_demographic_parity'] = float(conditional_demographic_parity(None, y_pred_t, sens_t, strata_t).item())

    return {
        'sensitive_cols': cfg['sensitive_cols'],
        'strata_cols': cfg['strata_cols'],
        'intersectional_sensitive_groups': int(len(np.unique(sens.reshape(-1, 1) if sens.ndim == 1 else sens.T, axis=0))) if sens.size > 0 else 0,
        'fairness': fairness,
    }


def run_hpo(X, y, mode, dataset_name='adult', random_state=42):
    """
        Run the HPO for the given dataset and mode.

        Parameters:
        - X: Features
        - y: Target
        - mode: One of 'mul', 'bin0', 'bin1', 'bin2'
        - dataset_name: Name of the dataset (default: 'adult')
        - random_state: Random seed (default: 42)

        Returns:
        - Dictionary containing HPO results and fairness metrics.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    if mode == 'mul':
        y_proc = y.copy()
    else:
        mappings = {
            'bin0': {0: 1, 1: 0, 2: 0},
            'bin1': {0: 0, 1: 1, 2: 0},
            'bin2': {0: 0, 1: 0, 2: 1},
        }
        if mode not in mappings:
            raise ValueError(f"Unsupported mode '{mode}'")
        y_proc = y.map(mappings[mode]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_proc, test_size=0.3, random_state=random_state, stratify=y_proc
    )

    best = hyperparams_tuning(X_train, y_train, target_map={})
    best_params = best['best_params']
    model = RandomForestClassifier(random_state=random_state, **best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        'dataset': dataset_name,
        'mode': mode,
        'n_samples': len(X),
        'positive_rate': float((y_proc == 1).mean()),
        'best_params': best_params,
        'best_score': best['best_score'],
    }
    result.update(_metric_snapshot(dataset_name, mode, X_test, y_test, y_pred))
    return result


def run_dataset_strategy(dataset_name, mode):
    """
        Run the HPO for a specific dataset and mode.

        Parameters:
        - dataset_name: Name of the dataset ('adult' or 'students')
        - mode: One of 'mul', 'bin0', 'bin1', 'bin2'

        Returns:
        - Dictionary containing HPO results and fairness metrics.
    """
    if dataset_name == 'adult':
        X, y = load_adult()
    elif dataset_name == 'students':
        X, y = load_students_success()
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    return run_hpo(X, y, mode, dataset_name=dataset_name)


def main():
    parser = argparse.ArgumentParser(description='Run the paper-style RF HPO grid for Adult/Students targets.')
    parser.add_argument('--dataset', choices=['adult', 'students'], required=True)
    parser.add_argument('--mode', choices=['mul', 'bin0', 'bin1', 'bin2'], required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    result = run_dataset_strategy(args.dataset, args.mode)
    payload = json.dumps(result, indent=2, sort_keys=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + '\n', encoding='utf-8')
    else:
        print(payload)


if __name__ == '__main__':
    main()
