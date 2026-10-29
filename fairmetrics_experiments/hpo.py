"""
    Hyperparameter optimization (HPO) utilities for fairness experiments.
    This module provides a lightweight interface for hyperparameter tuning used in the paper experiments.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def hyperparams_tuning(X, y, target_map):
    """
        Perform hyperparameter tuning for a RandomForestClassifier using GridSearchCV.

        Parameters:
        - X: pd.DataFrame
            Feature matrix.
        - y: pd.Series
            Target labels.
        - target_map: dict
            Mapping of target labels to integers.

        Returns:
        - dict
            Dictionary containing the best parameters and the best score.
    """

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [12, 15, 18, 20, None],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3],
        'max_features': [0.4, 0.5, 0.6, 'sqrt'],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid=param_grid, cv=10, n_jobs=-1, scoring='f1_macro', verbose=2,
    )
    grid.fit(X_train, y_train)
    return {'best_params': grid.best_params_, 'best_score': float(grid.best_score_)}
