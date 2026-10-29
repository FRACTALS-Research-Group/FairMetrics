"""
    Data loading and preprocessing utilities for fairness experiments.

    This module provides functions to load and preprocess commonly used datasets
    such as the Adult dataset and the Student Success dataset. It also includes
    helper functions for analyzing sensitive-feature distributions.
"""
import pandas as pd
import numpy as np


def find_distribution(X, sens_names):
    """
        Return the distribution of sensitive-feature intersections.
        
        Parameters:
        - X: Input feature matrix (DataFrame or array-like).
        - sens_names: List of sensitive feature column names.

        Returns:
        - DataFrame containing the counts of each combination of sensitive features.
    """
    if isinstance(X, pd.DataFrame):
        frame = X[sens_names].copy()
    else:
        frame = pd.DataFrame(X, columns=sens_names)

    counts = frame.groupby(list(frame.columns), dropna=False).size().reset_index(name='count')
    return counts


def load_adult():
    """
        Load the Adult dataset into (X, y) while preserving the logic used in paper experiments.
        Returns:
        - X: Feature matrix (DataFrame).
        - y: Target vector (Series).
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    X = data.frame.drop(columns=['class'])
    y = data.target

    X = X.copy()
    y = y.astype(str)
    y = y.map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
    X = X.apply(lambda s: s.astype('category').cat.codes if s.dtype == 'object' else s)

    return X, y


def load_students_success():
    """
        Load the student success dataset into (X, y) while preserving the logic used in paper experiments.

        Returns:
        - X: Feature matrix (DataFrame).
        - y: Target vector (Series).
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name='student-success', version=1, as_frame=True, parser='auto')
    X = data.frame.drop(columns=['target'])
    y = data.target

    X = X.copy()
    y = y.astype(str)
    y = y.map({'fail': 0, 'pass': 1})
    X = X.apply(lambda s: s.astype('category').cat.codes if s.dtype == 'object' else s)

    return X, y
