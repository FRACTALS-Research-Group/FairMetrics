"""
    Test utilities for the FairMetrics package.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _encode_categoricals(X):
    """
        Encode every non-numeric column (object, category, string, bool) as integer codes.

        Parameters:
        - X: pd.DataFrame
            Input DataFrame to encode.

        Returns:
        - pd.DataFrame
            DataFrame with encoded categorical columns.
    """
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype('category').cat.codes
    return X


def find_distribution(X, sens_names):
    """
        Return the distribution of sensitive-feature intersections.

        Parameters:
        - X: pd.DataFrame or np.ndarray
            Input data containing sensitive features.
        - sens_names: list of str
            Names of the sensitive feature columns.

        Returns:
        - pd.DataFrame
            DataFrame with the counts of each unique combination of sensitive features.
    """
    if isinstance(X, pd.DataFrame):
        frame = X[sens_names].copy()
    else:
        frame = pd.DataFrame(X, columns=sens_names)
    return frame.groupby(list(frame.columns), dropna=False).size().reset_index(name='count')


def _resolve_local_csv(*relative_paths):
    """
        Resolve the local CSV file path from the given relative paths.

        Parameters:
        - relative_paths: list of str
            Relative paths to check for the CSV file.

        Returns:
        - pathlib.Path or None
            Resolved path to the CSV file if it exists, otherwise None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for rel in relative_paths:
        candidate = (repo_root / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def load_students_success():
    """
        Load the student success dataset.

        Returns:
        - X: pd.DataFrame
            Encoded feature matrix.
        - y: pd.Series
            Target labels mapped to integers.
    """
    csv_path = _resolve_local_csv(
        '../datasets/student_success/student_success.csv',
        'datasets/student_success/student_success.csv',
    )
    if csv_path is not None:
        df = pd.read_csv(csv_path, sep=';')
    else:
        try:
            from ucimlrepo import fetch_ucirepo

            ds = fetch_ucirepo(id=697)
            df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        except Exception as exc:
            raise FileNotFoundError(
                "Student dataset not found. Put the CSV in ../datasets/student_success/student_success.csv "
                "or `pip install ucimlrepo` for the UCI fallback."
            ) from exc

    target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    y = df['Target'].map(target_map)
    X = df.drop(columns=['Target'])
    return _encode_categoricals(X), y


_OPENML_ADULT_COLUMNS = {
    'age': 'Age', 'workclass': 'Workclass', 'education': 'Education',
    'education-num': 'Education-Num', 'marital-status': 'Marital Status',
    'occupation': 'Occupation', 'relationship': 'Relationship', 'race': 'Race',
    'sex': 'Sex', 'capital-gain': 'Capital Gain', 'capital-loss': 'Capital Loss',
    'hours-per-week': 'Hours per week', 'native-country': 'Country', 'class': 'Target',
}


def load_adult():
    """
        Load the adult dataset with a 3-class target.

        Returns:
        - X: pd.DataFrame
            Encoded feature matrix.
        - y: pd.Series
            Target labels mapped to integers.
    """
    csv_path = _resolve_local_csv('../datasets/adult/my_adult.csv', 'datasets/adult/my_adult.csv')
    if csv_path is not None:
        features = [
            'Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num',
            'Marital Status', 'Occupation', 'Relationship', 'Race',
            'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week',
            'Country', 'Target',
        ]
        df = pd.read_csv(csv_path, sep=',', names=features)
    else:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(name='adult', version=2, as_frame=True, parser='auto').frame
        df = df.rename(columns=_OPENML_ADULT_COLUMNS)
        for col in df.select_dtypes(include='category'):
            df[col] = df[col].astype(object)

    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].str.strip().str.replace(r'\.$', '', regex=True)

    df = df.replace('?', np.nan)
    df = df.drop(columns=['Education'])
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    df['New_Target'] = np.select(
        [
            (df['Target'] == '<=50K') & (df['Education-Num'] <= 9),
            (df['Target'] == '<=50K') & (df['Education-Num'] > 9),
            (df['Target'] == '>50K'),
        ],
        [0, 1, 2],
    )
    df = df.drop(columns=['Target', 'Education-Num'])
    df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})

    categorical = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Country']
    df = pd.get_dummies(df, columns=categorical, dtype=int)

    y = df['New_Target']
    X = df.drop(columns=['New_Target'])
    return X, y

__all__ = [
    'find_distribution',
    'load_adult',
    'load_students_success',
]


if __name__ == '__main__':
    for name, loader in [('adult', load_adult), ('students', load_students_success)]:
        X, y = loader()
        print(name, X.shape, y.value_counts().sort_index().to_dict())