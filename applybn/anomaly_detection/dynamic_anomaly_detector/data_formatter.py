from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional

class TemporalDBNTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for creating a temporal windowed representation of tabular data
    for use with dynamic Bayesian networks (DBNs) or other time-dependent models.

    This transformer assumes that:
    - The input data has already been discretized (e.g., using KBinsDiscretizer).
    - Each row represents a time step for a given subject or unit.
    - The data is ordered correctly in time.

    Example:
    --------
    For input data:
        f1   f2
        0    10
        1    11
        2    12

    With window=2, the output will be:
        subject_id f1__0  f2__0  f1__1  f2__1
            0        0      10     1      11
            1        1      11     2      12

    """
    def __init__(self, window: int = 100, include_label: bool = True):
        """
        Initialize the transformer.

        Args:
            window: The size of the sliding temporal window.
            include_label: Whether to include the label (`y`) column in the transformed output.
        """
        self.window = window
        self.include_label = include_label

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit does nothing but is required by scikit-learn.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transforms the input DataFrame into a windowed representation.

        Args:
            X: Input features. Each row is a time step.
            y: Labels corresponding to each row of X (e.g., anomaly labels). Must be the same length as X.

        Returns:
            pd.DataFrame A DataFrame where each row is a flattened sliding window of the input.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        if self.include_label:
            if y is None:
                raise ValueError("Labels must be provided when include_label=True.")
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of rows.")
            X = X.copy()
            X["anomaly"] = y.values

        values = X.values
        if len(values) < self.window:
            raise ValueError(f"Input data must have at least {self.window} rows.")

        dfs = []
        for i, window_arr in enumerate(sliding_window_view(values, window_shape=(self.window, values.shape[1]))):
            window_flat = window_arr[0]
            col_names = [f"{col}__{i}" for col in X.columns]
            part_df = pd.DataFrame(window_flat, columns=col_names)
            dfs.append(part_df)

        final_df = pd.concat(dfs, axis=1).reset_index(names=["subject_id"])
        return final_df
