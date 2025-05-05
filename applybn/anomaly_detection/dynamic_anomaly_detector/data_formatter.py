from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import Optional, Literal


class TemporalDBNTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for creating a temporal windowed representation of tabular data
    for use with dynamic Bayesian networks (DBNs) or other time-dependent models.

    This transformer assumes that:

    - The input data has already been discretized (e.g., using KBinsDiscretizer).
    - Each row represents a time step for a given subject or unit.
    - The data is ordered correctly in time.

    Example:

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

    def __init__(
        self,
        window: float = 100,
        include_label: bool = True,
        stride: int = 1,
        gathering_strategy: None | Literal["any"] = "any",
    ):
        """
        Initialize the transformer.

        Args:
            window: If < 1, the size of the sliding temporal window. If > 1, number of rows in window.
            stride: The size of the sliding temporal stride.
            include_label: Whether to include the label (`y`) column in the transformed output.
        """
        self.window = window
        self.include_label = include_label
        self.gathering_strategy = gathering_strategy
        self.stride = stride

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit does nothing but is required by scikit-learn.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transforms the input DataFrame into a windowed representation with an optional stride.

        Args:
            X: Input features. Each row is a time step.
            y: Labels corresponding to each row of X (e.g., anomaly labels). Must be the same length as X.

        Returns:
            A DataFrame where each row is a flattened sliding window of the input.
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
        n_rows, n_features = values.shape
        if n_rows < self.window:
            raise ValueError(f"Input data must have at least {self.window} rows.")

        if self.window < 1:
            window_size = int(self.window * n_rows)
        else:
            window_size = self.window

        num_windows = (n_rows - window_size) // self.stride + 1

        dfs = []
        for i in range(0, num_windows * self.stride, self.stride):
            window = values[i : i + window_size]
            window_flat = window.flatten()
            col_names = [
                f"{col}__{j}" for j in range(1, window_size + 1) for col in X.columns
            ]
            part_df = pd.DataFrame([window_flat], columns=col_names)
            dfs.append(part_df)

        final_df = pd.concat(dfs, axis=0, ignore_index=True).reset_index(
            names=["subject_id"]
        )

        if self.gathering_strategy and self.include_label:
            final_df = self.aggregate_anomalies(final_df)

        return final_df

    def aggregate_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        This function aggregates anomalies. After transform there are a lot of cols with anomalies and
        gathering them into one target vector is required.
        Args:
            X: Sliced data

        Returns:
            Dataframe with target vector
        """
        X_ = X.copy()
        match self.gathering_strategy:
            case "any":
                anomaly_cols_names = [
                    col for col in X.columns if col.startswith("anomaly")
                ]
                anomalies_cols = X[anomaly_cols_names]

                aggregated = np.any(anomalies_cols, axis=1)
                X_.drop(anomaly_cols_names, axis=1, inplace=True)
                X_["anomaly"] = aggregated.astype(int)
                return X_
            case _:
                raise ValueError(
                    f"Unknown gathering strategy {self.gathering_strategy}."
                )
