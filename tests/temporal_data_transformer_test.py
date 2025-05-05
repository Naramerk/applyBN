import pytest
import pandas as pd
import numpy as np
from applybn.anomaly_detection.dynamic_anomaly_detector.data_formatter import (
    TemporalDBNTransformer,
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "f1": np.arange(10),
            "f2": np.arange(10, 20),
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    return X, y


def test_transform_basic_with_labels(sample_data):
    X, y = sample_data

    transformer = TemporalDBNTransformer(window=3, include_label=True)
    Xt = transformer.transform(X, y)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == len(X) - 3 + 1  # (n - window + 1)
    assert (
        Xt.shape[1] == (3 * (X.shape[1])) + 2
    )  # + 1 for anomaly label + 1 for subject_id

    # Check columns have "__" in names
    assert all("__" in col for col in Xt.columns[1:-1])


def test_transform_basic_without_labels(sample_data):
    X, _ = sample_data
    transformer = TemporalDBNTransformer(window=4, include_label=False)
    Xt = transformer.transform(X)

    assert Xt.shape[0] == len(X) - 3
    assert Xt.shape[1] == (4 * X.shape[1]) + 1


def test_transform_raises_for_small_input(sample_data):
    X, y = sample_data
    transformer = TemporalDBNTransformer(window=20, include_label=True)
    with pytest.raises(ValueError, match="at least 20 rows"):
        transformer.transform(X, y)


def test_transform_raises_for_non_dataframe():
    transformer = TemporalDBNTransformer(window=3)
    with pytest.raises(ValueError, match="pandas DataFrame"):
        transformer.transform(np.random.rand(10, 3))


def test_transform_raises_missing_labels(sample_data):
    X, _ = sample_data
    transformer = TemporalDBNTransformer(window=3, include_label=True)
    with pytest.raises(ValueError, match="Labels must be provided"):
        transformer.transform(X)


def test_transform_raises_mismatched_lengths(sample_data):
    X, y = sample_data
    transformer = TemporalDBNTransformer(window=3, include_label=True)
    with pytest.raises(ValueError, match="same number of rows"):
        transformer.transform(X.iloc[:-1], y)


def test_transform_windows_fraction(sample_data):
    X, y = sample_data
    row_numbers, _ = X.shape

    transformer_window_rel = TemporalDBNTransformer(window=0.5, include_label=True)
    true_windows_size = int(row_numbers * 0.5)
    Xt = transformer_window_rel.transform(X, y)

    assert Xt.shape[0] == row_numbers - true_windows_size + 1
