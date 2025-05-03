import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector import (
    FastTimeSeriesDetector,
)


@pytest.fixture
def input_df():
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "feat__1": [10, 12, 13, 11],
            "feat__2": [5, 4, 6, 7],
        }
    )


@pytest.fixture
def sliced_df():
    return pd.DataFrame(
        {
            "feat__1__0": [10, 12],
            "feat__2__0": [5, 4],
            "feat__1__1": [13, 11],
            "feat__2__1": [6, 7],
            "subject_id": [1, 2],
        }
    )


@patch(
    "applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.jpype.startJVM"
)
@patch(
    "applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.jpype.shutdownJVM"
)
@patch(
    "applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.os.remove"
)
@patch(
    "applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.pd.DataFrame.to_csv"
)
def test_fit_predict_with_dynamic_java_mock_and_calls(
    mock_to_csv, mock_remove, mock_shutdownJVM, mock_startJVM, input_df, sliced_df
):
    mock_scores = [[0.1, -5.0], [-0.1, -6.0]]
    detector = FastTimeSeriesDetector(
        abs_threshold=-4.5,
        rel_threshold=0.5,
        artificial_slicing=True,
        artificial_slicing_params={"window": 2},
    )

    with patch(
        "applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.TemporalDBNTransformer.fit_transform",
        return_value=sliced_df,
    ) as mock_transform:
        with patch(
            "applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector.FastTimeSeriesDetector.decision_function"
        ) as mock_decision:
            mock_decision.return_value = np.array(mock_scores)

            preds = detector.fit(input_df)

            # ✅ Assert return is correct
            assert isinstance(preds, np.ndarray)
            assert preds.tolist() == [0, 1]

            # ✅ Assert transformation was called once with original data
            mock_transform.assert_called_once_with(input_df)

            # ✅ Assert decision_function was called once with the transformed data
            mock_decision.assert_called_once()
            args, kwargs = mock_decision.call_args
            pd.testing.assert_frame_equal(args[0], sliced_df)
