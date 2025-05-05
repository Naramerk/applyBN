import jpype
import jpype.imports
import numpy as np
from jpype.types import *
import pandas as pd
import os
import shutil
import tempfile

from sklearn.exceptions import NotFittedError
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score

from applybn.anomaly_detection.dynamic_anomaly_detector.data_formatter import (
    TemporalDBNTransformer,
)

if not shutil.which("java"):
    raise NotImplementedError(
        "Java is not installed. In order to use the fast method you need to install it first."
    )


class FastTimeSeriesDetector:
    """
    A time-series anomaly detection model based on Dynamic Bayesian Network (DBN) structure learning
    implemented in Java and accessed via JPype.

    This class supports both pre-sliced DBN data and automatic transformation of raw tabular time-series data
    using a sliding window mechanism.
    """

    def __init__(
        self,
        abs_threshold: float = -4.5,
        rel_threshold: float = 0.8,
        num_parents: int = 3,
        artificial_slicing: bool = False,
        artificial_slicing_params: dict = None,
        scoring_function: str = "ll",
        markov_lag: int = 1,
        non_stationary: bool = False,
    ):
        """
        Initializes the FastTimeSeriesDetector.

        Args:
            abs_threshold: Absolute score below which values are flagged as anomalies.
            rel_threshold: Fraction of features with anomaly scores needed to flag the full sample.
            num_parents: Maximum number of parents allowed in the DBN structure.
            artificial_slicing: Whether to apply window-based transformation on the input data.
            artificial_slicing_params: Parameters for the TemporalDBNTransformer (e.g., window size).
            scoring_function: Scoring function used by the Java DBN learner ('ll' or 'MDL').
            markov_lag: The Markov lag (time distance) for DBN learning.
            non_stationary: Learn separate models for each transition instead of one shared model.
        """
        self.args = [
            "-p",
            str(num_parents),
            "-s",
            scoring_function,
            "-m",
            str(markov_lag),
            "-ns",
            str(non_stationary),
            "-pm",
        ]
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.join(base, "dbnod_modified.jar")

        if not jpype.isJVMStarted():
            jpype.startJVM(
                classpath=[
                    module_path,
                ]
            )

        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold
        self.artificial_slicing = artificial_slicing
        self.artificial_slicing_params = artificial_slicing_params

    def _is_fitted(self):
        return True if "scores_" in self.__dict__ else False

    @staticmethod
    def _validate_data(X):
        """
        Ensures the input DataFrame contains a 'subject_id' column and that all other
        column names follow the expected '__' naming convention for DBN inputs.

        Raises:
            TypeError: If required format is not met.
        """
        if "subject_id" not in X.columns:
            raise TypeError("subject_id column not found in data.")

        if not all("__" in col_name for col_name in X.columns.drop("subject_id")):
            raise TypeError(
                "Data type error. Column names must contain '__' characters."
            )

    def fit(self, X: pd.DataFrame):
        """
        Trains the DBN model using input data. If artificial slicing is enabled,
        performs time-window transformation before training.

        Args:
            X: Input data (time-series features).

        Returns:
            np.ndarray: Anomaly labels (0 for normal, 1 for anomalous).
        """
        if not self.artificial_slicing:
            self._validate_data(X)
        else:
            transformer = TemporalDBNTransformer(**self.artificial_slicing_params)
            X = transformer.fit_transform(X)

        self.scores_ = self.decision_function(X)

        return self

    def predict_scores(self, X: pd.DataFrame = None):
        """
        Computes raw anomaly scores from the trained DBN.

        Args:
            X: Input data. Not used in this implementation.

        Returns:
            np.ndarray: Raw scores.
        """
        if not self._is_fitted():
            raise NotFittedError("DBN model has not been fitted.")

        return self.scores_

    def calibrate(
        self,
        y_true: pd.Series | np.ndarray,
        calibration_bounds: dict | None = None,
        verbose: int = 1,
        calibration_params: dict = None,
    ):
        """
        A method to calibrate the DBN. Calibration means finding absolute and relative thresholds.
        Utilizes bayesian optimization.

        Args:
            y_true: values to calibrate on
            calibration_bounds: bound of calibration values. Must contain abs_thrs and rel_thrs keys.
            verbose: verbosity level.
            calibration_params: calibration parameters for optimization.

        """

        def func_to_optimize(abs_thrs, rel_thrs):
            self.abs_threshold = abs_thrs
            self.rel_threshold = rel_thrs
            preds = self.predict()
            return f1_score(y_true, preds)

        if calibration_params is None:
            calibration_params = dict(init_points=10, n_iter=100)

        if calibration_bounds is None:
            pbounds = {"abs_thrs": (-8, -2), "rel_thrs": (0.2, 0.95)}
        else:
            pbounds = calibration_bounds

        optimizer = BayesianOptimization(
            f=func_to_optimize, pbounds=pbounds, verbose=verbose
        )

        optimizer.maximize(**calibration_params)

        self.abs_threshold, self.rel_threshold = (
            optimizer.max["params"]["abs_thrs"],
            optimizer.max["params"]["rel_thrs"],
        )
        return self

    def predict(self, X: pd.DataFrame = None):
        """
        Trains the model and applies anomaly decision logic.

        Args:
            X: Input features. Not used.

        Returns:
            np.ndarray: Binary anomaly labels (1 = anomalous).
        """
        if not self._is_fitted():
            raise NotFittedError("DBN model has not been fitted.")

        thresholded = np.where((self.scores_ < self.abs_threshold), 1, 0)

        # Aggregate per-sample anomaly flags and compare against relative threshold
        self.anom_fractions_ = thresholded.mean(axis=0)
        return np.where(self.anom_fractions_ > self.rel_threshold, 1, 0)

    def decision_function(self, X: pd.DataFrame):
        """
        Calls the Java backend to score transitions using DBN inference.

        Args:
            X: Preprocessed DBN-compatible DataFrame.

        Returns:
            np.ndarray: 2D array of log-likelihood scores from the Java model.
        """
        from com.github.tDBN.cli import LearnFromFile

        # Write data to disk and call Java scoring
        with tempfile.NamedTemporaryFile() as tmpfile:
            X.to_csv(tmpfile, index=False)
            self.args.extend(["-i", tmpfile.name])
            result = LearnFromFile.ComputeScores(JArray(JString)(self.args))

            outlier_indexes, scores = result

            # Convert Java 2D double array into numpy
            py_2d_array = []
            for i in range(len(scores)):
                py_2d_array.append(list(scores[i]))

            scores = np.asarray(py_2d_array)
            return scores
