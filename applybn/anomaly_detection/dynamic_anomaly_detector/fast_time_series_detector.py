import jpype
import jpype.imports
import numpy as np
from jpype.types import *
import pandas as pd
import os
import shutil
from applybn.anomaly_detection.dynamic_anomaly_detector.data_formatter import TemporalDBNTransformer

if not shutil.which("java"):
    raise NotImplementedError("Java is not installed. In order to use the fast method you need to install it first.")

class FastTimeSeriesDetector:
    """
    A time-series anomaly detection model based on Dynamic Bayesian Network (DBN) structure learning
    implemented in Java and accessed via JPype.

    This class supports both pre-sliced DBN data and automatic transformation of raw tabular time-series data
    using a sliding window mechanism.
    """

    def __init__(self,
                 abs_threshold: float = -4.5,
                 rel_threshold: float = 0.8,
                 num_parents: int = None,
                 artificial_slicing: bool = False,
                 artificial_slicing_params: dict = None,
                 scoring_function: str = 'll',
                 markov_lag: int = 1,
                 non_stationary: bool = False,
                 parameters: bool = True,
                 intra_in: int = None):
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
            parameters: Whether to output DBN parameters.
            intra_in: In-degree limit for intra-slice connections.
        """
        self.args = [
            "-p", str(num_parents),
            "-s", scoring_function,
            "-m", markov_lag,
            "-ns", non_stationary,
            "--intra-in", intra_in,
            "-pm" if parameters else ""
        ]
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold
        self.artificial_slicing = artificial_slicing
        self.artificial_slicing_params = artificial_slicing_params

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
            raise TypeError("Data type error. Column names must contain '__' characters.")

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

        return self.fit_predict(X)

    def predict_scores(self, X: pd.DataFrame):
        """
        Computes raw anomaly scores from the trained DBN.

        Args:
            X: Input data.

        Returns:
            np.ndarray: Raw scores.
        """
        return self.decision_function(X)

    def fit_predict(self, X: pd.DataFrame):
        """
        Trains the model and applies anomaly decision logic.

        Args:
            X: Input features.

        Returns:
            np.ndarray: Binary anomaly labels (1 = anomalous).
        """
        self.scores_ = self.decision_function(X)
        thresholded = np.where((self.scores_ < self.abs_threshold), 1, 0)

        # Aggregate per-sample anomaly flags and compare against relative threshold
        anom_fractions = thresholded.mean(axis=0)
        return np.where(anom_fractions > self.rel_threshold, 1, 0)

    def decision_function(self, X: pd.DataFrame):
        """
        Calls the Java backend to score transitions using DBN inference.

        Args:
            X: Preprocessed DBN-compatible DataFrame.

        Returns:
            np.ndarray: 2D array of log-likelihood scores from the Java model.
        """
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.join(base, "dbnod_modified.jar")

        # Start JVM and load Java class
        jpype.startJVM(classpath=[module_path,])
        from com.github.tDBN.cli import LearnFromFile

        # Write data to disk and call Java scoring
        X.to_csv("temp.csv", index=False)
        self.args.extend(["-i", "temp.csv"])
        result = LearnFromFile.ComputeScores(JArray(JString)(self.args))

        outlier_indexes, scores = result

        # Convert Java 2D double array into numpy
        py_2d_array = []
        for i in range(len(scores)):
            py_2d_array.append(list(scores[i]))
        os.remove("temp.csv")

        jpype.shutdownJVM()
        return np.asarray(py_2d_array)
