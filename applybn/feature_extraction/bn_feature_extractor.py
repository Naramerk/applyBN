import numpy
from bamt.networks.hybrid_bn import HybridBN
import pandas as pd
import bamt.preprocessors as pp
from typing import Optional, List, Tuple
import logging
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from pgmpy.estimators import K2Score
import numpy as np
from bamt.preprocess.discretization import code_categories, get_nodes_type
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.discrete_bn import DiscreteBN
from scipy.stats import norm

class BNFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates features based on a Bayesian Network (BN).
    """
    def __init__(self,
                 black_list: Optional[List[Tuple[str, str]]] = []):
        self.bn = None
        self.variables: Optional[List[str]] = None
        self.black_list = black_list

        self.nodes_dict = {} # Dictionary to store nodes of the Bayesian Network
        self.target_name = ''

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the BNFeatureGenerator to the data.

        This involves:
        1.  Adding the target variable (if provided) to the input data.
        2.  Encoding categorical columns.
        3.  Discretizing continuous columns.
        4.  Creating a Bayesian Network based on the data types.
        5.  Learning the structure of the Bayesian Network (if known_structure is not provided).
        6.  Fitting the parameters of the Bayesian Network.

        Args:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target variable. If provided, it will be added to the input data and
                treated as a node in the Bayesian Network.

        Returns:
            self: The fitted BNFeatureGenerator object.
        """
        
        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy='kmeans',
            subsample=None
        )
        preprocessor = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

        if y is not(None):
            if y.name == '':
                y = y.rename('target')
            self.target_name = y.name
            X = pd.concat([X, y], axis=1).reset_index(drop=True)

        Z = X
        Z.reset_index(inplace=True, drop=True)
        cat_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if cat_columns:
            X, cat_encoder = code_categories(X, method="label", columns=cat_columns)
        discretized_data, est = preprocessor.apply(X)
        discretized_data = pd.DataFrame(discretized_data, columns=X.columns)
        info = preprocessor.info # Get information about the data types after preprocessing
        # Get types of nodes
        get_nodes_type = info['types']

        # Check for discrete and continuous columns
        values = get_nodes_type.values()
        has_discrete = any(value in ['disc', 'disc_num'] for value in values)
        has_continuous = any(value not in ['disc', 'disc_num'] for value in values)

        # If we have categorical columns, we have discrete data
        if cat_columns:
            has_discrete = True

        if has_continuous and not has_discrete:
            self.bn = ContinuousBN(use_mixture=False)
            logging.info("Using ContinuousBN")
        elif has_discrete and not has_continuous:
            self.bn = DiscreteBN()
            logging.info("Using DiscreteBN")
        else:
            self.bn = HybridBN(has_logit=True, use_mixture=False)
            logging.info("Using HybridBN")
        self.bn.add_nodes(info)
        params = {}
        if self.target_name:
            bl = self.create_black_list(X, self.target_name) # Edges to avoid
            params = {'bl_add': bl}
        self.bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params)
        self.bn.fit_parameters(Z) # Fit parameters of the Bayesian Network
        logging.info(self.bn.get_info())
        self.variables = list(X.columns)
        self.nodes_dict = {str(node.name): node for node in self.bn.nodes}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame `X` into a new DataFrame where each column
        represents the calculated feature based on the fitted BN.

        Args:
            X (pd.DataFrame) is the input DataFrame to transform.

        Returns:
            pd.DataFrame is a new DataFrame with lambda-features.
        """
        if not self.bn:
            logging.error(AttributeError,
                "Parameter learning wasn't done. Call fit method"
            )
            return pd.DataFrame()  # Return an empty DataFrame to avoid further errors

        # Handle categorical columns (if any)
        results = []
        # Process each feature (column) in the row using the BN
        for _, row in X.iterrows():
            row_probs = [self.process_feature(feat, row, X) for feat in self.variables]
            results.append(row_probs)
        if self.target_name:
            predictions = self.bn.predict(test=X, parall_count=4, progress_bar = False)
            predictions = pd.Series(predictions[self.target_name])
            result = pd.DataFrame(results, columns=['lambda_' + c for c in self.variables])
            result['lambda_' + self.target_name] = predictions
        else:
            result = pd.DataFrame(results, columns=['lambda_' + c for c in self.variables])
        return result

    def create_black_list(self, X: pd.DataFrame, y: Optional[str]):
        if not y:
            return []
        target_node = y
        black_list = [(target_node, (col)) for col in X.columns.to_list() if col != target_node]

        return black_list

    def val_to_str(self, val):
        """
        Converts a value to its string representation.
        """
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        elif isinstance(val, int):
            return str(val)
        elif isinstance(val, str):
            try:
                f = float(val)
                if f.is_integer():
                    return str(int(f))
                else:
                    return val
            except Exception:
                return val
        else:
            return str(val)

    def process_feature(self, feature: str, row: pd.Series, X):

        """
        Processes a single feature (node) in the Bayesian network for a given row of data.

        Args:
            feature (str): The name of the feature (node) being processed.
            row (pd.Series): A row from X.

        Returns:
            float: The probability or observed value depending on the node type.
        """
        if str(feature) == self.target_name:
            return 0.0001

        try:
            node = self.nodes_dict[str(feature)]

            #parents = node.cont_parents + node.disc_parents

            pvals = {}
            pvals_no_cont = []
            pvals_cont = []

            # Iterate through the continuous parents
            for p in node.cont_parents:
                pvals[p]=row[p]
                pvals_cont.append(row[p])

            # Iterate through the discrete parents
            for p in node.disc_parents:
                norm_val = self.val_to_str(row[p])
                pvals[p]=(norm_val)
                pvals_no_cont.append(norm_val)

            # Process discrete nodes
            if node.type == 'Discrete':
                vals = X[node.name].value_counts(normalize=True).sort_index()
                vals = [str(i) for i in vals.index.to_list()]
                return self._process_discrete_node(feature, row, pvals, vals)

            # Process non-discrete nodes
            else:
                vals = X[node.name].value_counts(normalize=True).sort_index()
                vals = [(i) for i in vals.index.to_list()]
                return self._process_non_discrete_node(feature, row, pvals, vals, pvals_no_cont)

        except Exception as e:
            logging.error(f"Error processing node {feature}: {e}")
            return 0.0001

    def _process_discrete_node(self, feature, row, pvals, vals):
        """
        Processes a discrete node.

        Args:
            node - the discrete node object.
            feature (str) - the name of the feature (node).
            row (pd.Series) - a row of data from the DataFrame.
            pvals (list) - list of parent values.
            vals - possible values of the 'feature'.

        Returns:
            float - value of a new feature.
        """

        obs_value = str((row[feature]))
        try:
            dist = self.bn.get_dist(str(feature), pvals=pvals)# list(map(str, pvals)))

        except:
            return 0.0001

        try:
            # Try to find the index of the observed value in the node's values
            idx = dist["vals"].index(obs_value)
            return dist["cprob"][idx]
        except:
            idx = vals.index((obs_value))
            return dist[idx]

    def _process_non_discrete_node(self, feature, row, pvals, vals, pvals_no_cont):
        """
        Processes a non-discrete node.

        Args:
            feature (str) - the name of the feature (node).
            row (pd.Series) - a row of data from the DataFrame.
            pvals (list) - list of parent values.
            vals - possible values of the 'feature'.

        Returns:
            float - value of a new feature.
        """
        obs_value = (row[feature])
        try:
            dist = self.bn.get_dist(str(feature), pvals=pvals)
        except:
            return 0.0001

        if isinstance(dist, dict) and len(dist) > 2 :
            try:
                # If the distribution is a tuple (these are a mean and variance)
                if isinstance(dist, dict) and len(dist) > 2:
                    mean, variance = dist['mean'], dist['variance']
                    sigma = np.sqrt(variance)

                prob = norm.cdf(obs_value, loc=mean, scale=sigma)

                if  numpy.isnan(prob):
                    return 0.0001
                return prob

            except:
                logging.warning(" dist not found for node %s:", feature)
                return 0.0001

        # If the distribution is an array (we're processing Logit node)
        elif isinstance(dist, dict):
            #dist = {'cprob': [0.18947368421052632, 0.2, 0.19210526315789472, 0.2026315789473684, 0.21578947368421053], 'vals': ['0', '1', '2', '3', '4']}
            try:
                # Try to find the index of the observed value in the node's values
                idx = dist["vals"].index(obs_value)
                return dist["cprob"][idx]
            except:
                idx = vals.index((obs_value))
                return dist[idx]

        elif isinstance(dist, tuple):

            try:
                # If the distribution is a tuple (these are a mean and variance)
                if len(dist) == 2:
                    mean, variance = dist
                    sigma = (variance)
                    prob = norm.cdf(obs_value, loc=mean, scale=sigma)
                    if  numpy.isnan(prob):
                        return 0.0001
                    return prob
                else:
                    logging.warning("unknown dist for node %s: %s", feature, dist)
                    return 0.0001
            except:
                logging.warning(" dist not found for node %s:", feature)
                return 0.0001
        elif isinstance(dist, np.ndarray):
            # If the distribution is a np.ndarray (we're processing conditional logit node)
            try:
                if "classes" in self.bn.distributions[feature]["hybcprob"][str(pvals_no_cont)]:
                    classes = self.bn.distributions[feature]["hybcprob"][str(pvals_no_cont)]["classes"]
                    if obs_value in classes:
                        idx = classes.index(obs_value)
                        prob = dist[idx]
                        return prob
                    else:
                        return 0.0001

            except KeyError:
                logging.warning(" dist not found for node %s:", feature)
                return 0.0001
        else:
            logging.warning(" dist not found for node %s:", feature)
            return 0.0001


        return 0.0001
