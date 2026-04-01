"""
Module implementing a simple Decision Tree Regressor from scratch.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

class DecisionTreeRegressor:
    """
    A simple Decision Tree Regressor implemented from scratch.
    
    This implementation follows a 'keep-it-simple' philosophy, supporting only
    max_depth as a hyperparameter and using variance reduction for splitting.
    """
    def __init__(self, max_depth: int = 3):
        """
        Initialize the Decision Tree Regressor.

        Args:
            max_depth (int): The maximum depth of the tree. Defaults to 3.
        """
        self.max_depth = max_depth
        self.tree: Optional[Union[float, Dict[str, Any]]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        Build the decision tree from the training set (X, y).

        Args:
            X (np.ndarray): The training input samples. Shape (n_samples, n_features).
            y (np.ndarray): The target values. Shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class or regression value for X.

        Args:
            X (np.ndarray): The input samples. Shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values. Shape (n_samples,).
        """
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Union[float, Dict[str, Any]]:
        """
        Recursively build the decision tree.

        Args:
            X (np.ndarray): Input samples for the current node.
            y (np.ndarray): Target values for the current node.
            depth (int): Current depth of the tree.

        Returns:
            Union[float, Dict[str, Any]]: A leaf value (float) or a node dictionary.
        """
        n_samples, n_features = X.shape
        
        # Base case: max depth reached or constant values
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return float(np.mean(y))

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return float(np.mean(y))

        # Split data
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Create node
        return {
            'feature_index': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_idxs], y[left_idxs], depth + 1),
            'right': self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        }

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split the data.

        Args:
            X (np.ndarray): Input samples.
            y (np.ndarray): Target values.

        Returns:
            Tuple[Optional[int], Optional[float]]: The index of the best feature and the threshold value.
            Returns (None, None) if no valid split is found.
        """
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None

        current_uncertainty = np.var(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            # Get unique values for potential thresholds
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_idxs = X[:, feature_idx] < threshold
                right_idxs = ~left_idxs
                
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue

                # Calculate weighted variance reduction
                n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
                var_left = np.var(y[left_idxs])
                var_right = np.var(y[right_idxs])
                
                weighted_var = (n_left * var_left + n_right * var_right) / n_samples
                gain = current_uncertainty - weighted_var

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _predict_one(self, x: np.ndarray, tree: Union[float, Dict[str, Any]]) -> float:
        """
        Predict the value for a single sample.

        Args:
            x (np.ndarray): A single input sample.
            tree (Union[float, Dict[str, Any]]): The current tree node or leaf value.

        Returns:
            float: The predicted value.
        """
        if not isinstance(tree, dict):
            return float(tree)
        
        if x[tree['feature_index']] < tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])
