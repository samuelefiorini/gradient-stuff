"""
Module defining pluggable loss functions for gradient boosting.
"""

import numpy as np
from abc import ABC, abstractmethod

EPS = np.finfo(float).eps


class LossFunction(ABC):
    """
    Abstract base class for a loss function.
    It must provide:
    1. The negative gradient (direction to move)
    2. The loss calculation (metric to track)
    3. The initial prediction (starting point)
    """

    @abstractmethod
    def negative_gradient(self, y_true, y_pred):
        pass

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def initial_prediction(self, y):
        pass


class MSELoss(LossFunction):
    """
    Mean Squared Error for Regression.
    L(y, F) = 0.5 * (y - F)^2
    """

    def negative_gradient(self, y_true, y_pred):
        return y_true - y_pred

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def initial_prediction(self, y):
        return np.mean(y)


class LogLoss(LossFunction):
    """
    Log Loss for Binary Classification.
    Predictions are in log-odds space (logits).
    """

    def _standard_logistic(self, x):
        # Convert log-odds to probability
        return 1 / (1 + np.exp(-x))

    def negative_gradient(self, y_true, y_pred):
        return y_true - self._standard_logistic(y_pred)

    def loss(self, y_true, y_pred):
        # Convert log-odds to probability
        p = self._standard_logistic(y_pred)
        # Clip to avoid log(0) error
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def initial_prediction(self, y):
        # Initial guess is the log-odds of the mean probability
        p = np.mean(y)
        # Avoid division by zero for pure classes
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))
