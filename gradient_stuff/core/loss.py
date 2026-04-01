"""
Module defining pluggable loss functions for gradient boosting.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.special import expit  # Sigmoid function

EPS = np.finfo(float).eps


class LossFunction(ABC):
    """
    Abstract base class for a loss function.
    
    It must provide methods for calculating the negative gradient (direction to move),
    the loss itself (metric to track), the initial prediction (starting point),
    and the hessian (second derivative).
    """

    @abstractmethod
    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the negative gradient of the loss function.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: The negative gradient vector.
        """
        pass

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scalar loss value.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated loss.
        """
        pass

    @abstractmethod
    def initial_prediction(self, y: np.ndarray) -> float:
        """
        Calculate the initial prediction (base score).

        Args:
            y (np.ndarray): Target values.

        Returns:
            float: The initial prediction value.
        """
        pass

    @abstractmethod
    def hessian(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the Hessian (second derivative) of the loss function.

        Args:
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: The Hessian vector.
        """
        pass


class MSELoss(LossFunction):
    """
    Mean Squared Error for Regression.
    L(y, F) = 0.5 * (y - F)^2
    """

    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate negative gradient for MSE: y_true - y_pred.
        """
        return y_true - y_pred

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MSE loss: mean((y_true - y_pred)^2).
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def initial_prediction(self, y: np.ndarray) -> float:
        """
        Initial prediction for MSE is the mean of target values.
        """
        return float(np.mean(y))

    def hessian(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Hessian for MSE is constant 1.
        """
        # The second derivative of 0.5*(y-pred)^2 with respect to pred is constant 1
        return np.ones_like(y_pred)


class LogLoss(LossFunction):
    """
    Log Loss for Binary Classification.
    Predictions are in log-odds space (logits).
    """

    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate negative gradient for LogLoss: y_true - sigmoid(y_pred).
        """
        return y_true - expit(y_pred)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Log Loss.
        """
        # Convert log-odds to probability
        p = expit(y_pred)
        # Clip to avoid log(0) error
        p = np.clip(p, EPS, 1 - EPS)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def initial_prediction(self, y: np.ndarray) -> float:
        """
        Initial prediction for LogLoss is the log-odds of the mean probability.
        """
        # Initial guess is the log-odds of the mean probability
        p = np.mean(y)
        # Avoid division by zero for pure classes
        p = np.clip(p, EPS, 1 - EPS)
        return float(np.log(p / (1 - p)))

    def hessian(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Hessian for LogLoss: p * (1 - p).
        """
        # Second Derivative (Hessian): p * (1 - p)
        p = expit(y_pred)
        return p * (1 - p)
