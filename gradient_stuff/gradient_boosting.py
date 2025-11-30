"""
Module implementing a simple Gradient Boosting framework with
pluggable loss functions and base estimator (weak learners).
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from .loss import MSELoss, LossFunction


class GradientBoosting:
    def __init__(
        self,
        loss: LossFunction = MSELoss(),
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
    ):
        """Vanilla Gradient Boosting framework.

        Args:
            loss (LossFunction, optional): Loss function to be optimized. Defaults to MSELoss().
            learning_rate (float, optional): Iteration shrinkage constant. Defaults to 0.1.
            n_estimators (int, optional): The number of boosting stages to perform. Defaults to 100.
            max_depth (int, optional): Maximum depth of the individual regression estimators. Defaults to 3.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth

        # Internal memory
        self.initial_pred_ = None
        self.trees_ = []
        self.gammas_ = []
        self.train_losses_ = []

    def fit(self, X, y):
        # 1. Initial Prediction (Base score)
        self.initial_pred_ = self.loss.initial_prediction(y)
        current_pred = np.full(len(y), self.initial_pred_)

        # 2. Boosting Loop
        for i in range(self.n_estimators):
            # A. Calculate Gradient (Pseudo-Residuals)
            residuals = self.loss.negative_gradient(y, current_pred)

            # B. Fit a Decision Tree to the Residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            new_pred = tree.predict(X)

            # C. Newton-Raphson Step
            # Use the Hessian to approximate the step.
            # Gamma = Sum(Gradient * TreePred) / Sum(Hessian * TreePred^2)

            # 1. Get Hessians (2nd Derivative)
            hessians = self.loss.hessian(current_pred)

            # 2. Calculate the optimal step size analytically
            numerator = np.sum(residuals * new_pred)
            denominator = np.sum(hessians * (new_pred**2))

            # Avoid division by zero
            if denominator == 0:
                gamma = 0
            else:
                gamma = numerator / denominator

            # D. Update Predictions
            # Prediction = Old + (Learning Rate * Gamma * Tree)
            current_pred += self.learning_rate * gamma * new_pred

            # Store tree and gamma
            self.trees_.append(tree)
            self.gammas_.append(gamma)

            # Track loss
            self.train_losses_.append(self.loss.loss(y, current_pred))

        return self

    def predict(self, X):
        # Start with base prediction
        preds = np.full(len(X), self.initial_pred_)

        # Add contributions from all trees
        for tree, gamma in zip(self.trees_, self.gammas_):
            preds += self.learning_rate * gamma * tree.predict(X)

        return preds
