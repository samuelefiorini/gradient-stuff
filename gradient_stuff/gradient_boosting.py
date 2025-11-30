"""
Module implementing a simple Gradient Boosting framework with
pluggable loss functions and base estimator (weak learners).
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone, BaseEstimator
from .loss import MSELoss, LossFunction
from typing import Optional


class GradientBoosting:
    def __init__(
        self,
        loss: LossFunction = MSELoss(),
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        base_estimator: Optional[BaseEstimator] = None,
    ):
        """Vanilla Gradient Boosting framework.

        Args:
            loss (LossFunction, optional): Loss function to be optimized. Defaults to MSELoss().
            learning_rate (float, optional): Iteration shrinkage constant. Defaults to 0.1.
            n_estimators (int, optional): The number of boosting stages to perform. Defaults to 100.
            base_estimator (Optional[BaseEstimator], optional): The base estimator to fit in a boosted fashion.
                Defaults to None corresponds to DecisionTreeRegressor.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss

        # The "Blueprint": An instance of a model that we will clone.
        # If None, default to a small Decision Tree.
        if base_estimator is None:
            self.base_estimator = DecisionTreeRegressor(max_depth=3)
        else:
            self.base_estimator = base_estimator

        # Internal memory
        self.initial_pred_ = None
        self.estimators_ = []
        self.train_losses_ = []

    def fit(self, X, y):
        # 1. Initialize with a constant value
        self.initial_pred_ = self.loss.initial_prediction(y)

        # Current prediction starts as the constant value for all samples
        current_pred = np.full(len(y), self.initial_pred_)

        # 2. The Boosting Loop
        for i in range(self.n_estimators):
            # A. Calculate the "Pseudo-Residuals"
            residuals = self.loss.negative_gradient(y, current_pred)

            # B. Clone the blueprint to create a fresh learner
            learner = clone(self.base_estimator)

            # Fit the learner to the residuals (X -> Residuals)
            learner.fit(X, residuals)

            # C. Update the model predictions
            learner_pred = learner.predict(X)
            current_pred += self.learning_rate * learner_pred

            # Store the learner
            self.estimators_.append(learner)

            # D. Track the loss (Generic now!)
            self.train_losses_.append(self.loss.loss(y, current_pred))

        return self

    def predict(self, X):
        # Start with the initial constant prediction
        y_pred = np.full(X.shape[0], self.initial_pred_)

        # Add the weighted contribution of every weak learner
        for learner in self.estimators_:
            y_pred += self.learning_rate * learner.predict(X)

        return y_pred
