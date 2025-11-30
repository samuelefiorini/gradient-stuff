import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from abc import ABC, abstractmethod

# ==========================================
# 1. The "Goal": Pluggable Loss Functions
# ==========================================


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

    def negative_gradient(self, y_true, y_pred):
        # Convert log-odds to probability
        p = 1 / (1 + np.exp(-y_pred))
        return y_true - p

    def loss(self, y_true, y_pred):
        # Convert log-odds to probability
        p = 1 / (1 + np.exp(-y_pred))
        # Clip to avoid log(0) error
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def initial_prediction(self, y):
        # Initial guess is the log-odds of the mean probability
        p = np.mean(y)
        # Avoid division by zero for pure classes
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))


# ==========================================
# 2. The "Meta" Algorithm: Gradient Boosting
# ==========================================


class SimpleGradientBoosting:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, loss=MSELoss(), base_estimator=None
    ):
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


# ==========================================
# 3. Demonstration & Visualization
# ==========================================

if __name__ == "__main__":
    # --- Generate Toy Data (Sine Wave + Noise) ---
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

    # --- Train the Custom Gradient Boosting Model ---
    gb = SimpleGradientBoosting(
        n_estimators=50,
        learning_rate=0.2,
        loss=MSELoss(),
        base_estimator=DecisionTreeRegressor(max_depth=2),
    )
    gb.fit(X, y)

    # --- Visualization Code ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: The Final Fit
    axes[0].scatter(X, y, color="black", label="Data", s=15, alpha=0.6)
    axes[0].plot(
        X, gb.predict(X), color="red", linewidth=2, label="Final GB Prediction"
    )
    axes[0].set_title("Final Model Fit")
    axes[0].legend()

    # Plot 2: How the model evolved (Step-by-step)
    axes[1].scatter(X, y, color="gray", alpha=0.3, s=10)
    current_pred = np.full(len(y), gb.initial_pred_)
    steps_to_plot = [0, 2, 10, 49]
    colors = ["green", "blue", "orange", "red"]

    for i, learner in enumerate(gb.estimators_):
        current_pred += gb.learning_rate * learner.predict(X)
        if i in steps_to_plot:
            axes[1].plot(
                X,
                current_pred,
                label=f"Iter {i + 1}",
                color=colors[steps_to_plot.index(i)],
                linewidth=1.5,
            )

    axes[1].set_title("Evolution of Prediction")
    axes[1].legend()

    # Plot 3: Training Loss Curve (Now generic!)
    axes[2].plot(gb.train_losses_, color="purple", linewidth=2)
    axes[2].set_title("Training Loss Curve")
    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # --- Demonstrate "Meta" Nature ---
    print("\n--- Training Component-wise Boosting (Linear Base Learner) ---")
    gb_linear = SimpleGradientBoosting(
        n_estimators=50, learning_rate=0.1, base_estimator=LinearRegression()
    )
    gb_linear.fit(X, y)
    print(f"Linear Boosting finished. Final Loss: {gb_linear.train_losses_[-1]:.4f}")
