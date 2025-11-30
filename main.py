"""
Main script to demonstrate the SimpleGradientBoosting model with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from gradient_stuff.loss import MSELoss
from gradient_stuff.gradient_boosting import GradientBoosting


def main():
    # --- Generate Toy Data (Sine Wave + Noise) ---
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

    # --- Train the Custom Gradient Boosting Model ---
    gb = GradientBoosting(
        n_estimators=50,
        learning_rate=0.2,
        loss=MSELoss(),
        base_estimator=DecisionTreeRegressor(max_depth=2),
    )
    gb.fit(X, y)

    print(f"Tree Boosting finished. Final Loss: {gb.train_losses_[-1]:.4f}")

    # --- Visualization Code ---
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

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
        n_estimators=50, learning_rate=0.2, base_estimator=LinearRegression()
    )
    gb_linear.fit(X, y)
    print(f"Linear Boosting finished. Final Loss: {gb_linear.train_losses_[-1]:.4f}")


if __name__ == "__main__":
    main()
