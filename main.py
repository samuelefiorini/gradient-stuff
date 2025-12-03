"""
Main script to demonstrate the SimpleGradientBoosting model with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from gradient_stuff.loss import MSELoss, LogLoss
from gradient_stuff.gradient_boosting import GradientBoosting
from sklearn.datasets import make_moons


def main():
    # --- 1. REGRESSION (1D Data) ---
    print("Training Regression Model (1D)...")
    np.random.seed(42)
    X_reg = np.linspace(0, 10, 100).reshape(-1, 1)
    y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.2, X_reg.shape[0])

    gb_reg = GradientBoosting(
        n_estimators=50, learning_rate=0.2, max_depth=2, loss=MSELoss()
    )
    gb_reg.fit(X_reg, y_reg)

    # --- 2. CLASSIFICATION (2D Data) ---
    print("Training Classification Model (2D)...")
    # Using make_moons to generate a non-linear 2D dataset
    X_clf, y_clf = make_moons(n_samples=300, noise=0.25, random_state=42)

    # We use a slightly deeper tree (max_depth=2 or 3) to capture interactions between feature 1 and 2
    gb_clf = GradientBoosting(
        n_estimators=50, learning_rate=0.5, max_depth=2, loss=LogLoss()
    )
    gb_clf.fit(X_clf, y_clf)

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Regression Fit
    axes[0, 0].set_title("Regression Fit (1D)")
    axes[0, 0].scatter(X_reg, y_reg, color="black", alpha=0.5, label="Data")
    axes[0, 0].plot(
        X_reg, gb_reg.predict(X_reg), color="red", linewidth=2, label="Prediction"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Regression Loss
    axes[0, 1].set_title("Regression Loss Curve")
    axes[0, 1].plot(gb_reg.train_losses_, color="blue")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Classification Decision Boundary (2D)
    # Create a meshgrid to visualize the decision boundary
    x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
    y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    # Flatten grid, predict, and reshape
    Z_logits = gb_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_probs = 1 / (1 + np.exp(-Z_logits))  # Convert to probability
    Z_probs = Z_probs.reshape(xx.shape)

    axes[1, 0].set_title("Classification: 2D Decision Boundary")
    contour = axes[1, 0].contourf(
        xx, yy, Z_probs, alpha=0.7, cmap="RdBu_r", levels=np.linspace(0, 1, 11)
    )
    fig.colorbar(contour, ax=axes[1, 0], label="Probability")

    # Scatter plot of actual data points
    axes[1, 0].scatter(
        X_clf[:, 0], X_clf[:, 1], c=y_clf, edgecolors="white", cmap="RdBu_r", s=40
    )
    axes[1, 0].set_xlabel("Feature 1")
    axes[1, 0].set_ylabel("Feature 2")

    # Plot 4: Classification Loss
    axes[1, 1].set_title("Classification Loss Curve")
    axes[1, 1].plot(gb_clf.train_losses_, color="green")
    axes[1, 1].set_ylabel("LogLoss")
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure.png")
    print("Plot saved to figure.png")


if __name__ == "__main__":
    main()
