"""
Numerical Optimization and Geometric Manifolds: Unit Simplex Projection

See `docs/simplex_projection.md` for a comprehensive mathematical 
derivation and explanation of the algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def project_simplex(v):
    """
    Project a vector v onto the unit probability simplex using the O(n log n) Sort-and-Pivot algorithm.

    Parameters
    ----------
    v : array-like, shape (n,)
        The vector to project.

    Returns
    -------
    x : ndarray, shape (n,)
        The projected vector on the unit simplex.
    """
    v = np.asarray(v)
    if v.ndim == 0:
        return np.ones_like(v)

    n = len(v)

    # 1. Sort v in descending order
    u = np.sort(v)[::-1]

    # 2. Compute cumulative sums of the sorted vector
    cssv = np.cumsum(u)

    # 3. Find the number of strictly positive components (rho)
    # the condition is u_j - 1/(j) * (cssv_j - 1) > 0. (with 1-based indexing for j)
    # In 0-based indexing: j goes from 1 to n, so we divide by index
    indices = np.arange(1, n + 1)
    cond = u - (cssv - 1.0) / indices > 0

    # 4. rho is the index of the last element where the condition holds
    rho = np.nonzero(cond)[0][-1]

    # 5. Compute the Lagrange multiplier (lambda_val)
    lambda_val = (cssv[rho] - 1.0) / (rho + 1.0)

    # 6. Thresholding (apply water-filling to get the final positive values)
    x = np.maximum(v - lambda_val, 0)
    return x


def test_project_simplex():
    """Run a few assertions to verify that project_simplex behaves correctly."""
    print("Running project_simplex assertions...")

    # Test 1: Uniform point (already on simplex)
    v1 = np.array([0.5, 0.5])
    x1 = project_simplex(v1)
    np.testing.assert_allclose(np.sum(x1), 1.0, err_msg="Sum must be 1")
    np.testing.assert_allclose(x1, v1)

    # Test 2: Needs uniform shifting (no clipping)
    v2 = np.array([1.0, 1.0, 1.0])
    x2 = project_simplex(v2)
    np.testing.assert_allclose(np.sum(x2), 1.0)
    np.testing.assert_allclose(x2, [1 / 3, 1 / 3, 1 / 3])

    # Test 3: Sparsity / Clipping (far off to one side)
    v3 = np.array([2.0, 0.2])
    x3 = project_simplex(v3)
    np.testing.assert_allclose(np.sum(x3), 1.0)
    np.testing.assert_allclose(x3, [1.0, 0.0])  # 0.2 got clipped

    # Test 4: Handling negative values
    v4 = np.array([-0.5, -0.1, -0.2])
    x4 = project_simplex(v4)
    np.testing.assert_allclose(np.sum(x4), 1.0)
    assert np.all(x4 >= 0), "All values must be non-negative"

    # Test 5: Arbitrary random vector
    v5 = np.random.randn(10) * 10
    x5 = project_simplex(v5)
    np.testing.assert_allclose(np.sum(x5), 1.0)
    assert np.all(x5 >= 0)

    print("All tests passed successfully!")


def visualize_projection(filename="simplex_projection.gif"):
    """
    Creates a 2D animation showing a gradient step landing outside the simplex
    and the projection path bringing it back to exactly the line x + y = 1.
    Highlights the sparsity effect when the path lands exactly on a corner (1,0).
    """
    print(f"Generating animation: {filename} ...")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Set up axes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.0, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Unit Simplex Projection in 2D (Sparsity Effect)", fontsize=14)

    # Draw simplex line (x+y=1 in the first quadrant)
    ax.plot(
        [0, 1], [1, 0], "g-", lw=4, label=r"Unit Simplex (Δ: $x_1+x_2=1, x \geq 0$)"
    )
    # Draw the extended line x+y=1 (dashed) to show constraint plane
    ax.plot(
        [-0.5, 2.5],
        [1.5, -1.5],
        "g--",
        alpha=0.3,
        label="Affine constraint ($x_1+x_2=1$)",
    )

    # Initial point exactly inside the simplex
    v0 = np.array([0.4, 0.6])
    ax.plot(v0[0], v0[1], "go", ms=6)  # start point marker

    # "Gradient step" to a point V outside the simplex.
    # Chosen to be very far to one side so it projects directly onto a corner.
    V = np.array([2.0, 0.2])

    # The projected point via our exact solver
    X = project_simplex(V)

    # Calculate lambda_opt to find the intermediate "dragged" point V_drag
    u = np.sort(V)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(1, len(V) + 1)
    cond = u - (cssv - 1.0) / indices > 0
    rho = np.nonzero(cond)[0][-1]
    lambda_val = (cssv[rho] - 1.0) / (rho + 1.0)

    # V_drag is where the lambda orthogonal projection pulls us before clipping (V - lambda)
    V_drag = V - lambda_val

    # Dynamic scatter points and lines for the animation
    (point,) = ax.plot([], [], "ro", ms=10, label="Current State")
    (path_line,) = ax.plot([], [], "r--", alpha=0.6, lw=2, label="Movement Path")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    fig.tight_layout()
    ax.grid(True, linestyle=":", alpha=0.6)

    # Prepare text box for dynamic updates explaining algorithm stages
    text_info = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9),
    )

    text_math = ax.text(0, 0, "", fontsize=12, fontweight="bold", color="black", zorder=10)

    def init():
        point.set_data([], [])
        path_line.set_data([], [])
        text_info.set_text("")
        text_math.set_text("")
        return point, path_line, text_info, text_math

    def update(frame):
        text_math.set_text("")
        # Phase 1: Frame 0 to 40 (Gradient Step from v0 to V)
        if frame <= 40:
            t = frame / 40.0
            curr = (1 - t) * v0 + t * V
            point.set_data([curr[0]], [curr[1]])
            path_line.set_data([v0[0], curr[0]], [v0[1], curr[1]])
            path_line.set_color("red")
            point.set_color("red")
            text_info.set_text(
                f"Stage 1: Gradient Step\n"
                f"Taking a step to $V$ outside Δ\n"
                f"$v = [{curr[0]:.2f}, {curr[1]:.2f}]$"
            )

        # Phase 2: Frame 41 to 55 (Pause at V outside)
        elif frame <= 55:
            curr = V
            point.set_data([curr[0]], [curr[1]])
            path_line.set_data([v0[0], curr[0]], [v0[1], curr[1]])
            text_info.set_text(
                f"Stage 2: Point out of bounds\n"
                f"Landed at $V = [{curr[0]:.2f}, {curr[1]:.2f}]$\n"
                f"Need to project back to Δ!"
            )

        # Phase 3: Frame 56 to 90 (Pull of λ: Orthogonal travel)
        elif frame <= 90:
            t = (frame - 55) / 35.0
            curr = (1 - t) * V + t * V_drag
            point.set_data([curr[0]], [curr[1]])

            # Show path from V to curr as orthogonal path (dashed purple)
            path_line.set_data([V[0], curr[0]], [V[1], curr[1]])
            path_line.set_color("purple")
            point.set_color("purple")

            text_math.set_position(((V[0] + curr[0]) / 2 + 0.05, (V[1] + curr[1]) / 2 + 0.05))
            text_math.set_color("purple")
            text_math.set_text(rf"$-\lambda \mathbf{{1}}$ ($\lambda={lambda_val:.2f}$ shift)")

            text_info.set_text(
                f"Stage 3: Pull of λ\n"
                f"Moving orthogonally toward $x_1+x_2=1$.\n"
                f"$v = [{curr[0]:.2f}, {curr[1]:.2f}]$"
            )

        # Phase 4: Frame 91 to 105 (Pause at V_drag in negative territory)
        elif frame <= 105:
            curr = V_drag
            point.set_data([curr[0]], [curr[1]])
            path_line.set_data([V[0], curr[0]], [V[1], curr[1]])
            text_info.set_text(
                f"Stage 4: Crossed the boundary!\n"
                f"V_drag = [{curr[0]:.2f}, {curr[1]:.2f}]\n"
                f"Notice that $x_2$ is negative!"
            )

        # Phase 5: Frame 106 to 140 (Reaction force μ pushes back up)
        elif frame <= 140:
            t = (frame - 105) / 35.0
            curr = (1 - t) * V_drag + t * X
            point.set_data([curr[0]], [curr[1]])

            # Show path from V_drag to curr as upward push (dashed blue)
            path_line.set_data([V_drag[0], curr[0]], [V_drag[1], curr[1]])
            path_line.set_color("blue")
            point.set_color("blue")

            text_math.set_position(((V_drag[0] + curr[0]) / 2 + 0.05, (V_drag[1] + curr[1]) / 2))
            text_math.set_color("blue")
            text_math.set_text(r"$+\mu$ (Upward force)")

            text_info.set_text(
                f"Stage 5: Reaction force μ\n"
                f"The $x_2=0$ solid wall pushes back!\n"
                f"$v = [{curr[0]:.2f}, {curr[1]:.2f}]$"
            )

        # Phase 6: Frame 141 to 160 (Highlight Sparsity at X)
        else:
            curr = X
            point.set_data([curr[0]], [curr[1]])
            point.set_color("blue")
            point.set_marker("*")
            point.set_markersize(15)
            text_info.set_text(
                f"Stage 6: Complete!\n"
                f"Sparsity Effect Hit: The projection\n"
                f"landed exactly on a corner!\n"
                f"$X = [{curr[0]:.2f}, {curr[1]:.2f}]$"
            )

        return point, path_line, text_info, text_math

    # 160 frames total
    ani = animation.FuncAnimation(fig, update, frames=160, init_func=init, blit=True)
    ani.save(filename, writer="pillow", fps=15)
    plt.close(fig)
    print(f"Animation saved to {filename}")


if __name__ == "__main__":
    test_project_simplex()
    visualize_projection()
