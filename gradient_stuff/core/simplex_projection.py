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


