# Unit Simplex Projection

This document explains the mathematical foundations, intuition, and algorithmic implementation of projecting an arbitrary vector onto the unit probability simplex.

## 1. What is the "Probability" Simplex?

A vector $x = [x_1, x_2, \dots, x_n]$ represents a valid discrete probability distribution if and only if:
1. Every probability is non-negative: $x_i \ge 0$ for all $i$.
2. The probabilities sum to $1$: $\sum_{i=1}^n x_i = 1$.

Geometrically, the **unit (or standard) simplex** is defined precisely by these two conditions. When we perform a "projection onto the unit simplex", we are finding the closest valid probability distribution to an arbitrary vector $v$ in the Euclidean ($L^2$) sense. This naturally creates a "sparsity effect," where non-likely options are assigned exactly $0$ probability (unlike Softmax, which leaves tiny residual probabilities everywhere).

**Formulation:**
$$ \min_x \frac{1}{2} \|x - v\|_2^2 \quad \text{subject to } \sum_i x_i = 1 \text{ and } x_i \ge 0 $$

---

## 2. The Lagrangian and Multipliers

To solve this constrained optimization problem, we formulate the Lagrangian:
$$ L(x, \lambda, \mu) = \frac{1}{2} \|x - v\|_2^2 + \lambda (\mathbf{1}^T x - 1) - \mu^T x $$

### Why are the Multiplier Coefficients Different?

- **The Equality Constraint ($\lambda$)**: There is only one global condition ($\sum x_i = 1$), so it requires a single scalar multiplier $\lambda \in \mathbb{R}$. This multiplier acts as a uniform shift (the "shadow price" or "cost" of the sum-to-one constraint).
- **The Inequality Constraint ($\mu$)**: The $x_i \ge 0$ requirement is actually $n$ separate independent constraints. Thus, it requires a vector of $n$ multipliers, $\mu = [\mu_1, \dots, \mu_n]^T$ where $\mu_i \ge 0$.
- **Signs**: By standard KKT convention, we rewrite inequalities as $g(x) \le 0$. Our constraint is $-x_i \le 0$, which yields the $+ \mu_i (-x_i) = - \mu^T x$ term.

---

## 3. KKT Conditions

### Stationarity
Taking the gradient of the Lagrangian with respect to $x$ and setting it to zero:
$$ \nabla_x L = (x - v) + \lambda \mathbf{1} - \mu = 0 \implies x_i = v_i - \lambda + \mu_i $$

### Complementary Slackness
Complementary Slackness provides the "either/or" rule for inequality constraints:
$$ \mu_i \cdot x_i = 0 \quad \text{for all } i $$

Because $\mu_i$ and $x_i$ must both be non-negative, at least one must be zero:
1. **Inactive Constraint ($x_i > 0$)**: The value naturally wants to be positive, so the boundary exerts no force ($\mu_i = 0$). Hence, $x_i = v_i - \lambda$.
2. **Active Constraint ($x_i = 0$)**: The value wants to go negative, so the boundary activates. The multiplier $\mu_i$ becomes positive, pushing back with exactly enough force ($\mu_i = \lambda - v_i$) to pin the value at zero.

Combining these states gives the exact thresholding solution:
$$ x_i = \max(v_i - \lambda, 0) $$

---

## 4. The "Water-Filling" Analogy

We must find the correct $\lambda$ such that the non-zero $x_i$ values sum to $1$. You can visualize solving for $\lambda$ as a "Water-Filling" (or "Flood") operation:

1. **The Landscape**: Imagine each original value $v_i$ as a solid vertical pillar mapping a given terrain.
2. **The Water Level**: A flood sweeps in. The horizontal water level is exactly $\lambda$.
3. **What Protrudes?**: The equation $x_i = \max(v_i - \lambda, 0)$ mathematically asks: "How much of pillar $i$ is sticking out above the water?"
   - **Pillar is taller than the water level** ($v_i > \lambda$): The exposed tip is $x_i$.
   - **Pillar is submerged** ($v_i \le \lambda$): You can't see it ($x_i = 0$).
4. **Enforcing the Constraint**: The goal is to raise or lower the water level $\lambda$ until the total combined height of all the exposed pillar tips equals exactly $1$.

---

## 5. Efficient $O(n \log n)$ "Sort-and-Pivot" Algorithm

The water-filling physics perfectly models the algorithmic solution:
1. **Sort**: Sort the values (pillars) descending into a vector $u$. We know the water will submerge the shortest pillars first.
2. **Pivot/Cutoff ($\rho$)**: Scan down from the tallest pillar. Continuously check: *"If I lower the water enough to fully expose the next tallest pillar, will the total exposed height exceed $1$?"*
   - Mathematically, find the last index $\rho$ where $u_\rho > \frac{1}{\rho}\left(\sum_{j=1}^\rho u_j - 1\right)$.
3. **Compute $\lambda$**: Every pillar shorter than $\rho$ will be fully submerged. We ignore them and compute the exact water level $\lambda$ using only the $\rho$ surviving tall pillars:
   $$ \lambda = \frac{1}{\rho} \left(\sum_{j=1}^\rho u_j - 1\right) $$
4. **Threshold**: Output the exposed pieces: $x = \max(v - \lambda, 0)$.
