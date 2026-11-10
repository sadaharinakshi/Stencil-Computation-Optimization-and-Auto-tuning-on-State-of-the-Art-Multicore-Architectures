# 02. Heat Stencil Fundamentals & Mathematical Formulation

---

## 1. What is a Stencil Computation?

In numerical analysis and scientific computing, a **stencil computation** is an iterative geometric pattern used to update values on a multidimensional spatial grid.

To compute the new value of an element at grid coordinate $(x, y, z)$ at time $t + \Delta t$, the stencil function reads:
1. The element's own value at time $t$.
2. The values of a specific set of neighboring elements at fixed geometric offsets at time $t$.

### The "Pattern" Motif
In the Berkeley "13 Dwarfs / Motifs of Parallel Computing" classification, stencil computations belong to the **Structured Grids** motif. The computation exhibits:
* **Static, regular data dependencies:** Every interior point uses the exact same pattern of neighbours.
* **High spatial locality:** Neighbours are geographically close in the physical domain.
* **Low computational intensity:** Very few arithmetic calculations are performed per byte of data fetched from memory.

---

## 2. The Physical Phenomenon: 3D Heat Diffusion

Heat diffusion is governed by **Fourier's Law of Thermal Conduction** and the conservation of energy. In three-dimensional space, the continuous partial differential equation (PDE) for isotropic heat conduction is:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2} \right)$$

Where:
* $T(x,y,z,t)$ is the temperature field at position $(x, y, z)$ and time $t$.
* $\nabla^2$ is the spatial **Laplacian operator** ($\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$).
* $\alpha = \frac{k}{\rho c_p}$ is the **thermal diffusivity** of the material (governed by thermal conductivity $k$, density $\rho$, and specific heat capacity $c_p$).

---

## 3. Discretization: Deriving the 7-Point Stencil

To solve the continuous PDE on a computer, we discretize space into a uniform Cartesian grid with grid spacing $h = \Delta x = \Delta y = \Delta z$, and time into discrete intervals $\Delta t$.

### Central Finite Difference Approximation
Using second-order Taylor series expansion, the second spatial derivative along the $X$-axis is approximated as:
$$\frac{\partial^2 T}{\partial x^2} \approx \frac{T(x+h, y, z) - 2T(x,y,z) + T(x-h, y, z)}{h^2}$$

Applying the same approximation to the $Y$ and $Z$ axes:
$$\nabla^2 T \approx \frac{T_{x+1} + T_{x-1} + T_{y+1} + T_{y-1} + T_{z+1} + T_{z-1} - 6T_{\text{center}}}{h^2}$$

Using a forward Euler difference in time:
$$\frac{\partial T}{\partial t} \approx \frac{T^{t+\Delta t} - T^t}{\Delta t}$$

Substituting into the heat equation and rearranging terms:
$$T^{t+\Delta t} = T^t + \frac{\alpha \Delta t}{h^2} \Big( T_{x+1} + T_{x-1} + T_{y+1} + T_{y-1} + T_{z+1} + T_{z-1} - 6T^t \Big)$$

Let the dimensionless diffusion parameter be:
$$\gamma = \frac{\alpha \Delta t}{h^2}$$

Factoring the center term $T^t$:
$$T^{t+\Delta t} = (1 - 6\gamma) T^t + \gamma \Big( T_{x-1} + T_{x+1} + T_{y-1} + T_{y+1} + T_{z-1} + T_{z+1} \Big)$$

Renaming $\gamma$ to $\alpha$ for convenience in code:
$$\mathbf{T_{\text{new}}[x,y,z] = (1 - 6\alpha) T[x,y,z] + \alpha \Big( T_{\text{left}} + T_{\text{right}} + T_{\text{top}} + T_{\text{bottom}} + T_{\text{back}} + T_{\text{front}} \Big)}$$

---

## 4. The 3D 7-Point Geometric Molecule

The stencil is named **7-Point** because exactly seven spatial values contribute to each update (1 center point + 6 orthogonal neighbours):

```
                     Z+1 (Front)
                         |
                         |   Y-1 (Top)
                         |  /
                         | /
                         |/
    X-1 (Left) ------- Center (x,y,z) ------- X+1 (Right)
                        /|
                       / |
                      /  |
            Y+1 (Bottom) |
                         |
                     Z-1 (Back)
```

In memory coordinates:
* **Center:** `in[x, y, z]`
* **Left:** `in[x - 1, y, z]`
* **Right:** `in[x + 1, y, z]`
* **Top:** `in[x, y - 1, z]`
* **Bottom:** `in[x, y + 1, z]`
* **Back:** `in[x, y, z - 1]`
* **Front:** `in[x, y, z + 1]`

---

## 5. Numerical Stability and the Choice of Alpha ($\alpha$)

In explicit finite-difference time-domain (FDTD) schemes, the numerical solution can become wildly unstable (temperatures oscillating towards infinity) if the time step $\Delta t$ is too large relative to spatial grid spacing $h$.

By the **CFL (Courant-Friedrichs-Lewy) / Von Neumann Stability Criterion**, an explicit 3D heat stencil is numerically stable **if and only if**:
$$\alpha \le \frac{1}{2d} = \frac{1}{2 \times 3} = \frac{1}{6} \approx 0.16667$$

### Why We Chose $\alpha = 0.10$:
* Setting $\alpha = 0.10$ guarantees that:
  $$(1 - 6\alpha) = 1.0 - 0.60 = 0.40 > 0$$
* All coefficients in the convex combination are strictly positive ($0.40 > 0$ and $0.10 > 0$).
* Their sum is exactly $0.40 + 6 \times 0.10 = 1.00$.
* This guarantees that maximum temperatures cannot exceed the initial boundary range, preventing overflow and ensuring unconditional numerical stability.

---

## 6. Boundary Conditions: Dirichlet Boundaries

In physical simulations, boundaries can be:
1. **Dirichlet:** Fixed, constant temperature at the boundaries (e.g., maintaining a constant ice-bath temperature of $0^\circ\text{C}$ on outer walls).
2. **Neumann:** Insulated boundaries (zero heat flux, $\frac{\partial T}{\partial n} = 0$).

### In Our Implementation: Fixed Dirichlet Conditions
* The outermost boundary cells are defined as:
  * $z = 0$ (Back plane) and $z = NZ - 1$ (Front plane)
  * $y = 0$ (Top row) and $y = NY - 1$ (Bottom row)
  * $x = 0$ (Left column) and $x = NX - 1$ (Right column)
* **Rule:** Boundary cells are **never updated**. Their temperatures are copied directly from `in` to `out`:
  $$\text{out}[\text{boundary}] = \text{in}[\text{boundary}]$$
* Only strictly interior points ($1 \le x \le NX-2, 1 \le y \le NY-2, 1 \le z \le NZ-2$) execute the 7-point update.
* **Count of Interior Points:** $(NX - 2) \times (NY - 2) \times (NZ - 2)$.

---

## 7. Arithmetic Operation Count (FLOP Breakdown)

For every interior point updated, let us count the exact arithmetic operations:
1. `in[x-1] + in[x+1]` $\implies$ 1 addition
2. `+ in[y-1]` $\implies$ 1 addition
3. `+ in[y+1]` $\implies$ 1 addition
4. `+ in[z-1]` $\implies$ 1 addition
5. `+ in[z+1]` $\implies$ 1 addition (Sum of 6 neighbours complete = 5 additions)
6. `alpha * (sum_neighbours)` $\implies$ 1 multiplication
7. `(1 - 6*alpha) * center` $\implies$ 1 multiplication
8. `term_center + term_neighbours` $\implies$ 1 addition

$$\mathbf{\text{Total Arithmetic Cost} = 6\text{ Additions} + 2\text{ Multiplications} = 8\text{ FLOPs per interior point.}}$$

### Compiler Flags and FMA Contraction
In modern GCC, the flag `-ffp-contract=off` is explicitly passed in our `Makefile`. This ensures that GCC does **not** fuse multiplications and additions into hardware FMA instructions (which would combine 1 mul + 1 add into 1 opcode), guaranteeing that the analytical 8 FLOP count remains strictly preserved across all benchmark evaluations.

