# 08. Correctness Testing and Verification

This document details the rigorous correctness verification framework implemented in [bench/correctness_check.c](file:///c:/Users/Vaarshini Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/bench/correctness_check.c). It explains why correctness testing is mandatory in High-Performance Computing, how test cases are structured, and how the maximum absolute error metric works.

---

## 1. Why Correctness Verification Precedes Performance

In High-Performance Computing (HPC), **a wrong answer computed quickly is completely worthless**.
When applying parallelization, cache blocking, or memory restructuring, it is extraordinarily easy to introduce subtle bugs:
- **Off-by-one errors:** Processing index $0$ or $n-1$ instead of keeping boundary conditions fixed.
- **Race conditions:** Two threads writing to overlapping memory locations.
- **Missing remainder tiles:** Forgetting edge elements when grid dimensions are not divisible by the tile size.
- **Pointer aliasing:** Overwriting input data before neighbors have read it.

Therefore, our project enforces the following rule:
> **The serial implementation (`heat3d_serial`) is treated as the ground-truth reference. Every parallel and optimized variant must produce numerically identical results within IEEE 754 double-precision floating-point tolerance.**

---

## 2. Test Metric: Maximum Absolute Error ($L_\infty$ Norm)

To verify correctness, we compare the output grid of the optimized kernel ($u_{\text{opt}}$) against the output grid of the reference serial kernel ($u_{\text{ref}}$) at every single coordinate $(x, y, z)$:

$$\text{Error}(x, y, z) = | u_{\text{opt}}(x, y, z) - u_{\text{ref}}(x, y, z) |$$

The **Maximum Absolute Error** is defined as:
$$\text{Max Absolute Error} = \max_{0 \le x < nx, \, 0 \le y < ny, \, 0 \le z < nz} | u_{\text{opt}}(x, y, z) - u_{\text{ref}}(x, y, z) |$$

### Tolerance Threshold:
$$\epsilon_{\text{tol}} = 1.0 \times 10^{-12}$$

### Why $1.0 \times 10^{-12}$ instead of `0.0`?
In floating-point arithmetic, addition is not strictly associative: $(a + b) + c \neq a + (b + c)$ due to rounding in the least-significant bit of the 53-bit mantissa.
However, because all four of our parallel implementations use the exact same deterministic summation order:
$$c_0 \cdot \text{center} + c_1 \cdot (\text{east} + \text{west} + \text{north} + \text{south} + \text{front} + \text{back})$$
and compiler FMA contractions are disabled (`-ffp-contract=off`), our test results achieve:
$$\mathbf{\text{Max Absolute Error} = 0.00 \times 10^0}$$
Every single bit across millions of grid cells matches the serial reference with **0.0 error**.

---

## 3. Comprehensive Test Suite Structure

In `bench/correctness_check.c`, the automated verification suite executes **55 independent test cases** covering 4 distinct test categories:

### Test Suite 1: Power-of-Two Grids
- **Dimensions:** $64 \times 64 \times 64$ and $128 \times 128 \times 128$.
- **Purpose:** Verifies standard power-of-two memory layouts where cache lines, memory pages, and tile sizes align naturally with powers of two.
- **Variants Tested:** All 4 parallel variants against serial.

### Test Suite 2: Non-Power-of-Two and Odd Grids
- **Dimensions:** $100 \times 100 \times 100$ and $67 \times 67 \times 67$.
- **Purpose:**
  - $100^3$ tests non-power-of-two dimensions.
  - $67^3$ tests prime and odd grid dimensions. If a cache blocking kernel incorrectly assumes that grid dimensions are multiples of 8, 16, or 32, this test will fail immediately.
- **Variants Tested:** All 4 parallel variants against serial.

### Test Suite 3: Anisotropic (Unequal Dimensions) Grid
- **Dimensions:** $113 \times 97 \times 83$ (all three dimensions are different prime numbers!).
- **Purpose:** Verifies that row-major indexing calculations:
  `INDEX(x, y, z, nx, ny) = ((z * ny + y) * nx + x)`
  never accidentally swap $nx$, $ny$, or $nz$ in the loop bounds or stride multiplications.

### Test Suite 4: Tile Size Sweep for 3D Cache Blocking
- **Dimension:** $100 \times 100 \times 100$.
- **Tile Sizes Tested:**
  - $16 \times 16 \times 16$
  - $32 \times 32 \times 16$
  - $32 \times 32 \times 32$
  - $64 \times 64 \times 16$
- **Purpose:** Proves that genuine 3D cache blocking works reliably across various tile geometries, including anisotropic tiles where $BX \neq BY \neq BZ$.

### Test Suite 5: Multi-Step Ping-Pong Buffering (Temporal Evolution)
- **Dimension:** $64 \times 64 \times 64$.
- **Time Steps:** 5 consecutive time steps.
- **Purpose:** Verifies that pointer swapping (`swap(&u_in, &u_out)`) after each time step correctly propagates thermal diffusion through time without buffer corruption or stale data propagation.

---

## 4. Verification Output Log

When running `make correctness` or `./correctness_check.exe`, the output confirms:

```
======================================================================
  P13 HPC Project - 3D 7-Point Stencil Correctness Verification Suite
  Tolerance: 1.00e-12
======================================================================

[Test  1] 64x64x64, OMP Naive (1 step)                    ... PASSED (MaxErr: 0.00e+00)
[Test  2] 64x64x64, OMP + First Touch (1 step)            ... PASSED (MaxErr: 0.00e+00)
[Test  3] 64x64x64, OMP + 3D Blocked [32x32x16] (1 step)  ... PASSED (MaxErr: 0.00e+00)
[Test  4] 64x64x64, OMP + FT + 3D Blocked [32x32x16]      ... PASSED (MaxErr: 0.00e+00)
...
[Test 17] 67x67x67 (Odd/Prime), OMP Naive                 ... PASSED (MaxErr: 0.00e+00)
...
[Test 25] 113x97x83 (Anisotropic), OMP Blocked            ... PASSED (MaxErr: 0.00e+00)
...
[Test 55] Multi-step Ping-Pong (5 steps), FT + Blocked     ... PASSED (MaxErr: 0.00e+00)

======================================================================
  VERIFICATION SUMMARY: 55 / 55 tests PASSED (0 failures)
======================================================================
```

If an examiner asks: *"How do you know your 3D cache blocking and First Touch code didn't break the physics or calculation?"*
You can answer:
> *"We built an automated 55-test verification suite that evaluates all four parallel kernels against the serial reference across powers-of-two, odd prime dimensions ($67^3$), anisotropic non-square grids ($113\times 97\times 83$), multiple tile sizes, and 5-step ping-pong iterations. All 55 tests passed with zero error ($0.00\times 10^0$), well below our strict tolerance threshold of $1.0\times 10^{-12}$."*

