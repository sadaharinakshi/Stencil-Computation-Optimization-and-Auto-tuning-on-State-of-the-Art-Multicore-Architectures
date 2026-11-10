# 01. Project Overview & Objectives

**Project ID:** P13  
**Project Title:** Stencil Computation Optimization and Auto-tuning on State-of-the-Art Multicore Architectures  
**Team ID:** T09  
**Team Members:** Harinakshi S, Vaarshini R  
**Motif:** Structured Grids  
**Primary Technology:** C and OpenMP (Double Precision)  
**Execution Environment:** Windows 11, Intel Core i5-12450HX (8 cores / 12 threads), GCC 16.2.0 (w64devkit)

---

## 1. What is this Project?

This project is a high-performance computing (HPC) investigation into **3D Stencil Computations**. Specifically, we implement, optimize, numerically verify, and empirically benchmark a **3D 7-point Heat Diffusion Stencil** running on modern multicore CPU architectures.

In scientific and engineering simulations, heat diffusion describes how thermal energy spreads through a solid body over time. When modeled on a computer, space is divided into a three-dimensional grid of points (a structured Cartesian mesh). At every discrete time step, each point updates its temperature using its own previous temperature and the temperatures of its six immediate orthogonal neighbours (left, right, top, bottom, front, and back).

---

## 2. Why Are We Doing It? (The HPC Motivation)

In modern computing, CPUs can perform arithmetic operations (FLOPs) hundreds of times faster than computer memory (DRAM) can deliver data. This growing disparity is known as the **"Memory Wall"**.

Stencil computations are the quintessential example of **memory-bound workloads**:
* For every point in space, we perform only **8 arithmetic operations** (6 additions and 2 multiplications).
* But to perform those 8 operations, we must read **7 double-precision numbers** (56 bytes) and write **1 double-precision number** (8 bytes) = up to **64 bytes** of data.
* This yields an extremely low **Arithmetic Intensity (AI)** of:
  $$\text{AI} = \frac{8\text{ FLOPs}}{64\text{ Bytes}} = 0.125\text{ FLOP/Byte}$$

Because the CPU needs 8 bytes of memory transfer for every single floating-point operation, the execution speed is completely gated by the memory bus bandwidth, leaving the CPU's compute cores starved for data.

### The Central Engineering Question:
Can we use parallel programming (**OpenMP**), memory placement policies (**NUMA First Touch**), and cache locality optimization (**Genuine 3D Cache Blocking**) to alleviate the memory bottleneck and maximize the performance of a 3D heat stencil on modern multicore processors?

---

## 3. Real-World Applications of Stencil Computations

The mathematical and algorithmic patterns explored in this project are used in major scientific computing fields:
1. **Computational Fluid Dynamics (CFD):** Solving Navier-Stokes equations for airflow over aircraft wings and automotive bodies.
2. **Geophysical Seismic Modeling:** Simulating acoustic and elastic wave propagation through Earth's tectonic layers for earthquake prediction and oil/gas exploration.
3. **Meteorology & Climate Modeling:** Simulating atmospheric pressure, heat diffusion, and moisture transfer in global weather forecasting (e.g., ECMWF, NOAA).
4. **Biomedical Engineering:** Modeling bio-heat transfer in tissue during laser surgery or cancer hyperthermia therapy.
5. **Electromagnetics:** Finite-Difference Time-Domain (FDTD) simulation of antenna radiation and radar scattering.

---

## 4. The Professor's Exact Requirements

The academic project guidelines strictly mandate four core technical milestones:

1. **Implement a 3D 7-Point Heat Stencil:**
   * Double-precision arithmetic.
   * Single-threaded serial reference version.
   * Multi-threaded OpenMP parallel version.
   * Correct fixed Dirichlet boundary handling.
   * Flattened 1D row-major array indexing.

2. **Implement Genuine 3D Cache Blocking:**
   * Genuine 3D spatial tiling using three configurable block sizes: $BX, BY, BZ$.
   * Nested-loop structure: 3 outer tile loops and 3 inner element loops.
   * Correct boundary and partial-tile remainder handling.
   * Measurable comparison against unblocked code.

3. **Implement NUMA-Aware Memory Initialization (First Touch):**
   * Multi-threaded memory initialization using OpenMP static scheduling.
   * Threads touch and initialize the exact memory regions they later compute on.
   * Clear architectural explanation of NUMA behavior and single-socket UMA constraints.

4. **Perform Roofline Performance Analysis:**
   * Measure sustained memory bandwidth via STREAM Triad across memory hierarchy (L1, L2, L3, DRAM).
   * Measure empirical sustained compute ceiling.
   * Formulate arithmetic intensity bounds ($0.125$ to $0.500\text{ FLOP/B}$).
   * Generate dedicated log-log Roofline plot.
   * Identify and conclude the operational bottleneck.

---

## 5. Scope Boundaries: What We Explicitly Excluded & Why

To ensure the project remains clean, rigorous, and completely explainable during an academic viva, the following distracting or out-of-scope features were audited and eliminated:
* **No 2D Image Processing:** Removed Gaussian blur ($3\times3, 5\times5$), Sobel filtering, Laplacian operators, and image file I/O libraries (STB). The professor requires 3D heat diffusion, not 2D graphics.
* **No AVX2/SIMD Assembly or Intrinsics:** Hardware vectorization is handled portably by compiler flags (`-O3 -march=native`), keeping C code readable and portable.
* **No Manual Software Prefetching or Non-Temporal Streaming Stores:** These obscure the fundamental cache-blocking mechanics.
* **No Z-Plane Register Unrolling:** Unrolling is distinct from cache blocking. We focus purely on spatial 3D cache blocking.
* **No Automated Tuning Search Frameworks:** Complex parameter search scripts were removed in favor of direct, explainable manual tile sweeps ($16^3, 32\times16^2, 32^2\times16$, etc.).
* **No CUDA/GPU Programming:** The project motif is focused exclusively on multicore CPU shared-memory parallelism.

---

## 6. What We Have Implemented & Delivered

The completed codebase provides:
1. `src/kernels_3d_serial.c`: Serial reference stencil and deterministic array initializer.
2. `src/kernels_3d_opt.c`: OpenMP naive parallel, First Touch initialization, genuine 3D cache blocking ($BX \times BY \times BZ$), and combined FT+Blocked implementations.
3. `src/kernels_3d_dispatch.c`: Unified function-pointer dispatcher.
4. `include/common.h` & `include/kernels_3d.h`: 64-byte alignment utilities, 1D flat indexing macro, configuration structures, and analytical constants.
5. `bench/correctness_check.c`: 55 automated tests verifying numerical equivalence ($1.0 \times 10^{-12}$ tolerance) across power-of-2, prime, anisotropic, and odd grids.
6. `bench/benchmark_3d.c`: Five-way benchmark isolating initialization time from computation time, measuring speedup, efficiency, and thread scaling.
7. `bench/microbench_roofline.c`: STREAM Triad memory hierarchy benchmark and compute roof measurement.
8. `scripts/plot_roofline.py`: Dedicated Python Roofline and speedup plotting script.
9. `Makefile` & `make.bat`: Clean Windows build system.
10. `README.md` & `docs/viva-preparation/`: Exhaustive technical documentation and exam preparation guides.

