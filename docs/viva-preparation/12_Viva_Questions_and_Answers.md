# 12. Complete Viva Questions and Answers (50 Comprehensive Questions)

This document contains 50 university viva and oral examination questions and answers covering the P13 HPC project. The questions range from basic concepts to deep technical implementations.

---

### Section 1: Project Overview and General HPC Concepts (Q1 – Q10)

#### Q1: What is the main title and objective of your HPC project?
**Answer:** The project title is **"Stencil Computation Optimization and Auto-tuning on State-of-the-Art Multicore Architectures"** (Project ID: P13, Team ID: T09). Our primary objective is to implement, optimize, and analyze a 3D 7-point heat diffusion stencil using C and OpenMP, specifically evaluating the performance impacts of OpenMP parallelization, NUMA-aware First Touch initialization, and genuine 3D cache blocking, and identifying performance bottlenecks using the Roofline Model.

#### Q2: What motif from the Berkeley "13 Dwarfs" of scientific computing does your project belong to?
**Answer:** It belongs to the **Structured Grids** motif. In structured grids, computation proceeds over a regular, multidimensional spatial grid where each point is updated as a linear combination of its immediate spatial neighbors.

#### Q3: What is a stencil computation in scientific computing?
**Answer:** A stencil computation updates each element in a multidimensional array according to a fixed spatial pattern (the "stencil") using the values of neighboring elements from the previous time step. They are widely used in computational fluid dynamics (CFD), heat transfer, seismic modeling, image processing, and numerical solutions of partial differential equations (PDEs).

#### Q4: What physical equation does your 3D 7-point stencil simulate?
**Answer:** It simulates the 3D Heat Diffusion Equation:
$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)$$
Using explicit Forward Euler time-stepping and standard second-order central spatial finite differences.

#### Q5: Which points are accessed in a 3D 7-point stencil?
**Answer:** To update cell $(x, y, z)$, the stencil accesses:
1. The **center** cell: $(x, y, z)$
2. Two neighbors along $X$: **left/west** $(x-1, y, z)$ and **right/east** $(x+1, y, z)$
3. Two neighbors along $Y$: **down/south** $(x, y-1, z)$ and **up/north** $(x, y+1, z)$
4. Two neighbors along $Z$: **back** $(x, y, z-1)$ and **front** $(x, y, z+1)$

#### Q6: Why do you maintain boundary values rather than updating them?
**Answer:** We enforce **Dirichlet boundary conditions**, where the temperature at the physical outer boundaries of the domain is held constant (e.g., fixed walls). Therefore, loops run strictly over interior points ($x, y, z \in [1, n-2]$), preserving the initial boundary values.

#### Q7: Why do you use double precision (`double`) rather than single precision (`float`)?
**Answer:** Double precision (IEEE 754 64-bit float) provides 53 bits of significand precision (~15–17 decimal digits), which prevents catastrophic cancellation and error accumulation over hundreds of thousands of simulation time steps in real-world numerical scientific modeling.

#### Q8: What are the four core requirements set by your professor?
**Answer:**
1. Implement a 3D 7-point heat stencil (serial reference and OpenMP parallel).
2. Implement genuine 3D cache blocking using $BX, BY, BZ$.
3. Implement NUMA-aware memory initialization using OpenMP First Touch.
4. Perform Roofline analysis to identify the hardware performance bottleneck.

#### Q9: What features did you explicitly remove during the project simplification?
**Answer:** We removed 2D image processing (Gaussian blur, Sobel, Laplacian), SIMD/AVX2 intrinsics, manual prefetching, non-temporal stores, Z-plane unrolling, automated heuristic tuning scripts, and CUDA, keeping the project 100% focused on the professor's four requirements.

#### Q10: What is double buffering (ping-pong buffering)?
**Answer:** In iterative time-stepping, reading and writing to the same array causes data hazards (reading updated values instead of previous-step values). Double buffering allocates two identical arrays (`u_in` and `u_out`). In step $t$, the kernel reads from `u_in` and writes to `u_out`. At the end of the step, the pointers are swapped so that `u_out` becomes the input for step $t+1$.

---

### Section 2: Data Layout and Memory Indexing (Q11 – Q18)

#### Q11: How is the 3D grid stored in memory?
**Answer:** As a single, contiguous 1D flattened array of size $nx \times ny \times nz$ allocated with 64-byte alignment using `aligned_alloc()` (or `_aligned_malloc()` on Windows).

#### Q12: Why use a 1D flattened array instead of pointer-to-pointers (`double***`)?
**Answer:** A `double***` requires allocating $nz$ pointers, each pointing to $ny$ pointers, each pointing to $nx$ doubles. This causes:
1. Massive pointer-chasing overhead (3 pointer dereferences per element access).
2. Fragmented heap memory, ruining spatial locality.
3. Inability for the compiler to vectorize or prefetch memory.
A 1D flattened array is contiguous in physical memory, highly cache-friendly, and vectorizable.

#### Q13: What is the exact formula used to convert 3D coordinates $(x, y, z)$ to a 1D index?
**Answer:** We use row-major ordering:
$$\text{INDEX}(x, y, z, nx, ny) = ((z \times ny + y) \times nx + x)$$
In C: `#define INDEX(x, y, z, nx, ny) ((((size_t)(z) * (size_t)(ny) + (size_t)(y)) * (size_t)(nx)) + (size_t)(x))`

#### Q14: Which coordinate has unit stride (stride-1) in your memory layout?
**Answer:** The $X$ coordinate. Adjacent values $(x, y, z)$ and $(x+1, y, z)$ sit in consecutive 8-byte memory addresses.

#### Q15: What is the memory stride between adjacent $Y$ neighbors $(x, y, z)$ and $(x, y+1, z)$?
**Answer:** The stride is $nx \times 8$ bytes (one full row of doubles).

#### Q16: What is the memory stride between adjacent $Z$ neighbors $(x, y, z)$ and $(x, y, z+1)$?
**Answer:** The stride is $nx \times ny \times 8$ bytes (one full 2D plane of doubles).

#### Q17: Why do you cast coordinate variables to `size_t` inside the `INDEX` macro?
**Answer:** For large 3D grids (e.g., $600 \times 600 \times 600 = 216,000,000$ doubles), the linear index exceeds $2^{31} - 1$ ($2.14 \times 10^9$ bytes). If evaluated with standard 32-bit signed integers, arithmetic overflow occurs, causing memory corruption or segmentation faults. `size_t` is 64-bit on x86-64 platforms.

#### Q18: What is the purpose of the C99 `restrict` keyword on array pointers?
**Answer:** `restrict` tells the C compiler that the pointers (`u_in` and `u_out`) do not alias each other in memory. This guarantees to the compiler that writing to `u_out[center]` will never alter the value read from any `u_in[...]` address, enabling aggressive register caching and SIMD instruction scheduling.

---

### Section 3: OpenMP Parallelization (Q19 – Q26)

#### Q19: What OpenMP directive did you use to parallelize the naive stencil?
**Answer:**
```c
#pragma omp parallel for schedule(static) collapse(2) default(none) \
    shared(u_in, u_out, nx, ny, nz, c0, c1)
```

#### Q20: What does `collapse(2)` do and why did you choose 2 instead of 1 or 3?
**Answer:** `collapse(2)` merges the two outermost loops ($Z$ and $Y$) into a single iteration space of size $(nz-2) \times (ny-2)$.
- Collapsing 2 loops provides thousands of fine-grained iterations, giving perfect load balance across all 12 threads.
- We do not collapse 3 because keeping the innermost $X$ loop separate preserves contiguous unit-stride memory access for hardware prefetchers and vectorization.

#### Q21: Why is `schedule(static)` used instead of `schedule(dynamic)`?
**Answer:**
1. Stencil work per grid cell is strictly uniform (zero branches, identical FLOP count), so dynamic scheduling would only add work-queue lock overhead.
2. `schedule(static)` guarantees deterministic, reproducible thread-to-iteration mapping, which is required for NUMA First Touch memory alignment.

#### Q22: What does `default(none)` do?
**Answer:** It forces the programmer to explicitly specify the data-sharing clause (`shared` or `private`) for every variable in the construct. Any unlisted variable triggers a compile-time error, preventing accidental race conditions or uninitialized variables.

#### Q23: Why are the loop index variables ($z, y, x$) not listed in `private(...)`?
**Answer:** Because they are declared inside the loop header (e.g., `for (size_t z = 1; ...)`), which according to the OpenMP specification automatically gives them private scope per thread.

#### Q24: Is there any race condition when multiple threads read from `u_in`?
**Answer:** No. Multiple threads can read from the same memory location simultaneously without any synchronization or data hazard. Race conditions only occur when at least one thread writes to a shared location concurrently.

#### Q25: Why is there no race condition on `u_out`?
**Answer:** Because each thread is assigned a mutually exclusive, non-overlapping subset of coordinates $(x, y, z)$. Every index `u_out[center]` is written by exactly one thread.

#### Q26: How are time steps synchronized?
**Answer:** There is an implicit barrier at the end of `#pragma omp parallel for`. All threads must finish computing step $t$ before the master thread swaps the pointers and launches step $t+1$.

---

### Section 4: First Touch and NUMA Architecture (Q27 – Q33)

#### Q27: What is NUMA?
**Answer:** Non-Uniform Memory Access. A computer architecture used in multi-socket servers where each CPU socket has its own dedicated local memory controller and DRAM. Accessing local DRAM is fast; accessing remote DRAM attached to another socket across an interconnect (e.g., Intel UPI) has higher latency and limited bandwidth.

#### Q28: What is the "First Touch" page allocation policy?
**Answer:** When virtual memory is allocated via `malloc()`, physical memory pages are not assigned immediately. The operating system kernel assigns a physical DRAM page frame only when a memory page is first written to (page faulted). Under First Touch, the kernel allocates that page in the DRAM bank local to the CPU core executing the writing thread.

#### Q29: What happens if you initialize memory serially on a NUMA system?
**Answer:** The master thread touches every page during initialization, causing 100% of the physical memory pages to be placed on Socket 0's local DRAM. During parallel computation, worker threads on other sockets must fetch all their data across the inter-socket bus, degrading performance.

#### Q30: How does your project implement First Touch?
**Answer:** In `first_touch_init_3d()`, we initialize the array using `#pragma omp parallel for schedule(static) collapse(2)` with the exact same thread count, loop structure, and scheduling as the compute kernel.

#### Q31: Is your test machine a NUMA system?
**Answer:** No. We queried the Windows operating system API (`GetNumaHighestNodeNumber`), which reported 0 (meaning **1 NUMA node**). The Intel Core i5-12450HX is a single-socket processor with an integrated memory controller, making it a **UMA (Uniform Memory Access)** system.

#### Q32: If your machine is UMA, why did First Touch show an ~8x initialization speedup?
**Answer:** Because First Touch parallelizes the **operating system page fault handling and page zeroing routines** across all 12 CPU threads simultaneously, and parallelizes memory bus write traffic during array initialization, reducing init time from 460 ms to 53 ms.

#### Q33: How would you verify NUMA on a Linux HPC cluster?
**Answer:** Using the Linux command `numactl --hardware` or `lscpu`, which lists the number of NUMA nodes, socket distribution, and memory distance matrix.

---

### Section 5: 3D Cache Blocking (Q34 – Q41)

#### Q34: What is 3D cache blocking (tiling)?
**Answer:** It is a loop transformation technique that partitions a 3D grid into smaller sub-volumes of size $BX \times BY \times BZ$ so that the active working set of grid points fits entirely inside the CPU cache (L1 or L2), eliminating capacity cache misses.

#### Q35: How many nested loops are in your 3D cache-blocked kernel?
**Answer:** **6 nested loops**:
- 3 outer tile loops: `bz_start`, `by_start`, `bx_start` (step size: tile dimensions).
- 3 inner element loops: `z`, `y`, `x` (traversing elements inside the tile).

#### Q36: How do you handle grid dimensions that are not evenly divisible by the tile size?
**Answer:** We use boundary clamping:
`const size_t bz_end = (bz_start + bz < nz - 1) ? bz_start + bz : nz - 1;`
This ensures partial remainder tiles at the grid boundaries are processed safely without accessing out-of-bounds memory.

#### Q37: What default tile size did you use?
**Answer:** $BX = 32$, $BY = 32$, $BZ = 16$. This creates a tile working set of $32 \times 32 \times 16 \times 8 \text{ bytes} \approx 131 \text{ KB}$, which fits comfortably within the 1.25 MB L2 cache of the Intel P-cores.

#### Q38: Why is genuine 3D blocking better than 2D (Y-Z) blocking?
**Answer:** 2D blocking leaves the $X$ dimension unblocked. If the grid is very wide ($nx = 2048$ or $4096$), a single row exceeds L1 cache capacity. 3D blocking tiles $X$ as well, ensuring cache residency regardless of domain size.

#### Q39: Why did 3D blocking not beat naive OpenMP on your $200^3$ benchmark?
**Answer:**
1. Blocking along $X$ truncates continuous streaming rows into short 32-element segments, disrupting the hardware stream prefetcher.
2. The 6-nested-loop structure introduces index clamping and branch overhead.
3. For a single time step, every point in the global array must still be loaded from DRAM at least once. Cache blocking provides its greatest advantages on massive grids ($512^3+$) or when combined with temporal blocking across multiple time steps.

#### Q40: What is temporal cache blocking?
**Answer:** Computing multiple consecutive time steps ($t \dots t+K$) on a single cache-resident tile before writing the results back to DRAM. This reuses data in cache multiple times, dramatically increasing Arithmetic Intensity.

#### Q41: Why did you implement spatial blocking instead of temporal blocking?
**Answer:** The professor's syllabus specifically requested spatial 3D cache blocking ($BX, BY, BZ$). Spatial blocking is simpler, preserves clear code structure, and directly illustrates cache tiling principles without the complex skewed diamond time-space dependencies of temporal blocking.

---

### Section 6: Roofline Model and Performance Analysis (Q42 – Q50)

#### Q42: What is Arithmetic Intensity?
**Answer:** The ratio of total floating-point operations (FLOPs) to total bytes transferred between the processor and main DRAM memory:
$$I = \frac{\text{FLOPs}}{\text{Bytes transferred}}$$
Units: FLOPs/byte.

#### Q43: How many FLOPs are in one update of your 3D 7-point stencil?
**Answer:** Exactly **8 FLOPs**: 1 multiply ($c_0 \cdot \text{center}$), 5 additions (summing 6 neighbors), 1 multiply ($c_1 \cdot \text{sum}$), and 1 final addition.

#### Q44: What are the theoretical memory traffic bounds for your stencil?
**Answer:**
- **Zero Cache Reuse (Upper Bound):** $7 \text{ reads} \times 8\text{B} + 1 \text{ write} \times 8\text{B} = 64 \text{ bytes/point}$. Arithmetic Intensity = $8/64 = \mathbf{0.125 \text{ FLOP/B}}$.
- **Ideal Cache Reuse (Lower Bound):** $1 \text{ read} \times 8\text{B} + 1 \text{ write} \times 8\text{B} = 16 \text{ bytes/point}$. Arithmetic Intensity = $8/16 = \mathbf{0.500 \text{ FLOP/B}}$.
- **Actual x86-64 Operational Intensity:** $\approx \mathbf{0.34 \text{ FLOP/B}}$ due to 64-byte cache line spatial reuse along $X$ and write-allocate traffic.

#### Q45: What was the measured peak DRAM bandwidth of your machine?
**Answer:** **32.13 GB/s**, measured using the STREAM Triad microbenchmark with OpenMP on a 96 MB buffer exceeding L3 cache.

#### Q46: What is the Ridge Point of your test machine?
**Answer:**
$$I^* = \frac{\text{Peak Compute}}{\text{Peak Bandwidth}} = \frac{36.50 \text{ GFLOP/s}}{32.13 \text{ GB/s}} = \mathbf{1.136 \text{ FLOP/byte}}$$

#### Q47: Is your 3D heat stencil compute-bound or memory-bound? Prove it.
**Answer:** It is strictly **Memory Bandwidth Bound**.
Our stencil's operational Arithmetic Intensity is $\approx 0.34 \text{ FLOP/byte}$, which is far below the machine's ridge point of $1.136 \text{ FLOP/byte}$ ($0.34 \ll 1.136$). Therefore, performance is bounded by the slanted memory bandwidth roof, not the CPU execution units.

#### Q48: What was your maximum achieved stencil performance, and what fraction of the DRAM ceiling did it reach?
**Answer:** Our OpenMP + First Touch implementation achieved **11.94 GFLOP/s** on the $200^3$ grid. At an ideal Arithmetic Intensity of $0.50 \text{ FLOP/B}$, the maximum attainable DRAM performance is $32.13 \text{ GB/s} \times 0.50 = 16.07 \text{ GFLOP/s}$. Our kernel achieved:
$$\frac{11.94}{16.07} \approx \mathbf{74.3\% \text{ of the physical memory bandwidth ceiling}}$$

#### Q49: How was correctness verified?
**Answer:** We implemented an automated 55-case test suite in `correctness_check.c` that compares every parallel variant against the serial reference across power-of-two grids, odd prime grids ($67^3$), anisotropic non-square grids ($113\times 97\times 83$), 4 tile sizes, and 5-step time iterations. All 55 tests passed with a Maximum Absolute Error of **$0.00 \times 10^0$** (tolerance $1.0 \times 10^{-12}$).

#### Q50: If you had more time or a GPU, what would be the next optimization?
**Answer:**
1. Implement **Temporal Tiling** (trapezoidal or diamond tiling) to reuse data across multiple time steps in cache, pushing Arithmetic Intensity from 0.34 up to 1.5+ FLOP/B.
2. Port the stencil to CUDA or OpenMP target offloading for execution on an NVIDIA GPU, leveraging high-bandwidth GPU memory (HBM/GDDR) offering 800+ GB/s of bandwidth.

