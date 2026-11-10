# Stencil Computation Optimization on Multicore Architectures
## 3D 7-Point Heat Diffusion Stencil with OpenMP, NUMA First-Touch & 3D Cache Blocking

* **Project ID:** P13
* **Project Title:** Stencil Computation Optimization and Auto-tuning on State-of-the-Art Multicore Architectures
* **Team ID:** T09
* **Team Members:** Harinakshi S, Vaarshini R
* **Motif:** Structured Grids
* **Primary Technology:** C & OpenMP (Standard IEEE 754 Double Precision)

---

## 1. Project Overview & Physical Problem

This project implements, optimizes, and analyzes a **3D 7-point Heat Diffusion Stencil** on multicore CPU architectures. Heat diffusion in a solid medium is governed by the continuous heat conduction differential equation. Discretizing the spatial and temporal domains using standard second-order central finite differences yields the explicit 7-point stencil update equation.

In a 3D Cartesian grid of dimensions $NX \times NY \times NZ$, every interior grid cell $(x, y, z)$ updates its temperature based on its current temperature and the temperatures of its 6 immediate orthogonal neighbours:
* **Left:** $(x - 1, y, z)$
* **Right:** $(x + 1, y, z)$
* **Top:** $(x, y - 1, z)$
* **Bottom:** $(x, y + 1, z)$
* **Back:** $(x, y, z - 1)$
* **Front:** $(x, y, z + 1)$

### Mathematical Formulation
$$\begin{aligned}
T_{\text{new}}[x,y,z] = (1 - 6\alpha) \cdot T[x,y,z] + \alpha \cdot \Big( & T[x-1,y,z] + T[x+1,y,z] \\
+ & T[x,y-1,z] + T[x,y+1,z] \\
+ & T[x,y,z-1] + T[x,y,z+1] \Big)
\end{aligned}$$

* **Diffusion Coefficient ($\alpha$):** Set to $\alpha = 0.10$, satisfying the numerical stability criterion for 3D explicit stencils ($\alpha \le 1/6 \approx 0.1667$).
* **Dirichlet Boundary Conditions:** Outer boundary cells ($x=0, x=NX-1, y=0, y=NY-1, z=0, z=NZ-1$) remain fixed at their initial values.
* **Double Buffering:** Two separate allocated grids (`current_grid` and `next_grid`) are used to prevent read-after-write data hazards. Pointers are swapped after each time step.

---

## 2. 3D Grid Representation & 1D Flat Indexing

The 3D grid is stored in contiguous, 64-byte cache-aligned memory using row-major ordering. Z is the outermost plane dimension, Y is the row dimension, and X is the innermost column dimension:

```c
#define INDEX(x, y, z, nx, ny) (((size_t)(z) * (size_t)(ny) + (size_t)(y)) * (size_t)(nx) + (size_t)(x))
```

* **Spatial Locality:** Accessing adjacent elements along the X dimension ($x$ and $x \pm 1$) has unit stride ($\Delta \text{stride} = 1$). Each 64-byte cache line loaded brings in 8 contiguous `double` values.
* **Strided Dimensions:** Y neighbours are separated by $\pm NX$ elements, and Z neighbours are separated by $\pm (NX \times NY)$ elements.

---

## 3. The Five Implementation Variants

The project implements five progressive configurations to isolate the performance impact of parallel execution, memory placement, and cache blocking:

### 1. Serial Reference (`heat3d_serial`)
* Single-threaded reference implementation.
* Traverses the grid with three nested loops: $Z$ outer, $Y$ middle, $X$ inner.
* Acts as the gold standard for all correctness verifications.

### 2. Basic OpenMP (`heat3d_omp`)
* Parallelizes the outermost spatial dimension ($Z$-planes) across threads:
  ```c
  #pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int z = 1; z < nz - 1; ++z) { ... }
  ```
* Static scheduling partitions contiguous slices of $Z$-planes deterministically across threads.

### 3. OpenMP + First Touch (`heat3d_omp_numa`)
* On NUMA (Non-Uniform Memory Access) systems, physical memory pages are allocated on the NUMA node of the thread that **first writes** to them.
* Uses OpenMP parallel loop with `schedule(static)` to initialize arrays, ensuring threads touch the same memory regions they subsequently compute on.
* **Important Hardware Note:** On single-socket client processors (such as Intel Core i5 with 1 NUMA node / UMA), memory access latency is uniform. First Touch parallelization still yields a massive ~9x speedup in initialization time (page fault handling and memory write saturation), but does not alter physical NUMA locality because only one memory controller exists.

### 4. OpenMP + Genuine 3D Cache Blocking (`heat3d_blocked`)
* Solves the capacity eviction problem inherent in 3D stencils.
* In an unblocked $200^3$ grid, each plane is $200 \times 200 \times 8\text{ B} = 320\text{ KB}$. Accessing $(z-1)$, $z$, and $(z+1)$ sweeps megabytes of data through cache, evicting previous lines before temporal reuse occurs.
* **Genuine 3D blocking** partitions the volume into small 3D tiles of size $BX \times BY \times BZ$:
  ```c
  /* 3 Outer Tile Loops */
  #pragma omp parallel for num_threads(nthreads) schedule(static) collapse(2)
  for (int bz_start = 1; bz_start < nz - 1; bz_start += tile_z) {
      for (int by_start = 1; by_start < ny - 1; by_start += tile_y) {
          for (int bx_start = 1; bx_start < nx - 1; bx_start += tile_x) {
              /* 3 Inner Element Loops */
              for (int z = bz_start; z < z_end; ++z) {
                  for (int y = by_start; y < y_end; ++y) {
                      for (int x = bx_start; x < x_end; ++x) {
                          out[...] = c0 * center + c1 * neighbours;
                      }
                  }
              }
          }
      }
  }
  ```
* For tile size $32 \times 32 \times 16$: working set $\approx 3 \times (32 \times 32 \times 16) \times 8\text{ B} \approx 393\text{ KB}$, fitting comfortably inside L2/L3 cache slices!

### 5. OpenMP + First Touch + 3D Cache Blocking (`heat3d_numa_blocked`)
* Combines parallel First Touch initialization with genuine 3D cache-blocked computation.

---

## 4. Analytical Arithmetic Intensity & Memory Traffic Model

### FLOP Count per Interior Update
Each interior grid update requires:
* 5 additions (summing 6 neighbours)
* 1 multiplication by $\alpha$
* 1 multiplication of center cell by $(1 - 6\alpha)$
* 1 addition (center term + neighbour term)
* **Total = 8 FLOPs per interior update** (6 additions + 2 multiplications).

### Theoretical Memory Traffic vs. Actual DRAM Traffic
1. **Theoretical Worst-Case Upper Bound (No Cache Reuse):**
   * 7 reads ($7 \times 8\text{ B} = 56\text{ B}$) + 1 write ($8\text{ B}$) = **64 Bytes/update**.
   * $\text{Arithmetic Intensity}_{\text{no-reuse}} = \frac{8\text{ FLOPs}}{64\text{ Bytes}} = \mathbf{0.125\text{ FLOP/Byte}}$.
   *(With write-allocate, writing an uncached line adds another 8 B load = 72 B).*
2. **Theoretical Best-Case Lower Bound (Ideal Cache Reuse):**
   * Each cell loaded from DRAM once ($8\text{ B}$) and written to DRAM once ($8\text{ B}$) = **16 Bytes/update**.
   * $\text{Arithmetic Intensity}_{\text{ideal}} = \frac{8\text{ FLOPs}}{16\text{ Bytes}} = \mathbf{0.500\text{ FLOP/Byte}}$.
3. **Actual DRAM Traffic:**
   * Falls strictly between 16 B and 64 B. The hardware spatial cache line fetch (64 bytes = 8 doubles) guarantees that 7 out of 8 contiguous X loads hit in L1 cache regardless of blocking. 3D cache blocking extends this reuse to the Y and Z dimensions.

---

## 5. Numerical Correctness Verification

The verification suite (`bench/correctness_check.c`) compares all 4 parallel implementations against the serial reference:
$$\text{Max Absolute Error} = \max_{i} |T_{\text{reference}}[i] - T_{\text{parallel}}[i]|$$

### Verification Results (Double-Precision Tolerance: $1.0 \times 10^{-12}$)
| Test Grid Dimensions | Grid Type | Tile Sizes Tested | Max Absolute Error | Status |
|---|---|---|---|---|
| $64 \times 64 \times 64$ | Power-of-2 | $16^3, 32^2\times16, 64\times32\times16, 16\times32^2$ | $0.00 \times 10^0$ | **PASSED** |
| $67 \times 67 \times 67$ | Odd / Prime | $16^3, 32^2\times16, 64\times32\times16, 16\times32^2$ | $0.00 \times 10^0$ | **PASSED** |
| $100 \times 100 \times 100$ | Non-power-of-2 | $16^3, 32^2\times16, 64\times32\times16, 16\times32^2$ | $0.00 \times 10^0$ | **PASSED** |
| $113 \times 97 \times 83$ | Anisotropic Primes | $16^3, 32^2\times16, 64\times32\times16, 16\times32^2$ | $0.00 \times 10^0$ | **PASSED** |
| $128 \times 128 \times 128$ | Standard Divisible | $16^3, 32^2\times16, 64\times32\times16, 16\times32^2$ | $0.00 \times 10^0$ | **PASSED** |
| Multi-Step (5 steps) | Temporal Double Buffer | $32 \times 32 \times 16$ | $0.00 \times 10^0$ | **PASSED** |

**Total: 55 / 55 tests passed with zero error.**

---

## 6. Empirical Benchmark Results

Measured on 12th Gen Intel Core i5-12450HX (12 OpenMP threads, 10 time-steps, median of 10 trials):

### A. Five-Way Comparison (Grid: $200 \times 200 \times 200$, 61.0 MB/array)
| Implementation | Init Time (ms) | Compute Time (ms) | Achieved GFLOP/s | Effective Bandwidth | Speedup | Parallel Efficiency |
|---|---|---|---|---|---|---|
| **1. Serial Reference** | 460.0 ms | 9.50 ms | 6.54 GF/s | 13.07 GB/s | 1.00x | 100.0% |
| **2. OpenMP (Naive)** | 451.0 ms | 5.70 ms | 10.89 GF/s | 21.79 GB/s | 1.67x | 13.9% |
| **3. OpenMP + First Touch** | **62.0 ms** | **5.20 ms** | **11.94 GF/s** | **23.88 GB/s** | **1.83x** | **15.2%** |
| **4. OpenMP + 3D Blocked** | 483.0 ms | 6.50 ms | 9.55 GF/s | 19.11 GB/s | 1.46x | 12.2% |
| **5. OpenMP + FT + 3D Blocked** | **53.0 ms** | 6.30 ms | 9.86 GF/s | 19.71 GB/s | 1.51x | 12.6% |

### Key Viva Insights:
1. **Initialisation Impact:** First Touch reduces memory initialization time from ~450–480 ms down to ~53–62 ms (**~8x faster**) by distributing physical page fault servicing and memory writes across all 12 cores simultaneously.
2. **DRAM Bandwidth Saturation:** OpenMP achieves up to 23.88 GB/s sustained memory throughput, saturating the laptop's dual-channel DDR4/DDR5 DRAM bus.

### B. Strong Thread Scaling (Grid: $160 \times 160 \times 160$)
| Threads | Naive OpenMP Compute (ms) | Naive Speedup | Naive Efficiency | 3D Blocked Compute (ms) | 3D Blocked Speedup | 3D Blocked Efficiency |
|---|---|---|---|---|---|---|
| 1 | 4.50 ms | 1.00x | 100.0% | 6.80 ms | 1.00x | 100.0% |
| 2 | 3.30 ms | 1.36x | 68.2% | 4.50 ms | 1.51x | 75.6% |
| 4 | 2.90 ms | 1.55x | 38.8% | 3.90 ms | 1.74x | 43.6% |
| 8 | 2.80 ms | **1.61x** | 20.1% | 4.10 ms | 1.66x | 20.7% |
| 12 | 3.30 ms | 1.36x | 11.4% | 3.90 ms | **1.74x** | 14.5% |

*Self-relative parallel efficiency: $\text{Efficiency}(P) = \frac{T(1)}{P \times T(P)} \times 100\%$, where $T(1)$ is the 1-thread execution of that specific algorithm.*

### Architectural Insight: Why 8 Threads Outperforms 12 Threads on Naive OpenMP
The Intel Core i5-12450HX is a **hybrid processor** with 4 Performance Cores (8 threads via Hyper-Threading) and 4 Efficient Cores (4 single threads). Under static OpenMP scheduling, each thread receives an equal number of planes ($160 / 12 \approx 13$ planes). Because E-cores operate at lower clock speeds and IPC, faster P-core threads finish early and must wait at the implicit barrier for the slower E-core threads. At 8 threads, threads run exclusively on faster cores, avoiding this core-asymmetry synchronization penalty.

### C. 3D Tile Size Exploration ($160^3$ Grid)
| Tile Dimensions ($BX \times BY \times BZ$) | Tile Working Set (KB) | Compute Time (ms) | Achieved GFLOP/s | Speedup vs. 1-Thread Blocked |
|---|---|---|---|---|
| $16 \times 16 \times 16$ | 96.0 KB | 3.70 ms | 8.53 GF/s | 1.84x |
| **$32 \times 16 \times 16$** | **192.0 KB** | **3.70 ms** | **8.53 GF/s** | **1.84x (Optimal Tile)** |
| $32 \times 32 \times 16$ | 384.0 KB | 3.90 ms | 8.09 GF/s | 1.74x |
| $64 \times 32 \times 16$ | 768.0 KB | 3.70 ms | 8.53 GF/s | 1.84x |
| $32 \times 32 \times 32$ | 768.0 KB | 4.40 ms | 7.17 GF/s | 1.55x |
| $64 \times 64 \times 16$ | 1536.0 KB | 4.20 ms | 7.51 GF/s | 1.62x |

---

## 7. Roofline Model & Bottleneck Analysis

### A. Measured Hardware Ceilings
* **STREAM Triad Measured Memory Bandwidth:**
  * L1 Data Cache: **575.96 GB/s**
  * L2 Cache: **199.92 GB/s**
  * L3 Cache: **61.38 GB/s**
  * Sustained DRAM Bandwidth: **32.13 GB/s**
* **Compute Performance Ceilings:**
  * Theoretical Max Turbo Peak (Intel i5-12450HX): **380.80 GFLOP/s** *(Documented hardware spec)*
  * Theoretical Base Frequency Peak: **211.20 GFLOP/s**
  * Empirical Double-Precision Compute Peak: **36.50 GFLOP/s** *(Measured multi-threaded FMA)*

### B. DRAM Ridge Point
$$\text{Ridge Point} = \frac{\text{Empirical Compute Roof}}{\text{DRAM Bandwidth}} = \frac{36.50\text{ GFLOP/s}}{32.13\text{ GB/s}} = \mathbf{1.14\text{ FLOP/Byte}}$$
*(Or relative to theoretical base peak: $211.2 / 32.13 = \mathbf{6.57\text{ FLOP/Byte}}$).*

### C. Roofline Attainable Ceilings (Evaluated against Physical DRAM Ceiling)
The physical DRAM bandwidth of $32.13\text{ GB/s}$ dictates an attainable ceiling of **$16.07\text{ GFLOP/s}$** at ideal cache reuse ($0.500\text{ FLOP/B}$) and **$4.02\text{ GFLOP/s}$** at zero reuse ($0.125\text{ FLOP/B}$):

| Variant | Measured GFLOP/s | Operational AI Range (FLOP/B) | DRAM Roof (Ideal Reuse) | % of Ideal DRAM Roof | Performance Bottleneck |
|---|---|---|---|---|---|
| **1. Serial Reference** | 6.54 GF/s | 0.125 – 0.500 | 16.07 GF/s | 40.7% | **Memory Bandwidth Bound** |
| **2. OpenMP (Naive)** | 10.89 GF/s | 0.125 – 0.500 | 16.07 GF/s | 67.8% | **Memory Bandwidth Bound** |
| **3. OpenMP + First Touch** | 11.94 GF/s | 0.125 – 0.500 | 16.07 GF/s | 74.3% | **Memory Bandwidth Bound** |
| **4. OpenMP + 3D Blocked** | 9.55 GF/s | 0.125 – 0.500 | 16.07 GF/s | 59.5% | **Memory Bandwidth Bound** |
| **5. OpenMP + FT + 3D Blocked** | 9.86 GF/s | 0.125 – 0.500 | 16.07 GF/s | 61.4% | **Memory Bandwidth Bound** |

### Understanding the Cache-Reuse Effect:
Why does naive OpenMP achieve $10.89\text{ GFLOP/s}$, which is above $4.02\text{ GFLOP/s}$?
The 4.02 GFLOP/s figure assumes **zero cache reuse** (every neighbour read misses and hits DRAM, 64 B/pt). In reality, hardware cache lines load 64 bytes (8 doubles) at once, ensuring that 7 out of 8 X-elements hit in L1 cache even without software blocking. This reduces actual DRAM traffic well below 64 B/pt, shifting operational intensity to $\approx 0.34\text{ FLOP/B}$, where $10.89\text{ GFLOP/s}$ operates at $\approx 67.8\%$ of the physical DRAM bus ceiling.

### D. Final Bottleneck Conclusion
Because the operational arithmetic intensity of the 3D 7-point stencil ($0.125 - 0.500\text{ FLOP/Byte}$) is far to the left of the DRAM ridge point ($1.14\text{ FLOP/Byte}$), **the 3D 7-point heat diffusion stencil is strictly MEMORY-BANDWIDTH BOUND**. Performance is gated by memory bus throughput, not floating-point execution units.

---

## 8. Build & Execution Instructions

The project uses the bundled MinGW-w64 toolchain in `tools/w64devkit/`:

```bash
# 1. Compile all binaries (zero warnings, -O3, OpenMP)
tools\w64devkit\bin\make.exe all

# 2. Run numerical correctness verification (55 tests)
tools\w64devkit\bin\make.exe correctness

# 3. Run 5-way performance benchmarks and scaling studies
tools\w64devkit\bin\make.exe bench

# 4. Measure STREAM memory hierarchy bandwidth and generate Roofline plots
tools\w64devkit\bin\make.exe roofline

# 5. Clean generated binaries
tools\w64devkit\bin\make.exe clean
```

---

## 9. Viva Preparation: Questions & Confident Answers

**Q1: What is a 7-point stencil?**  
> *A:* In a 3D grid, each interior cell updates using its own value plus its 6 immediate orthogonal neighbours (left, right, top, bottom, back, front). It performs 6 additions and 2 multiplications (8 FLOPs per point).

**Q2: Why is the 3D stencil memory-bandwidth bound?**  
> *A:* Its arithmetic intensity is very low (at most 0.5 FLOPs per byte transferred). The CPU can execute math much faster than RAM can supply the data.

**Q3: How does 3D cache blocking work?**  
> *A:* In large grids, entire 2D planes are too big for L2/L3 cache, causing back/front planes to be evicted before they can be reused. 3D blocking divides the grid into $BX \times BY \times BZ$ sub-cubes that fit in cache, preserving temporal locality.

**Q4: What is First Touch, and what does it do on NUMA vs. UMA?**  
> *A:* Operating systems assign physical pages to the NUMA node of the thread that first writes to them. On multi-socket NUMA systems, parallel initialisation ensures memory is local to each socket. On single-socket UMA systems, all cores share one memory controller, but parallel initialisation still speeds up array startup by ~9x through parallel page faulting.

**Q5: What is the Roofline Model?**  
> *A:* A performance model plotting GFLOP/s against Arithmetic Intensity (FLOP/byte). It has two ceilings: a slanted memory bandwidth ceiling and a flat compute ceiling. The intersection is the "ridge point". Since our stencil lies to the left of the ridge point, it is strictly memory-bound.
