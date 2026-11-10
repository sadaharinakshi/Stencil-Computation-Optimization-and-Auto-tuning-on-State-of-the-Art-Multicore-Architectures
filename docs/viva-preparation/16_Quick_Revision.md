# 16. Quick Revision Cheat Sheet (One-Page High-Yield Summary)

Use this one-page cheat sheet for rapid revision right before your viva or presentation.

---

## 1. Core Project Facts
- **Project ID / Team ID:** P13 / Team T09 (Harinakshi S, Vaarshini R)
- **Title:** Stencil Computation Optimization on Multicore Architectures
- **Motif:** Structured Grids (Berkeley 13 Dwarfs)
- **Application:** 3D 7-Point Heat Diffusion Equation ($\frac{\partial u}{\partial t} = \alpha \nabla^2 u$)
- **Data Type:** IEEE 754 Double Precision (`double`, 8 bytes per cell)
- **Hardware Testbed:** Intel Core i5-12450HX (4 P-cores + 4 E-cores = 8 physical cores, 12 logical threads, 12 MB L3 cache, 1 NUMA node / UMA)

---

## 2. Four Professor's Requirements & Key Source Code Locations
1. **3D 7-Point Stencil:** [src/kernels_3d_serial.c](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/src/kernels_3d_serial.c) (`heat3d_serial`) and [src/kernels_3d_opt.c](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/src/kernels_3d_opt.c) (`heat3d_omp`).
2. **Genuine 3D Cache Blocking ($BX, BY, BZ$):** [src/kernels_3d_opt.c](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/src/kernels_3d_opt.c) (`heat3d_blocked`) using 6 nested loops with boundary clamping.
3. **NUMA-Aware First Touch:** [src/kernels_3d_opt.c](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/src/kernels_3d_opt.c) (`first_touch_init_3d`) using `#pragma omp parallel for schedule(static) collapse(2)`.
4. **Roofline Model:** [bench/microbench_roofline.c](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/bench/microbench_roofline.c) and [scripts/plot_roofline.py](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/scripts/plot_roofline.py).

---

## 3. Critical Formulas & Numbers

### 1D Index Formula (Row-Major):
$$\text{INDEX}(x, y, z, nx, ny) = ((z \times ny + y) \times nx + x)$$

### Stencil FLOP Count per Point:
$$\mathbf{8 \text{ FLOP}} \quad (1 \text{ multiply} + 5 \text{ adds} + 1 \text{ multiply} + 1 \text{ add})$$

### Arithmetic Intensity ($I = \text{FLOPs} / \text{Bytes}$):
- **Theoretical Upper Bound (Zero Reuse):** $8 / 64 = \mathbf{0.125 \text{ FLOP/B}}$
- **Theoretical Lower Bound (Ideal Reuse):** $8 / 16 = \mathbf{0.500 \text{ FLOP/B}}$
- **Actual Operational Intensity (x86-64):** $\approx \mathbf{0.34 \text{ FLOP/B}}$

### Measured Hardware Ceilings (STREAM Triad):
- **L1 Data Bandwidth:** $575.96 \text{ GB/s}$
- **L2 Cache Bandwidth:** $199.92 \text{ GB/s}$
- **L3 Cache Bandwidth:** $61.38 \text{ GB/s}$
- **Sustained DRAM Bandwidth:** $\mathbf{32.13 \text{ GB/s}}$
- **Empirical DP Compute Peak:** $36.50 \text{ GFLOP/s}$
- **Ridge Point ($I^*$):** $36.50 / 32.13 = \mathbf{1.136 \text{ FLOP/byte}}$

### Maximum Attainable DRAM Performance:
$$\text{DRAM Ceiling} = 32.13 \text{ GB/s} \times 0.50 \text{ FLOP/B} = \mathbf{16.07 \text{ GFLOP/s}}$$

---

## 4. Benchmark Summary ($200^3$ Grid, 10 Steps)

| Configuration | Init Time | Init Speedup | Compute Time | Achieved GFLOP/s | % of DRAM Ceiling |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Serial Reference** | 460 ms | 1.00x | 9.50 ms | 6.54 GF/s | 40.7% |
| **Naive OpenMP** | 451 ms | 1.02x | 5.70 ms | 10.89 GF/s | 67.8% |
| **OpenMP + First Touch** | **62 ms** | **7.42x** | **5.20 ms** | **11.94 GF/s** | **74.3%** |
| **OpenMP + 3D Blocked** | 483 ms | 0.95x | 6.50 ms | 9.55 GF/s | 59.4% |
| **OpenMP + FT + Blocked** | **53 ms** | **8.68x** | **6.30 ms** | **9.86 GF/s** | **61.4%** |

---

## 5. High-Frequency Viva "Gotchas"

| Trap / Examiner Question | High-Yield Defending Answer |
|:---|:---|
| *"Is your laptop NUMA?"* | **No, it is a single-socket UMA system (1 NUMA node).** First Touch accelerates init by 8.7x because it parallelizes OS page fault zeroing across 12 cores, and the code is structured correctly for multi-socket NUMA servers. |
| *"Why did 8 threads beat 12 threads?"* | **Intel hybrid P-core/E-core asymmetry.** Under `schedule(static)`, threads 9–12 run on slower E-cores, forcing the fast P-core threads to wait at the OpenMP barrier. |
| *"Why did blocking not beat naive on $200^3$?"* | **Single-step spatial streaming.** Naive OpenMP streams contiguous rows of 198 doubles with near-100% hardware prefetcher efficiency; blocking along $X$ ($BX=32$) truncates streams and adds 6-loop index branch overhead. Blocking shines on massive grids ($512^3+$) or with temporal blocking. |
| *"Is your code correct?"* | **Yes.** Automated 55-test suite verified against serial reference across power-of-2, prime ($67^3$), anisotropic ($113\times 97\times 83$), 4 tile sizes, and 5-step iterations with **$0.00\times 10^0$ error** (tolerance $1.0\times 10^{-12}$). |
| *"Is it compute or memory bound?"* | **Strictly Memory Bandwidth Bound.** Actual Arithmetic Intensity ($0.34 \text{ FLOP/B}$) is far below the machine ridge point ($1.136 \text{ FLOP/B}$). |

