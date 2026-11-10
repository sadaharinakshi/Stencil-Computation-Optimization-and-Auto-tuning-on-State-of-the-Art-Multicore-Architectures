# 11. Experimental Results and Architectural Interpretation

This document provides a detailed breakdown of all empirical benchmark data collected on the target test platform (Intel Core i5-12450HX, Windows 11). It synthesizes the grid scaling data, tile size sweeps, thread scaling experiments, and provides the architectural explanations for why each variant performed the way it did.

---

## 1. Test Hardware and Environment Specifications

- **Processor:** Intel Core i5-12450HX (Alder Lake Architecture)
- **Cores & Threads:** 8 physical cores (4 P-cores + 4 E-cores), 12 logical threads via Intel Hyper-Threading on P-cores.
  - P-cores: Golden Cove microarchitecture, 32 KB L1D / 1.25 MB L2 per core, up to 4.4 GHz turbo.
  - E-cores: Gracemont microarchitecture, 32 KB L1D / 2.0 MB L2 shared per 4-core cluster, up to 3.1 GHz turbo.
- **L3 Cache (LLC):** 12 MB shared Intel Smart Cache.
- **System Memory:** 16 GB DDR4/DDR5 dual-channel.
- **Measured Peak DRAM Bandwidth:** 32.13 GB/s (STREAM Triad microbenchmark).
- **Operating System:** Microsoft Windows 11 Home (64-bit).
- **Compiler:** GCC 16.2.0 (w64devkit distribution).
- **Compilation Flags:** `-O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude`.
- **NUMA Configuration:** Single-socket, **1 NUMA node (UMA)** confirmed via Windows API.

---

## 2. 5-Variant Benchmark Comparison on $128^3$ and $200^3$ Grids

### Benchmark Table: $200 \times 200 \times 200$ Grid (61.0 MB per array, 122 MB active footprint, 10 time steps)

| Variant ID | Configuration Description | Init Time (ms) | Init Speedup | Compute Time (ms) | Achieved GFLOP/s | Compute Speedup |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Serial Reference Baseline | 460.0 ms | 1.00x | 9.50 ms | 6.54 GF/s | 1.00x |
| **2** | Naive OpenMP (Static, Collapse 2) | 451.0 ms | 1.02x | 5.70 ms | 10.89 GF/s | 1.67x |
| **3** | **OpenMP + First Touch** | **62.0 ms** | **7.42x** | **5.20 ms** | **11.94 GF/s** | **1.83x** |
| **4** | OpenMP + 3D Blocked ($32 \times 32 \times 16$) | 483.0 ms | 0.95x | 6.50 ms | 9.55 GF/s | 1.46x |
| **5** | **OpenMP + FT + 3D Blocked** | **53.0 ms** | **8.68x** | **6.30 ms** | **9.86 GF/s** | **1.51x** |

### Benchmark Table: $128 \times 128 \times 128$ Grid (16.0 MB per array, 32 MB active footprint, 10 time steps)

| Variant ID | Configuration Description | Init Time (ms) | Compute Time (ms) | Achieved GFLOP/s | Compute Speedup |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | Serial Reference Baseline | 120.0 ms | 2.50 ms | 6.40 GF/s | 1.00x |
| **2** | Naive OpenMP | 118.0 ms | 1.50 ms | 10.67 GF/s | 1.67x |
| **3** | **OpenMP + First Touch** | **16.0 ms** | **1.40 ms** | **11.43 GF/s** | **1.79x** |
| **4** | OpenMP + 3D Blocked | 125.0 ms | 1.80 ms | 8.89 GF/s | 1.39x |
| **5** | **OpenMP + FT + 3D Blocked** | **14.0 ms** | **1.70 ms** | **9.41 GF/s** | **1.47x** |

---

## 3. Strong Scaling Analysis: Thread Count Sweeps

We evaluated strong scaling on a $160 \times 160 \times 160$ grid (31.25 MB per array, 62.5 MB total footprint) by varying OpenMP threads from 1 to 12.
Efficiency is computed using **self-relative parallel efficiency**:
$$E(P) = \frac{T(1)}{P \times T(P)} \times 100\%$$

| Threads ($P$) | Naive OMP Time | Naive Speedup | Naive Efficiency | Blocked OMP Time | Blocked Speedup | Blocked Efficiency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 4.50 ms | 1.00x | 100.0% | 6.80 ms | 1.00x | 100.0% |
| **2** | 3.30 ms | 1.36x | 68.2% | 4.50 ms | 1.51x | 75.6% |
| **4** | 2.90 ms | 1.55x | 38.8% | 3.90 ms | 1.74x | 43.6% |
| **8** | **2.80 ms** | **1.61x** | **20.1%** | 4.10 ms | 1.66x | 20.7% |
| **12** | 3.30 ms | 1.36x | 11.4% | **3.90 ms** | **1.74x** | **14.5%** |

### Key Observations from Thread Scaling:
1. **Saturation at 4 to 8 Threads:**
   Going from 1 thread (4.50 ms) to 4 threads (2.90 ms) achieves the vast majority of available speedup. Beyond 4 threads, the single dual-channel memory bus (~32 GB/s) becomes fully saturated with read/write requests.
2. **The 8-Thread vs. 12-Thread Asymmetry:**
   On Naive OpenMP, 8 threads runs in **2.80 ms**, whereas 12 threads slows down to **3.30 ms**.
   - *Why?* Because threads 9–12 run on Gracemont Efficient cores (E-cores) which have lower clock speeds and smaller caches. Under `schedule(static)`, all 12 threads receive equal-sized chunks of work. The 8 P-core threads finish early and waste time waiting at the OpenMP barrier for the 4 slower E-core threads.

---

## 4. Tile Size Sweep for 3D Cache Blocking

On a $160 \times 160 \times 160$ grid, we tested 4 different 3D tile configurations:

| Tile Size ($BX \times BY \times BZ$) | Tile Working Footprint | Compute Time (ms) | Achieved GFLOP/s |
|:---:|:---:|:---:|:---:|
| $16 \times 16 \times 16$ | 32.8 KB (fits in L1D) | 4.10 ms | 7.80 GF/s |
| **$32 \times 32 \times 16$ (Default)** | **131 KB (fits in L2)** | **3.90 ms** | **8.20 GF/s** |
| $32 \times 32 \times 32$ | 262 KB (fits in L2) | 4.20 ms | 7.61 GF/s |
| $64 \times 64 \times 16$ | 524 KB (fits in L2) | 4.40 ms | 7.27 GF/s |

### Architectural Insight:
- $32 \times 32 \times 16$ provides the optimal balance between keeping the tile working set inside the fast L2 cache (~1.25 MB per P-core) while keeping the contiguous inner $X$ loop long enough ($BX=32$) to allow SIMD vectorization and hardware prefetching.
- Extremely small tiles ($16^3$) incur too much loop indexing overhead.
- Overly large tiles ($64 \times 64 \times 16$) approach the capacity limits of the cache hierarchy.

