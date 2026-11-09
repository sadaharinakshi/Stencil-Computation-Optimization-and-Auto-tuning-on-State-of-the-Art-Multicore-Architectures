# HPC Stencil Optimization Project Notes

## Machine Specifications (Auto-Detected)
- **CPU**: 12th Gen Intel(R) Core(TM) i5-1235U
- **Topology**: 10 physical cores (2 Performance-cores @ up to 4.4 GHz with SMT, 8 Efficient-cores @ up to 3.3 GHz) -> 12 logical hardware threads
- **Cache Hierarchy**:
  - L1 Data: 48 KB (P-core) / 32 KB (E-core)
  - L2: 1.25 MB per P-core, 2 MB shared per 4 E-core cluster
  - L3 (LLC): 12 MB shared Intel Smart Cache
- **Vector Extensions**: AVX2 (256-bit vector width: 8 single-precision float / 4 double-precision float), FMA3 (2 ops/cycle per vector lane). AVX-512 is not supported.
- **NUMA Nodes**: 1 (Single-socket Uniform Memory Access architecture).
  - *Note*: NUMA first-touch initialization is fully implemented in variant v2 as per the project specification, but its impact is expected to be minimal on this single-socket architecture compared to multi-socket NUMA systems.
- **Operating System**: Windows 11 64-bit
- **Compiler**: GCC 16.2.0 (`x86_64-w64-mingw32`), OpenMP 5.2
- **Python**: 3.13.2 (with NumPy, SciPy, Matplotlib, Seaborn, Pillow)

---

## Phase Log and Architectural Decisions

### Phase 3: Auto-Tuning and Parameter Space Exploration (COMPLETED)
- **Search Spaces Explored**:
  - Tile dimensions: $TX \in [16, 32, 64, 128, 256, 512]$, $TY \in [8, 16, 32, 64, 128, 256]$, $TZ \in [4, 8, 16, 32]$.
  - Thread counts: $1 \dots 12$ threads.
  - Scheduling strategies: Static, dynamic, guided with chunk sizes $0, 8, 16, 32$.
  - Vectorization: AVX2 SIMD enabled vs disabled.
  - Software Prefetch distance: $0, 8, 16, 32$ cache elements.
- **2D Tile Size Sweep (Gauss 5x5, 2048x2048)**:
  - Optimal Tile Size: **$TX = 64, TY = 32$**
  - Performance: **137.01 GFLOP/s** (1.50 ms), achieving **22.37 GB/s** effective memory throughput.
  - Saved heatmap: `plots/tuning_heatmap_2d.png` and `plots/tuning_heatmap_2d.pdf`.
- **Multi-Parameter Search Speedup**:
  - Untuned Baseline (1 thread, no SIMD): 49.00 ms (4.19 GFLOP/s).
  - Best Tuned Configuration (10 threads, SIMD on, Prefetch 16, $TX=256, TY=128$): 2.20 ms.
  - **Tuning Speedup: 22.27x speedup over untuned baseline!**
  - All trials recorded in `results/tuning_2d_trials.csv` and `results/tuning_2d_heatmap.csv`.
- **3D Heat Stencil Auto-Tuning ($128^3$)**:
  - Optimal blocking parameters: **$TZ = 16, TY = 64$**
  - Performance: **16.78 GFLOP/s** (1.00 ms), achieving **33.55 GB/s** effective bandwidth (L2/L3 cache reuse).
  - Saved 3D heatmap: `plots/tuning_heatmap_3d.png` and `plots/tuning_heatmap_3d.pdf`.

- **Variants Implemented and Verified**:
  - `v0_serial`: Serial naive baseline.
  - `v1_omp_naive`: OpenMP parallel-for with static schedule across rows/planes.
  - `v2_first_touch`: NUMA-aware first-touch initialization touching memory pages with the compute schedule.
  - `v3_tiling`: Spatial cache blocking ($TX \times TY$ in 2D, $TZ \times TY$ in 3D).
  - `v4_simd_window`: AVX2 `#pragma omp simd` unit-stride vectorization + sliding window register reuse across consecutive rows/planes.
  - `v5_prefetch_nt`: Software prefetching (`__builtin_prefetch`) + non-temporal streaming stores (`_mm_sfence`).
  - `v6_unroll_reg` (3D only): Datta et al. Z-unrolling by 2, computing $(z, z+1)$ simultaneously to share intermediate planes in registers and cut vertical loads by 33%.
- **Correctness Verification**:
  - All variants verified against serial reference (`correctness_check 6`).
  - Maximum absolute error: $0.000\text{e}+00$ across all kernels (tolerance: $10^{-5}$ float, $10^{-12}$ double).
- **Key 2D Measurements (4096 x 4096, 12 Threads)**:
  - **Gauss 5x5**:
    - v0 Serial: 183.0 ms (4.49 GFLOP/s, 0.73 GB/s)
    - v1 OMP Naive: 31.0 ms (26.52 GFLOP/s, 5.90x speedup)
    - v2 First Touch: 30.0 ms (27.40 GFLOP/s, 6.10x speedup)
    - v3 Tiling: 37.0 ms (22.22 GFLOP/s, 4.95x speedup)
    - v4 SIMD Window: 9.67 ms (85.04 GFLOP/s, 18.93x speedup)
    - v5 Prefetch/NT: **8.00 ms (102.76 GFLOP/s, 22.88x speedup)**
  - **Gauss 3x3**:
    - v0 Serial: 70.0 ms (4.07 GFLOP/s)
    - v1 OMP Naive: 16.0 ms (17.83 GFLOP/s, 4.38x)
    - v4 SIMD Window: **7.33 ms (38.89 GFLOP/s, 9.55x speedup)**
  - **Sobel 3x3**:
    - v0 Serial: 48.0 ms (5.59 GFLOP/s)
    - v1 OMP Naive: 9.67 ms (27.77 GFLOP/s, 4.97x)
    - v4 SIMD Window: **7.14 ms (37.58 GFLOP/s, 6.72x speedup)**
  - **Laplacian 3x3**:
    - v0 Serial: 30.0 ms (3.36 GFLOP/s)
    - v1 OMP Naive: 7.80 ms (12.91 GFLOP/s, 3.85x)
    - v4 SIMD Window: **7.00 ms (14.38 GFLOP/s, 4.29x speedup)**
- **Key 3D Measurements ($256^3$ Double Precision Grid, 128 MB buffer)**:
  - v0 Serial: 34.0 ms (3.95 GFLOP/s, 7.90 GB/s)
  - v1 OMP Naive: 22.0 ms (6.10 GFLOP/s, 1.55x)
  - v3 Tiling ($TZ=16, TY=32$): **16.0 ms (8.39 GFLOP/s, 16.78 GB/s, 2.12x speedup)**
  - v6 Z-Unrolling: 22.0 ms (6.10 GFLOP/s, 1.55x)
  - *Observation*: In 3D at $N=256$, cache blocking (v3) provides the highest speedup (2.12x) by keeping active planes in L2/L3 cache. Beyond 4 threads, DRAM bandwidth saturates at ~16.8 GB/s on the mobile DDR4/LPDDR5 memory bus.
- **Strong Scaling Observations**:
  - Memory-bound stencils (3D and 2D Gauss3/Laplacian) scale linearly up to 4-6 threads before hitting memory controller saturation.
  - Compute-intensive stencils (Gauss 5x5 with 49 flops/pixel) scale continuously across all 12 threads reaching 102.76 GFLOP/s.

- **Repo Structure**: Created standard HPC project directory layout:
  - `include/`: Common headers (`common.h`, `image_io.h`, `kernels_2d.h`, `kernels_3d.h`), stb libraries (`stb_image.h`, `stb_image_write.h`).
  - `src/`: Kernel implementations (`kernels_2d_serial.c`, `kernels_3d_serial.c`, `image_io.c`, `kernels_2d_opt.c`, `kernels_3d_opt.c`, dispatchers).
  - `bench/`: Benchmarks and validation harnesses (`correctness_check.c`, `synthetic_gen.c`, `benchmark_2d.c`, `benchmark_3d.c`, `microbench_roofline.c`).
  - `scripts/`: Detection and automation scripts (`detect_hardware.c`, tuning, plotting).
  - `results/`: CSV outputs and verification images.
  - `plots/`: Output charts and visualizations.
- **Kernels Implemented**:
  - `gauss3_v0_serial`: 3x3 Gaussian blur (17 FLOPs/pixel, 9 muls + 8 adds, border clamping).
  - `gauss5_v0_serial`: 5x5 Gaussian blur (49 FLOPs/pixel, 25 muls + 24 adds, border clamping).
  - `sobel_v0_serial`: 3x3 Sobel edge detection (Gx + Gy + magnitude sqrt, 16 FLOPs/pixel).
  - `laplace_v0_serial`: 3x3 discrete 5-point Laplacian (6 FLOPs/pixel).
  - `heat3d_v0_serial`: 3-D 7-point heat equation stencil on double precision (8 FLOPs/grid point as in Datta et al. SC 2008).
- **Image Generation & I/O**:
  - Single-channel float32 image representation with border clamping.
  - Synthetic images generated for sizes $1024^2$, $4096^2$, $8192^2$, and $16384^2$ (up to 1024 MB buffer) with Gaussian noise ($\sigma=20$) and Salt-and-Pepper noise ($p=0.05$).
  - Verified visual output PNGs saved to `results/visual_*.png`.
- **Validation**:
  - Correctness verification harness checks max absolute error ($< 10^{-5}$ for single precision, $< 10^{-12}$ for double precision) and relative $L_2$ error. Reference serial v0 passed.

## Phase 4: Roofline Model & Bottleneck Analysis

- **Hardware Memory Subsystem Detection**:
  - Detected 16 GB Dual-Channel DDR4/LPDDR4 memory:
    - Slot 1: SK Hynix HMAA1GS6CJR6N-XN (8 GB, 3200 MT/s).
    - Slot 2: Micron Technology MT40A1G16TB-062E:F (8 GB, 3200 MT/s).
  - Theoretical peak DRAM bandwidth: $2 \times 64\text{ bits} \times 3200\text{ MT/s} = 51.2\text{ GB/s}$.
  - Measured OpenMP STREAM Triad bandwidth: **$20.93\text{ GB/s}$** on 720 MB array ($>4\times$ the 12 MB L3 cache).

- **Theoretical & Measured Compute Ceilings**:
  - **Platform**: Intel Core i5-1235U Hybrid Architecture (2 Performance Golden Cove cores + 8 Efficient Gracemont cores):
    - P-cores: 2 FMA units $\times$ 8 SP vector lanes $\times$ 2 FLOPs = $32\text{ SP FLOP/cycle}$.
    - E-cores: 1 FMA unit $\times$ 8 SP vector lanes $\times$ 2 FLOPs = $16\text{ SP FLOP/cycle}$.
  - **Theoretical Max Turbo Peak**:
    - P-cores @ 4.4 GHz: $2 \times 4.4 \times 32 = 281.6\text{ GFLOP/s}$.
    - E-cores @ 3.3 GHz: $8 \times 3.3 \times 16 = 422.4\text{ GFLOP/s}$.
    - **Total Turbo Peak**: **$704.00\text{ GFLOP/s}$**.
  - **Theoretical Base Clock Peak**:
    - P-cores @ 1.3 GHz: $2 \times 1.3 \times 32 = 83.2\text{ GFLOP/s}$.
    - E-cores @ 0.9 GHz: $8 \times 0.9 \times 16 = 115.2\text{ GFLOP/s}$.
    - **Total Base Peak**: **$198.40\text{ GFLOP/s}$**.
  - **Empirical AVX2 FMA Compute Ceilings**:
    - Initial Burst (~0.3s runtime): **$558.76\text{ GFLOP/s}$**.
    - **Sustained Compute Roof ($\ge 10\text{s}$ continuous run with 12 threads)**: **$471.68\text{ GFLOP/s}$** (throttled slightly from burst due to sustained multi-core power/thermal limits).
  - **Empirical Scalar Peak (no SIMD)**: **$37.09\text{ GFLOP/s}$** (8 independent scalar accumulators).

- **Memory & Cache Hierarchy Bandwidth (STREAM Triad with QueryPerformanceCounter)**:
  - **L1 Data Cache** (~24 KB/thread working set): **$774.43\text{ GB/s}$**.
  - **L2 Cache** (~1 MB/thread working set): **$134.06\text{ GB/s}$**.
  - **L3 Cache** (2.88 MB/thread slice of 12 MB shared L3): **$39.82\text{ GB/s}$**.
  - **DRAM** (60 MB/thread = 720 MB total array $>4\times$ L3): **$20.93\text{ GB/s}$**.
  - **Hierarchy Verification**: $\text{L1 } (774.4) > \text{L2 } (134.1) > \text{L3 } (39.8) > \text{DRAM } (20.9\text{ GB/s})$ — **PASSED**.
  - **Machine Balance Knee Point**: $\approx 22.5\text{ FLOP/byte}$ ($471.68\text{ GFLOP/s} / 20.93\text{ GB/s}$).

- **Arithmetic Intensity (AI) Analytical Formulations**:
  - **Gauss 3x3**: 17 FLOPs / 8 bytes (ideal) = $2.125\text{ FLOP/byte}$; no reuse = $0.425\text{ FLOP/byte}$.
  - **Gauss 5x5**: 49 FLOPs / 8 bytes (ideal) = $6.125\text{ FLOP/byte}$; no reuse = $0.471\text{ FLOP/byte}$.
  - **Sobel 3x3**: 16 FLOPs / 8 bytes (ideal) = $2.000\text{ FLOP/byte}$; no reuse = $0.308\text{ FLOP/byte}$.
  - **Laplacian 3x3**: 6 FLOPs / 8 bytes (ideal) = $0.750\text{ FLOP/byte}$; no reuse = $0.250\text{ FLOP/byte}$.
  - **Heat 3D 7-point**: 8 FLOPs / 16 bytes (ideal double) = $0.500\text{ FLOP/byte}$; no reuse = $0.125\text{ FLOP/byte}$.

- **Attained Performance & Efficiency Relative to Attainable Roof**:
  - **Gauss 5x5**: Attains **$102.76\text{ GFLOP/s}$** ($80.2\%$ of attainable DRAM roof of $128.20\text{ GFLOP/s}$) in `v5_prefetch_nt`.
  - **Gauss 3x3**: Attains **$38.89\text{ GFLOP/s}$** ($87.4\%$ of attainable DRAM roof of $44.48\text{ GFLOP/s}$) in `v4_simd_window`.
  - **Sobel 3x3**: Attains **$37.58\text{ GFLOP/s}$** ($89.8\%$ of attainable DRAM roof of $41.86\text{ GFLOP/s}$) in `v4_simd_window`.
  - **Laplacian 3x3**: Attains **$14.38\text{ GFLOP/s}$** ($91.6\%$ of attainable DRAM roof of $15.70\text{ GFLOP/s}$) in `v4_simd_window`.
  - **Heat 3D 7-point**: Attains **$8.39\text{ GFLOP/s}$** ($80.2\%$ of attainable DRAM roof of $10.46\text{ GFLOP/s}$) in `v3_tiling`.

- **Key Architectural Takeaways**:
  1. All stencil kernels implemented operate below the platform's machine balance threshold ($22.5\text{ FLOP/byte}$), making them fundamentally memory-bandwidth bound in DRAM regimes.
  2. Cache tiling (`v3`), SIMD sliding window (`v4`), and software prefetching (`v5`) shift memory traffic from DRAM towards L2/L3 caches, enabling compute-heavy kernels like Gauss 5x5 to break past naive bandwidth barriers and sustain over $100\text{ GFLOP/s}$.
  3. All data and plots verified and saved:
     - `results/roofline_ceilings.csv`
     - `results/roofline_summary.csv`
     - `plots/roofline_model.png` & `plots/roofline_model.pdf`
     - `plots/speedup_variants_2d.png` & `.pdf`
     - `plots/speedup_variants_3d.png` & `.pdf`
     - `plots/thread_scaling_2d.png` & `.pdf`
     - `plots/thread_scaling_3d.png` & `.pdf`


