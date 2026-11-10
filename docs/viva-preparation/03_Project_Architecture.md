# 03. Project Architecture & Program Flow

---

## 1. Directory Structure Overview

The project is structured into clean, modular directories adhering to professional HPC conventions:

```
Stencil-Computation/
│
├── src/                               # Core computation kernels
│   ├── kernels_3d_serial.c            # Single-threaded reference & array initialization
│   ├── kernels_3d_opt.c               # OpenMP naive, First Touch, 3D cache blocking
│   └── kernels_3d_dispatch.c          # Uniform function-pointer dispatcher
│
├── include/                           # Public headers & declarations
│   ├── common.h                       # Aligned allocation, 5-variant enum, constants
│   └── kernels_3d.h                   # Function prototypes, 1D indexing macro
│
├── bench/                             # Testing and benchmarking entry points
│   ├── correctness_check.c            # 55-case automated numerical validation suite
│   ├── benchmark_3d.c                 # Five-way benchmark, thread scaling, tile sweeps
│   └── microbench_roofline.c          # STREAM Triad bandwidth & compute roof benchmark
│
├── scripts/                           # Post-processing and visualization
│   └── plot_roofline.py               # Generates Roofline model and speedup charts
│
├── results/                           # Raw measurement data (CSVs & logs)
│   ├── bench_3d.csv                   # Main 5-way performance comparison metrics
│   ├── bench_3d_scaling.csv           # Strong thread scaling data (1 to 12 threads)
│   ├── correctness_log.txt            # Maximum absolute error validation log
│   ├── roofline_ceilings.csv          # Measured DRAM, L3, L2, L1 bandwidths & compute roof
│   └── roofline_summary.csv           # AI, attainable ceilings, bottleneck summary
│
├── plots/                             # Generated vector & bitmap plots
│   ├── roofline_3d.png                # Dedicated 3D Stencil Roofline diagram (PNG)
│   ├── roofline_3d.pdf                # Vector PDF format of Roofline diagram
│   └── speedup_3d.png                 # Stencil execution time & speedup bar chart
│
├── docs/viva-preparation/             # Complete academic learning and viva guide
│   ├── README.md                      # Index linking all 16 study modules
│   └── 01_Project_Overview.md ...     # Detailed viva preparation modules
│
├── tools/w64devkit/                   # Bundled standalone MinGW-w64 toolchain (GCC 16.2.0)
├── Makefile                           # Automated compilation rules (make all, bench, etc.)
├── make.bat                           # Windows CMD wrapper for make.exe
└── README.md                          # Main project guide with quick-start instructions
```

---

## 2. Component Responsibility & Inter-Module Interaction

```
                        +----------------------+
                        |   include/common.h   |
                        |   Analytical model,  |
                        |   structs, 64B alloc |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |  include/kernels_3d.h|
                        |  1D Index macro, fn  |
                        |      declarations    |
                        +----------+-----------+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+------------------+     +-------------------+     +--------------------+
|src/kernels_3d_   |     |src/kernels_3d_    |     |src/kernels_3d_     |
|   serial.c       |     |      opt.c        |     |   dispatch.c       |
| Serial reference |     | OpenMP, FT init,  |     | Uniform function-  |
|  & serial init   |     | 3D Cache Blocking |     | pointer interface  |
+--------+---------+     +---------+---------+     +---------+----------+
         |                         |                         |
         +-------------------------+-------------------------+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+------------------+     +-------------------+     +--------------------+
|bench/correctness_|     |bench/benchmark_   |     |bench/microbench_   |
|     check.c      |     |      3d.c         |     |    roofline.c      |
| Validates vs     |     | Measures 5-way    |     | Measures STREAM    |
| serial reference |     | time, speedup,    |     | triad bandwidth    |
| (55 test cases)  |     | scaling to CSV    |     | & compute roof     |
+--------+---------+     +---------+---------+     +---------+----------+
         |                         |                         |
         v                         v                         v
  correctness_log.txt         bench_3d.csv          roofline_ceilings.csv
                                   |                         |
                                   +------------+------------+
                                                |
                                                v
                                    +-----------------------+
                                    |scripts/plot_roofline.py|
                                    +-----------+-----------+
                                                |
                                                v
                                      plots/roofline_3d.png
                                      plots/speedup_3d.png
                                      roofline_summary.csv
```

---

## 3. Build System & Compilation Pipeline

The project is compiled using the bundled MinGW-w64 toolchain (`tools/w64devkit/bin/gcc.exe`).

### Compiler Flags Explained:
```makefile
CFLAGS := -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude
LDFLAGS := -fopenmp -lm
```
1. `-O3`: Maximum standard optimization level (loop unrolling, auto-vectorization, dead-code elimination, inlining).
2. `-march=native`: Emits CPU instructions tailored specifically to the host processor (Intel Alder Lake), enabling optimal instruction pipeline scheduling.
3. `-ffp-contract=off`: Disables automatic fusion of multiply and add into hardware FMA instructions. This guarantees that our analytical operation count of exactly 8 FLOPs per interior point is preserved.
4. `-fopenmp`: Enables OpenMP multi-threading pragmas and links the OpenMP runtime library (`libgomp`).
5. `-Wall -Wextra`: Enables rigorous compiler diagnostic warnings.
6. `-Iinclude`: Adds `include/` to the preprocessor header search path.
7. `-lm`: Links the C standard math library (`libm`).

### Build Targets in Makefile:
* `make all`: Compiles the three standalone executables (`correctness_check.exe`, `bench_3d.exe`, `microbench_roofline.exe`).
* `make correctness`: Compiles and executes the 55-case numerical validation suite.
* `make bench`: Compiles and runs the five-way benchmark, strong scaling, and tile sweeps.
* `make roofline`: Compiles and runs STREAM triad microbenchmarks and generates Roofline plots via Python.
* `make clean`: Removes all compiled `.exe` binaries.

---

## 4. Execution Data Flow (From Input to Output)

### Execution Sequence for a Benchmark Run:
1. **Memory Allocation:**
   Two double-precision buffers (`in` and `out`) of size $NX \times NY \times NZ \times 8\text{ bytes}$ are allocated using `hpc_aligned_alloc(64, ...)`.
2. **Phase A — Initialization Timing:**
   * A timer probe starts: `t_init_start = omp_get_wtime()`.
   * For serial/naive variants: `serial_init_3d()` runs on a single thread.
   * For First Touch variants: `first_touch_init_3d()` runs in parallel using `#pragma omp parallel for schedule(static)`.
   * Timer probe stops: `InitTime_ms = (t_init_end - t_init_start) * 1000.0`.
3. **Phase B — Warmup Runs:**
   The kernel executes 2 warmup time steps to pre-populate instruction caches and warm up the CPU turbo clock.
4. **Phase C — Timed Trials (Double Buffering):**
   * For 10 timed trials:
     * High-resolution timer starts: `t0 = omp_get_wtime()`.
     * For 10 time steps:
       * Kernel updates `next = f(curr)`.
       * Pointers are swapped: `tmp = curr; curr = next; next = tmp;`.
     * Timer stops: `t1 = omp_get_wtime()`.
     * Per-step computation time is recorded: `trial_time[t] = (t1 - t0) / num_steps`.
5. **Phase D — Statistical Processing:**
   * Trial execution times are sorted using `qsort()`.
   * The **median time** is extracted to eliminate transient operating system interrupts or thermal throttling spikes.
   * GFLOP/s, effective bandwidth, and speedups are computed.
6. **Phase E — Data Persistence:**
   Metrics are written as structured rows into `results/bench_3d.csv`.
7. **Phase F — Post-Processing & Plotting:**
   `scripts/plot_roofline.py` reads `results/roofline_ceilings.csv` and `results/bench_3d.csv`, generates matplotlib figures, and saves `plots/roofline_3d.png` and `plots/speedup_3d.png`.

