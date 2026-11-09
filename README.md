# HPC Stencil Optimization Project

## Build & Run Instructions

### Prerequisites
- Windows 10/11 with 64-bit processor
- Python 3.x (`numpy`, `matplotlib`, `scipy` pre-installed)
- **w64devkit** (bundled at `tools/w64devkit/`; already downloaded)

---

## Quick Start

All commands use `tools\w64devkit\bin\make.exe` via the wrapper:

```bat
make.bat all
```

Or use the full path directly:

```bat
tools\w64devkit\bin\make.exe all
```

---

## Available Make Targets

| Target     | Description |
|-----------|-------------|
| `make all`      | Build all executables (`correctness_check`, `bench_2d`, `bench_3d`, `bench_tune`, `microbench_roofline`) |
| `make bench`    | Build + run all 2D and 3D benchmark suites; outputs to `results/bench_2d_variants.csv`, `results/bench_2d_scaling.csv`, `results/bench_3d_*.csv` |
| `make tune`     | Build + run auto-tuner; outputs `results/tuning_*.csv` and `plots/tuning_heatmap_*.png` |
| `make roofline` | Build + run STREAM triad and FMA microbenchmarks; generates `results/roofline_ceilings.csv` and `plots/roofline_model.png` |
| `make plots`    | Generate all performance plots from existing CSVs into `plots/` |
| `make clean`    | Remove all built executables |

---

## Running Individual Programs

```bat
# Correctness verification for all variants v0..v5 (and v6 for 3D)
.\correctness_check.exe 6

# Synthetic image generator (confirms all 4 image sizes)
.\synthetic_gen.exe

# 2D Stencil Benchmarks (all kernels, all variants, scaling)
.\bench_2d.exe

# 3D Heat Stencil Benchmarks
.\bench_3d.exe

# Single tunable run (used by auto-tuner)
.\bench_tune.exe --kernel gauss5 --dim 2048 --tx 64 --ty 32 --threads 12 --simd 1 --prefetch 16

# Peak Compute (FMA) + Memory Bandwidth (STREAM Triad) Ceilings
.\microbench_roofline.exe
```

---

## Results Files

| File | Contents |
|------|----------|
| `results/bench_2d_variants.csv` | Per-kernel, per-variant performance: time, Mpixels/s, GFLOP/s, GB/s, speedup |
| `results/bench_2d_scaling.csv` | Strong scaling (1..12 threads) for 2D kernels |
| `results/bench_3d_variants.csv` | Per-variant 3D stencil performance |
| `results/bench_3d_scaling.csv` | Strong scaling for 3D stencil |
| `results/tuning_2d_heatmap.csv` | Exhaustive TX×TY tile size sweep results |
| `results/tuning_2d_trials.csv` | All random search tuning trials |
| `results/tuning_3d_trials.csv` | 3D heat stencil tile search results |
| `results/roofline_ceilings.csv` | Measured bandwidth and peak GFLOP/s ceilings |
| `results/roofline_summary.csv` | Per-kernel: AI, GFLOP/s, % of applicable roof, limiting ceiling |

## Plots

| File | Description |
|------|-------------|
| `plots/roofline_model.png` | Log-log roofline with all kernels and optimization steps |
| `plots/speedup_variants_2d.png` | Grouped bar chart of speedup ladder (v0→v5), all 4 kernels |
| `plots/speedup_variants_3d.png` | 3D stencil speedup bar (v0→v6) |
| `plots/thread_scaling_2d.png` | Strong scaling speedup and parallel efficiency (2D) |
| `plots/thread_scaling_3d.png` | Strong scaling speedup and parallel efficiency (3D) |
| `plots/tuning_heatmap_2d.png` | TX×TY performance heatmap (Gauss5x5 2048×2048) |
| `plots/tuning_heatmap_3d.png` | TZ×TY performance heatmap (Heat3D 128^3) |

