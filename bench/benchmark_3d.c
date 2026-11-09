#include "common.h"
#include "kernels_3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

typedef struct {
    double median_sec;
    double min_sec;
    double max_sec;
    double gstencils_per_sec;
    double gflops;
    double effective_gb_s;
} Benchmark3DResult;

Benchmark3DResult run_benchmark_3d(VariantType vtype, int nx, int ny, int nz,
                                   const TuningConfig* cfg, int warmups, int trials) {
    kernel_3d_fn fn = get_kernel_3d(vtype);
    Benchmark3DResult res = {0};
    if (!fn) return res;

    size_t num_grid = (size_t)nx * ny * nz;
    double* in = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
    double* out = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));

    double c0 = 0.5, c1 = 1.0 / 12.0;

    if (vtype == VAR_V2_NUMA_FIRST_TOUCH) {
        first_touch_init_3d(in, nx, ny, nz, 1.0, cfg);
        first_touch_init_3d(out, nx, ny, nz, 0.0, cfg);
    } else {
        for (size_t i = 0; i < num_grid; ++i) {
            in[i] = sin((double)i * 0.001);
        }
        memset(out, 0, num_grid * sizeof(double));
    }

    /* Warmup and dynamically calibrate repeats for high timer resolution */
    for (int w = 0; w < warmups; ++w) {
        fn(in, out, nx, ny, nz, c0, c1, cfg);
    }

    double t_test0 = omp_get_wtime();
    fn(in, out, nx, ny, nz, c0, c1, cfg);
    double t_test1 = omp_get_wtime();
    double single_run = t_test1 - t_test0;

    int repeats = 1;
    if (single_run < 0.03 && single_run > 0.0) {
        repeats = (int)(0.03 / single_run);
        if (repeats < 1) repeats = 1;
        if (repeats > 200) repeats = 200;
    } else if (single_run <= 0.0) {
        repeats = 20;
    }

    /* Timed trials */
    double* times = (double*)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; ++t) {
        double t0 = omp_get_wtime();
        for (int r = 0; r < repeats; ++r) {
            fn(in, out, nx, ny, nz, c0, c1, cfg);
        }
        double t1 = omp_get_wtime();
        times[t] = (t1 - t0) / (double)repeats;
    }

    qsort(times, trials, sizeof(double), cmp_double);
    res.median_sec = times[trials / 2];
    res.min_sec = times[0];
    res.max_sec = times[trials - 1];

    double flops = STENCIL3D_FLOPS_PER_GRID * (double)num_grid;
    double ideal_bytes = STENCIL3D_IDEAL_BYTES * (double)num_grid;

    res.gstencils_per_sec = ((double)num_grid / res.median_sec) * 1e-9;
    res.gflops = (flops / res.median_sec) * 1e-9;
    res.effective_gb_s = (ideal_bytes / res.median_sec) * 1e-9;

    free(times);
    hpc_aligned_free(in);
    hpc_aligned_free(out);
    return res;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("=====================================================\n");
    printf("      3D HEAT STENCIL BENCHMARK SUITE (Phase 2)       \n");
    printf("=====================================================\n");

    FILE* f_var = fopen("results/bench_3d_variants.csv", "w");
    if (!f_var) {
        fprintf(stderr, "Error creating results/bench_3d_variants.csv\n");
        return 1;
    }
    fprintf(f_var, "Kernel,Variant,GridN,Threads,MedianSec,MinSec,MaxSec,GStencilsSec,GFLOPs,EffectiveGBs,Speedup\n");

    VariantType variants[7] = {
        VAR_V0_SERIAL,
        VAR_V1_OMP_NAIVE,
        VAR_V2_NUMA_FIRST_TOUCH,
        VAR_V3_TILING,
        VAR_V4_SIMD_WINDOW,
        VAR_V5_PREFETCH_NT,
        VAR_V6_UNROLL_REG
    };

    int grid_sizes[] = {128, 256};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    int max_threads = omp_get_max_threads();

    /* 1. Benchmark all variants (v0 -> v6) at full thread count */
    printf("\n--- Section 1: 3D Stencil Ladder (v0 -> v6) at full threads (%d) ---\n", max_threads);
    for (int s = 0; s < num_sizes; ++s) {
        int n = grid_sizes[s];
        printf("\nGrid Size: %dx%dx%d (double precision, %.1f MB)\n",
               n, n, n, (double)n * n * n * sizeof(double) / (1024.0 * 1024.0));
        printf("%-16s %-10s %-14s %-10s %-10s %-8s\n",
               "Variant", "Time(ms)", "GStencil/s", "GFLOP/s", "GB/s", "Speedup");
        printf("------------------------------------------------------------------\n");

        double serial_time = 1.0;
        for (int v = 0; v < 7; ++v) {
            VariantType vtype = variants[v];
            TuningConfig cfg = default_tuning_config();
            cfg.num_threads = (vtype == VAR_V0_SERIAL) ? 1 : max_threads;

            Benchmark3DResult res = run_benchmark_3d(vtype, n, n, n, &cfg, 3, 10);
            if (vtype == VAR_V0_SERIAL) {
                serial_time = res.median_sec;
            }
            double speedup = serial_time / res.median_sec;

            printf("%-16s %8.2f ms %12.3f %10.2f %10.2f %7.2fx\n",
                   variant_name(vtype),
                   res.median_sec * 1000.0,
                   res.gstencils_per_sec,
                   res.gflops,
                   res.effective_gb_s,
                   speedup);

            fprintf(f_var, "Heat7pt,%s,%d,%d,%.6f,%.6f,%.6f,%.3f,%.2f,%.2f,%.2f\n",
                    variant_name(vtype), n, cfg.num_threads,
                    res.median_sec, res.min_sec, res.max_sec,
                    res.gstencils_per_sec, res.gflops, res.effective_gb_s, speedup);
            fflush(f_var);
        }
    }
    fclose(f_var);
    printf("Saved 3D variant benchmark results to results/bench_3d_variants.csv\n");

    /* 2. Strong Thread Scaling Study */
    printf("\n--- Section 2: 3D Strong Thread Scaling Study (1..%d threads) ---\n", max_threads);
    FILE* f_scale = fopen("results/bench_3d_scaling.csv", "w");
    if (!f_scale) {
        fprintf(stderr, "Error creating results/bench_3d_scaling.csv\n");
        return 1;
    }
    fprintf(f_scale, "Kernel,Variant,GridN,Threads,MedianSec,GFLOPs,Speedup,Efficiency\n");

    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12};
    int num_tc = sizeof(thread_counts) / sizeof(thread_counts[0]);
    int scale_n = 256;

    VariantType test_vars[2] = {VAR_V1_OMP_NAIVE, VAR_V6_UNROLL_REG};
    for (int vi = 0; vi < 2; ++vi) {
        VariantType vtype = test_vars[vi];
        double t1 = 1.0;

        for (int ti = 0; ti < num_tc; ++ti) {
            int threads = thread_counts[ti];
            if (threads > max_threads) continue;

            TuningConfig cfg = default_tuning_config();
            cfg.num_threads = threads;

            Benchmark3DResult res = run_benchmark_3d(vtype, scale_n, scale_n, scale_n, &cfg, 2, 10);
            if (threads == 1) {
                t1 = res.median_sec;
            }
            double speedup = t1 / res.median_sec;
            double efficiency = (speedup / (double)threads) * 100.0;

            printf("[Scaling 3D] %-16s Threads=%2d Time=%7.2f ms Speedup=%5.2fx Eff=%5.1f%%\n",
                   variant_name(vtype), threads,
                   res.median_sec * 1000.0, speedup, efficiency);

            fprintf(f_scale, "Heat7pt,%s,%d,%d,%.6f,%.2f,%.2f,%.2f\n",
                    variant_name(vtype), scale_n, threads,
                    res.median_sec, res.gflops, speedup, efficiency);
            fflush(f_scale);
        }
    }
    fclose(f_scale);
    printf("Saved 3D scaling results to results/bench_3d_scaling.csv\n");

    return 0;
}
