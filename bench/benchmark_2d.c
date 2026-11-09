#include "common.h"
#include "kernels_2d.h"
#include "image_io.h"
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
    double mpixels_per_sec;
    double gflops;
    double effective_gb_s;
} BenchmarkResult;

BenchmarkResult run_benchmark_2d(Kernel2DType ktype, VariantType vtype,
                                 int width, int height,
                                 const TuningConfig* cfg,
                                 int warmups, int trials) {
    kernel_2d_fn fn = get_kernel_2d(ktype, vtype);
    BenchmarkResult res = {0};
    if (!fn) return res;

    size_t num_pixels = (size_t)width * height;
    float* in = allocate_image_float(width, height);
    float* out = allocate_image_float(width, height);

    if (vtype == VAR_V2_NUMA_FIRST_TOUCH) {
        first_touch_init_2d(in, width, height, 128.0f, cfg);
        first_touch_init_2d(out, width, height, 0.0f, cfg);
    } else {
        generate_synthetic_image(in, width, height, NOISE_GAUSSIAN, 20.0f, 12345);
        memset(out, 0, num_pixels * sizeof(float));
    }

    /* Warmup and dynamically calibrate repeats for high timer resolution */
    for (int w = 0; w < warmups; ++w) {
        fn(in, out, width, height, cfg);
    }

    double t_test0 = omp_get_wtime();
    fn(in, out, width, height, cfg);
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
            fn(in, out, width, height, cfg);
        }
        double t1 = omp_get_wtime();
        times[t] = (t1 - t0) / (double)repeats;
    }

    qsort(times, trials, sizeof(double), cmp_double);
    res.median_sec = times[trials / 2];
    res.min_sec = times[0];
    res.max_sec = times[trials - 1];

    double flops = get_kernel_flops_per_pixel(ktype) * (double)num_pixels;
    double ideal_bytes = get_kernel_ideal_bytes_per_pixel(ktype) * (double)num_pixels;

    res.mpixels_per_sec = ((double)num_pixels / res.median_sec) * 1e-6;
    res.gflops = (flops / res.median_sec) * 1e-9;
    res.effective_gb_s = (ideal_bytes / res.median_sec) * 1e-9;

    free(times);
    free_image_float(in);
    free_image_float(out);
    return res;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("=====================================================\n");
    printf("      2D IMAGE STENCIL BENCHMARK SUITE (Phase 2)      \n");
    printf("=====================================================\n");

    FILE* f_var = fopen("results/bench_2d_variants.csv", "w");
    if (!f_var) {
        fprintf(stderr, "Error creating results/bench_2d_variants.csv\n");
        return 1;
    }
    fprintf(f_var, "Kernel,Variant,Width,Height,Threads,MedianSec,MinSec,MaxSec,MpixelsSec,GFLOPs,EffectiveGBs,Speedup\n");

    Kernel2DType kernels[4] = {KERNEL_GAUSS3, KERNEL_GAUSS5, KERNEL_SOBEL, KERNEL_LAPLACE};
    VariantType variants[6] = {
        VAR_V0_SERIAL,
        VAR_V1_OMP_NAIVE,
        VAR_V2_NUMA_FIRST_TOUCH,
        VAR_V3_TILING,
        VAR_V4_SIMD_WINDOW,
        VAR_V5_PREFETCH_NT
    };

    int test_sizes[] = {1024, 4096};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int max_threads = omp_get_max_threads();

    /* 1. Benchmark all variants at full thread count */
    printf("\n--- Section 1: Optimization Ladder (v0 -> v5) at full threads (%d) ---\n", max_threads);
    for (int s = 0; s < num_sizes; ++s) {
        int dim = test_sizes[s];
        printf("\nImage Dimension: %dx%d\n", dim, dim);
        printf("%-10s %-16s %-10s %-12s %-10s %-10s %-8s\n",
               "Kernel", "Variant", "Time(ms)", "Mpixels/s", "GFLOP/s", "GB/s", "Speedup");
        printf("--------------------------------------------------------------------------\n");

        for (int k = 0; k < 4; ++k) {
            Kernel2DType ktype = kernels[k];
            double serial_time = 1.0;

            for (int v = 0; v < 6; ++v) {
                VariantType vtype = variants[v];
                TuningConfig cfg = default_tuning_config();
                cfg.num_threads = (vtype == VAR_V0_SERIAL) ? 1 : max_threads;

                /* Warmup 3, Trials 10 */
                BenchmarkResult res = run_benchmark_2d(ktype, vtype, dim, dim, &cfg, 3, 10);
                if (vtype == VAR_V0_SERIAL) {
                    serial_time = res.median_sec;
                }
                double speedup = serial_time / res.median_sec;

                printf("%-10s %-16s %8.2f ms %12.2f %10.2f %10.2f %7.2fx\n",
                       kernel_name(ktype), variant_name(vtype),
                       res.median_sec * 1000.0,
                       res.mpixels_per_sec,
                       res.gflops,
                       res.effective_gb_s,
                       speedup);

                fprintf(f_var, "%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f\n",
                        kernel_name(ktype), variant_name(vtype), dim, dim,
                        cfg.num_threads,
                        res.median_sec, res.min_sec, res.max_sec,
                        res.mpixels_per_sec, res.gflops, res.effective_gb_s, speedup);
                fflush(f_var);
            }
        }
    }
    fclose(f_var);
    printf("Saved variant benchmark results to results/bench_2d_variants.csv\n");

    /* 2. Strong Thread Scaling Study */
    printf("\n--- Section 2: Strong Thread Scaling Study (1..%d threads) ---\n", max_threads);
    FILE* f_scale = fopen("results/bench_2d_scaling.csv", "w");
    if (!f_scale) {
        fprintf(stderr, "Error creating results/bench_2d_scaling.csv\n");
        return 1;
    }
    fprintf(f_scale, "Kernel,Variant,Size,Threads,MedianSec,GFLOPs,Speedup,Efficiency\n");

    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12};
    int num_tc = sizeof(thread_counts) / sizeof(thread_counts[0]);
    int scale_dim = 4096;

    for (int k = 0; k < 4; ++k) {
        Kernel2DType ktype = kernels[k];
        /* Test both v1 (naive OpenMP) and v4 (SIMD optimized) */
        VariantType test_vars[2] = {VAR_V1_OMP_NAIVE, VAR_V4_SIMD_WINDOW};

        for (int vi = 0; vi < 2; ++vi) {
            VariantType vtype = test_vars[vi];
            double t1 = 1.0;

            for (int ti = 0; ti < num_tc; ++ti) {
                int threads = thread_counts[ti];
                if (threads > max_threads) continue;

                TuningConfig cfg = default_tuning_config();
                cfg.num_threads = threads;

                BenchmarkResult res = run_benchmark_2d(ktype, vtype, scale_dim, scale_dim, &cfg, 2, 10);
                if (threads == 1) {
                    t1 = res.median_sec;
                }
                double speedup = t1 / res.median_sec;
                double efficiency = (speedup / (double)threads) * 100.0;

                printf("[Scaling] %-10s %-15s Threads=%2d Time=%7.2f ms Speedup=%5.2fx Eff=%5.1f%%\n",
                       kernel_name(ktype), variant_name(vtype), threads,
                       res.median_sec * 1000.0, speedup, efficiency);

                fprintf(f_scale, "%s,%s,%d,%d,%.6f,%.2f,%.2f,%.2f\n",
                        kernel_name(ktype), variant_name(vtype), scale_dim, threads,
                        res.median_sec, res.gflops, speedup, efficiency);
                fflush(f_scale);
            }
        }
    }
    fclose(f_scale);
    printf("Saved scaling results to results/bench_2d_scaling.csv\n");

    return 0;
}
