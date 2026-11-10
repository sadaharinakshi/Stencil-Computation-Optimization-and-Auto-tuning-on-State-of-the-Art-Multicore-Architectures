#include "common.h"
#include "kernels_3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * ============================================================================
 * 3D 7-Point Heat Stencil Benchmark Suite
 * ============================================================================
 *
 * Compares all 5 required implementations:
 *   1. Serial 3D Stencil Reference
 *   2. Basic OpenMP 3D Stencil
 *   3. OpenMP + First Touch Initialisation
 *   4. OpenMP + Genuine 3D Cache Blocking
 *   5. OpenMP + First Touch + Genuine 3D Cache Blocking
 *
 * Key Measurement Features:
 * - Separates memory initialisation time from stencil computation time.
 * - Uses identical initial conditions across all 5 configurations.
 * - Multi-iteration ping-pong time-stepping (10 steps) to ensure representative
 *   cache behaviour and eliminate transient cold-start biases.
 * - Records median, minimum, and maximum execution times across 10 trials.
 * - Computes speedup relative to the serial reference.
 * - Evaluates parallel efficiency.
 * - Performs strong thread scaling (1, 2, 4, 8, max_threads).
 * - Tests multiple 3D tile configurations.
 * - Saves structured CSV data to results/bench_3d.csv and results/bench_3d_scaling.csv.
 * ============================================================================
 */

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

typedef struct {
    double init_sec;          /* Time spent on memory allocation and initialisation */
    double median_compute_sec;/* Median stencil execution time per step */
    double min_compute_sec;
    double max_compute_sec;
    double total_time_sec;    /* Total multi-step execution time */
    double gflops;            /* GFLOP/s based on 8 FLOPs per interior point */
    double ideal_gb_s;        /* Effective bandwidth under ideal cache reuse (16 B/pt) */
    double noreuse_gb_s;      /* Effective bandwidth under no cache reuse (64 B/pt) */
    double speedup;
    double efficiency;
} BenchResult;

/*
 * Run benchmark for a specific variant.
 * Measures initialisation time separately from stencil computation time.
 */
static BenchResult run_benchmark_variant(VariantType vtype, int nx, int ny, int nz,
                                         int num_steps, const StencilConfig* cfg,
                                         int warmups, int trials, double serial_time) {
    BenchResult res = {0};
    size_t num_grid = (size_t)nx * ny * nz;
    size_t interior_points = (size_t)(nx - 2) * (size_t)(ny - 2) * (size_t)(nz - 2);

    /* Allocate buffers */
    double* in  = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
    double* out = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
    if (!in || !out) {
        fprintf(stderr, "Error: Memory allocation failed for benchmark\n");
        return res;
    }

    /*
     * 1. Measure Initialisation Time Separately
     * All variants use identical initial mathematical pattern: sin(i * 0.001).
     * First-touch variants use OpenMP static loop; others use serial single-threaded loop.
     */
    double t_init_start = omp_get_wtime();
    if (vtype == VAR_OMP_NUMA || vtype == VAR_OMP_NUMA_BLOCKED) {
        first_touch_init_3d(in, out, nx, ny, nz, cfg->num_threads);
    } else {
        serial_init_3d(in, out, nx, ny, nz);
    }
    double t_init_end = omp_get_wtime();
    res.init_sec = t_init_end - t_init_start;

    kernel_3d_fn fn = get_kernel_3d(vtype);

    /* 2. Warmup Runs */
    for (int w = 0; w < warmups; ++w) {
        double* curr = in;
        double* next = out;
        for (int s = 0; s < num_steps; ++s) {
            fn(curr, next, nx, ny, nz, cfg);
            double* tmp = curr; curr = next; next = tmp;
        }
    }

    /* 3. Timed Trials (Multi-Step Stencil Execution) */
    double* trial_times = (double*)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; ++t) {
        /* Re-initialize buffers before each trial to maintain identical state */
        if (vtype == VAR_OMP_NUMA || vtype == VAR_OMP_NUMA_BLOCKED) {
            first_touch_init_3d(in, out, nx, ny, nz, cfg->num_threads);
        } else {
            serial_init_3d(in, out, nx, ny, nz);
        }

        double* curr = in;
        double* next = out;

        double t0 = omp_get_wtime();
        for (int s = 0; s < num_steps; ++s) {
            fn(curr, next, nx, ny, nz, cfg);
            double* tmp = curr; curr = next; next = tmp;
        }
        double t1 = omp_get_wtime();
        trial_times[t] = t1 - t0;
    }

    qsort(trial_times, trials, sizeof(double), cmp_double);
    res.total_time_sec = trial_times[trials / 2];
    res.median_compute_sec = res.total_time_sec / (double)num_steps;
    res.min_compute_sec = trial_times[0] / (double)num_steps;
    res.max_compute_sec = trial_times[trials - 1] / (double)num_steps;

    /* Compute FLOPs and Bandwidth Metrics based on interior update points */
    double total_flops = STENCIL3D_FLOPS_PER_POINT * (double)interior_points * (double)num_steps;
    double ideal_bytes = STENCIL3D_IDEAL_BYTES * (double)interior_points * (double)num_steps;
    double noreuse_bytes = STENCIL3D_NOREUSE_BYTES * (double)interior_points * (double)num_steps;

    res.gflops = (total_flops / res.total_time_sec) * 1e-9;
    res.ideal_gb_s = (ideal_bytes / res.total_time_sec) * 1e-9;
    res.noreuse_gb_s = (noreuse_bytes / res.total_time_sec) * 1e-9;

    if (serial_time > 0.0) {
        res.speedup = serial_time / res.median_compute_sec;
        int threads = (vtype == VAR_SERIAL) ? 1 : cfg->num_threads;
        res.efficiency = (res.speedup / (double)threads) * 100.0;
    } else {
        res.speedup = 1.0;
        res.efficiency = 100.0;
    }

    free(trial_times);
    hpc_aligned_free(in);
    hpc_aligned_free(out);
    return res;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int max_threads = omp_get_max_threads();
    int num_steps = 10;   /* 10 time steps per benchmark trial */
    int warmups = 2;
    int trials = 10;

    printf("========================================================================================\n");
    printf("        3D 7-POINT HEAT STENCIL BENCHMARK SUITE (Five-Way Comparison)                   \n");
    printf("========================================================================================\n");
    printf("Detected OpenMP Max Threads: %d\n", max_threads);
    printf("Benchmark Configuration: %d steps/trial, %d warmups, %d timed trials (taking median)\n",
           num_steps, warmups, trials);
    printf("Separating Initialisation Time from Stencil Computation Time.\n\n");

    /*
     * Part 1: Main 5-Way Comparison on Benchmark Grid
     * Test grid: 128^3 (2,097,152 doubles, ~16 MB per array, 32 MB working set)
     * and 200^3 (8,000,000 doubles, ~64 MB per array, 128 MB working set > L3 cache)
     */
    int benchmark_sizes[] = {128, 200};
    int num_bm_sizes = sizeof(benchmark_sizes) / sizeof(benchmark_sizes[0]);

    FILE* f_main = fopen("results/bench_3d.csv", "w");
    if (!f_main) {
        fprintf(stderr, "Error creating results/bench_3d.csv\n");
        return 1;
    }
    fprintf(f_main, "GridN,Variant,Threads,TileX,TileY,TileZ,Steps,InitTime_ms,ComputeTime_ms,GFLOPs,IdealGBs,NoReuseGBs,Speedup,Efficiency\n");

    for (int s = 0; s < num_bm_sizes; ++s) {
        int n = benchmark_sizes[s];
        printf("========================================================================================\n");
        printf("Grid Dimension: %d x %d x %d (%.1f MB per array, %d time-steps)\n",
               n, n, n, (double)n * n * n * sizeof(double) / (1024.0 * 1024.0), num_steps);
        printf("========================================================================================\n");
        printf("%-30s %10s %12s %10s %10s %8s %8s\n",
               "Implementation", "Init(ms)", "Compute(ms)", "GFLOP/s", "Ideal GB/s", "Speedup", "Eff(%)");
        printf("----------------------------------------------------------------------------------------\n");

        double serial_time = 0.0;

        for (int v = 0; v < NUM_VARIANTS; ++v) {
            VariantType vtype = (VariantType)v;
            StencilConfig cfg = default_stencil_config();
            cfg.num_threads = (vtype == VAR_SERIAL) ? 1 : max_threads;
            cfg.num_steps = num_steps;
            cfg.bx = 32; cfg.by = 32; cfg.bz = 16;

            BenchResult res = run_benchmark_variant(vtype, n, n, n, num_steps, &cfg,
                                                   warmups, trials, serial_time);

            if (vtype == VAR_SERIAL) {
                serial_time = res.median_compute_sec;
                res.speedup = 1.0;
                res.efficiency = 100.0;
            }

            printf("%-30s %10.2f %12.2f %10.2f %10.2f %7.2fx %7.1f%%\n",
                   variant_name(vtype),
                   res.init_sec * 1000.0,
                   res.median_compute_sec * 1000.0,
                   res.gflops,
                   res.ideal_gb_s,
                   res.speedup,
                   res.efficiency);

            fprintf(f_main, "%d,%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.1f\n",
                    n, variant_short_name(vtype), cfg.num_threads,
                    (vtype >= VAR_OMP_BLOCKED) ? cfg.bx : 0,
                    (vtype >= VAR_OMP_BLOCKED) ? cfg.by : 0,
                    (vtype >= VAR_OMP_BLOCKED) ? cfg.bz : 0,
                    num_steps,
                    res.init_sec * 1000.0,
                    res.median_compute_sec * 1000.0,
                    res.gflops,
                    res.ideal_gb_s,
                    res.noreuse_gb_s,
                    res.speedup,
                    res.efficiency);
            fflush(f_main);
        }
        printf("\n");
    }
    fclose(f_main);
    printf("Saved main benchmark results to results/bench_3d.csv\n\n");

    /*
     * Part 2: Strong Thread Scaling Study
     * Compare Naive OpenMP vs 3D Blocked across thread counts (1, 2, 4, 8, max_threads).
     */
    printf("========================================================================================\n");
    printf("        SECTION 2: STRONG THREAD SCALING STUDY (Grid: 160 x 160 x 160)                  \n");
    printf("========================================================================================\n");

    FILE* f_scale = fopen("results/bench_3d_scaling.csv", "w");
    if (!f_scale) {
        fprintf(stderr, "Error creating results/bench_3d_scaling.csv\n");
        return 1;
    }
    fprintf(f_scale, "GridN,Variant,Threads,ComputeTime_ms,GFLOPs,Speedup,Efficiency\n");

    int scale_threads[] = {1, 2, 4, 8, 12};
    int num_tc = sizeof(scale_threads) / sizeof(scale_threads[0]);
    int scale_n = 160;

    /* Get serial reference for 160^3 */
    StencilConfig cfg_s = default_stencil_config();
    cfg_s.num_threads = 1;
    cfg_s.num_steps = num_steps;
    BenchResult s_res = run_benchmark_variant(VAR_SERIAL, scale_n, scale_n, scale_n,
                                             num_steps, &cfg_s, warmups, trials, 0.0);
    double scale_serial_time = s_res.median_compute_sec;

    VariantType scale_variants[] = {VAR_OMP, VAR_OMP_BLOCKED};
    for (int vi = 0; vi < 2; ++vi) {
        VariantType vtype = scale_variants[vi];
        printf("\nScaling Implementation: %s\n", variant_name(vtype));
        printf("%-10s %14s %12s %10s %12s\n", "Threads", "Compute(ms)", "GFLOP/s", "Speedup", "Efficiency");
        printf("----------------------------------------------------------------\n");

        double t1_base = 0.0;
        for (int ti = 0; ti < num_tc; ++ti) {
            int tc = scale_threads[ti];
            if (tc > max_threads) continue;

            StencilConfig cfg = default_stencil_config();
            cfg.num_threads = tc;
            cfg.num_steps = num_steps;
            cfg.bx = 32; cfg.by = 32; cfg.bz = 16;

            BenchResult res = run_benchmark_variant(vtype, scale_n, scale_n, scale_n,
                                                   num_steps, &cfg, warmups, trials, scale_serial_time);

            if (ti == 0 || tc == 1) {
                t1_base = res.median_compute_sec;
            }
            double self_speedup = t1_base / res.median_compute_sec;
            double self_efficiency = (self_speedup / (double)tc) * 100.0;

            printf("%-10d %14.2f %12.2f %9.2fx %11.1f%%\n",
                   tc, res.median_compute_sec * 1000.0, res.gflops, self_speedup, self_efficiency);

            fprintf(f_scale, "%d,%s,%d,%.3f,%.3f,%.2f,%.1f\n",
                    scale_n, variant_short_name(vtype), tc,
                    res.median_compute_sec * 1000.0, res.gflops, self_speedup, self_efficiency);
            fflush(f_scale);
        }
    }
    fclose(f_scale);
    printf("\nSaved thread scaling results to results/bench_3d_scaling.csv\n\n");

    /*
     * Part 3: 3D Tile Size Exploration
     * Test different BX x BY x BZ combinations for 3D blocking.
     */
    printf("========================================================================================\n");
    printf("        SECTION 3: 3D CACHE BLOCKING TILE SIZE COMPARISON                               \n");
    printf("========================================================================================\n");

    int test_tiles[][3] = {
        {16, 16, 16},
        {32, 16, 16},
        {32, 32, 16},
        {64, 32, 16},
        {32, 32, 32},
        {64, 64, 16}
    };
    int num_tiles = sizeof(test_tiles) / sizeof(test_tiles[0]);

    printf("%-20s %10s %10s %12s %10s %10s\n",
           "Tile Dimensions", "WorkingSet", "Compute(ms)", "GFLOP/s", "Speedup", "Ideal GB/s");
    printf("----------------------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tiles; ++t) {
        int bx = test_tiles[t][0];
        int by = test_tiles[t][1];
        int bz = test_tiles[t][2];
        double working_set_kb = (double)(bx * by * bz * sizeof(double) * 3) / 1024.0;

        StencilConfig cfg = default_stencil_config();
        cfg.num_threads = max_threads;
        cfg.num_steps = num_steps;
        cfg.bx = bx; cfg.by = by; cfg.bz = bz;

        BenchResult res = run_benchmark_variant(VAR_OMP_BLOCKED, scale_n, scale_n, scale_n,
                                               num_steps, &cfg, warmups, trials, scale_serial_time);

        printf("BX=%-2d BY=%-2d BZ=%-2d %8.1f KB %12.2f %10.2f %9.2fx %10.2f\n",
               bx, by, bz, working_set_kb,
               res.median_compute_sec * 1000.0, res.gflops, res.speedup, res.ideal_gb_s);
    }

    printf("========================================================================================\n");
    printf("Benchmark suite completed successfully.\n");
    return 0;
}
