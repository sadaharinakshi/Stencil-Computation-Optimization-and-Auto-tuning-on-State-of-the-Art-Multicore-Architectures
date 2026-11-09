#include "common.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/*
 * High-Precision AVX2 FMA Compute Kernel
 * Computes 16 independent vector accumulators per iteration:
 * 16 accumulators * 8 SP floats * 2 FLOPs (FMA) = 256 FLOPs / iteration.
 * Marked noinline and combines all accumulators into output to guarantee
 * full execution without dead-code elimination.
 */
__attribute__((noinline))
void fma_vector_kernel(size_t iters, float* out) {
    __m256 acc0  = _mm256_set1_ps(1.0f);
    __m256 acc1  = _mm256_set1_ps(1.01f);
    __m256 acc2  = _mm256_set1_ps(1.02f);
    __m256 acc3  = _mm256_set1_ps(1.03f);
    __m256 acc4  = _mm256_set1_ps(1.04f);
    __m256 acc5  = _mm256_set1_ps(1.05f);
    __m256 acc6  = _mm256_set1_ps(1.06f);
    __m256 acc7  = _mm256_set1_ps(1.07f);
    __m256 acc8  = _mm256_set1_ps(1.08f);
    __m256 acc9  = _mm256_set1_ps(1.09f);
    __m256 acc10 = _mm256_set1_ps(1.10f);
    __m256 acc11 = _mm256_set1_ps(1.11f);
    __m256 acc12 = _mm256_set1_ps(1.12f);
    __m256 acc13 = _mm256_set1_ps(1.13f);
    __m256 acc14 = _mm256_set1_ps(1.14f);
    __m256 acc15 = _mm256_set1_ps(1.15f);

    __m256 mult = _mm256_set1_ps(1.0000001f);
    __m256 add  = _mm256_set1_ps(0.0000001f);

    for (size_t i = 0; i < iters; ++i) {
        acc0  = _mm256_fmadd_ps(acc0,  mult, add);
        acc1  = _mm256_fmadd_ps(acc1,  mult, add);
        acc2  = _mm256_fmadd_ps(acc2,  mult, add);
        acc3  = _mm256_fmadd_ps(acc3,  mult, add);
        acc4  = _mm256_fmadd_ps(acc4,  mult, add);
        acc5  = _mm256_fmadd_ps(acc5,  mult, add);
        acc6  = _mm256_fmadd_ps(acc6,  mult, add);
        acc7  = _mm256_fmadd_ps(acc7,  mult, add);
        acc8  = _mm256_fmadd_ps(acc8,  mult, add);
        acc9  = _mm256_fmadd_ps(acc9,  mult, add);
        acc10 = _mm256_fmadd_ps(acc10, mult, add);
        acc11 = _mm256_fmadd_ps(acc11, mult, add);
        acc12 = _mm256_fmadd_ps(acc12, mult, add);
        acc13 = _mm256_fmadd_ps(acc13, mult, add);
        acc14 = _mm256_fmadd_ps(acc14, mult, add);
        acc15 = _mm256_fmadd_ps(acc15, mult, add);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc4 = _mm256_add_ps(acc4, acc5);
    acc6 = _mm256_add_ps(acc6, acc7);
    acc8 = _mm256_add_ps(acc8, acc9);
    acc10 = _mm256_add_ps(acc10, acc11);
    acc12 = _mm256_add_ps(acc12, acc13);
    acc14 = _mm256_add_ps(acc14, acc15);

    acc0 = _mm256_add_ps(acc0, acc2);
    acc4 = _mm256_add_ps(acc4, acc6);
    acc8 = _mm256_add_ps(acc8, acc10);
    acc12 = _mm256_add_ps(acc12, acc14);

    acc0 = _mm256_add_ps(acc0, acc4);
    acc8 = _mm256_add_ps(acc8, acc12);

    acc0 = _mm256_add_ps(acc0, acc8);
    _mm256_storeu_ps(out, acc0);
}

/*
 * Scalar Single-Precision Compute Kernel (No SIMD)
 * 8 independent scalar accumulators = 16 FLOPs / iteration.
 */
__attribute__((noinline))
void scalar_kernel(size_t iters, float* out) {
    float acc0 = 1.0f, acc1 = 1.01f, acc2 = 1.02f, acc3 = 1.03f;
    float acc4 = 1.04f, acc5 = 1.05f, acc6 = 1.06f, acc7 = 1.07f;
    float mult = 1.0000001f, add = 0.0000001f;

    for (size_t i = 0; i < iters; ++i) {
        acc0 = acc0 * mult + add;
        acc1 = acc1 * mult + add;
        acc2 = acc2 * mult + add;
        acc3 = acc3 * mult + add;
        acc4 = acc4 * mult + add;
        acc5 = acc5 * mult + add;
        acc6 = acc6 * mult + add;
        acc7 = acc7 * mult + add;
    }
    out[0] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}

/*
 * STREAM Triad Benchmark for Hierarchy Levels (L1, L2, L3, DRAM)
 * Uses thread-local allocated arrays of size n_per_thread * sizeof(double).
 * Bytes transferred per iteration = 3 arrays * sizeof(double) * total_n = 24 * total_n.
 * Calibrated using QueryPerformanceCounter until measurement duration exceeds target_sec (>100ms).
 */
double measure_stream_hierarchy(const char* name, size_t n_per_thread, int threads, double target_sec) {
    size_t total_n = n_per_thread * (size_t)threads;
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    double total_bytes_per_iter = 3.0 * (double)sizeof(double) * (double)total_n;

    /* 1. Calibration probe with 200 iterations */
    int iters = 200;
    #pragma omp parallel num_threads(threads)
    {
        double* a = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
        double* b = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
        double* c = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
        double scalar = 3.14159;
        for (size_t i = 0; i < n_per_thread; ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 3.0; }

        #pragma omp barrier
        #pragma omp single
        {
            QueryPerformanceCounter(&t0);
        }

        for (int it = 0; it < iters; ++it) {
            #pragma omp simd
            for (size_t i = 0; i < n_per_thread; ++i) {
                a[i] = b[i] + scalar * c[i];
            }
        }

        #pragma omp barrier
        #pragma omp single
        {
            QueryPerformanceCounter(&t1);
        }

        volatile double sink = a[0];
        (void)sink;
        hpc_aligned_free(a);
        hpc_aligned_free(b);
        hpc_aligned_free(c);
    }

    double probe_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    if (probe_sec < 1e-7) probe_sec = 1e-7;
    long long calc_iters = (long long)(target_sec / probe_sec * (double)iters);
    if (calc_iters < 10) calc_iters = 10;
    iters = (int)calc_iters;

    /* 2. Measured Trials (3 repeats, take max bandwidth) */
    double best_bw = 0.0;
    double last_elapsed = 0.0;

    for (int trial = 0; trial < 3; ++trial) {
        #pragma omp parallel num_threads(threads)
        {
            double* a = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
            double* b = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
            double* c = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
            double scalar = 3.14159;
            for (size_t i = 0; i < n_per_thread; ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 3.0; }

            #pragma omp barrier
            #pragma omp single
            {
                QueryPerformanceCounter(&t0);
            }

            for (int it = 0; it < iters; ++it) {
                #pragma omp simd
                for (size_t i = 0; i < n_per_thread; ++i) {
                    a[i] = b[i] + scalar * c[i];
                }
            }

            #pragma omp barrier
            #pragma omp single
            {
                QueryPerformanceCounter(&t1);
            }

            volatile double sink = a[0];
            (void)sink;
            hpc_aligned_free(a);
            hpc_aligned_free(b);
            hpc_aligned_free(c);
        }

        double elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
        last_elapsed = elapsed;
        double bw = (total_bytes_per_iter * (double)iters / elapsed) * 1e-9;
        if (bw > best_bw) best_bw = bw;
    }

    printf("  %-5s: Working set = %6.1f KB/thread (%6.1f MB total), %7d iters, %6.3f s => %7.2f GB/s\n",
           name,
           (double)(n_per_thread * sizeof(double) * 3) / 1024.0,
           (double)(total_n * sizeof(double) * 3) / (1024.0 * 1024.0),
           iters, last_elapsed, best_bw);

    return best_bw;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int max_threads = omp_get_max_threads();

    printf("=====================================================\n");
    printf("     ROOFLINE CEILINGS MICROBENCHMARK (Phase 4)       \n");
    printf("=====================================================\n");
    printf("Detected OpenMP Max Threads: %d\n", max_threads);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    /* -------------------------------------------------------------
     * 1. Theoretical Peak Compute Calculations
     * Intel Core i5-1235U:
     *   2 P-cores (Golden Cove): 2 FMA units * 8 SP lanes * 2 FLOPs = 32 FLOP/cycle
     *   8 E-cores (Gracemont):   1 FMA unit  * 8 SP lanes * 2 FLOPs = 16 FLOP/cycle
     *
     * Max Turbo:
     *   P-cores @ 4.4 GHz: 2 * 4.4 * 32 = 281.6 GFLOP/s
     *   E-cores @ 3.3 GHz: 8 * 3.3 * 16 = 422.4 GFLOP/s
     *   Total Max Turbo Peak = 704.00 GFLOP/s
     *
     * Base Clock:
     *   P-cores @ 1.3 GHz: 2 * 1.3 * 32 = 83.2 GFLOP/s
     *   E-cores @ 0.9 GHz: 8 * 0.9 * 16 = 115.2 GFLOP/s
     *   Total Base Clock Peak = 198.40 GFLOP/s
     * ------------------------------------------------------------- */
    double theo_turbo_peak = 704.0;
    double theo_base_peak  = 198.4;

    printf("\nTheoretical Peak Compute (Intel i5-1235U Hybrid 2P+8E):\n");
    printf("  Max Turbo Peak (P@4.4GHz, E@3.3GHz):              %7.2f GFLOP/s\n", theo_turbo_peak);
    printf("  Base Clock Peak (P@1.3GHz, E@0.9GHz):             %7.2f GFLOP/s\n", theo_base_peak);

    /* -------------------------------------------------------------
     * 2. Empirical AVX2 FMA Compute Benchmarks: Burst & Sustained
     * ------------------------------------------------------------- */
    printf("\nMeasuring Empirical AVX2 FMA Compute Ceilings...\n");

    /* Burst measurement (~0.3s) */
    size_t burst_iters = 50000000;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    #pragma omp parallel num_threads(max_threads)
    {
        float out[8];
        fma_vector_kernel(burst_iters, out);
        volatile float sink = out[0];
        (void)sink;
    }
    QueryPerformanceCounter(&t1);
    double burst_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double burst_gflops = ((double)burst_iters * 256.0 * (double)max_threads / burst_sec) * 1e-9;
    printf("  Measured Burst FMA Peak (~0.3s run):             %7.2f GFLOP/s\n", burst_gflops);

    /* Sustained measurement (>= 10.0s continuous run) */
    size_t chunk_iters = 50000000;
    size_t total_sustained_iters = 0;
    QueryPerformanceCounter(&t0);
    double cur_sec = 0.0;
    while (cur_sec < 10.0) {
        #pragma omp parallel num_threads(max_threads)
        {
            float out[8];
            fma_vector_kernel(chunk_iters, out);
            volatile float sink = out[0];
            (void)sink;
        }
        total_sustained_iters += chunk_iters;
        QueryPerformanceCounter(&t1);
        cur_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    }
    double sustained_gflops = ((double)total_sustained_iters * 256.0 * (double)max_threads / cur_sec) * 1e-9;
    printf("  Measured Sustained FMA Peak (>=10s run, %5.2fs):   %7.2f GFLOP/s\n", cur_sec, sustained_gflops);

    /* -------------------------------------------------------------
     * 3. Empirical Scalar Compute Benchmark
     * ------------------------------------------------------------- */
    size_t scalar_iters = 100000000;
    QueryPerformanceCounter(&t0);
    #pragma omp parallel num_threads(max_threads)
    {
        float out[8];
        scalar_kernel(scalar_iters, out);
        volatile float sink = out[0];
        (void)sink;
    }
    QueryPerformanceCounter(&t1);
    double scalar_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double scalar_gflops = ((double)scalar_iters * 16.0 * (double)max_threads / scalar_sec) * 1e-9;
    printf("  Measured Peak Scalar Compute:                     %7.2f GFLOP/s\n", scalar_gflops);

    /* -------------------------------------------------------------
     * 4. Memory & Cache Bandwidth Hierarchy (L1, L2, L3, DRAM)
     * All tests use thread-local working sets and run >= 150-300 ms.
     * DRAM buffer: 2.5M doubles/array/thread * 8 bytes * 3 arrays * 12 threads = 720 MB (> 4x 12 MB L3).
     * ------------------------------------------------------------- */
    printf("\nMeasuring Memory Bandwidth Ceilings via STREAM Triad Hierarchy...\n");

    /* L1: 1000 doubles/array * 8 bytes * 3 arrays = 24 KB / thread (< 32-48 KB L1D) */
    double bw_l1 = measure_stream_hierarchy("L1", 1000, max_threads, 0.20);

    /* L2: 40000 doubles/array * 8 bytes * 3 arrays = 960 KB / thread (~1 MB) */
    double bw_l2 = measure_stream_hierarchy("L2", 40000, max_threads, 0.20);

    /* L3: 120000 doubles/array * 8 bytes * 3 arrays = 2.88 MB / thread (shared 12 MB L3) */
    double bw_l3 = measure_stream_hierarchy("L3", 120000, max_threads, 0.25);

    /* DRAM: 2,500,000 doubles/array * 8 bytes * 3 arrays = 60 MB / thread (720 MB total) */
    double bw_dram = measure_stream_hierarchy("DRAM", 2500000, max_threads, 0.35);

    int hierarchy_valid = (bw_l1 > bw_l2 && bw_l2 > bw_l3 && bw_l3 > bw_dram);
    printf("\nHierarchy verification: L1 (%.1f) > L2 (%.1f) > L3 (%.1f) > DRAM (%.1f): %s\n",
           bw_l1, bw_l2, bw_l3, bw_dram, hierarchy_valid ? "PASSED" : "FAILED");

    /* -------------------------------------------------------------
     * 5. Save results to results/roofline_ceilings.csv
     * ------------------------------------------------------------- */
    FILE* f = fopen("results/roofline_ceilings.csv", "w");
    if (!f) {
        fprintf(stderr, "Cannot open results/roofline_ceilings.csv for writing\n");
        return 1;
    }

    fprintf(f, "CeilingType,Value,Unit,Description\n");
    fprintf(f, "TheoreticalPeakComputeTurbo,%.2f,GFLOP/s,Intel i5-1235U Hybrid (2P@4.4GHz*32 + 8E@3.3GHz*16) Max Turbo Peak\n", theo_turbo_peak);
    fprintf(f, "TheoreticalPeakComputeBase,%.2f,GFLOP/s,Intel i5-1235U Hybrid (2P@1.3GHz*32 + 8E@0.9GHz*16) Base Clock Peak\n", theo_base_peak);
    fprintf(f, "MeasuredBurstFMAPeak,%.2f,GFLOP/s,Empirical AVX2 FMA Initial Burst (~0.3s) at Max Turbo\n", burst_gflops);
    fprintf(f, "MeasuredSustainedFMAPeak,%.2f,GFLOP/s,Empirical AVX2 FMA Sustained (>=10s continuous run) Compute Roof\n", sustained_gflops);
    fprintf(f, "MeasuredScalarPeak,%.2f,GFLOP/s,Empirical Scalar Single Precision Peak\n", scalar_gflops);

    if (hierarchy_valid) {
        fprintf(f, "L1Bandwidth,%.2f,GB/s,STREAM Triad L1 Data Cache (~24 KB/thread working set)\n", bw_l1);
        fprintf(f, "L2Bandwidth,%.2f,GB/s,STREAM Triad L2 Cache (~1 MB/thread working set)\n", bw_l2);
        fprintf(f, "L3Bandwidth,%.2f,GB/s,STREAM Triad L3 Cache (2.88 MB/thread slice of 12 MB shared L3)\n", bw_l3);
    }
    fprintf(f, "DRAMBandwidth,%.2f,GB/s,STREAM Triad DRAM (720 MB total array > 4x L3 cache)\n", bw_dram);

    fclose(f);
    printf("\nSuccessfully wrote updated ceilings to results/roofline_ceilings.csv\n");
    return 0;
}
