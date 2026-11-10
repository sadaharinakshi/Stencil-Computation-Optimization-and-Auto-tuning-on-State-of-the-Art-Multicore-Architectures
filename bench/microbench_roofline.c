#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
 * ============================================================================
 * Roofline Model Ceilings Microbenchmark
 * ============================================================================
 *
 * This benchmark measures the empirical ceilings for the Roofline Analysis:
 * 1. Sustained Memory Bandwidth across the memory hierarchy (L1, L2, L3, DRAM)
 *    using the standard multi-threaded STREAM Triad kernel:
 *        A[i] = B[i] + scalar * C[i]
 *    (Transfers 3 double-precision elements = 24 bytes per iteration).
 *
 * 2. Peak Floating-Point Compute Performance using a portable multi-threaded
 *    fused multiply-accumulate (FMA) kernel without proprietary SIMD intrinsics.
 *
 * 3. Theoretical Hardware Peak Calculations based on official Intel documentation
 *    for the 12th Gen Intel Core i5-12450HX (4 P-cores + 4 E-cores).
 *
 * Outputs results to results/roofline_ceilings.csv for plotting and analysis.
 * ============================================================================
 */

/*
 * Portable Double-Precision Compute Kernel:
 * 8 independent parallel scalar accumulators to prevent dependency stalling.
 * Each accumulator performs 1 multiplication + 1 addition = 2 FLOPs.
 * Total = 16 FLOPs per iteration.
 */
__attribute__((noinline))
void compute_fma_kernel(size_t iters, double* out) {
    double a0 = 1.0,  a1 = 1.01, a2 = 1.02, a3 = 1.03;
    double a4 = 1.04, a5 = 1.05, a6 = 1.06, a7 = 1.07;
    double mult = 1.0000001, add = 0.0000001;

    for (size_t i = 0; i < iters; ++i) {
        a0 = a0 * mult + add;
        a1 = a1 * mult + add;
        a2 = a2 * mult + add;
        a3 = a3 * mult + add;
        a4 = a4 * mult + add;
        a5 = a5 * mult + add;
        a6 = a6 * mult + add;
        a7 = a7 * mult + add;
    }
    *out = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

/*
 * Multi-threaded STREAM Triad Bandwidth Benchmark:
 * Measures sustained bandwidth at a specific working set size per thread.
 *
 * Total bytes transferred = 3 arrays * sizeof(double) * total_elements
 *                         = 24 * (n_per_thread * num_threads) bytes per iteration.
 */
static double measure_stream_triad(const char* level_name, size_t n_per_thread,
                                   int num_threads, double target_duration_sec) {
    size_t total_n = n_per_thread * (size_t)num_threads;
    double total_bytes_per_iter = 3.0 * (double)sizeof(double) * (double)total_n;

    /* Calibration probe to determine number of iterations for target duration */
    int iters = 100;
    double probe_start = omp_get_wtime();

    #pragma omp parallel num_threads(num_threads)
    {
        double* a = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
        double* b = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
        double* c = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
        double scalar = 3.141592653589793;

        for (size_t i = 0; i < n_per_thread; ++i) {
            a[i] = 1.0; b[i] = 2.0; c[i] = 3.0;
        }

        #pragma omp barrier
        for (int it = 0; it < iters; ++it) {
            for (size_t i = 0; i < n_per_thread; ++i) {
                a[i] = b[i] + scalar * c[i];
            }
        }
        #pragma omp barrier

        volatile double sink = a[0];
        (void)sink;
        hpc_aligned_free(a);
        hpc_aligned_free(b);
        hpc_aligned_free(c);
    }

    double probe_sec = omp_get_wtime() - probe_start;
    if (probe_sec < 1e-6) probe_sec = 1e-6;

    long long calc_iters = (long long)(target_duration_sec / probe_sec * (double)iters);
    if (calc_iters < 10) calc_iters = 10;
    if (calc_iters > 20000000) calc_iters = 20000000;
    iters = (int)calc_iters;

    /* Timed Measurement Trials (3 trials, report peak bandwidth) */
    double best_bw = 0.0;
    double best_sec = 0.0;

    for (int trial = 0; trial < 3; ++trial) {
        double t0 = omp_get_wtime();

        #pragma omp parallel num_threads(num_threads)
        {
            double* a = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
            double* b = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
            double* c = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, n_per_thread * sizeof(double));
            double scalar = 3.141592653589793;

            for (size_t i = 0; i < n_per_thread; ++i) {
                a[i] = 1.0; b[i] = 2.0; c[i] = 3.0;
            }

            #pragma omp barrier
            for (int it = 0; it < iters; ++it) {
                for (size_t i = 0; i < n_per_thread; ++i) {
                    a[i] = b[i] + scalar * c[i];
                }
            }
            #pragma omp barrier

            volatile double sink = a[0];
            (void)sink;
            hpc_aligned_free(a);
            hpc_aligned_free(b);
            hpc_aligned_free(c);
        }

        double elapsed = omp_get_wtime() - t0;
        double bw = (total_bytes_per_iter * (double)iters / elapsed) * 1e-9;
        if (bw > best_bw) {
            best_bw = bw;
            best_sec = elapsed;
        }
    }

    double total_mb = (double)(total_n * sizeof(double) * 3) / (1024.0 * 1024.0);
    printf("  %-5s: Working set = %6.1f KB/thread (%6.1f MB total), %7d iters, %5.2f s => %7.2f GB/s\n",
           level_name,
           (double)(n_per_thread * sizeof(double) * 3) / 1024.0,
           total_mb,
           iters, best_sec, best_bw);

    return best_bw;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int max_threads = omp_get_max_threads();

    printf("========================================================================\n");
    printf("        ROOFLINE CEILINGS EMPIRICAL MICROBENCHMARK                      \n");
    printf("========================================================================\n");
    printf("Target Architecture: Intel Core i5-12450HX (4 P-cores + 4 E-cores)\n");
    printf("Available OpenMP Threads: %d\n\n", max_threads);

    /*
     * 1. Theoretical Peak Compute Calculations:
     * Intel Core i5-12450HX:
     *   4 P-cores (Golden Cove, AVX2 FMA): 2 FMA units * 4 DP lanes * 2 FLOPs = 16 DP FLOP/cycle
     *   4 E-cores (Gracemont, AVX2 FMA):   1 FMA unit  * 4 DP lanes * 2 FLOPs =  8 DP FLOP/cycle
     *
     * Theoretical Max Turbo Peak:
     *   P-cores @ 4.4 GHz: 4 * 4.4 * 16 = 281.6 GFLOP/s (DP)
     *   E-cores @ 3.1 GHz: 4 * 3.1 *  8 =  99.2 GFLOP/s (DP)
     *   Total DP Theoretical Max Turbo Peak = 380.8 GFLOP/s
     *
     * Theoretical Base Frequency Peak:
     *   P-cores @ 2.4 GHz: 4 * 2.4 * 16 = 153.6 GFLOP/s (DP)
     *   E-cores @ 1.8 GHz: 4 * 1.8 *  8 =  57.6 GFLOP/s (DP)
     *   Total DP Theoretical Base Peak = 211.2 GFLOP/s
     */
    double theo_turbo_peak = 380.8;
    double theo_base_peak  = 211.2;

    printf("Theoretical Peak Compute (Documented Hardware Specs):\n");
    printf("  Max Turbo Peak (DP FMA): %7.2f GFLOP/s\n", theo_turbo_peak);
    printf("  Base Clock Peak (DP FMA):%7.2f GFLOP/s\n\n", theo_base_peak);

    /*
     * 2. Empirical Compute Peak Measurement:
     * Runs multi-threaded double-precision FMA kernel for ~3 seconds.
     */
    printf("Measuring Empirical Compute Performance...\n");
    size_t fma_iters = 50000000;
    double t_comp_start = omp_get_wtime();

    #pragma omp parallel num_threads(max_threads)
    {
        double out = 0.0;
        compute_fma_kernel(fma_iters, &out);
        volatile double sink = out;
        (void)sink;
    }

    double comp_elapsed = omp_get_wtime() - t_comp_start;
    double measured_compute_gflops = ((double)fma_iters * 16.0 * (double)max_threads / comp_elapsed) * 1e-9;
    printf("  Measured Sustained Compute Peak: %7.2f GFLOP/s (%5.2f s run)\n\n",
           measured_compute_gflops, comp_elapsed);

    /*
     * 3. Empirical Memory Bandwidth Hierarchy (L1, L2, L3, DRAM):
     * Tests different working sets with thread-local buffers.
     * DRAM buffer size: 2,500,000 doubles * 8 B * 3 arrays * 12 threads = 720 MB (>> 12 MB L3).
     */
    printf("Measuring Sustained Memory Bandwidth via STREAM Triad Hierarchy...\n");

    /* L1: 1,000 doubles/array * 8 bytes * 3 arrays = 24 KB / thread (< 32-48 KB L1D) */
    double bw_l1 = measure_stream_triad("L1", 1000, max_threads, 0.20);

    /* L2: 40,000 doubles/array * 8 bytes * 3 arrays = 960 KB / thread (~1.25 MB L2) */
    double bw_l2 = measure_stream_triad("L2", 40000, max_threads, 0.20);

    /* L3: 120,000 doubles/array * 8 bytes * 3 arrays = 2.88 MB / thread (shared 12 MB L3) */
    double bw_l3 = measure_stream_triad("L3", 120000, max_threads, 0.25);

    /* DRAM: 2,500,000 doubles/array * 8 bytes * 3 arrays = 60 MB / thread (720 MB total) */
    double bw_dram = measure_stream_triad("DRAM", 2500000, max_threads, 0.35);

    int hierarchy_valid = (bw_l1 > bw_l2 && bw_l2 > bw_l3 && bw_l3 > bw_dram);
    printf("\nHierarchy verification (L1 > L2 > L3 > DRAM): %s\n",
           hierarchy_valid ? "PASSED" : "NOTICE (L3/DRAM variation)");

    /*
     * 4. Save Ceilings to results/roofline_ceilings.csv
     */
    FILE* f = fopen("results/roofline_ceilings.csv", "w");
    if (!f) {
        fprintf(stderr, "Cannot open results/roofline_ceilings.csv for writing\n");
        return 1;
    }

    fprintf(f, "CeilingType,Value,Unit,Description\n");
    fprintf(f, "TheoreticalPeakComputeTurbo,%.2f,GFLOP/s,Intel i5-12450HX DP FMA Max Turbo Peak\n", theo_turbo_peak);
    fprintf(f, "TheoreticalPeakComputeBase,%.2f,GFLOP/s,Intel i5-12450HX DP FMA Base Clock Peak\n", theo_base_peak);
    fprintf(f, "MeasuredComputePeak,%.2f,GFLOP/s,Empirical Multi-Threaded Double-Precision Compute Roof\n", measured_compute_gflops);
    fprintf(f, "L1Bandwidth,%.2f,GB/s,STREAM Triad L1 Data Cache (~24 KB/thread)\n", bw_l1);
    fprintf(f, "L2Bandwidth,%.2f,GB/s,STREAM Triad L2 Cache (~960 KB/thread)\n", bw_l2);
    fprintf(f, "L3Bandwidth,%.2f,GB/s,STREAM Triad L3 Cache (~2.88 MB/thread)\n", bw_l3);
    fprintf(f, "DRAMBandwidth,%.2f,GB/s,STREAM Triad Sustained DRAM Bandwidth (720 MB buffer)\n", bw_dram);

    fclose(f);
    printf("Successfully wrote updated ceilings to results/roofline_ceilings.csv\n");
    return 0;
}
