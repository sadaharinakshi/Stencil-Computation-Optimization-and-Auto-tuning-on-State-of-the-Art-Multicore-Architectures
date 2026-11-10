#include "common.h"
#include "kernels_3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * ============================================================================
 * 3D 7-Point Heat Stencil Numerical Correctness Verification
 * ============================================================================
 *
 * Verifies that all 4 optimized implementations:
 *   - OpenMP (Naive)
 *   - OpenMP + First Touch
 *   - OpenMP + 3D Cache Blocking
 *   - OpenMP + First Touch + 3D Cache Blocking
 *
 * produce bitwise or numerically identical outputs to the Serial Reference.
 *
 * Tests:
 * 1. Standard divisible grid sizes: 64^3, 128^3
 * 2. Non-divisible prime/odd grid sizes: 67^3, 100^3, 113^3 (verifies boundary/partial tile logic)
 * 3. Multiple 3D tile configurations: (16x16x16), (32x32x16), (64x32x16), (16x32x32)
 * 4. Multi-step iterations (1 step and 5 steps with ping-pong buffer swaps)
 *
 * Tolerance for IEEE 754 double precision: 1.0e-12.
 * ============================================================================
 */

typedef struct {
    double max_abs_diff;
    double l2_rel_diff;
    int passed;
} CheckResult;

static CheckResult compare_grids(const double* ref, const double* test, size_t n, double tol) {
    CheckResult res;
    res.max_abs_diff = 0.0;
    res.passed = 1;

    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double diff = fabs(ref[i] - test[i]);
        if (diff > res.max_abs_diff) {
            res.max_abs_diff = diff;
        }
        sum_sq_diff += diff * diff;
        sum_sq_ref += ref[i] * ref[i];
    }

    res.l2_rel_diff = (sum_sq_ref > 0.0) ? sqrt(sum_sq_diff / sum_sq_ref) : sqrt(sum_sq_diff);
    if (res.max_abs_diff > tol) {
        res.passed = 0;
    }
    return res;
}

/*
 * Execute multi-step simulation with pointer ping-pong double buffering.
 */
static void run_multi_step(VariantType vtype, double* buf_a, double* buf_b,
                           int nx, int ny, int nz, int steps, const StencilConfig* cfg) {
    kernel_3d_fn fn = get_kernel_3d(vtype);
    double* curr = buf_a;
    double* next = buf_b;

    for (int s = 0; s < steps; ++s) {
        fn(curr, next, nx, ny, nz, cfg);
        /* Swap pointers after each time step */
        double* tmp = curr;
        curr = next;
        next = tmp;
    }

    /* Ensure final output is in buf_b if odd steps, or copy to buf_b if in buf_a */
    if (curr != buf_b) {
        size_t total = (size_t)nx * ny * nz;
        memcpy(buf_b, curr, total * sizeof(double));
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    const double TOLERANCE = 1.0e-12;
    int total_tests = 0;
    int passed_tests = 0;

    printf("==================================================================\n");
    printf("     3D HEAT STENCIL NUMERICAL CORRECTNESS VERIFICATION SUITE      \n");
    printf("==================================================================\n");
    printf("Double-precision tolerance: %.1e\n", TOLERANCE);
    printf("Comparing all 4 parallel implementations against Serial Reference.\n\n");

    FILE* flog = fopen("results/correctness_log.txt", "w");
    if (!flog) {
        fprintf(stderr, "Warning: Could not open results/correctness_log.txt for writing.\n");
    }

    /* Grid test configurations */
    int test_grids[][3] = {
        {64, 64, 64},       /* Divisible standard size */
        {67, 67, 67},       /* Prime/odd non-divisible size */
        {100, 100, 100},    /* Non-power-of-2 size */
        {113, 97, 83},      /* Anisotropic prime non-divisible dimensions */
        {128, 128, 128}     /* Larger power-of-2 size */
    };
    int num_grids = sizeof(test_grids) / sizeof(test_grids[0]);

    /* Tile size configurations to verify genuine 3D blocking with partial boundaries */
    int test_tiles[][3] = {
        {16, 16, 16},
        {32, 32, 16},
        {64, 32, 16},
        {16, 32, 32}
    };
    int num_tiles = sizeof(test_tiles) / sizeof(test_tiles[0]);

    VariantType parallel_variants[4] = {
        VAR_OMP,
        VAR_OMP_NUMA,
        VAR_OMP_BLOCKED,
        VAR_OMP_NUMA_BLOCKED
    };

    for (int g = 0; g < num_grids; ++g) {
        int nx = test_grids[g][0];
        int ny = test_grids[g][1];
        int nz = test_grids[g][2];
        size_t num_grid = (size_t)nx * ny * nz;

        printf("------------------------------------------------------------------\n");
        printf(">>> Testing Grid: %d x %d x %d (%zu elements, %.2f MB)\n",
               nx, ny, nz, num_grid, (double)num_grid * sizeof(double) / (1024.0 * 1024.0));
        printf("------------------------------------------------------------------\n");

        double* in_ref  = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
        double* out_ref = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
        double* in_test = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
        double* out_test= (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));

        if (!in_ref || !out_ref || !in_test || !out_test) {
            fprintf(stderr, "Error: Memory allocation failed for grid %dx%dx%d\n", nx, ny, nz);
            continue;
        }

        /* 1. Single-step correctness check for non-blocked variants */
        serial_init_3d(in_ref, out_ref, nx, ny, nz);
        StencilConfig cfg_ref = default_stencil_config();
        heat3d_serial(in_ref, out_ref, nx, ny, nz, cfg_ref.alpha);

        for (int v = 0; v < 4; ++v) {
            VariantType vtype = parallel_variants[v];

            /* If it's a blocked variant, test all tile configurations */
            int tile_runs = (vtype == VAR_OMP_BLOCKED || vtype == VAR_OMP_NUMA_BLOCKED) ? num_tiles : 1;

            for (int t = 0; t < tile_runs; ++t) {
                StencilConfig cfg = default_stencil_config();
                if (vtype == VAR_OMP_BLOCKED || vtype == VAR_OMP_NUMA_BLOCKED) {
                    cfg.bx = test_tiles[t][0];
                    cfg.by = test_tiles[t][1];
                    cfg.bz = test_tiles[t][2];
                }

                /* Initialize test buffers */
                if (vtype == VAR_OMP_NUMA || vtype == VAR_OMP_NUMA_BLOCKED) {
                    first_touch_init_3d(in_test, out_test, nx, ny, nz, cfg.num_threads);
                } else {
                    serial_init_3d(in_test, out_test, nx, ny, nz);
                }

                kernel_3d_fn fn = get_kernel_3d(vtype);
                fn(in_test, out_test, nx, ny, nz, &cfg);

                CheckResult res = compare_grids(out_ref, out_test, num_grid, TOLERANCE);
                total_tests++;
                if (res.passed) passed_tests++;

                char tile_str[32] = "";
                if (vtype == VAR_OMP_BLOCKED || vtype == VAR_OMP_NUMA_BLOCKED) {
                    snprintf(tile_str, sizeof(tile_str), " [Tile %dx%dx%d]", cfg.bx, cfg.by, cfg.bz);
                }

                printf("  %-28s%-18s : MaxDiff = %9.2e  L2Rel = %9.2e => [%s]\n",
                       variant_name(vtype), tile_str,
                       res.max_abs_diff, res.l2_rel_diff,
                       res.passed ? "PASSED" : "FAILED");

                if (flog) {
                    fprintf(flog, "Grid=%dx%dx%d Variant=%s Tile=%dx%dx%d MaxDiff=%.3e L2Rel=%.3e Status=%s\n",
                            nx, ny, nz, variant_name(vtype), cfg.bx, cfg.by, cfg.bz,
                            res.max_abs_diff, res.l2_rel_diff, res.passed ? "PASSED" : "FAILED");
                }
            }
        }

        /* 2. Multi-step ping-pong double buffering check (5 steps) */
        int steps = 5;
        serial_init_3d(in_ref, out_ref, nx, ny, nz);
        run_multi_step(VAR_SERIAL, in_ref, out_ref, nx, ny, nz, steps, &cfg_ref);

        /* Test blocked variant over multiple steps */
        StencilConfig cfg_ms = default_stencil_config();
        cfg_ms.bx = 32; cfg_ms.by = 32; cfg_ms.bz = 16;
        serial_init_3d(in_test, out_test, nx, ny, nz);
        run_multi_step(VAR_OMP_BLOCKED, in_test, out_test, nx, ny, nz, steps, &cfg_ms);

        CheckResult res_ms = compare_grids(out_ref, out_test, num_grid, TOLERANCE);
        total_tests++;
        if (res_ms.passed) passed_tests++;

        printf("  Multi-Step (5 steps ping-pong) [Tile 32x32x16]: MaxDiff = %9.2e  L2Rel = %9.2e => [%s]\n",
               res_ms.max_abs_diff, res_ms.l2_rel_diff, res_ms.passed ? "PASSED" : "FAILED");

        if (flog) {
            fprintf(flog, "Grid=%dx%dx%d MultiStep=5 Variant=OMP_BLOCKED Tile=32x32x16 MaxDiff=%.3e Status=%s\n",
                    nx, ny, nz, res_ms.max_abs_diff, res_ms.passed ? "PASSED" : "FAILED");
        }

        hpc_aligned_free(in_ref);
        hpc_aligned_free(out_ref);
        hpc_aligned_free(in_test);
        hpc_aligned_free(out_test);
    }

    if (flog) fclose(flog);

    printf("\n==================================================================\n");
    printf("CORRECTNESS SUMMARY: %d / %d tests passed (Tolerance: %.1e)\n",
           passed_tests, total_tests, TOLERANCE);
    printf("==================================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
