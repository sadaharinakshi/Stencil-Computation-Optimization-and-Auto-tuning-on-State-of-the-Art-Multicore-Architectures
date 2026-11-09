#include "common.h"
#include "kernels_2d.h"
#include "kernels_3d.h"
#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double max_abs_diff;
    double l2_rel_diff;
    int passed;
} CheckResult;

static CheckResult compare_buffers_float(const float* ref, const float* test, size_t n, double tol) {
    CheckResult res;
    res.max_abs_diff = 0.0;
    res.passed = 1;

    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double diff = fabs((double)ref[i] - (double)test[i]);
        if (diff > res.max_abs_diff) {
            res.max_abs_diff = diff;
        }
        sum_sq_diff += diff * diff;
        sum_sq_ref += (double)ref[i] * (double)ref[i];
    }

    res.l2_rel_diff = (sum_sq_ref > 0.0) ? sqrt(sum_sq_diff / sum_sq_ref) : sqrt(sum_sq_diff);
    if (res.max_abs_diff > tol) {
        res.passed = 0;
    }
    return res;
}

static CheckResult compare_buffers_double(const double* ref, const double* test, size_t n, double tol) {
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

void run_visual_verification(void) {
    printf("=== Generating Visual Test Images (Phase 1) ===\n");
    int w = 512, h = 512;
    float* clean = allocate_image_float(w, h);
    float* noisy_g = allocate_image_float(w, h);
    float* noisy_sp = allocate_image_float(w, h);
    float* out = allocate_image_float(w, h);

    generate_synthetic_image(clean, w, h, NOISE_NONE, 0.0f, 42);
    generate_synthetic_image(noisy_g, w, h, NOISE_GAUSSIAN, 25.0f, 42);
    generate_synthetic_image(noisy_sp, w, h, NOISE_SALT_PEPPER, 0.08f, 42);

    save_image_float("results/visual_clean.png", clean, w, h);
    save_image_float("results/visual_noisy_gaussian.png", noisy_g, w, h);
    save_image_float("results/visual_noisy_sp.png", noisy_sp, w, h);

    TuningConfig cfg = default_tuning_config();

    /* Gaussian 3x3 on Gaussian noisy image */
    gauss3_v0_serial(noisy_g, out, w, h, &cfg);
    save_image_float("results/visual_gauss3_denoised.png", out, w, h);

    /* Gaussian 5x5 on Gaussian noisy image */
    gauss5_v0_serial(noisy_g, out, w, h, &cfg);
    save_image_float("results/visual_gauss5_denoised.png", out, w, h);

    /* Sobel on clean image */
    sobel_v0_serial(clean, out, w, h, &cfg);
    save_image_float("results/visual_sobel_clean.png", out, w, h);

    /* Sobel on Gaussian noisy image */
    sobel_v0_serial(noisy_g, out, w, h, &cfg);
    save_image_float("results/visual_sobel_noisy.png", out, w, h);

    /* Laplacian on clean image */
    laplace_v0_serial(clean, out, w, h, &cfg);
    save_image_float("results/visual_laplace_clean.png", out, w, h);

    free_image_float(clean);
    free_image_float(noisy_g);
    free_image_float(noisy_sp);
    free_image_float(out);
    printf("Saved visual outputs to results/visual_*.png successfully.\n");
}

int main(int argc, char** argv) {
    int max_variant = VAR_V0_SERIAL;
    if (argc > 1) {
        max_variant = atoi(argv[1]);
    }

    printf("=====================================================\n");
    printf("    HPC STENCIL CORRECTNESS CHECKER (up to v%d)       \n", max_variant);
    printf("=====================================================\n");

    /* 1. Generate visual PNGs */
    run_visual_verification();

    /* 2. Numerical Correctness for 2-D Kernels */
    const int w = 1024, h = 1024;
    size_t num_pixels = (size_t)w * h;
    float* in_2d = allocate_image_float(w, h);
    float* ref_2d = allocate_image_float(w, h);
    float* test_2d = allocate_image_float(w, h);

    generate_synthetic_image(in_2d, w, h, NOISE_GAUSSIAN, 20.0f, 999);

    TuningConfig cfg = default_tuning_config();
    int all_passed = 1;
    const double TOL_FLOAT = 1e-5;

    Kernel2DType kernels[4] = {KERNEL_GAUSS3, KERNEL_GAUSS5, KERNEL_SOBEL, KERNEL_LAPLACE};

    for (int k = 0; k < 4; ++k) {
        Kernel2DType ktype = kernels[k];
        kernel_2d_fn ref_fn = get_kernel_2d(ktype, VAR_V0_SERIAL);
        if (!ref_fn) {
            fprintf(stderr, "Error: missing reference kernel for %s\n", kernel_name(ktype));
            return 1;
        }

        ref_fn(in_2d, ref_2d, w, h, &cfg);

        for (int v = 1; v <= max_variant; ++v) {
            kernel_2d_fn test_fn = get_kernel_2d(ktype, (VariantType)v);
            if (!test_fn) continue;

            memset(test_2d, 0, num_pixels * sizeof(float));
            test_fn(in_2d, test_2d, w, h, &cfg);

            CheckResult res = compare_buffers_float(ref_2d, test_2d, num_pixels, TOL_FLOAT);
            printf("[2D %-10s %-15s] MaxAbsDiff = %.3e, RelL2 = %.3e => %s\n",
                   kernel_name(ktype), variant_name((VariantType)v),
                   res.max_abs_diff, res.l2_rel_diff,
                   res.passed ? "PASSED" : "FAILED");
            if (!res.passed) all_passed = 0;
        }
    }

    /* 3. Numerical Correctness for 3-D 7-Point Stencil */
    const int nx = 128, ny = 128, nz = 128;
    size_t num_grid = (size_t)nx * ny * nz;
    double* in_3d = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
    double* ref_3d = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));
    double* test_3d = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, num_grid * sizeof(double));

    for (size_t i = 0; i < num_grid; ++i) {
        in_3d[i] = sin((double)i * 0.001);
    }

    double c0 = 0.5, c1 = 0.08333333333333333; /* 1.0 / 12.0 */
    kernel_3d_fn ref_3d_fn = get_kernel_3d(VAR_V0_SERIAL);
    ref_3d_fn(in_3d, ref_3d, nx, ny, nz, c0, c1, &cfg);

    const double TOL_DOUBLE = 1e-12;
    for (int v = 1; v <= max_variant; ++v) {
        kernel_3d_fn test_3d_fn = get_kernel_3d((VariantType)v);
        if (!test_3d_fn) continue;

        memset(test_3d, 0, num_grid * sizeof(double));
        test_3d_fn(in_3d, test_3d, nx, ny, nz, c0, c1, &cfg);

        CheckResult res = compare_buffers_double(ref_3d, test_3d, num_grid, TOL_DOUBLE);
        printf("[3D %-10s %-15s] MaxAbsDiff = %.3e, RelL2 = %.3e => %s\n",
               "Heat7pt", variant_name((VariantType)v),
               res.max_abs_diff, res.l2_rel_diff,
               res.passed ? "PASSED" : "FAILED");
        if (!res.passed) all_passed = 0;
    }

    free_image_float(in_2d);
    free_image_float(ref_2d);
    free_image_float(test_2d);
    hpc_aligned_free(in_3d);
    hpc_aligned_free(ref_3d);
    hpc_aligned_free(test_3d);

    if (all_passed) {
        printf("\n>>> ALL CORRECTNESS CHECKS PASSED (Tolerance float: 1e-5, double: 1e-12) <<<\n");
        return 0;
    } else {
        printf("\n>>> ONE OR MORE CORRECTNESS CHECKS FAILED <<<\n");
        return 1;
    }
}

