#include "common.h"
#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int sizes[] = {1024, 4096, 8192, 16384};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== Synthetic Image Generator (Phase 1) ===\n");
    for (int i = 0; i < num_sizes; ++i) {
        int dim = sizes[i];
        size_t bytes = (size_t)dim * dim * sizeof(float);
        printf("Generating synthetic image %dx%d (%.2f MB)... ", dim, dim, bytes / (1024.0 * 1024.0));
        fflush(stdout);

        double t0 = omp_get_wtime();
        float* img = allocate_image_float(dim, dim);
        if (!img) {
            printf("FAILED (out of memory)\n");
            continue;
        }

        /* Generate Gaussian noise image */
        generate_synthetic_image(img, dim, dim, NOISE_GAUSSIAN, 20.0f, 12345);
        double t1 = omp_get_wtime();

        /* Compute quick checksum to verify content */
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int p = 0; p < dim * 10; ++p) {
            sum += img[p];
        }

        printf("Done in %.3f s (checksum sample = %.1f)\n", t1 - t0, sum);

        /* Save PNG for 1024 size */
        if (dim == 1024) {
            save_image_float("results/synthetic_1024_gaussian.png", img, dim, dim);
            /* Also generate and save Salt & Pepper */
            generate_synthetic_image(img, dim, dim, NOISE_SALT_PEPPER, 0.05f, 54321);
            save_image_float("results/synthetic_1024_sp.png", img, dim, dim);
            printf("  Saved results/synthetic_1024_gaussian.png and synthetic_1024_sp.png\n");
        }

        free_image_float(img);
    }
    printf("Synthetic image generation test completed successfully.\n");
    return 0;
}

