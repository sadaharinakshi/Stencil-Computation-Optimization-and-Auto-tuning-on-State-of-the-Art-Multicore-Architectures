#include "common.h"
#include "kernels_2d.h"
#include "kernels_3d.h"
#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    char kernel_str[64] = "gauss3";
    int dim = 2048;
    int tx = 64;
    int ty = 32;
    int tz = 16;
    int threads = omp_get_max_threads();
    int sched_type = 0; /* static */
    int chunk_size = 0;
    int use_simd = 1;
    int prefetch_dist = 16;
    int trials = 5;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--kernel") && i + 1 < argc) strncpy(kernel_str, argv[++i], 63);
        else if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tx") && i + 1 < argc) tx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ty") && i + 1 < argc) ty = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tz") && i + 1 < argc) tz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sched") && i + 1 < argc) sched_type = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--simd") && i + 1 < argc) use_simd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefetch") && i + 1 < argc) prefetch_dist = atoi(argv[++i]);
    }

    TuningConfig cfg;
    cfg.tx = tx;
    cfg.ty = ty;
    cfg.tz = tz;
    cfg.num_threads = threads;
    cfg.sched_type = sched_type;
    cfg.chunk_size = chunk_size;
    cfg.use_simd = use_simd;
    cfg.prefetch_dist = prefetch_dist;

    /* Set OpenMP schedule */
    omp_sched_t osched = omp_sched_static;
    if (sched_type == 1) osched = omp_sched_dynamic;
    else if (sched_type == 2) osched = omp_sched_guided;
    omp_set_schedule(osched, chunk_size > 0 ? chunk_size : 1);

    if (!strcmp(kernel_str, "heat3d")) {
        /* 3D Stencil */
        int nx = dim, ny = dim, nz = dim;
        size_t total = (size_t)nx * ny * nz;
        double* in = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, total * sizeof(double));
        double* out = (double*)hpc_aligned_alloc(ALIGNMENT_BYTES, total * sizeof(double));
        for (size_t i = 0; i < total; ++i) in[i] = 1.0;

        kernel_3d_fn fn = use_simd ? get_kernel_3d(VAR_V3_TILING) : get_kernel_3d(VAR_V1_OMP_NAIVE);
        if (prefetch_dist > 0 && use_simd) fn = get_kernel_3d(VAR_V5_PREFETCH_NT);

        /* Calibration */
        double t_c0 = omp_get_wtime();
        fn(in, out, nx, ny, nz, 0.5, 1.0/12.0, &cfg);
        double t_c1 = omp_get_wtime();
        double srun = t_c1 - t_c0;
        int reps = 1;
        if (srun < 0.02 && srun > 0.0) reps = (int)(0.02 / srun);
        if (reps < 1) reps = 1;

        double* tlist = (double*)malloc(trials * sizeof(double));
        for (int t = 0; t < trials; ++t) {
            double t0 = omp_get_wtime();
            for (int r = 0; r < reps; ++r) {
                fn(in, out, nx, ny, nz, 0.5, 1.0/12.0, &cfg);
            }
            double t1 = omp_get_wtime();
            tlist[t] = (t1 - t0) / (double)reps;
        }
        qsort(tlist, trials, sizeof(double), cmp_double);
        double med = tlist[trials / 2];
        double gflops = (STENCIL3D_FLOPS_PER_GRID * (double)total / med) * 1e-9;
        double gbs = (STENCIL3D_IDEAL_BYTES * (double)total / med) * 1e-9;

        printf("RESULT: time_ms=%.3f, gflops=%.2f, gb_s=%.2f\n", med * 1000.0, gflops, gbs);

        free(tlist);
        hpc_aligned_free(in);
        hpc_aligned_free(out);
    } else {
        /* 2D Kernel */
        Kernel2DType ktype = KERNEL_GAUSS3;
        if (!strcmp(kernel_str, "gauss5")) ktype = KERNEL_GAUSS5;
        else if (!strcmp(kernel_str, "sobel")) ktype = KERNEL_SOBEL;
        else if (!strcmp(kernel_str, "laplace")) ktype = KERNEL_LAPLACE;

        size_t total = (size_t)dim * dim;
        float* in = allocate_image_float(dim, dim);
        float* out = allocate_image_float(dim, dim);
        generate_synthetic_image(in, dim, dim, NOISE_GAUSSIAN, 20.0f, 12345);

        kernel_2d_fn fn = use_simd ? get_kernel_2d(ktype, VAR_V3_TILING) : get_kernel_2d(ktype, VAR_V1_OMP_NAIVE);
        if (prefetch_dist > 0 && use_simd) fn = get_kernel_2d(ktype, VAR_V5_PREFETCH_NT);

        /* Calibration */
        double t_c0 = omp_get_wtime();
        fn(in, out, dim, dim, &cfg);
        double t_c1 = omp_get_wtime();
        double srun = t_c1 - t_c0;
        int reps = 1;
        if (srun < 0.02 && srun > 0.0) reps = (int)(0.02 / srun);
        if (reps < 1) reps = 1;

        double* tlist = (double*)malloc(trials * sizeof(double));
        for (int t = 0; t < trials; ++t) {
            double t0 = omp_get_wtime();
            for (int r = 0; r < reps; ++r) {
                fn(in, out, dim, dim, &cfg);
            }
            double t1 = omp_get_wtime();
            tlist[t] = (t1 - t0) / (double)reps;
        }
        qsort(tlist, trials, sizeof(double), cmp_double);
        double med = tlist[trials / 2];
        double gflops = (get_kernel_flops_per_pixel(ktype) * (double)total / med) * 1e-9;
        double gbs = (get_kernel_ideal_bytes_per_pixel(ktype) * (double)total / med) * 1e-9;

        printf("RESULT: time_ms=%.3f, gflops=%.2f, gb_s=%.2f\n", med * 1000.0, gflops, gbs);

        free(tlist);
        free_image_float(in);
        free_image_float(out);
    }
    return 0;
}

