#ifndef KERNELS_3D_H
#define KERNELS_3D_H

#include "common.h"

typedef void (*kernel_3d_fn)(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v0: Serial Naive Reference */
void heat3d_v0_serial(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v1: OpenMP Parallel-For Naive */
void heat3d_v1_omp(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v2: NUMA First-Touch Init */
void first_touch_init_3d(double* buf, int nx, int ny, int nz, double val, const TuningConfig* cfg);
void heat3d_v2_numa(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v3: Cache Blocking / 3D Tiling (TX, TY, TZ) */
void heat3d_v3_tiled(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v4: SIMD Vectorization + Register Reuse across planes */
void heat3d_v4_simd(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v5: Software Prefetch + Non-Temporal Stores */
void heat3d_v5_prefetch(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* v6: Datta et al. Z-unrolling / Register Blocking */
void heat3d_v6_unroll_reg(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg);

/* Dispatcher */
kernel_3d_fn get_kernel_3d(VariantType vtype);

#endif /* KERNELS_3D_H */

