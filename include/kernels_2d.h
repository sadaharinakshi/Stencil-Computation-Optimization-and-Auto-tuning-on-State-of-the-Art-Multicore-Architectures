#ifndef KERNELS_2D_H
#define KERNELS_2D_H

#include "common.h"

typedef void (*kernel_2d_fn)(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* v0: Serial Naive Reference Kernels */
void gauss3_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void gauss5_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void sobel_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void laplace_v0_serial(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* v1: OpenMP Parallel-For Naive */
void gauss3_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void gauss5_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void sobel_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void laplace_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* v2: NUMA First-Touch Init + Thread Pinning */
void first_touch_init_2d(float* buf, int width, int height, float val, const TuningConfig* cfg);
void gauss3_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void gauss5_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void sobel_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void laplace_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* v3: Cache Blocking / 2D Tiling (TX, TY) */
void gauss3_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void gauss5_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void sobel_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void laplace_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* v4: SIMD Vectorization + Sliding Window / Register Reuse */
void gauss3_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void gauss5_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void sobel_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void laplace_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* v5: Software Prefetch + Non-Temporal Streaming Stores */
void gauss3_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void gauss5_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void sobel_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg);
void laplace_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg);

/* Dispatcher */
kernel_2d_fn get_kernel_2d(Kernel2DType ktype, VariantType vtype);

#endif /* KERNELS_2D_H */

