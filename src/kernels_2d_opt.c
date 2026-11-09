#include "kernels_2d.h"
#include <math.h>
#include <string.h>
#include <immintrin.h>

/* Reference: K. Datta et al., "Stencil Computation Optimization and Auto-tuning
 * on State-of-the-Art Multicore Architectures", SC 2008.
 */

static inline int clamp(int val, int max_val) {
    if (val < 0) return 0;
    if (val >= max_val) return max_val - 1;
    return val;
}

/* =========================================================================
 * VARIANT 1: OpenMP Parallel-For Naive (Datta et al. Section 4.2)
 * Parallelize the outer row-loop across available OpenMP threads.
 * ========================================================================= */

void gauss3_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    const float k[3][3] = {
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
        {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
    };
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int py = clamp(y + dy, height);
                for (int dx = -1; dx <= 1; ++dx) {
                    int px = clamp(x + dx, width);
                    sum += in[py * width + px] * k[dy + 1][dx + 1];
                }
            }
            out[y * width + x] = sum;
        }
    }
}

void gauss5_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    const float inv = 1.0f / 256.0f;
    const float k[5][5] = {
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 6.0f*inv, 24.0f*inv, 36.0f*inv, 24.0f*inv, 6.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv }
    };
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                int py = clamp(y + dy, height);
                for (int dx = -2; dx <= 2; ++dx) {
                    int px = clamp(x + dx, width);
                    sum += in[py * width + px] * k[dy + 2][dx + 2];
                }
            }
            out[y * width + x] = sum;
        }
    }
}

void sobel_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        int ym1 = clamp(y - 1, height);
        int yp1 = clamp(y + 1, height);
        for (int x = 0; x < width; ++x) {
            int xm1 = clamp(x - 1, width);
            int xp1 = clamp(x + 1, width);

            float p00 = in[ym1 * width + xm1];
            float p01 = in[ym1 * width + x  ];
            float p02 = in[ym1 * width + xp1];

            float p10 = in[y   * width + xm1];
            float p12 = in[y   * width + xp1];

            float p20 = in[yp1 * width + xm1];
            float p21 = in[yp1 * width + x  ];
            float p22 = in[yp1 * width + xp1];

            float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
            float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

            out[y * width + x] = sqrtf(gx * gx + gy * gy);
        }
    }
}

void laplace_v1_omp(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        int ym1 = clamp(y - 1, height);
        int yp1 = clamp(y + 1, height);
        for (int x = 0; x < width; ++x) {
            int xm1 = clamp(x - 1, width);
            int xp1 = clamp(x + 1, width);

            float top    = in[ym1 * width + x];
            float bottom = in[yp1 * width + x];
            float left   = in[y * width + xm1];
            float right  = in[y * width + xp1];
            float center = in[y * width + x];

            out[y * width + x] = top + bottom + left + right - 4.0f * center;
        }
    }
}

/* =========================================================================
 * VARIANT 2: NUMA First-Touch Initialization (Datta et al. Section 4.2)
 * Guarantee that memory pages are mapped near the computing core by
 * touching them in parallel with identical thread scheduling.
 * ========================================================================= */

void first_touch_init_2d(float* buf, int width, int height, float val, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            buf[y * width + x] = val;
        }
    }
}

void gauss3_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    gauss3_v1_omp(in, out, width, height, cfg);
}
void gauss5_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    gauss5_v1_omp(in, out, width, height, cfg);
}
void sobel_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    sobel_v1_omp(in, out, width, height, cfg);
}
void laplace_v2_numa(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    laplace_v1_omp(in, out, width, height, cfg);
}

/* =========================================================================
 * VARIANT 3: Cache Blocking / 2D Tiling (TX, TY) (Datta et al. Section 4.3)
 * Divide domain into TX x TY tiles to maximize L1/L2 cache residency.
 * Fast inner loop for interior tile points with branch-free clamping.
 * ========================================================================= */

void gauss3_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int tx = (cfg && cfg->tx > 0) ? cfg->tx : 64;
    int ty = (cfg && cfg->ty > 0) ? cfg->ty : 32;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    const float k00 = 1.0f/16.0f, k01 = 2.0f/16.0f, k02 = 1.0f/16.0f;
    const float k10 = 2.0f/16.0f, k11 = 4.0f/16.0f, k12 = 2.0f/16.0f;
    const float k20 = 1.0f/16.0f, k21 = 2.0f/16.0f, k22 = 1.0f/16.0f;

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int by = 0; by < height; by += ty) {
        int y_end = (by + ty < height) ? (by + ty) : height;
        for (int bx = 0; bx < width; bx += tx) {
            int x_end = (bx + tx < width) ? (bx + tx) : width;

            for (int y = by; y < y_end; ++y) {
                if (y == 0 || y == height - 1) {
                    /* Boundary row: clamp */
                    int ym1 = clamp(y - 1, height);
                    int yp1 = clamp(y + 1, height);
                    for (int x = bx; x < x_end; ++x) {
                        int xm1 = clamp(x - 1, width);
                        int xp1 = clamp(x + 1, width);
                        float sum = in[ym1 * width + xm1] * k00 + in[ym1 * width + x] * k01 + in[ym1 * width + xp1] * k02
                                  + in[y   * width + xm1] * k10 + in[y   * width + x] * k11 + in[y   * width + xp1] * k12
                                  + in[yp1 * width + xm1] * k20 + in[yp1 * width + x] * k21 + in[yp1 * width + xp1] * k22;
                        out[y * width + x] = sum;
                    }
                } else {
                    /* Interior row */
                    int ym1 = y - 1;
                    int yp1 = y + 1;
                    const float* r0 = &in[ym1 * width];
                    const float* r1 = &in[y   * width];
                    const float* r2 = &in[yp1 * width];
                    float* rout = &out[y * width];

                    for (int x = bx; x < x_end; ++x) {
                        if (x == 0 || x == width - 1) {
                            int xm1 = clamp(x - 1, width);
                            int xp1 = clamp(x + 1, width);
                            rout[x] = r0[xm1]*k00 + r0[x]*k01 + r0[xp1]*k02
                                    + r1[xm1]*k10 + r1[x]*k11 + r1[xp1]*k12
                                    + r2[xm1]*k20 + r2[x]*k21 + r2[xp1]*k22;
                        } else {
                            rout[x] = r0[x-1]*k00 + r0[x]*k01 + r0[x+1]*k02
                                    + r1[x-1]*k10 + r1[x]*k11 + r1[x+1]*k12
                                    + r2[x-1]*k20 + r2[x]*k21 + r2[x+1]*k22;
                        }
                    }
                }
            }
        }
    }
}

void gauss5_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int tx = (cfg && cfg->tx > 0) ? cfg->tx : 64;
    int ty = (cfg && cfg->ty > 0) ? cfg->ty : 32;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    const float inv = 1.0f / 256.0f;
    const float k[5][5] = {
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 6.0f*inv, 24.0f*inv, 36.0f*inv, 24.0f*inv, 6.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv }
    };

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int by = 0; by < height; by += ty) {
        int y_end = (by + ty < height) ? (by + ty) : height;
        for (int bx = 0; bx < width; bx += tx) {
            int x_end = (bx + tx < width) ? (bx + tx) : width;

            for (int y = by; y < y_end; ++y) {
                for (int x = bx; x < x_end; ++x) {
                    float sum = 0.0f;
                    for (int dy = -2; dy <= 2; ++dy) {
                        int py = clamp(y + dy, height);
                        for (int dx = -2; dx <= 2; ++dx) {
                            int px = clamp(x + dx, width);
                            sum += in[py * width + px] * k[dy + 2][dx + 2];
                        }
                    }
                    out[y * width + x] = sum;
                }
            }
        }
    }
}

void sobel_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int tx = (cfg && cfg->tx > 0) ? cfg->tx : 64;
    int ty = (cfg && cfg->ty > 0) ? cfg->ty : 32;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int by = 0; by < height; by += ty) {
        int y_end = (by + ty < height) ? (by + ty) : height;
        for (int bx = 0; bx < width; bx += tx) {
            int x_end = (bx + tx < width) ? (bx + tx) : width;

            for (int y = by; y < y_end; ++y) {
                int ym1 = clamp(y - 1, height);
                int yp1 = clamp(y + 1, height);
                for (int x = bx; x < x_end; ++x) {
                    int xm1 = clamp(x - 1, width);
                    int xp1 = clamp(x + 1, width);

                    float p00 = in[ym1 * width + xm1];
                    float p01 = in[ym1 * width + x  ];
                    float p02 = in[ym1 * width + xp1];

                    float p10 = in[y   * width + xm1];
                    float p12 = in[y   * width + xp1];

                    float p20 = in[yp1 * width + xm1];
                    float p21 = in[yp1 * width + x  ];
                    float p22 = in[yp1 * width + xp1];

                    float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                    float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

                    out[y * width + x] = sqrtf(gx * gx + gy * gy);
                }
            }
        }
    }
}

void laplace_v3_tiled(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int tx = (cfg && cfg->tx > 0) ? cfg->tx : 64;
    int ty = (cfg && cfg->ty > 0) ? cfg->ty : 32;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int by = 0; by < height; by += ty) {
        int y_end = (by + ty < height) ? (by + ty) : height;
        for (int bx = 0; bx < width; bx += tx) {
            int x_end = (bx + tx < width) ? (bx + tx) : width;

            for (int y = by; y < y_end; ++y) {
                int ym1 = clamp(y - 1, height);
                int yp1 = clamp(y + 1, height);
                for (int x = bx; x < x_end; ++x) {
                    int xm1 = clamp(x - 1, width);
                    int xp1 = clamp(x + 1, width);

                    float top    = in[ym1 * width + x];
                    float bottom = in[yp1 * width + x];
                    float left   = in[y * width + xm1];
                    float right  = in[y * width + xp1];
                    float center = in[y * width + x];

                    out[y * width + x] = top + bottom + left + right - 4.0f * center;
                }
            }
        }
    }
}

/* =========================================================================
 * VARIANT 4: SIMD Vectorization + Sliding Window / Register Reuse
 * (Datta et al. Section 4.4 & 4.5)
 * - Interior rows executed with AVX / #pragma omp simd on aligned pointers.
 * - Sliding window across rows: row pointers kept in registers.
 * - Horizontal unit-stride SIMD loop processes 8 float32 elements per iteration.
 * ========================================================================= */

void gauss3_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    const float k00 = 1.0f/16.0f, k01 = 2.0f/16.0f, k02 = 1.0f/16.0f;
    const float k10 = 2.0f/16.0f, k11 = 4.0f/16.0f, k12 = 2.0f/16.0f;
    const float k20 = 1.0f/16.0f, k21 = 2.0f/16.0f, k22 = 1.0f/16.0f;

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y == 0 || y == height - 1) {
            int ym1 = clamp(y - 1, height);
            int yp1 = clamp(y + 1, height);
            for (int x = 0; x < width; ++x) {
                int xm1 = clamp(x - 1, width);
                int xp1 = clamp(x + 1, width);
                out[y * width + x] = in[ym1*width+xm1]*k00 + in[ym1*width+x]*k01 + in[ym1*width+xp1]*k02
                                   + in[y  *width+xm1]*k10 + in[y  *width+x]*k11 + in[y  *width+xp1]*k12
                                   + in[yp1*width+xm1]*k20 + in[yp1*width+x]*k21 + in[yp1*width+xp1]*k22;
            }
        } else {
            /* Sliding window pointers for row y-1, row y, row y+1 */
            const float* __restrict__ r0 = &in[(y - 1) * width];
            const float* __restrict__ r1 = &in[y * width];
            const float* __restrict__ r2 = &in[(y + 1) * width];
            float* __restrict__ rout = &out[y * width];

            /* Left border */
            rout[0] = r0[0]*k00 + r0[0]*k01 + r0[1]*k02
                    + r1[0]*k10 + r1[0]*k11 + r1[1]*k12
                    + r2[0]*k20 + r2[0]*k21 + r2[1]*k22;

            /* SIMD Vectorized Interior Loop */
            #pragma omp simd
            for (int x = 1; x < width - 1; ++x) {
                rout[x] = r0[x-1]*k00 + r0[x]*k01 + r0[x+1]*k02
                        + r1[x-1]*k10 + r1[x]*k11 + r1[x+1]*k12
                        + r2[x-1]*k20 + r2[x]*k21 + r2[x+1]*k22;
            }

            /* Right border */
            rout[width - 1] = r0[width - 2]*k00 + r0[width - 1]*k01 + r0[width - 1]*k02
                            + r1[width - 2]*k10 + r1[width - 1]*k11 + r1[width - 1]*k12
                            + r2[width - 2]*k20 + r2[width - 1]*k21 + r2[width - 1]*k22;
        }
    }
}

void gauss5_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    const float inv = 1.0f / 256.0f;
    const float k[5][5] = {
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 6.0f*inv, 24.0f*inv, 36.0f*inv, 24.0f*inv, 6.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv }
    };

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y < 2 || y >= height - 2) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int py = clamp(y + dy, height);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int px = clamp(x + dx, width);
                        sum += in[py * width + px] * k[dy + 2][dx + 2];
                    }
                }
                out[y * width + x] = sum;
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 2) * width];
            const float* __restrict__ r1 = &in[(y - 1) * width];
            const float* __restrict__ r2 = &in[y * width];
            const float* __restrict__ r3 = &in[(y + 1) * width];
            const float* __restrict__ r4 = &in[(y + 2) * width];
            float* __restrict__ rout = &out[y * width];

            /* Borders */
            for (int x = 0; x < 2; ++x) {
                float sum = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int py = clamp(y + dy, height);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int px = clamp(x + dx, width);
                        sum += in[py * width + px] * k[dy + 2][dx + 2];
                    }
                }
                rout[x] = sum;
            }

            /* Interior SIMD */
            #pragma omp simd
            for (int x = 2; x < width - 2; ++x) {
                rout[x] = r0[x-2]*k[0][0] + r0[x-1]*k[0][1] + r0[x]*k[0][2] + r0[x+1]*k[0][3] + r0[x+2]*k[0][4]
                        + r1[x-2]*k[1][0] + r1[x-1]*k[1][1] + r1[x]*k[1][2] + r1[x+1]*k[1][3] + r1[x+2]*k[1][4]
                        + r2[x-2]*k[2][0] + r2[x-1]*k[2][1] + r2[x]*k[2][2] + r2[x+1]*k[2][3] + r2[x+2]*k[2][4]
                        + r3[x-2]*k[3][0] + r3[x-1]*k[3][1] + r3[x]*k[3][2] + r3[x+1]*k[3][3] + r3[x+2]*k[3][4]
                        + r4[x-2]*k[4][0] + r4[x-1]*k[4][1] + r4[x]*k[4][2] + r4[x+1]*k[4][3] + r4[x+2]*k[4][4];
            }

            for (int x = width - 2; x < width; ++x) {
                float sum = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int py = clamp(y + dy, height);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int px = clamp(x + dx, width);
                        sum += in[py * width + px] * k[dy + 2][dx + 2];
                    }
                }
                rout[x] = sum;
            }
        }
    }
}

void sobel_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y == 0 || y == height - 1) {
            int ym1 = clamp(y - 1, height);
            int yp1 = clamp(y + 1, height);
            for (int x = 0; x < width; ++x) {
                int xm1 = clamp(x - 1, width);
                int xp1 = clamp(x + 1, width);

                float p00 = in[ym1 * width + xm1];
                float p01 = in[ym1 * width + x  ];
                float p02 = in[ym1 * width + xp1];
                float p10 = in[y   * width + xm1];
                float p12 = in[y   * width + xp1];
                float p20 = in[yp1 * width + xm1];
                float p21 = in[yp1 * width + x  ];
                float p22 = in[yp1 * width + xp1];

                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                out[y * width + x] = sqrtf(gx * gx + gy * gy);
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 1) * width];
            const float* __restrict__ r1 = &in[y * width];
            const float* __restrict__ r2 = &in[(y + 1) * width];
            float* __restrict__ rout = &out[y * width];

            /* Left pixel */
            {
                float p00 = r0[0], p01 = r0[0], p02 = r0[1];
                float p10 = r1[0],              p12 = r1[1];
                float p20 = r2[0], p21 = r2[0], p22 = r2[1];
                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[0] = sqrtf(gx * gx + gy * gy);
            }

            /* Vectorized Interior */
            #pragma omp simd
            for (int x = 1; x < width - 1; ++x) {
                float p00 = r0[x - 1];
                float p01 = r0[x];
                float p02 = r0[x + 1];
                float p10 = r1[x - 1];
                float p12 = r1[x + 1];
                float p20 = r2[x - 1];
                float p21 = r2[x];
                float p22 = r2[x + 1];

                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[x] = sqrtf(gx * gx + gy * gy);
            }

            /* Right pixel */
            {
                int xm1 = width - 2, x = width - 1, xp1 = width - 1;
                float p00 = r0[xm1], p01 = r0[x], p02 = r0[xp1];
                float p10 = r1[xm1],             p12 = r1[xp1];
                float p20 = r2[xm1], p21 = r2[x], p22 = r2[xp1];
                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[x] = sqrtf(gx * gx + gy * gy);
            }
        }
    }
}

void laplace_v4_simd(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y == 0 || y == height - 1) {
            int ym1 = clamp(y - 1, height);
            int yp1 = clamp(y + 1, height);
            for (int x = 0; x < width; ++x) {
                int xm1 = clamp(x - 1, width);
                int xp1 = clamp(x + 1, width);
                float top    = in[ym1 * width + x];
                float bottom = in[yp1 * width + x];
                float left   = in[y * width + xm1];
                float right  = in[y * width + xp1];
                float center = in[y * width + x];
                out[y * width + x] = top + bottom + left + right - 4.0f * center;
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 1) * width];
            const float* __restrict__ r1 = &in[y * width];
            const float* __restrict__ r2 = &in[(y + 1) * width];
            float* __restrict__ rout = &out[y * width];

            rout[0] = r0[0] + r2[0] + r1[0] + r1[1] - 4.0f * r1[0];

            #pragma omp simd
            for (int x = 1; x < width - 1; ++x) {
                rout[x] = r0[x] + r2[x] + r1[x - 1] + r1[x + 1] - 4.0f * r1[x];
            }

            rout[width - 1] = r0[width - 1] + r2[width - 1] + r1[width - 2] + r1[width - 1] - 4.0f * r1[width - 1];
        }
    }
}

/* =========================================================================
 * VARIANT 5: Software Prefetch + Non-Temporal Stores (Datta et al. Section 4.6)
 * - Software prefetch brings future rows into L1/L2 cache before execution.
 * - Non-temporal streaming stores (_mm_stream_ps) avoid cache pollution
 *   and eliminate Read-For-Ownership (RFO) overhead on output buffers.
 * ========================================================================= */

void gauss3_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    int pdist = (cfg && cfg->prefetch_dist > 0) ? cfg->prefetch_dist : 16;
    const float k00 = 1.0f/16.0f, k01 = 2.0f/16.0f, k02 = 1.0f/16.0f;
    const float k10 = 2.0f/16.0f, k11 = 4.0f/16.0f, k12 = 2.0f/16.0f;
    const float k20 = 1.0f/16.0f, k21 = 2.0f/16.0f, k22 = 1.0f/16.0f;

    const __m256 vk00 = _mm256_set1_ps(k00), vk01 = _mm256_set1_ps(k01), vk02 = _mm256_set1_ps(k02);
    const __m256 vk10 = _mm256_set1_ps(k10), vk11 = _mm256_set1_ps(k11), vk12 = _mm256_set1_ps(k12);
    const __m256 vk20 = _mm256_set1_ps(k20), vk21 = _mm256_set1_ps(k21), vk22 = _mm256_set1_ps(k22);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y == 0 || y == height - 1) {
            int ym1 = clamp(y - 1, height);
            int yp1 = clamp(y + 1, height);
            for (int x = 0; x < width; ++x) {
                int xm1 = clamp(x - 1, width);
                int xp1 = clamp(x + 1, width);
                out[y * width + x] = in[ym1*width+xm1]*k00 + in[ym1*width+x]*k01 + in[ym1*width+xp1]*k02
                                   + in[y  *width+xm1]*k10 + in[y  *width+x]*k11 + in[y  *width+xp1]*k12
                                   + in[yp1*width+xm1]*k20 + in[yp1*width+x]*k21 + in[yp1*width+xp1]*k22;
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 1) * width];
            const float* __restrict__ r1 = &in[y * width];
            const float* __restrict__ r2 = &in[(y + 1) * width];
            const float* __restrict__ r_next = (y + 2 < height) ? &in[(y + 2) * width] : r2;
            float* __restrict__ rout = &out[y * width];

            rout[0] = r0[0]*k00 + r0[0]*k01 + r0[1]*k02
                    + r1[0]*k10 + r1[0]*k11 + r1[1]*k12
                    + r2[0]*k20 + r2[0]*k21 + r2[1]*k22;

            int x = 1;
            /* Scalar peel until destination is 32-byte aligned for _mm256_stream_ps */
            while (x < width - 1 && (((uintptr_t)&rout[x]) & 31)) {
                rout[x] = r0[x-1]*k00 + r0[x]*k01 + r0[x+1]*k02
                        + r1[x-1]*k10 + r1[x]*k11 + r1[x+1]*k12
                        + r2[x-1]*k20 + r2[x]*k21 + r2[x+1]*k22;
                x++;
            }

            /* Vectorized stream loop with software prefetching */
            for (; x <= width - 9; x += 8) {
                __builtin_prefetch(&r_next[x + pdist], 0, 1);
                __m256 r0m = _mm256_loadu_ps(&r0[x - 1]);
                __m256 r0c = _mm256_loadu_ps(&r0[x]);
                __m256 r0p = _mm256_loadu_ps(&r0[x + 1]);

                __m256 r1m = _mm256_loadu_ps(&r1[x - 1]);
                __m256 r1c = _mm256_loadu_ps(&r1[x]);
                __m256 r1p = _mm256_loadu_ps(&r1[x + 1]);

                __m256 r2m = _mm256_loadu_ps(&r2[x - 1]);
                __m256 r2c = _mm256_loadu_ps(&r2[x]);
                __m256 r2p = _mm256_loadu_ps(&r2[x + 1]);

                __m256 sum = _mm256_mul_ps(r0m, vk00);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r0c, vk01));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r0p, vk02));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r1m, vk10));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r1c, vk11));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r1p, vk12));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r2m, vk20));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r2c, vk21));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(r2p, vk22));

                _mm256_stream_ps(&rout[x], sum);
            }

            for (; x < width - 1; ++x) {
                rout[x] = r0[x-1]*k00 + r0[x]*k01 + r0[x+1]*k02
                        + r1[x-1]*k10 + r1[x]*k11 + r1[x+1]*k12
                        + r2[x-1]*k20 + r2[x]*k21 + r2[x+1]*k22;
            }

            rout[width - 1] = r0[width - 2]*k00 + r0[width - 1]*k01 + r0[width - 1]*k02
                            + r1[width - 2]*k10 + r1[width - 1]*k11 + r1[width - 1]*k12
                            + r2[width - 2]*k20 + r2[width - 1]*k21 + r2[width - 1]*k22;
        }
    }
    _mm_sfence();
}

void gauss5_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    int pdist = (cfg && cfg->prefetch_dist > 0) ? cfg->prefetch_dist : 16;
    const float inv = 1.0f / 256.0f;
    const float k[5][5] = {
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 6.0f*inv, 24.0f*inv, 36.0f*inv, 24.0f*inv, 6.0f*inv },
        { 4.0f*inv, 16.0f*inv, 24.0f*inv, 16.0f*inv, 4.0f*inv },
        { 1.0f*inv,  4.0f*inv,  6.0f*inv,  4.0f*inv, 1.0f*inv }
    };

    __m256 vk[5][5];
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            vk[r][c] = _mm256_set1_ps(k[r][c]);
        }
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y < 2 || y >= height - 2) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int py = clamp(y + dy, height);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int px = clamp(x + dx, width);
                        sum += in[py * width + px] * k[dy + 2][dx + 2];
                    }
                }
                out[y * width + x] = sum;
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 2) * width];
            const float* __restrict__ r1 = &in[(y - 1) * width];
            const float* __restrict__ r2 = &in[y * width];
            const float* __restrict__ r3 = &in[(y + 1) * width];
            const float* __restrict__ r4 = &in[(y + 2) * width];
            const float* __restrict__ r_next = (y + 3 < height) ? &in[(y + 3) * width] : r4;
            float* __restrict__ rout = &out[y * width];

            int x = 0;
            while (x < 2) {
                float sum = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int py = clamp(y + dy, height);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int px = clamp(x + dx, width);
                        sum += in[py * width + px] * k[dy + 2][dx + 2];
                    }
                }
                rout[x] = sum;
                x++;
            }

            while (x < width - 2 && (((uintptr_t)&rout[x]) & 31)) {
                float sum = r0[x-2]*k[0][0] + r0[x-1]*k[0][1] + r0[x]*k[0][2] + r0[x+1]*k[0][3] + r0[x+2]*k[0][4]
                          + r1[x-2]*k[1][0] + r1[x-1]*k[1][1] + r1[x]*k[1][2] + r1[x+1]*k[1][3] + r1[x+2]*k[1][4]
                          + r2[x-2]*k[2][0] + r2[x-1]*k[2][1] + r2[x]*k[2][2] + r2[x+1]*k[2][3] + r2[x+2]*k[2][4]
                          + r3[x-2]*k[3][0] + r3[x-1]*k[3][1] + r3[x]*k[3][2] + r3[x+1]*k[3][3] + r3[x+2]*k[3][4]
                          + r4[x-2]*k[4][0] + r4[x-1]*k[4][1] + r4[x]*k[4][2] + r4[x+1]*k[4][3] + r4[x+2]*k[4][4];
                rout[x] = sum;
                x++;
            }

            for (; x <= width - 10; x += 8) {
                __builtin_prefetch(&r_next[x + pdist], 0, 1);
                __m256 acc = _mm256_setzero_ps();

                #define G5_ROW(rp, r_idx) \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(&(rp)[x-2]), vk[r_idx][0])); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(&(rp)[x-1]), vk[r_idx][1])); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(&(rp)[x  ]), vk[r_idx][2])); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(&(rp)[x+1]), vk[r_idx][3])); \
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(&(rp)[x+2]), vk[r_idx][4]));

                G5_ROW(r0, 0);
                G5_ROW(r1, 1);
                G5_ROW(r2, 2);
                G5_ROW(r3, 3);
                G5_ROW(r4, 4);
                #undef G5_ROW

                _mm256_stream_ps(&rout[x], acc);
            }

            for (; x < width - 2; ++x) {
                float sum = r0[x-2]*k[0][0] + r0[x-1]*k[0][1] + r0[x]*k[0][2] + r0[x+1]*k[0][3] + r0[x+2]*k[0][4]
                          + r1[x-2]*k[1][0] + r1[x-1]*k[1][1] + r1[x]*k[1][2] + r1[x+1]*k[1][3] + r1[x+2]*k[1][4]
                          + r2[x-2]*k[2][0] + r2[x-1]*k[2][1] + r2[x]*k[2][2] + r2[x+1]*k[2][3] + r2[x+2]*k[2][4]
                          + r3[x-2]*k[3][0] + r3[x-1]*k[3][1] + r3[x]*k[3][2] + r3[x+1]*k[3][3] + r3[x+2]*k[3][4]
                          + r4[x-2]*k[4][0] + r4[x-1]*k[4][1] + r4[x]*k[4][2] + r4[x+1]*k[4][3] + r4[x+2]*k[4][4];
                rout[x] = sum;
            }

            for (x = width - 2; x < width; ++x) {
                float sum = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int py = clamp(y + dy, height);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int px = clamp(x + dx, width);
                        sum += in[py * width + px] * k[dy + 2][dx + 2];
                    }
                }
                rout[x] = sum;
            }
        }
    }
    _mm_sfence();
}

void sobel_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    int pdist = (cfg && cfg->prefetch_dist > 0) ? cfg->prefetch_dist : 16;
    const __m256 two = _mm256_set1_ps(2.0f);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y == 0 || y == height - 1) {
            int ym1 = clamp(y - 1, height);
            int yp1 = clamp(y + 1, height);
            for (int x = 0; x < width; ++x) {
                int xm1 = clamp(x - 1, width);
                int xp1 = clamp(x + 1, width);

                float p00 = in[ym1 * width + xm1];
                float p01 = in[ym1 * width + x  ];
                float p02 = in[ym1 * width + xp1];
                float p10 = in[y   * width + xm1];
                float p12 = in[y   * width + xp1];
                float p20 = in[yp1 * width + xm1];
                float p21 = in[yp1 * width + x  ];
                float p22 = in[yp1 * width + xp1];

                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                out[y * width + x] = sqrtf(gx * gx + gy * gy);
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 1) * width];
            const float* __restrict__ r1 = &in[y * width];
            const float* __restrict__ r2 = &in[(y + 1) * width];
            const float* __restrict__ r_next = (y + 2 < height) ? &in[(y + 2) * width] : r2;
            float* __restrict__ rout = &out[y * width];

            {
                float p00 = r0[0], p01 = r0[0], p02 = r0[1];
                float p10 = r1[0],              p12 = r1[1];
                float p20 = r2[0], p21 = r2[0], p22 = r2[1];
                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[0] = sqrtf(gx * gx + gy * gy);
            }

            int x = 1;
            while (x < width - 1 && (((uintptr_t)&rout[x]) & 31)) {
                float p00 = r0[x - 1], p01 = r0[x], p02 = r0[x + 1];
                float p10 = r1[x - 1],              p12 = r1[x + 1];
                float p20 = r2[x - 1], p21 = r2[x], p22 = r2[x + 1];
                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[x] = sqrtf(gx * gx + gy * gy);
                x++;
            }

            for (; x <= width - 9; x += 8) {
                __builtin_prefetch(&r_next[x + pdist], 0, 1);
                __m256 p00 = _mm256_loadu_ps(&r0[x - 1]);
                __m256 p01 = _mm256_loadu_ps(&r0[x]);
                __m256 p02 = _mm256_loadu_ps(&r0[x + 1]);

                __m256 p10 = _mm256_loadu_ps(&r1[x - 1]);
                __m256 p12 = _mm256_loadu_ps(&r1[x + 1]);

                __m256 p20 = _mm256_loadu_ps(&r2[x - 1]);
                __m256 p21 = _mm256_loadu_ps(&r2[x]);
                __m256 p22 = _mm256_loadu_ps(&r2[x + 1]);

                __m256 gx = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(p02, _mm256_mul_ps(two, p12)), p22),
                                          _mm256_add_ps(_mm256_add_ps(p00, _mm256_mul_ps(two, p10)), p20));

                __m256 gy = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(p20, _mm256_mul_ps(two, p21)), p22),
                                          _mm256_add_ps(_mm256_add_ps(p00, _mm256_mul_ps(two, p01)), p02));

                __m256 mag = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(gx, gx), _mm256_mul_ps(gy, gy)));
                _mm256_stream_ps(&rout[x], mag);
            }

            for (; x < width - 1; ++x) {
                float p00 = r0[x - 1], p01 = r0[x], p02 = r0[x + 1];
                float p10 = r1[x - 1],              p12 = r1[x + 1];
                float p20 = r2[x - 1], p21 = r2[x], p22 = r2[x + 1];
                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[x] = sqrtf(gx * gx + gy * gy);
            }

            {
                int xm1 = width - 2, lx = width - 1, xp1 = width - 1;
                float p00 = r0[xm1], p01 = r0[lx], p02 = r0[xp1];
                float p10 = r1[xm1],               p12 = r1[xp1];
                float p20 = r2[xm1], p21 = r2[lx], p22 = r2[xp1];
                float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
                float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);
                rout[lx] = sqrtf(gx * gx + gy * gy);
            }
        }
    }
    _mm_sfence();
}

void laplace_v5_prefetch(const float* in, float* out, int width, int height, const TuningConfig* cfg) {
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    int pdist = (cfg && cfg->prefetch_dist > 0) ? cfg->prefetch_dist : 16;
    const __m256 four = _mm256_set1_ps(4.0f);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int y = 0; y < height; ++y) {
        if (y == 0 || y == height - 1) {
            int ym1 = clamp(y - 1, height);
            int yp1 = clamp(y + 1, height);
            for (int x = 0; x < width; ++x) {
                int xm1 = clamp(x - 1, width);
                int xp1 = clamp(x + 1, width);
                float top    = in[ym1 * width + x];
                float bottom = in[yp1 * width + x];
                float left   = in[y * width + xm1];
                float right  = in[y * width + xp1];
                float center = in[y * width + x];
                out[y * width + x] = top + bottom + left + right - 4.0f * center;
            }
        } else {
            const float* __restrict__ r0 = &in[(y - 1) * width];
            const float* __restrict__ r1 = &in[y * width];
            const float* __restrict__ r2 = &in[(y + 1) * width];
            const float* __restrict__ r_next = (y + 2 < height) ? &in[(y + 2) * width] : r2;
            float* __restrict__ rout = &out[y * width];

            rout[0] = r0[0] + r2[0] + r1[0] + r1[1] - 4.0f * r1[0];

            int x = 1;
            while (x < width - 1 && (((uintptr_t)&rout[x]) & 31)) {
                rout[x] = r0[x] + r2[x] + r1[x - 1] + r1[x + 1] - 4.0f * r1[x];
                x++;
            }

            for (; x <= width - 9; x += 8) {
                __builtin_prefetch(&r_next[x + pdist], 0, 1);
                __m256 top = _mm256_loadu_ps(&r0[x]);
                __m256 bot = _mm256_loadu_ps(&r2[x]);
                __m256 left = _mm256_loadu_ps(&r1[x - 1]);
                __m256 right = _mm256_loadu_ps(&r1[x + 1]);
                __m256 center = _mm256_loadu_ps(&r1[x]);

                __m256 sum = _mm256_add_ps(top, bot);
                sum = _mm256_add_ps(sum, left);
                sum = _mm256_add_ps(sum, right);
                __m256 res = _mm256_sub_ps(sum, _mm256_mul_ps(four, center));
                _mm256_stream_ps(&rout[x], res);
            }

            for (; x < width - 1; ++x) {
                rout[x] = r0[x] + r2[x] + r1[x - 1] + r1[x + 1] - 4.0f * r1[x];
            }

            rout[width - 1] = r0[width - 1] + r2[width - 1] + r1[width - 2] + r1[width - 1] - 4.0f * r1[width - 1];
        }
    }
    _mm_sfence();
}
