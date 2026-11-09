#include "kernels_3d.h"
#include <immintrin.h>
#include <string.h>

/* Reference: K. Datta et al., "Stencil Computation Optimization and Auto-tuning
 * on State-of-the-Art Multicore Architectures", SC 2008.
 */

/* =========================================================================
 * VARIANT 1: OpenMP Parallel-For Naive (Datta et al. Section 4.2)
 * Parallelize the outermost spatial dimension (Z planes).
 * ========================================================================= */

void heat3d_v1_omp(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    const size_t plane_size = (size_t)nx * ny;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    /* Boundary planes z = 0 and z = nz - 1 */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(nz - 1) * plane_size + i] = in[(nz - 1) * plane_size + i];
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        const size_t z_offset = (size_t)z * plane_size;
        const size_t zm1_offset = (size_t)(z - 1) * plane_size;
        const size_t zp1_offset = (size_t)(z + 1) * plane_size;

        for (int x = 0; x < nx; ++x) {
            out[z_offset + x] = in[z_offset + x];
            out[z_offset + (ny - 1) * nx + x] = in[z_offset + (ny - 1) * nx + x];
        }

        for (int y = 1; y < ny - 1; ++y) {
            const size_t row_offset = z_offset + (size_t)y * nx;
            const size_t ym1_offset = z_offset + (size_t)(y - 1) * nx;
            const size_t yp1_offset = z_offset + (size_t)(y + 1) * nx;

            out[row_offset] = in[row_offset];
            out[row_offset + nx - 1] = in[row_offset + nx - 1];

            for (int x = 1; x < nx - 1; ++x) {
                double center = in[row_offset + x];
                double left   = in[row_offset + x - 1];
                double right  = in[row_offset + x + 1];
                double top    = in[ym1_offset + x];
                double bottom = in[yp1_offset + x];
                double back   = in[zm1_offset + y * nx + x];
                double front  = in[zp1_offset + y * nx + x];

                double neighbors = left + right + top + bottom + back + front;
                out[row_offset + x] = c0 * center + c1 * neighbors;
            }
        }
    }
}

/* =========================================================================
 * VARIANT 2: NUMA First-Touch Initialization (Datta et al. Section 4.2)
 * Parallel memory touch in 3D using the same Z-plane partitioning.
 * ========================================================================= */

void first_touch_init_3d(double* buf, int nx, int ny, int nz, double val, const TuningConfig* cfg) {
    const size_t plane_size = (size_t)nx * ny;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 0; z < nz; ++z) {
        size_t offset = (size_t)z * plane_size;
        for (size_t i = 0; i < plane_size; ++i) {
            buf[offset + i] = val;
        }
    }
}

void heat3d_v2_numa(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    heat3d_v1_omp(in, out, nx, ny, nz, c0, c1, cfg);
}

/* =========================================================================
 * VARIANT 3: Cache Blocking / 3D Tiling (TZ, TY) (Datta et al. Section 4.3)
 * Spatial tiling in Z and Y dimensions (with unit-stride X stream).
 * ========================================================================= */

void heat3d_v3_tiled(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    const size_t plane_size = (size_t)nx * ny;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    int tz = (cfg && cfg->tz > 0) ? cfg->tz : 16;
    int ty = (cfg && cfg->ty > 0) ? cfg->ty : 32;

    /* Copy boundaries */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(nz - 1) * plane_size + i] = in[(nz - 1) * plane_size + i];
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static) collapse(2)
    for (int bz = 1; bz < nz - 1; bz += tz) {
        for (int by = 1; by < ny - 1; by += ty) {
            int z_end = (bz + tz < nz - 1) ? (bz + tz) : (nz - 1);
            int y_end = (by + ty < ny - 1) ? (by + ty) : (ny - 1);

            for (int z = bz; z < z_end; ++z) {
                const size_t z_offset = (size_t)z * plane_size;
                const size_t zm1_offset = (size_t)(z - 1) * plane_size;
                const size_t zp1_offset = (size_t)(z + 1) * plane_size;

                for (int y = by; y < y_end; ++y) {
                    const size_t row_offset = z_offset + (size_t)y * nx;
                    const size_t ym1_offset = z_offset + (size_t)(y - 1) * nx;
                    const size_t yp1_offset = z_offset + (size_t)(y + 1) * nx;

                    out[row_offset] = in[row_offset];
                    out[row_offset + nx - 1] = in[row_offset + nx - 1];

                    for (int x = 1; x < nx - 1; ++x) {
                        double center = in[row_offset + x];
                        double left   = in[row_offset + x - 1];
                        double right  = in[row_offset + x + 1];
                        double top    = in[ym1_offset + x];
                        double bottom = in[yp1_offset + x];
                        double back   = in[zm1_offset + y * nx + x];
                        double front  = in[zp1_offset + y * nx + x];

                        out[row_offset + x] = c0 * center + c1 * (left + right + top + bottom + back + front);
                    }
                }
            }
        }
    }

    /* Copy y=0 and y=ny-1 boundary rows */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        const size_t z_offset = (size_t)z * plane_size;
        for (int x = 0; x < nx; ++x) {
            out[z_offset + x] = in[z_offset + x];
            out[z_offset + (ny - 1) * nx + x] = in[z_offset + (ny - 1) * nx + x];
        }
    }
}

/* =========================================================================
 * VARIANT 4: SIMD Vectorization + Plane Register Reuse
 * (Datta et al. Section 4.4 & 4.5)
 * - Vectorize unit-stride X dimension using #pragma omp simd
 * - Maintain pointers to z-1, z, z+1 planes to facilitate register reuse
 * ========================================================================= */

void heat3d_v4_simd(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    const size_t plane_size = (size_t)nx * ny;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(nz - 1) * plane_size + i] = in[(nz - 1) * plane_size + i];
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        const double* __restrict__ p_prev = &in[(size_t)(z - 1) * plane_size];
        const double* __restrict__ p_curr = &in[(size_t)z * plane_size];
        const double* __restrict__ p_next = &in[(size_t)(z + 1) * plane_size];
        double* __restrict__ p_out = &out[(size_t)z * plane_size];

        for (int x = 0; x < nx; ++x) {
            p_out[x] = p_curr[x];
            p_out[(ny - 1) * nx + x] = p_curr[(ny - 1) * nx + x];
        }

        for (int y = 1; y < ny - 1; ++y) {
            const double* __restrict__ r_curr = &p_curr[y * nx];
            const double* __restrict__ r_prev_y = &p_curr[(y - 1) * nx];
            const double* __restrict__ r_next_y = &p_curr[(y + 1) * nx];
            const double* __restrict__ r_back = &p_prev[y * nx];
            const double* __restrict__ r_front = &p_next[y * nx];
            double* __restrict__ r_out = &p_out[y * nx];

            r_out[0] = r_curr[0];
            r_out[nx - 1] = r_curr[nx - 1];

            #pragma omp simd
            for (int x = 1; x < nx - 1; ++x) {
                double center = r_curr[x];
                double left   = r_curr[x - 1];
                double right  = r_curr[x + 1];
                double top    = r_prev_y[x];
                double bottom = r_next_y[x];
                double back   = r_back[x];
                double front  = r_front[x];

                r_out[x] = c0 * center + c1 * (left + right + top + bottom + back + front);
            }
        }
    }
}

/* =========================================================================
 * VARIANT 5: Software Prefetch + Non-Temporal Stores (Datta et al. Section 4.6)
 * ========================================================================= */

void heat3d_v5_prefetch(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    const size_t plane_size = (size_t)nx * ny;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
    int pdist = (cfg && cfg->prefetch_dist > 0) ? cfg->prefetch_dist : 8;

    __m256d vc0 = _mm256_set1_pd(c0);
    __m256d vc1 = _mm256_set1_pd(c1);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(nz - 1) * plane_size + i] = in[(nz - 1) * plane_size + i];
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        const double* __restrict__ p_prev = &in[(size_t)(z - 1) * plane_size];
        const double* __restrict__ p_curr = &in[(size_t)z * plane_size];
        const double* __restrict__ p_next = &in[(size_t)(z + 1) * plane_size];
        double* __restrict__ p_out = &out[(size_t)z * plane_size];

        for (int x = 0; x < nx; ++x) {
            p_out[x] = p_curr[x];
            p_out[(ny - 1) * nx + x] = p_curr[(ny - 1) * nx + x];
        }

        for (int y = 1; y < ny - 1; ++y) {
            const double* __restrict__ r_curr = &p_curr[y * nx];
            const double* __restrict__ r_prev_y = &p_curr[(y - 1) * nx];
            const double* __restrict__ r_next_y = &p_curr[(y + 1) * nx];
            const double* __restrict__ r_back = &p_prev[y * nx];
            const double* __restrict__ r_front = &p_next[y * nx];
            double* __restrict__ r_out = &p_out[y * nx];

            r_out[0] = r_curr[0];

            int x = 1;
            while (x < nx - 1 && (((uintptr_t)&r_out[x]) & 31)) {
                double center = r_curr[x];
                double neigh = r_curr[x - 1] + r_curr[x + 1] + r_prev_y[x] + r_next_y[x] + r_back[x] + r_front[x];
                r_out[x] = c0 * center + c1 * neigh;
                x++;
            }

            for (; x <= nx - 5; x += 4) {
                __builtin_prefetch(&r_front[x + pdist], 0, 1);
                __m256d c = _mm256_loadu_pd(&r_curr[x]);
                __m256d left = _mm256_loadu_pd(&r_curr[x - 1]);
                __m256d right = _mm256_loadu_pd(&r_curr[x + 1]);
                __m256d top = _mm256_loadu_pd(&r_prev_y[x]);
                __m256d bot = _mm256_loadu_pd(&r_next_y[x]);
                __m256d back = _mm256_loadu_pd(&r_back[x]);
                __m256d front = _mm256_loadu_pd(&r_front[x]);

                __m256d neigh = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(left, right), _mm256_add_pd(top, bot)), _mm256_add_pd(back, front));
                __m256d res = _mm256_add_pd(_mm256_mul_pd(vc0, c), _mm256_mul_pd(vc1, neigh));
                _mm256_stream_pd(&r_out[x], res);
            }

            for (; x < nx - 1; ++x) {
                double center = r_curr[x];
                double neigh = r_curr[x - 1] + r_curr[x + 1] + r_prev_y[x] + r_next_y[x] + r_back[x] + r_front[x];
                r_out[x] = c0 * center + c1 * neigh;
            }

            r_out[nx - 1] = r_curr[nx - 1];
        }
    }
    _mm_sfence();
}

/* =========================================================================
 * VARIANT 6: Z-Unrolling / Register Blocking (Datta et al. Section 4.7)
 * Unroll by 2 in Z: computes slice z and slice z+1 together.
 * Slices z and z+1 share intermediate planes, reducing vertical loads by 33%.
 * ========================================================================= */

void heat3d_v6_unroll_reg(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    const size_t plane_size = (size_t)nx * ny;
    int nthreads = cfg && cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();

    /* Copy boundaries z = 0 and z = nz - 1 */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(nz - 1) * plane_size + i] = in[(nz - 1) * plane_size + i];
    }

    /* Unroll by 2 in Z */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 2; z += 2) {
        const double* __restrict__ p0 = &in[(size_t)(z - 1) * plane_size];
        const double* __restrict__ p1 = &in[(size_t)z * plane_size];
        const double* __restrict__ p2 = &in[(size_t)(z + 1) * plane_size];
        const double* __restrict__ p3 = &in[(size_t)(z + 2) * plane_size];

        double* __restrict__ out1 = &out[(size_t)z * plane_size];
        double* __restrict__ out2 = &out[(size_t)(z + 1) * plane_size];

        /* Y boundaries */
        for (int x = 0; x < nx; ++x) {
            out1[x] = p1[x];
            out1[(ny - 1) * nx + x] = p1[(ny - 1) * nx + x];
            out2[x] = p2[x];
            out2[(ny - 1) * nx + x] = p2[(ny - 1) * nx + x];
        }

        for (int y = 1; y < ny - 1; ++y) {
            const double* __restrict__ r0 = &p0[y * nx];
            const double* __restrict__ r1 = &p1[y * nx];
            const double* __restrict__ r2 = &p2[y * nx];
            const double* __restrict__ r3 = &p3[y * nx];

            const double* __restrict__ r1_ym1 = &p1[(y - 1) * nx];
            const double* __restrict__ r1_yp1 = &p1[(y + 1) * nx];
            const double* __restrict__ r2_ym1 = &p2[(y - 1) * nx];
            const double* __restrict__ r2_yp1 = &p2[(y + 1) * nx];

            double* __restrict__ o1 = &out1[y * nx];
            double* __restrict__ o2 = &out2[y * nx];

            o1[0] = r1[0]; o1[nx - 1] = r1[nx - 1];
            o2[0] = r2[0]; o2[nx - 1] = r2[nx - 1];

            #pragma omp simd
            for (int x = 1; x < nx - 1; ++x) {
                /* Point (z, y, x) uses p0, p1, p2 */
                double c1_val = r1[x];
                double neigh1 = r1[x - 1] + r1[x + 1] + r1_ym1[x] + r1_yp1[x] + r0[x] + r2[x];
                o1[x] = c0 * c1_val + c1 * neigh1;

                /* Point (z+1, y, x) uses p1, p2, p3 (p1 and p2 shared in registers!) */
                double c2_val = r2[x];
                double neigh2 = r2[x - 1] + r2[x + 1] + r2_ym1[x] + r2_yp1[x] + r1[x] + r3[x];
                o2[x] = c0 * c2_val + c1 * neigh2;
            }
        }
    }

    /* Handle remainder slice if nz - 2 is odd */
    if ((nz - 2) % 2 != 0) {
        int z = nz - 2;
        const double* __restrict__ p_prev = &in[(size_t)(z - 1) * plane_size];
        const double* __restrict__ p_curr = &in[(size_t)z * plane_size];
        const double* __restrict__ p_next = &in[(size_t)(z + 1) * plane_size];
        double* __restrict__ p_out = &out[(size_t)z * plane_size];

        for (int x = 0; x < nx; ++x) {
            p_out[x] = p_curr[x];
            p_out[(ny - 1) * nx + x] = p_curr[(ny - 1) * nx + x];
        }

        for (int y = 1; y < ny - 1; ++y) {
            const double* __restrict__ r_curr = &p_curr[y * nx];
            const double* __restrict__ r_prev_y = &p_curr[(y - 1) * nx];
            const double* __restrict__ r_next_y = &p_curr[(y + 1) * nx];
            const double* __restrict__ r_back = &p_prev[y * nx];
            const double* __restrict__ r_front = &p_next[y * nx];
            double* __restrict__ r_out = &p_out[y * nx];

            r_out[0] = r_curr[0];
            r_out[nx - 1] = r_curr[nx - 1];

            #pragma omp simd
            for (int x = 1; x < nx - 1; ++x) {
                double center = r_curr[x];
                double left   = r_curr[x - 1];
                double right  = r_curr[x + 1];
                double top    = r_prev_y[x];
                double bottom = r_next_y[x];
                double back   = r_back[x];
                double front  = r_front[x];

                r_out[x] = c0 * center + c1 * (left + right + top + bottom + back + front);
            }
        }
    }
}
