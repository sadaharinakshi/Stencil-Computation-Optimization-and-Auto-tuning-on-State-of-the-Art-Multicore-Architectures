#include "kernels_3d.h"

/*
 * ============================================================================
 * Kernel Dispatcher for 3D 7-Point Heat Stencil
 * ============================================================================
 *
 * Provides a unified function pointer interface (kernel_3d_fn) for the
 * five required implementations:
 *   1. Serial Reference
 *   2. Basic OpenMP
 *   3. OpenMP + First Touch
 *   4. OpenMP + 3D Cache Blocking (BX x BY x BZ)
 *   5. OpenMP + First Touch + 3D Cache Blocking
 * ============================================================================
 */

static void wrap_serial(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg) {
    double alpha = (cfg && cfg->alpha > 0.0) ? cfg->alpha : STENCIL3D_ALPHA_DEFAULT;
    heat3d_serial(in, out, nx, ny, nz, alpha);
}

static void wrap_omp(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg) {
    double alpha = (cfg && cfg->alpha > 0.0) ? cfg->alpha : STENCIL3D_ALPHA_DEFAULT;
    int nthreads = (cfg && cfg->num_threads > 0) ? cfg->num_threads : omp_get_max_threads();
    heat3d_omp(in, out, nx, ny, nz, alpha, nthreads);
}

static void wrap_omp_numa(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg) {
    double alpha = (cfg && cfg->alpha > 0.0) ? cfg->alpha : STENCIL3D_ALPHA_DEFAULT;
    int nthreads = (cfg && cfg->num_threads > 0) ? cfg->num_threads : omp_get_max_threads();
    heat3d_omp_numa(in, out, nx, ny, nz, alpha, nthreads);
}

static void wrap_omp_blocked(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg) {
    double alpha = (cfg && cfg->alpha > 0.0) ? cfg->alpha : STENCIL3D_ALPHA_DEFAULT;
    int nthreads = (cfg && cfg->num_threads > 0) ? cfg->num_threads : omp_get_max_threads();
    int bx = (cfg && cfg->bx > 0) ? cfg->bx : 32;
    int by = (cfg && cfg->by > 0) ? cfg->by : 32;
    int bz = (cfg && cfg->bz > 0) ? cfg->bz : 16;
    heat3d_blocked(in, out, nx, ny, nz, alpha, bx, by, bz, nthreads);
}

static void wrap_omp_numa_blocked(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg) {
    double alpha = (cfg && cfg->alpha > 0.0) ? cfg->alpha : STENCIL3D_ALPHA_DEFAULT;
    int nthreads = (cfg && cfg->num_threads > 0) ? cfg->num_threads : omp_get_max_threads();
    int bx = (cfg && cfg->bx > 0) ? cfg->bx : 32;
    int by = (cfg && cfg->by > 0) ? cfg->by : 32;
    int bz = (cfg && cfg->bz > 0) ? cfg->bz : 16;
    heat3d_numa_blocked(in, out, nx, ny, nz, alpha, bx, by, bz, nthreads);
}

kernel_3d_fn get_kernel_3d(VariantType vtype) {
    switch (vtype) {
        case VAR_SERIAL:           return wrap_serial;
        case VAR_OMP:              return wrap_omp;
        case VAR_OMP_NUMA:         return wrap_omp_numa;
        case VAR_OMP_BLOCKED:      return wrap_omp_blocked;
        case VAR_OMP_NUMA_BLOCKED: return wrap_omp_numa_blocked;
        default:                   return NULL;
    }
}
