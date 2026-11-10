#ifndef KERNELS_3D_H
#define KERNELS_3D_H

#include "common.h"

/*
 * 1D Flat Indexing Macro:
 * Grid is stored linearly in row-major order: Z is outermost (planes),
 * Y is middle (rows), and X is innermost (columns, unit stride).
 *
 * index = (z * ny + y) * nx + x
 */
#define INDEX(x, y, z, nx, ny) (((size_t)(z) * (size_t)(ny) + (size_t)(y)) * (size_t)(nx) + (size_t)(x))

/* Memory Initialization Routines */
void serial_init_3d(double* in, double* out, int nx, int ny, int nz);
void first_touch_init_3d(double* in, double* out, int nx, int ny, int nz, int num_threads);

/* 1. Serial 3D 7-Point Stencil (Reference) */
void heat3d_serial(const double* in, double* out, int nx, int ny, int nz, double alpha);

/* 2. OpenMP 3D 7-Point Stencil (Naive Parallel Z-Planes) */
void heat3d_omp(const double* in, double* out, int nx, int ny, int nz, double alpha, int num_threads);

/* 3. OpenMP 3D 7-Point Stencil with First-Touch Initialisation */
void heat3d_omp_numa(const double* in, double* out, int nx, int ny, int nz, double alpha, int num_threads);

/* 4. OpenMP 3D 7-Point Stencil with Genuine 3D Cache Blocking (BX x BY x BZ) */
void heat3d_blocked(const double* in, double* out, int nx, int ny, int nz, double alpha,
                    int bx, int by, int bz, int num_threads);

/* 5. OpenMP 3D 7-Point Stencil with First-Touch + Genuine 3D Cache Blocking */
void heat3d_numa_blocked(const double* in, double* out, int nx, int ny, int nz, double alpha,
                         int bx, int by, int bz, int num_threads);

/* Unified Kernel Function Pointer Type */
typedef void (*kernel_3d_fn)(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg);

/* Dispatcher returning the corresponding kernel function pointer */
kernel_3d_fn get_kernel_3d(VariantType vtype);

#endif /* KERNELS_3D_H */
