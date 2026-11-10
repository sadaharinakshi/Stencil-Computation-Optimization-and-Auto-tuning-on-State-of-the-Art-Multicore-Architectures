#include "kernels_3d.h"

/*
 * ============================================================================
 * Optimized 3D 7-Point Heat Stencil Implementations
 * ============================================================================
 *
 * This file implements the optimizations required by the project:
 *
 * 1. NUMA-Aware First Touch Memory Initialisation (first_touch_init_3d)
 * 2. Basic OpenMP Parallel Stencil (heat3d_omp)
 * 3. OpenMP Stencil with First Touch (heat3d_omp_numa)
 * 4. Genuine 3D Cache Blocking (heat3d_blocked) using BX x BY x BZ
 * 5. Combined First Touch + 3D Cache Blocking (heat3d_numa_blocked)
 * ============================================================================
 */

/*
 * ============================================================================
 * 1. NUMA-Aware First Touch Initialisation
 * ============================================================================
 *
 * Why First Touch matters on NUMA (Non-Uniform Memory Access) systems:
 * In modern multi-socket server architectures, each CPU socket has its own
 * integrated memory controller and directly attached local RAM (a NUMA node).
 * Accessing local RAM has low latency and high bandwidth; accessing remote RAM
 * across the inter-socket interconnect (Intel UPI or AMD Infinity Fabric)
 * incurs substantial latency penalties and consumes interconnect bandwidth.
 *
 * Modern operating systems (Linux, Windows) use a "First-Touch" page allocation
 * policy: physical memory pages (typically 4 KB or 2 MB) are NOT assigned to a
 * physical NUMA node at malloc() time. Instead, the page is mapped to the NUMA
 * node of whichever thread FIRST writes to it.
 *
 * If a single serial thread initialises an entire multi-gigabyte grid, all
 * physical pages are allocated on Node 0's memory banks. When a parallel OpenMP
 * loop later executes across multiple sockets, threads on Node 1 must make remote
 * memory accesses across the interconnect for every stencil tap.
 *
 * By parallelising the initialisation with the exact same OpenMP static schedule
 * as the computation loop, each thread touches only its own partition, ensuring
 * that pages are physically allocated in local RAM.
 *
 * NOTE ON UMA ARCHITECTURES:
 * On single-socket client processors (such as the 12th Gen Intel Core i5-12450HX),
 * there is only one NUMA node (Node 0). All cores share the same memory controller.
 * In this case, parallel initialisation provides faster startup due to multi-threaded
 * memory bandwidth saturation, but does not provide multi-socket NUMA locality gains.
 */
void first_touch_init_3d(double* in, double* out, int nx, int ny, int nz, int num_threads) {
    int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const size_t plane_size = (size_t)nx * ny;

    /*
     * We distribute the outermost dimension (Z-planes) using static scheduling.
     * This matches the thread-to-plane distribution used in the naive OpenMP kernel.
     */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 0; z < nz; ++z) {
        size_t z_offset = (size_t)z * plane_size;
        for (int y = 0; y < ny; ++y) {
            size_t row_offset = z_offset + (size_t)y * nx;
            for (int x = 0; x < nx; ++x) {
                size_t idx = row_offset + (size_t)x;
                double val = sin((double)idx * 0.001);
                in[idx]  = val;
                out[idx] = val;
            }
        }
    }
}

/*
 * ============================================================================
 * 2. Basic OpenMP Parallel Stencil (heat3d_omp)
 * ============================================================================
 *
 * Parallelises the outermost spatial loop (Z-planes) across OpenMP threads.
 * Uses static scheduling for deterministic, low-overhead chunk assignment.
 */
void heat3d_omp(const double* in, double* out, int nx, int ny, int nz, double alpha, int num_threads) {
    int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const double c0 = 1.0 - 6.0 * alpha;
    const double c1 = alpha;
    const size_t plane_size = (size_t)nx * ny;

    /* Step 1: Copy boundary Z-planes (z = 0 and z = nz - 1) in parallel */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(size_t)(nz - 1) * plane_size + i] = in[(size_t)(nz - 1) * plane_size + i];
    }

    /* Step 2: Parallel loop over interior Z-planes */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        const size_t z_curr  = (size_t)z * plane_size;
        const size_t z_back  = (size_t)(z - 1) * plane_size;
        const size_t z_front = (size_t)(z + 1) * plane_size;

        /* Copy Y boundary rows */
        for (int x = 0; x < nx; ++x) {
            out[z_curr + x] = in[z_curr + x];
            out[z_curr + (size_t)(ny - 1) * nx + x] = in[z_curr + (size_t)(ny - 1) * nx + x];
        }

        /* Iterate through interior Y-rows */
        for (int y = 1; y < ny - 1; ++y) {
            const size_t row_curr = z_curr + (size_t)y * nx;
            const size_t row_top  = z_curr + (size_t)(y - 1) * nx;
            const size_t row_bot  = z_curr + (size_t)(y + 1) * nx;

            /* Copy X boundary columns */
            out[row_curr] = in[row_curr];
            out[row_curr + nx - 1] = in[row_curr + nx - 1];

            /* Compute stencil on interior points */
            for (int x = 1; x < nx - 1; ++x) {
                double center = in[row_curr + x];
                double left   = in[row_curr + x - 1];
                double right  = in[row_curr + x + 1];
                double top    = in[row_top  + x];
                double bottom = in[row_bot  + x];
                double back   = in[z_back + (size_t)y * nx + x];
                double front  = in[z_front + (size_t)y * nx + x];

                double neighbours = left + right + top + bottom + back + front;
                out[row_curr + x] = c0 * center + c1 * neighbours;
            }
        }
    }
}

/*
 * ============================================================================
 * 3. OpenMP 3D Stencil with First-Touch (heat3d_omp_numa)
 * ============================================================================
 *
 * Computation kernel is identical to heat3d_omp. The NUMA locality benefit
 * is realised because first_touch_init_3d was previously called to physically
 * map the pages to the local NUMA memory of each processing thread.
 */
void heat3d_omp_numa(const double* in, double* out, int nx, int ny, int nz, double alpha, int num_threads) {
    heat3d_omp(in, out, nx, ny, nz, alpha, num_threads);
}

/*
 * ============================================================================
 * 4. Genuine 3D Cache Blocking (heat3d_blocked)
 * ============================================================================
 *
 * Why 3D Cache Blocking improves performance:
 * A 3D stencil requires 7 points: center, left, right, top, bottom, back, front.
 * In a large grid (e.g. 256 x 256 x 256 of doubles = 128 MB), each plane is
 * 256 x 256 x 8 bytes = 512 KB.
 *
 * In naive unblocked traversal:
 * When computing at (x, y, z), we read the back plane at (x, y, z - 1) and front
 * plane at (x, y, z + 1). By the time the loop advances from z to z + 1, an
 * entire 512 KB plane has been swept through cache. If the working set exceeds
 * cache capacity (L1=32-48KB, L2=1.25MB per core), the data is evicted to DRAM
 * before it can be reused as the center or back point in subsequent planes!
 *
 * Genuine 3D Cache Blocking partitions the 3D volume into sub-cubes of size
 * BX x BY x BZ:
 * - Working set per tile ~= 3 * (BX * BY * BZ) * 8 bytes.
 * - For BX=32, BY=32, BZ=16: 32 x 32 x 16 = 16,384 elements * 8 bytes ~= 131 KB.
 * This easily fits inside the high-speed L2 cache, ensuring high temporal cache
 * reuse between the z, z-1, and z+1 planes!
 *
 * LOOP STRUCTURE:
 * 3 Outer Tile Loops:   bz (step BZ), by (step BY), bx (step BX)
 * 3 Inner Element Loops: z (inside tile), y (inside tile), x (inside tile)
 */
void heat3d_blocked(const double* in, double* out, int nx, int ny, int nz, double alpha,
                    int bx, int by, int bz, int num_threads) {
    int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const double c0 = 1.0 - 6.0 * alpha;
    const double c1 = alpha;
    const size_t plane_size = (size_t)nx * ny;

    /* Ensure tile sizes are at least 1 */
    const int tile_x = (bx > 0) ? bx : 32;
    const int tile_y = (by > 0) ? by : 32;
    const int tile_z = (bz > 0) ? bz : 16;

    /*
     * Step 1: Copy outer boundary conditions in parallel before tile execution.
     * This ensures all boundary cells are correctly preserved.
     */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];                                                         /* z = 0 */
        out[(size_t)(nz - 1) * plane_size + i] = in[(size_t)(nz - 1) * plane_size + i]; /* z = nz-1 */
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        size_t z_curr = (size_t)z * plane_size;
        for (int x = 0; x < nx; ++x) {
            out[z_curr + x] = in[z_curr + x];                                   /* y = 0 */
            out[z_curr + (size_t)(ny - 1) * nx + x] = in[z_curr + (size_t)(ny - 1) * nx + x]; /* y = ny-1 */
        }
        for (int y = 1; y < ny - 1; ++y) {
            size_t row = z_curr + (size_t)y * nx;
            out[row] = in[row];                                                 /* x = 0 */
            out[row + nx - 1] = in[row + nx - 1];                               /* x = nx-1 */
        }
    }

    /*
     * Step 2: Genuine 3D Cache-Blocked Stencil Loops
     *
     * The outer tile loops iterate through 3D blocks.
     * OpenMP parallelises across the BZ and BY tile dimensions using collapse(2).
     * The BX tile loop remains within the thread to preserve contiguous memory stream
     * along the X axis.
     */
    #pragma omp parallel for num_threads(nthreads) schedule(static) collapse(2)
    for (int bz_start = 1; bz_start < nz - 1; bz_start += tile_z) {
        for (int by_start = 1; by_start < ny - 1; by_start += tile_y) {
            for (int bx_start = 1; bx_start < nx - 1; bx_start += tile_x) {

                /* Compute tile boundary bounds, safely handling partial boundary tiles */
                int z_end = (bz_start + tile_z < nz - 1) ? (bz_start + tile_z) : (nz - 1);
                int y_end = (by_start + tile_y < ny - 1) ? (by_start + tile_y) : (ny - 1);
                int x_end = (bx_start + tile_x < nx - 1) ? (bx_start + tile_x) : (nx - 1);

                /*
                 * 3 Inner loops iterating through elements within the current 3D tile.
                 * Data within this tile fits in the core's private cache hierarchy.
                 */
                for (int z = bz_start; z < z_end; ++z) {
                    const size_t z_curr  = (size_t)z * plane_size;
                    const size_t z_back  = (size_t)(z - 1) * plane_size;
                    const size_t z_front = (size_t)(z + 1) * plane_size;

                    for (int y = by_start; y < y_end; ++y) {
                        const size_t row_curr = z_curr + (size_t)y * nx;
                        const size_t row_top  = z_curr + (size_t)(y - 1) * nx;
                        const size_t row_bot  = z_curr + (size_t)(y + 1) * nx;

                        for (int x = bx_start; x < x_end; ++x) {
                            double center = in[row_curr + x];
                            double left   = in[row_curr + x - 1];
                            double right  = in[row_curr + x + 1];
                            double top    = in[row_top  + x];
                            double bottom = in[row_bot  + x];
                            double back   = in[z_back + (size_t)y * nx + x];
                            double front  = in[z_front + (size_t)y * nx + x];

                            double neighbours = left + right + top + bottom + back + front;
                            out[row_curr + x] = c0 * center + c1 * neighbours;
                        }
                    }
                }
            }
        }
    }
}

/*
 * ============================================================================
 * 5. Combined First-Touch + Genuine 3D Cache Blocking (heat3d_numa_blocked)
 * ============================================================================
 */
void heat3d_numa_blocked(const double* in, double* out, int nx, int ny, int nz, double alpha,
                         int bx, int by, int bz, int num_threads) {
    heat3d_blocked(in, out, nx, ny, nz, alpha, bx, by, bz, num_threads);
}
