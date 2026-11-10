#include "kernels_3d.h"

/*
 * ============================================================================
 * Serial Reference 3D 7-Point Heat Stencil Implementation
 * ============================================================================
 *
 * Problem Description:
 * In a 3D heat diffusion simulation, a solid domain is discretized into a 3D
 * Cartesian grid of size NX x NY x NZ. Each grid cell holds a double-precision
 * temperature value.
 *
 * In each time step, every interior cell (1 <= x < NX-1, 1 <= y < NY-1, 1 <= z < NZ-1)
 * updates its temperature based on its current temperature and the conduction
 * from its 6 immediate orthogonal neighbours:
 *   - Left:   (x - 1, y, z)
 *   - Right:  (x + 1, y, z)
 *   - Top:    (x, y - 1, z)
 *   - Bottom: (x, y + 1, z)
 *   - Back:   (x, y, z - 1)
 *   - Front:  (x, y, z + 1)
 *
 * Mathematical Update:
 *   T_new[x,y,z] = (1 - 6*alpha) * T[x,y,z]
 *                + alpha * ( T[x-1,y,z] + T[x+1,y,z]
 *                          + T[x,y-1,z] + T[x,y+1,z]
 *                          + T[x,y,z-1] + T[x,y,z+1] )
 *
 * Boundary Condition:
 * Dirichlet boundary condition is applied: all boundary cells (x=0, x=NX-1,
 * y=0, y=NY-1, z=0, z=NZ-1) remain fixed at their initial values.
 *
 * Array Storage:
 * Stored as a contiguous 1D flattened array with row-major ordering:
 *   index = ((size_t)z * ny + y) * nx + x
 * ============================================================================
 */

/*
 * Single-threaded serial memory initialisation.
 * Initialises both 'in' (current) and 'out' (next) arrays with a deterministic pattern.
 */
void serial_init_3d(double* in, double* out, int nx, int ny, int nz) {
    size_t total_elements = (size_t)nx * ny * nz;
    for (size_t i = 0; i < total_elements; ++i) {
        double val = sin((double)i * 0.001);
        in[i] = val;
        out[i] = val;
    }
}

/*
 * 1. Serial 3D 7-Point Stencil Reference Kernel
 */
void heat3d_serial(const double* in, double* out, int nx, int ny, int nz, double alpha) {
    const double c0 = 1.0 - 6.0 * alpha;
    const double c1 = alpha;
    const size_t plane_size = (size_t)nx * ny;

    /*
     * Step 1: Preserve Dirichlet boundary conditions.
     * Copy the outer boundary planes (z = 0 and z = nz - 1).
     */
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];                                       /* z = 0 plane */
        out[(size_t)(nz - 1) * plane_size + i] = in[(size_t)(nz - 1) * plane_size + i]; /* z = nz-1 plane */
    }

    /*
     * Step 2: Iterate through all interior Z-planes (1 <= z < nz - 1).
     */
    for (int z = 1; z < nz - 1; ++z) {
        const size_t z_curr = (size_t)z * plane_size;
        const size_t z_back = (size_t)(z - 1) * plane_size;
        const size_t z_front = (size_t)(z + 1) * plane_size;

        /* Copy Y boundary rows (y = 0 and y = ny - 1) for the current plane */
        for (int x = 0; x < nx; ++x) {
            out[z_curr + x] = in[z_curr + x];                               /* y = 0 row */
            out[z_curr + (size_t)(ny - 1) * nx + x] = in[z_curr + (size_t)(ny - 1) * nx + x]; /* y = ny-1 row */
        }

        /* Iterate through interior Y-rows (1 <= y < ny - 1) */
        for (int y = 1; y < ny - 1; ++y) {
            const size_t row_curr = z_curr + (size_t)y * nx;
            const size_t row_top  = z_curr + (size_t)(y - 1) * nx;
            const size_t row_bot  = z_curr + (size_t)(y + 1) * nx;

            /* Copy X boundary columns (x = 0 and x = nx - 1) for the current row */
            out[row_curr] = in[row_curr];                               /* x = 0 column */
            out[row_curr + nx - 1] = in[row_curr + nx - 1];             /* x = nx-1 column */

            /*
             * Innermost loop: Iterate through interior X-columns (1 <= x < nx - 1).
             * Unit stride in X ensures sequential memory access for maximum spatial locality.
             */
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
