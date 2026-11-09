#include "kernels_3d.h"

/* Reference 3D 7-point Heat Stencil (Serial Naive)
 * Grid: nx * ny * nz in double precision.
 * Flops: 8 flops per interior point (6 additions + 2 multiplications).
 * Boundary: Dirichlet (boundary copied from in to out).
 */
void heat3d_v0_serial(const double* in, double* out, int nx, int ny, int nz, double c0, double c1, const TuningConfig* cfg) {
    (void)cfg;
    const size_t plane_size = (size_t)nx * ny;

    /* Copy boundary planes z = 0 and z = nz - 1 */
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(nz - 1) * plane_size + i] = in[(nz - 1) * plane_size + i];
    }

    for (int z = 1; z < nz - 1; ++z) {
        const size_t z_offset = (size_t)z * plane_size;
        const size_t zm1_offset = (size_t)(z - 1) * plane_size;
        const size_t zp1_offset = (size_t)(z + 1) * plane_size;

        /* Copy boundary rows y = 0 and y = ny - 1 */
        for (int x = 0; x < nx; ++x) {
            out[z_offset + x] = in[z_offset + x];
            out[z_offset + (ny - 1) * nx + x] = in[z_offset + (ny - 1) * nx + x];
        }

        for (int y = 1; y < ny - 1; ++y) {
            const size_t row_offset = z_offset + (size_t)y * nx;
            const size_t ym1_offset = z_offset + (size_t)(y - 1) * nx;
            const size_t yp1_offset = z_offset + (size_t)(y + 1) * nx;

            /* Boundary columns x = 0 and x = nx - 1 */
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

