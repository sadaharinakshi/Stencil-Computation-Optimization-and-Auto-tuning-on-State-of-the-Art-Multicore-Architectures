#include "kernels_3d.h"
#include <stdio.h>

kernel_3d_fn get_kernel_3d(VariantType vtype) {
    switch (vtype) {
        case VAR_V0_SERIAL:           return heat3d_v0_serial;
        case VAR_V1_OMP_NAIVE:        return heat3d_v1_omp;
        case VAR_V2_NUMA_FIRST_TOUCH: return heat3d_v2_numa;
        case VAR_V3_TILING:           return heat3d_v3_tiled;
        case VAR_V4_SIMD_WINDOW:      return heat3d_v4_simd;
        case VAR_V5_PREFETCH_NT:      return heat3d_v5_prefetch;
        case VAR_V6_UNROLL_REG:       return heat3d_v6_unroll_reg;
        default:                      return NULL;
    }
}

