#include "kernels_2d.h"
#include <stdio.h>

kernel_2d_fn get_kernel_2d(Kernel2DType ktype, VariantType vtype) {
    switch (ktype) {
        case KERNEL_GAUSS3:
            switch (vtype) {
                case VAR_V0_SERIAL:           return gauss3_v0_serial;
                case VAR_V1_OMP_NAIVE:        return gauss3_v1_omp;
                case VAR_V2_NUMA_FIRST_TOUCH: return gauss3_v2_numa;
                case VAR_V3_TILING:           return gauss3_v3_tiled;
                case VAR_V4_SIMD_WINDOW:      return gauss3_v4_simd;
                case VAR_V5_PREFETCH_NT:      return gauss3_v5_prefetch;
                default:                      return NULL;
            }
        case KERNEL_GAUSS5:
            switch (vtype) {
                case VAR_V0_SERIAL:           return gauss5_v0_serial;
                case VAR_V1_OMP_NAIVE:        return gauss5_v1_omp;
                case VAR_V2_NUMA_FIRST_TOUCH: return gauss5_v2_numa;
                case VAR_V3_TILING:           return gauss5_v3_tiled;
                case VAR_V4_SIMD_WINDOW:      return gauss5_v4_simd;
                case VAR_V5_PREFETCH_NT:      return gauss5_v5_prefetch;
                default:                      return NULL;
            }
        case KERNEL_SOBEL:
            switch (vtype) {
                case VAR_V0_SERIAL:           return sobel_v0_serial;
                case VAR_V1_OMP_NAIVE:        return sobel_v1_omp;
                case VAR_V2_NUMA_FIRST_TOUCH: return sobel_v2_numa;
                case VAR_V3_TILING:           return sobel_v3_tiled;
                case VAR_V4_SIMD_WINDOW:      return sobel_v4_simd;
                case VAR_V5_PREFETCH_NT:      return sobel_v5_prefetch;
                default:                      return NULL;
            }
        case KERNEL_LAPLACE:
            switch (vtype) {
                case VAR_V0_SERIAL:           return laplace_v0_serial;
                case VAR_V1_OMP_NAIVE:        return laplace_v1_omp;
                case VAR_V2_NUMA_FIRST_TOUCH: return laplace_v2_numa;
                case VAR_V3_TILING:           return laplace_v3_tiled;
                case VAR_V4_SIMD_WINDOW:      return laplace_v4_simd;
                case VAR_V5_PREFETCH_NT:      return laplace_v5_prefetch;
                default:                      return NULL;
            }
        default:
            return NULL;
    }
}

