#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
#include <malloc.h>
static inline void* hpc_aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static inline void hpc_aligned_free(void* ptr) {
    _aligned_free(ptr);
}
#else
static inline void* hpc_aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
}
static inline void hpc_aligned_free(void* ptr) {
    free(ptr);
}
#endif

#define ALIGNMENT_BYTES 64

typedef enum {
    KERNEL_GAUSS3,
    KERNEL_GAUSS5,
    KERNEL_SOBEL,
    KERNEL_LAPLACE
} Kernel2DType;

typedef enum {
    VAR_V0_SERIAL = 0,
    VAR_V1_OMP_NAIVE = 1,
    VAR_V2_NUMA_FIRST_TOUCH = 2,
    VAR_V3_TILING = 3,
    VAR_V4_SIMD_WINDOW = 4,
    VAR_V5_PREFETCH_NT = 5,
    VAR_V6_UNROLL_REG = 6
} VariantType;

/* Structure for tuning configuration */
typedef struct {
    int tx;          /* Tile size X */
    int ty;          /* Tile size Y */
    int tz;          /* Tile size Z (3D only) */
    int num_threads; /* OpenMP threads */
    int sched_type;  /* 0: static, 1: dynamic, 2: guided */
    int chunk_size;  /* Schedule chunk size */
    int use_simd;    /* 0 or 1 */
    int prefetch_dist; /* Prefetch distance in elements (e.g. 0, 8, 16, 32) */
    int unroll_factor; /* Unroll factor (e.g. 1, 2, 4) */
} TuningConfig;

static inline TuningConfig default_tuning_config(void) {
    TuningConfig cfg;
    cfg.tx = 64;
    cfg.ty = 32;
    cfg.tz = 16;
    cfg.num_threads = omp_get_max_threads();
    cfg.sched_type = 0; /* static */
    cfg.chunk_size = 0; /* default */
    cfg.use_simd = 1;
    cfg.prefetch_dist = 16;
    cfg.unroll_factor = 2;
    return cfg;
}

/* Analytical Flops and Byte Counts */
static inline double get_kernel_flops_per_pixel(Kernel2DType k) {
    switch (k) {
        case KERNEL_GAUSS3:  return 17.0; /* 9 muls + 8 adds */
        case KERNEL_GAUSS5:  return 49.0; /* 25 muls + 24 adds */
        case KERNEL_SOBEL:   return 16.0; /* 6 for Gx, 6 for Gy, 4 for mag/sqrt */
        case KERNEL_LAPLACE: return 6.0;  /* 4 adds, 1 mul, 1 sub */
        default: return 0.0;
    }
}

static inline double get_kernel_ideal_bytes_per_pixel(Kernel2DType k) {
    (void)k;
    return 8.0; /* float: 1 read (4B) + 1 write (4B) assuming perfect cache reuse */
}

static inline double get_kernel_noreuse_bytes_per_pixel(Kernel2DType k) {
    switch (k) {
        case KERNEL_GAUSS3:  return 40.0; /* 9 reads * 4B + 1 write * 4B */
        case KERNEL_GAUSS5:  return 104.0;/* 25 reads * 4B + 1 write * 4B */
        case KERNEL_SOBEL:   return 52.0; /* 12 reads * 4B (Gx+Gy) + 1 write * 4B */
        case KERNEL_LAPLACE: return 24.0; /* 5 reads * 4B + 1 write * 4B */
        default: return 8.0;
    }
}

/* 3D 7-Point Stencil Constants (double) */
#define STENCIL3D_FLOPS_PER_GRID 8.0  /* 6 adds + 2 muls (Datta et al. SC08) */
#define STENCIL3D_IDEAL_BYTES    16.0 /* 1 double read (8B) + 1 double write (8B) */
#define STENCIL3D_NOREUSE_BYTES  64.0 /* 7 double reads (56B) + 1 double write (8B) */

static inline const char* kernel_name(Kernel2DType k) {
    switch (k) {
        case KERNEL_GAUSS3:  return "Gauss3x3";
        case KERNEL_GAUSS5:  return "Gauss5x5";
        case KERNEL_SOBEL:   return "Sobel";
        case KERNEL_LAPLACE: return "Laplacian";
        default: return "Unknown";
    }
}

static inline const char* variant_name(VariantType v) {
    switch (v) {
        case VAR_V0_SERIAL:           return "v0_serial";
        case VAR_V1_OMP_NAIVE:        return "v1_omp_naive";
        case VAR_V2_NUMA_FIRST_TOUCH: return "v2_first_touch";
        case VAR_V3_TILING:           return "v3_tiling";
        case VAR_V4_SIMD_WINDOW:      return "v4_simd_window";
        case VAR_V5_PREFETCH_NT:      return "v5_prefetch_nt";
        case VAR_V6_UNROLL_REG:       return "v6_unroll_reg";
        default: return "unknown";
    }
}

#endif /* COMMON_H */

