#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/*
 * Memory Alignment:
 * 64-byte alignment matches standard x86-64 cache line size, preventing false
 * sharing at thread partition boundaries and ensuring aligned vector loads.
 */
#define ALIGNMENT_BYTES 64

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

/*
 * ============================================================================
 * 3D 7-Point Heat Stencil Analytical Model (Double Precision: 8 bytes/element)
 * ============================================================================
 *
 * Update equation:
 *   T_new[x,y,z] = (1 - 6*alpha) * T[x,y,z]
 *                + alpha * ( T[x-1,y,z] + T[x+1,y,z]
 *                          + T[x,y-1,z] + T[x,y+1,z]
 *                          + T[x,y,z-1] + T[x,y,z+1] )
 *
 * FLOP Count per Interior Point Update:
 *   - 5 additions to sum the 6 spatial neighbours
 *   - 1 multiplication by alpha
 *   - 1 multiplication of center cell by (1 - 6*alpha)
 *   - 1 addition combining center and neighbour terms
 *   Total = 6 additions + 2 multiplications = 8 FLOPs per interior grid update.
 *
 * Memory Traffic Model & Cache Reuse Explanation:
 * ------------------------------------------------
 * Distinguishing Theoretical Bounds from Actual DRAM Traffic:
 *
 * 1. Theoretical Worst-Case Upper Bound (Cold / Zero Cache Reuse):
 *    If every memory access missed all cache levels and required a DRAM fetch:
 *    - 7 reads (center + 6 neighbours) * 8 bytes = 56 bytes read.
 *    - 1 write (updated center) * 8 bytes = 8 bytes written.
 *    Total = 64 bytes/update.
 *    (On standard write-allocate architectures, writing an uncached line also
 *     triggers a read-for-ownership (RFO), adding an extra 8 bytes = 72 bytes).
 *    Arithmetic Intensity (No Reuse) = 8 FLOPs / 64 bytes = 0.125 FLOP/byte.
 *
 * 2. Theoretical Best-Case Lower Bound (Ideal Cache Reuse):
 *    If temporal and spatial cache locality is 100% effective:
 *    - Each grid cell is transferred from DRAM into cache exactly once (8 bytes).
 *    - The updated grid cell is written back to DRAM once (8 bytes).
 *    Total = 16 bytes/update (or 24 bytes with write-allocate).
 *    Arithmetic Intensity (Ideal Reuse) = 8 FLOPs / 16 bytes = 0.500 FLOP/byte.
 *
 * 3. Actual DRAM Traffic:
 *    Falls between 16 and 64 bytes/update. In an unblocked large grid (e.g. 256^3),
 *    the large stride between plane (z-1) and plane (z+1) exceeds cache capacity,
 *    causing capacity evictions and driving actual traffic closer to the 64-byte upper bound.
 *    Genuine 3D cache blocking (BX x BY x BZ) keeps sub-volumes in fast L2/L3 cache,
 *    reducing DRAM traffic towards the ideal 16-24 bytes/update.
 */
#define STENCIL3D_FLOPS_PER_POINT 8.0
#define STENCIL3D_NOREUSE_BYTES   64.0  /* Theoretical worst-case upper bound */
#define STENCIL3D_IDEAL_BYTES     16.0  /* Theoretical ideal lower bound */
#define STENCIL3D_ALPHA_DEFAULT   0.10  /* Documented stable explicit heat diffusion coefficient */

/*
 * The 5 Required Implementation Variants:
 * 1. Serial 3D stencil reference
 * 2. Basic OpenMP 3D stencil
 * 3. OpenMP 3D stencil + First Touch initialisation
 * 4. OpenMP 3D stencil + Genuine 3D Cache Blocking
 * 5. OpenMP 3D stencil + First Touch + Genuine 3D Cache Blocking
 */
typedef enum {
    VAR_SERIAL = 0,
    VAR_OMP = 1,
    VAR_OMP_NUMA = 2,
    VAR_OMP_BLOCKED = 3,
    VAR_OMP_NUMA_BLOCKED = 4
} VariantType;

#define NUM_VARIANTS 5

/* Configuration structure for 3D stencil execution and benchmarking */
typedef struct {
    int bx;           /* 3D Tile dimension X */
    int by;           /* 3D Tile dimension Y */
    int bz;           /* 3D Tile dimension Z */
    int num_threads;  /* OpenMP thread count */
    int num_steps;    /* Number of time-step iterations */
    double alpha;     /* Diffusion coefficient */
} StencilConfig;

static inline StencilConfig default_stencil_config(void) {
    StencilConfig cfg;
    cfg.bx = 32;      /* Default X tile: 32 doubles (256 bytes) */
    cfg.by = 32;      /* Default Y tile: 32 rows */
    cfg.bz = 16;      /* Default Z tile: 16 planes */
    cfg.num_threads = omp_get_max_threads();
    cfg.num_steps = 1;/* Default 1 time step */
    cfg.alpha = STENCIL3D_ALPHA_DEFAULT;
    return cfg;
}

static inline const char* variant_name(VariantType v) {
    switch (v) {
        case VAR_SERIAL:           return "1. Serial Reference";
        case VAR_OMP:              return "2. OpenMP (Naive)";
        case VAR_OMP_NUMA:         return "3. OpenMP + First Touch";
        case VAR_OMP_BLOCKED:      return "4. OpenMP + 3D Blocked";
        case VAR_OMP_NUMA_BLOCKED: return "5. OpenMP + FT + 3D Blocked";
        default:                   return "Unknown";
    }
}

static inline const char* variant_short_name(VariantType v) {
    switch (v) {
        case VAR_SERIAL:           return "serial";
        case VAR_OMP:              return "omp";
        case VAR_OMP_NUMA:         return "omp_ft";
        case VAR_OMP_BLOCKED:      return "omp_blocked";
        case VAR_OMP_NUMA_BLOCKED: return "omp_ft_blocked";
        default:                   return "unknown";
    }
}

#endif /* COMMON_H */
