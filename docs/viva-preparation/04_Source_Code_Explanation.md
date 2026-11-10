# 04. Source Code Deep Dive & Function Walkthrough

---

## 1. Header Files Overview

### `include/common.h`
This header contains shared platform definitions, memory alignment helpers, analytical constants, and configuration structs.

#### Aligned Memory Allocation:
```c
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
```
* **Why 64 bytes?** x86-64 processors fetch and evict data in 64-byte chunks (cache lines). If an array begins at an unaligned address, vector instructions and cache prefetchers experience split-load penalties. Furthermore, when multiple threads process adjacent slices of an array, 64-byte alignment prevents **false sharing** (two threads modifying data within the same cache line).
* **Cross-Platform Portability:** On Windows/MinGW, `_aligned_malloc()` and `_aligned_free()` from `<malloc.h>` are required. On Linux, `posix_memalign()` and standard `free()` are used.

#### Analytical Constants & Variant Enum:
```c
#define STENCIL3D_FLOPS_PER_POINT 8.0
#define STENCIL3D_NOREUSE_BYTES   64.0  /* Theoretical worst-case upper bound */
#define STENCIL3D_IDEAL_BYTES     16.0  /* Theoretical ideal lower bound */
#define STENCIL3D_ALPHA_DEFAULT   0.10

typedef enum {
    VAR_SERIAL = 0,
    VAR_OMP = 1,
    VAR_OMP_NUMA = 2,
    VAR_OMP_BLOCKED = 3,
    VAR_OMP_NUMA_BLOCKED = 4
} VariantType;
```

---

### `include/kernels_3d.h`
Contains function declarations and the primary indexing macro:

```c
#define INDEX(x, y, z, nx, ny) (((size_t)(z) * (size_t)(ny) + (size_t)(y)) * (size_t)(nx) + (size_t)(x))
```
* **Why `size_t` casts?** In a large grid (e.g., $1024 \times 1024 \times 1024$), the total element count is $2^{30} \approx 1.07 \times 10^9$. If standard 32-bit `int` arithmetic is used, coordinates like $z \times ny \times nx$ can overflow a signed 32-bit integer ($2^{31}-1 \approx 2.14 \times 10^9$) during offset calculations. Casting to 64-bit unsigned `size_t` prevents silent integer overflow.

---

## 2. Serial Implementation (`src/kernels_3d_serial.c`)

### Memory Initialization (`serial_init_3d`):
```c
void serial_init_3d(double* in, double* out, int nx, int ny, int nz) {
    size_t total_elements = (size_t)nx * ny * nz;
    for (size_t i = 0; i < total_elements; ++i) {
        double val = sin((double)i * 0.001);
        in[i] = val;
        out[i] = val;
    }
}
```
* **Purpose:** Initialises the entire grid on a single thread using a deterministic mathematical function `sin(i * 0.001)`.
* **Identical State:** Both `in` and `out` arrays are initialized with identical initial temperatures so boundary cells on `out` already hold their permanent Dirichlet boundary values.

### The Serial Reference Stencil (`heat3d_serial`):
```c
void heat3d_serial(const double* in, double* out, int nx, int ny, int nz, double alpha) {
    const double c0 = 1.0 - 6.0 * alpha;
    const double c1 = alpha;
    const size_t plane_size = (size_t)nx * ny;

    /* Step 1: Copy boundary Z-planes (z = 0 and z = nz - 1) */
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(size_t)(nz - 1) * plane_size + i] = in[(size_t)(nz - 1) * plane_size + i];
    }

    /* Step 2: Iterate through all interior Z-planes */
    for (int z = 1; z < nz - 1; ++z) {
        const size_t z_curr  = (size_t)z * plane_size;
        const size_t z_back  = (size_t)(z - 1) * plane_size;
        const size_t z_front = (size_t)(z + 1) * plane_size;

        /* Copy Y boundary rows (y = 0 and y = ny - 1) */
        for (int x = 0; x < nx; ++x) {
            out[z_curr + x] = in[z_curr + x];
            out[z_curr + (size_t)(ny - 1) * nx + x] = in[z_curr + (size_t)(ny - 1) * nx + x];
        }

        /* Iterate through interior Y-rows */
        for (int y = 1; y < ny - 1; ++y) {
            const size_t row_curr = z_curr + (size_t)y * nx;
            const size_t row_top  = z_curr + (size_t)(y - 1) * nx;
            const size_t row_bot  = z_curr + (size_t)(y + 1) * nx;

            /* Copy X boundary columns (x = 0 and x = nx - 1) */
            out[row_curr] = in[row_curr];
            out[row_curr + nx - 1] = in[row_curr + nx - 1];

            /* Innermost loop: Unit stride along X */
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
```

#### Code Mechanics Explained:
1. `c0 = 1.0 - 6.0 * alpha` and `c1 = alpha`: Precomputed outside all loops. For $\alpha = 0.10$, $c0 = 0.40$ and $c1 = 0.10$.
2. **Loop Order ($Z \to Y \to X$):**
   * $Z$ is outermost: step size $= NX \times NY$ elements.
   * $Y$ is middle: step size $= NX$ elements.
   * $X$ is innermost: step size $= 1$ element (unit stride).
   * **Why this order?** In C row-major layout, memory is laid out contiguously along $X$. Iterating along $X$ in the innermost loop ensures maximum **spatial cache locality**. When `in[row_curr + x]` is accessed, the CPU hardware prefetcher fetches the next 64 bytes (8 doubles), so subsequent iterations hit directly in L1 data cache.
3. **Boundary Handling:**
   * Step 1 copies the top and bottom $Z$ planes.
   * Inside the $Z$ loop, outer $Y$ rows are copied.
   * Inside the $Y$ loop, the first ($x=0$) and last ($x=NX-1$) columns are copied.
   * This guarantees complete Dirichlet boundary integrity with zero out-of-bounds access.

---

## 3. Parallel & Optimized Implementations (`src/kernels_3d_opt.c`)

### 1. NUMA First Touch Initialization (`first_touch_init_3d`):
```c
void first_touch_init_3d(double* in, double* out, int nx, int ny, int nz, int num_threads) {
    int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const size_t plane_size = (size_t)nx * ny;

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
```
* **The Directive:** `#pragma omp parallel for schedule(static)` distributes the $Z$-planes evenly across threads.
* **Why it matters:** On operating systems adhering to the First Touch policy (Linux, Windows Server), memory pages are mapped to the NUMA node where the executing thread runs. Because the computation loop also partitions planes statically along $Z$, each thread later accesses memory already located on its local memory channel.

---

### 2. Basic OpenMP Parallel Stencil (`heat3d_omp`):
```c
void heat3d_omp(const double* in, double* out, int nx, int ny, int nz, double alpha, int num_threads) {
    int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const double c0 = 1.0 - 6.0 * alpha;
    const double c1 = alpha;
    const size_t plane_size = (size_t)nx * ny;

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(size_t)(nz - 1) * plane_size + i] = in[(size_t)(nz - 1) * plane_size + i];
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        ... /* Same interior loops as serial */
    }
}
```
* **Coarse-Grained Parallelism:** Parallelizing at the outermost $Z$ loop minimizes OpenMP fork/join overhead. Each thread receives a large chunk of contiguous planes (e.g., $200 / 12 \approx 16$ planes per thread).
* **Thread Safety:** Every thread writes to completely disjoint memory addresses in `out`. Because threads read from the unchanged `in` array, there are zero data races.

---

### 3. Genuine 3D Cache Blocking (`heat3d_blocked`):
This is the core optimization required by the professor.

```c
void heat3d_blocked(const double* in, double* out, int nx, int ny, int nz, double alpha,
                    int bx, int by, int bz, int num_threads) {
    int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const double c0 = 1.0 - 6.0 * alpha;
    const double c1 = alpha;
    const size_t plane_size = (size_t)nx * ny;

    const int tile_x = (bx > 0) ? bx : 32;
    const int tile_y = (by > 0) ? by : 32;
    const int tile_z = (bz > 0) ? bz : 16;

    /* Step 1: Parallel boundary copy */
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < plane_size; ++i) {
        out[i] = in[i];
        out[(size_t)(nz - 1) * plane_size + i] = in[(size_t)(nz - 1) * plane_size + i];
    }
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int z = 1; z < nz - 1; ++z) {
        size_t z_curr = (size_t)z * plane_size;
        for (int x = 0; x < nx; ++x) {
            out[z_curr + x] = in[z_curr + x];
            out[z_curr + (size_t)(ny - 1) * nx + x] = in[z_curr + (size_t)(ny - 1) * nx + x];
        }
        for (int y = 1; y < ny - 1; ++y) {
            size_t row = z_curr + (size_t)y * nx;
            out[row] = in[row];
            out[row + nx - 1] = in[row + nx - 1];
        }
    }

    /* Step 2: 3 Outer Tile Loops */
    #pragma omp parallel for num_threads(nthreads) schedule(static) collapse(2)
    for (int bz_start = 1; bz_start < nz - 1; bz_start += tile_z) {
        for (int by_start = 1; by_start < ny - 1; by_start += tile_y) {
            for (int bx_start = 1; bx_start < nx - 1; bx_start += tile_x) {

                /* Safe boundary clamp for incomplete edge tiles */
                int z_end = (bz_start + tile_z < nz - 1) ? (bz_start + tile_z) : (nz - 1);
                int y_end = (by_start + tile_y < ny - 1) ? (by_start + tile_y) : (ny - 1);
                int x_end = (bx_start + tile_x < nx - 1) ? (bx_start + tile_x) : (nx - 1);

                /* 3 Inner Element Loops */
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
```

#### Detailed Breakdown of the 6 Loops:
1. `collapse(2)` on outer `bz_start` and `by_start`: Flattens the outer two tile loops into a single iteration space of size $\lceil (NZ-2)/BZ \rceil \times \lceil (NY-2)/BY \rceil$, providing ample work chunks for 12 OpenMP threads.
2. `bx_start` loop: Runs sequentially within each thread. This ensures that a thread executes tiles along the contiguous $X$-direction, maximizing L1 cache line hardware streaming.
3. `z_end, y_end, x_end`: Compute the dynamic upper limit. If a dimension is not evenly divisible by the tile size (e.g., $NZ=67, BZ=16$), the remainder tile stops safely at $NZ-1$ rather than overshooting into memory.
4. Three inner loops (`z`, `y`, `x`): Sweep the sub-volume. Because the tile working set is small ($\approx 192\text{ KB}$ for $32 \times 16 \times 16$), the $Z-1$, $Z$, and $Z+1$ planes stay resident in the core's private L2 cache throughout the tile's computation.

---

## 4. Kernel Dispatcher (`src/kernels_3d_dispatch.c`)

Implements a clean function-pointer dispatcher mapping `VariantType` to uniform wrapper signatures:

```c
typedef void (*kernel_3d_fn)(const double* in, double* out, int nx, int ny, int nz, const StencilConfig* cfg);

kernel_3d_fn get_kernel_3d(VariantType vtype) {
    switch (vtype) {
        case VAR_SERIAL:           return wrap_serial;
        case VAR_OMP:              return wrap_omp;
        case VAR_OMP_NUMA:         return wrap_omp_numa;
        case VAR_OMP_BLOCKED:      return wrap_omp_blocked;
        case VAR_OMP_NUMA_BLOCKED: return wrap_numa_blocked;
        default:                   return NULL;
    }
}
```
* **Benefit:** Allows the benchmark harness and correctness checker to test all 5 variants cleanly without code duplication.

