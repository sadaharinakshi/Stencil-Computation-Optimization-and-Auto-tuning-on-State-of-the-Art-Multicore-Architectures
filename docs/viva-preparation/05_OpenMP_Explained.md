# 05. OpenMP Parallelization Explained

This document explains how OpenMP is used in the P13 HPC project to parallelize the 3D 7-point heat stencil computation. It breaks down the directives, scheduling policies, loop transformations, and synchronization semantics used in our implementation.

---

## 1. What is OpenMP?

**OpenMP (Open Multi-Processing)** is an API for shared-memory parallel programming in C, C++, and Fortran. It uses compiler directives (`#pragma omp`), runtime library routines (such as `omp_get_wtime()`), and environment variables (such as `OMP_NUM_THREADS`).

### Fork-Join Model
OpenMP follows the **fork-join execution model**:
1. The program starts as a single execution thread called the **master thread** (Thread 0).
2. When the master thread encounters a parallel construct (`#pragma omp parallel`), it creates a team of worker threads (**fork**).
3. The parallel region is executed concurrently by all threads in the team.
4. At the end of the parallel construct, there is an implicit barrier where all threads synchronize. Only the master thread continues execution beyond this point (**join**).

```
Master Thread ──────────────────┬───────────────────────────────┬───────────────────>
                                │  Fork into worker threads     │  Join & barrier
Worker Thread 1                 ├───────────────────────────────┤
Worker Thread 2                 ├───────────────────────────────┤
Worker Thread 3                 └───────────────────────────────┘
                                <────── Parallel Region ────────>
```

---

## 2. Naive OpenMP 3D Stencil (`heat3d_omp`)

In [src/kernels_3d_opt.c](file:///c:/Users/Vaarshini%20Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/src/kernels_3d_opt.c), the basic parallel stencil is implemented as follows:

```c
void heat3d_omp(const double *restrict u_in,
                double *restrict u_out,
                size_t nx, size_t ny, size_t nz,
                const StencilConfig *config)
{
    const double c0 = config->c0;
    const double c1 = config->c1;

    #pragma omp parallel for schedule(static) collapse(2) default(none) \
        shared(u_in, u_out, nx, ny, nz, c0, c1)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t center = INDEX(x, y, z, nx, ny);
                u_out[center] = c0 * u_in[center] +
                                c1 * (u_in[INDEX(x + 1, y,     z,     nx, ny)] +
                                      u_in[INDEX(x - 1, y,     z,     nx, ny)] +
                                      u_in[INDEX(x,     y + 1, z,     nx, ny)] +
                                      u_in[INDEX(x,     y - 1, z,     nx, ny)] +
                                      u_in[INDEX(x,     y,     z + 1, nx, ny)] +
                                      u_in[INDEX(x,     y,     z - 1, nx, ny)]);
            }
        }
    }
}
```

---

## 3. Dissecting the Pragmas and Clauses

### A. `#pragma omp parallel for`
This combines two directives:
- `#pragma omp parallel`: Spawns a team of threads.
- `#pragma omp for`: Distributes the iterations of the following loop among the active threads.

Combining them avoids the overhead of creating a parallel region separately if only a single loop is being parallelized.

### B. `collapse(2)`
The `collapse(n)` clause instructs the compiler to collapse $n$ nested loops into a single linear iteration space before partitioning iterations among threads.
- In our code, we collapse the `z` loop (range: $1 \dots nz-2$) and the `y` loop (range: $1 \dots ny-2$).
- Total collapsed iteration count: $(nz - 2) \times (ny - 2)$.
- **Why collapse 2 loops instead of 1?**
  - If we only parallelized the outermost `z` loop on a $64 \times 64 \times 64$ grid, there are only 62 iterations. Dividing 62 iterations among 12 threads gives ~5 iterations per thread, which can create load imbalance.
  - By collapsing `z` and `y`, the total iteration space becomes $62 \times 62 = 3844$ iterations. Dividing 3844 iterations among 12 threads gives ~320 iterations per thread, ensuring much finer granularity and excellent load balance.
- **Why NOT collapse all 3 loops (`collapse(3)`)?**
  - Collapsing the innermost `x` loop breaks contiguous vectorization and prefetching. Keeping the `x` loop uncollapsed ensures that each thread processes contiguous sequences of memory addresses along the stride-1 dimension. This enables compiler auto-vectorization and hardware streaming prefetchers.

### C. `schedule(static)`
OpenMP provides multiple scheduling policies:
1. `static`: Divides the total iteration space into chunks and assigns them to threads at loop startup in a deterministic, round-robin manner. If no chunk size is specified, iterations are divided evenly into $P$ contiguous chunks (where $P$ is the number of threads).
2. `dynamic`: Threads grab chunks of iterations dynamically from a shared work queue at runtime. This adds synchronization overhead.
3. `guided`: Chunk size decreases exponentially as iterations progress.

**Why `schedule(static)` is mandatory for our project:**
- The computational work per grid cell is strictly identical (8 FLOPs, 7 memory reads, 1 memory write). There is zero data-dependent branching or conditional logic. Therefore, static scheduling provides near-zero scheduling overhead.
- **NUMA First Touch alignment:** To achieve NUMA locality, the thread that performs the computation on a slice of memory must be the exact same thread that first wrote to that slice during initialization. Static scheduling guarantees deterministic thread-to-iteration mapping across both the initialization phase and the computation phase.

### D. `default(none)` and Scope Clauses
- `default(none)` forces the programmer to explicitly state the data-sharing attribute (`shared` or `private`) of every variable referenced in the parallel construct.
- If any variable is forgotten, the compiler throws a compile-time error. This prevents accidental race conditions or uninitialized private variables.
- `shared(u_in, u_out, nx, ny, nz, c0, c1)`:
  - Input array pointer `u_in` is read-only and safely shared across all threads.
  - Output array pointer `u_out` is written by threads, but **each thread writes to disjoint memory indices** because iterations are partitioned disjointly. Therefore, no race conditions occur on `u_out`.
  - Grid dimensions and stencil coefficients are read-only constants.
- Loop counters (`z`, `y`, `x`) and the local variable `center` are declared inside the loop scope in C99 style, making them implicitly `private` to each thread.

---

## 4. Race Condition Analysis: Why is this Code Race-Free?

A **race condition** occurs when two or more threads access the same memory location concurrently, and at least one of the accesses is a write.

In our stencil update:
1. **Reads from `u_in`:** Multiple threads may read from neighboring cells in `u_in` at boundary boundaries between thread chunks (e.g., Thread 0 reads `u_in[..., z]` while Thread 1 reads `u_in[..., z]` as its neighbor `z-1`). Because all accesses to `u_in` are strictly **read operations**, concurrent reads are completely safe and do not cause race conditions.
2. **Writes to `u_out`:** Every thread computes updates for a strictly unique, disjoint set of `(x, y, z)` coordinates. No two threads ever write to the same `u_out[center]` index.
3. **Double Buffering:** In multi-step simulations, `u_in` and `u_out` are swapped only after the parallel loop completes (at the implicit OpenMP barrier). Threads never read from and write to the same buffer in the same time step.

---

## 5. Thread Count and Scalability on Intel Core i5-12450HX

Our benchmarking evaluated strong scaling on a $160 \times 160 \times 160$ grid across thread counts $P \in \{1, 2, 4, 8, 12\}$:

| Threads ($P$) | Naive OMP Time | Speedup vs 1T | Parallel Efficiency |
|:---:|:---:|:---:|:---:|
| 1 | 4.50 ms | 1.00x | 100.0% |
| 2 | 3.30 ms | 1.36x | 68.2% |
| 4 | 2.90 ms | 1.55x | 38.8% |
| 8 | **2.80 ms** | **1.61x** | 20.1% |
| 12 | 3.30 ms | 1.36x | 11.4% |

### Why does 8 threads outperform 12 threads?
The Intel Core i5-12450HX features an asymmetric hybrid architecture:
- **4 Performance Cores (P-cores):** High clock frequency (up to 4.4 GHz), 2 threads per core via Hyper-Threading (8 logical threads total), larger L1/L2 caches.
- **4 Efficient Cores (E-cores):** Lower clock frequency (up to 3.1 GHz), single-threaded (4 logical threads total), smaller caches.
- Total logical threads = $8 + 4 = 12$.

When running with 12 threads under `schedule(static)`, OpenMP assigns an equal fraction ($1/12$) of the collapsed iteration space to every thread.
However, the 4 E-core threads execute significantly slower than the 8 P-core threads. Because of the implicit barrier at the end of `#pragma omp parallel for`, the fast P-core threads finish early and sit idle, waiting for the slower E-core threads to finish.

Furthermore, all cores share the same dual-channel DDR4/DDR5 memory bus (~32.13 GB/s peak measured). Once 4 to 8 threads fully saturate the memory bus bandwidth, adding more threads cannot increase throughput and only introduces thread scheduling overhead and cache contention.

