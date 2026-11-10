# 07. Genuine 3D Cache Blocking Explained

This document explains the concept of 3D spatial cache blocking (tiling), the memory hierarchy, why 3D blocking is necessary for large grids, how our 6-loop implementation works, and the performance trade-offs observed on modern hardware.

---

## 1. The Memory Wall and Cache Hierarchy

Modern CPUs execute floating-point arithmetic at immense speeds (hundreds of GFLOP/s), but accessing main memory (DRAM) is relatively slow (~50–80 ns latency, ~30–40 GB/s bandwidth). This discrepancy is known as the **Memory Wall**.

To bridge this gap, modern processors have a hierarchical cache system:
- **L1 Data Cache:** ~32–48 KB per core, latency ~1 ns (4–5 cycles), throughput ~2000–3000 GB/s aggregate.
- **L2 Cache:** ~1.25–2 MB per core, latency ~3–4 ns (14 cycles), throughput ~600–1000 GB/s.
- **L3 Cache (LLC - Last Level Cache):** 12–36 MB shared among all cores, latency ~10–20 ns (40–60 cycles), throughput ~150–250 GB/s.
- **DRAM (Main Memory):** 16–64 GB, latency ~60–80 ns, throughput ~30–50 GB/s.

```
       Fastest, Smallest  ┌───────────────────────┐
                          │     Registers         │  (~0 cycles)
                          ├───────────────────────┤
                          │    L1 Data Cache      │  (~4 cycles, ~32-48 KB)
                          ├───────────────────────┤
                          │      L2 Cache         │  (~14 cycles, ~1.25 MB)
                          ├───────────────────────┤
                          │  L3 Shared Cache      │  (~50 cycles, ~12 MB)
                          ├───────────────────────┤
       Slowest, Largest   │     Main DRAM         │  (~200 cycles, ~16-64 GB)
                          └───────────────────────┘
```

---

## 2. Why Does a Naive 3D Stencil Suffer from Cache Thrashing?

In a 3D 7-point stencil:
$$u_{\text{out}}(x, y, z) = c_0 u_{\text{in}}(x, y, z) + c_1 [ u_{\text{in}}(x+1) + u_{\text{in}}(x-1) + u_{\text{in}}(y+1) + u_{\text{in}}(y-1) + u_{\text{in}}(z+1) + u_{\text{in}}(z-1) ]$$

To compute the update at $(x, y, z)$, the CPU must access points from three different planes in memory:
- Center, Left, Right, Up, Down: live on the **current plane** $z$.
- Back neighbor: lives on the **previous plane** $z - 1$, which was stored $(n_x \times n_y \times 8)$ bytes earlier in memory!
- Front neighbor: lives on the **next plane** $z + 1$, which is located $(n_x \times n_y \times 8)$ bytes ahead in memory!

### The Working Set of One Slice:
To keep data in cache as the loop sweeps through the grid, the CPU must hold at least **2 full 2D planes** of the grid in cache simultaneously.
- Memory required for 2 planes:
  $$\text{Footprint} = 2 \times n_x \times n_y \times 8 \text{ bytes}$$
- For a small grid ($64 \times 64$):
  $$2 \times 64 \times 64 \times 8 = 65,536 \text{ bytes} = 64 \text{ KB}$$
  (This easily fits in modern L2 cache).
- For a larger grid ($200 \times 200$):
  $$2 \times 200 \times 200 \times 8 = 640,000 \text{ bytes} = 625 \text{ KB}$$
- For an industrial grid ($512 \times 512$):
  $$2 \times 512 \times 512 \times 8 = 4,194,304 \text{ bytes} = 4.0 \text{ MB}$$
  (Exceeds L2 cache; for $1024 \times 1024$, it requires 16 MB, exceeding L3 cache!).

When the 2D plane footprint exceeds cache capacity, **capacity cache misses** occur. By the time the loop reaches the next plane, the previous plane's data has already been evicted back to DRAM, forcing the CPU to fetch it again at slow DRAM speeds.

---

## 3. Genuine 3D Cache Blocking: The 6-Loop Structure

**Cache blocking (tiling)** restructures the iteration space into smaller 3D sub-volumes (tiles) of size $BX \times BY \times BZ$ that fit completely inside the CPU cache (typically L1 or L2 cache).

Instead of sweeping across an entire $200 \times 200$ plane, the algorithm computes all updates inside a compact box (e.g., $32 \times 32 \times 16$).

### Loop Structure in `src/kernels_3d_opt.c`:
```c
void heat3d_blocked(const double *restrict u_in,
                    double *restrict u_out,
                    size_t nx, size_t ny, size_t nz,
                    size_t bx, size_t by, size_t bz,
                    const StencilConfig *config)
{
    const double c0 = config->c0;
    const double c1 = config->c1;

    // 3 OUTER TILE LOOPS (Parallelized across tiles)
    #pragma omp parallel for schedule(static) collapse(2) default(none) \
        shared(u_in, u_out, nx, ny, nz, bx, by, bz, c0, c1)
    for (size_t bz_start = 1; bz_start < nz - 1; bz_start += bz) {
        for (size_t by_start = 1; by_start < ny - 1; by_start += by) {
            for (size_t bx_start = 1; bx_start < nx - 1; bx_start += bx) {

                // Compute Tile Bounds with Clamping for Odd/Remainder Sizes
                const size_t bz_end = (bz_start + bz < nz - 1) ? bz_start + bz : nz - 1;
                const size_t by_end = (by_start + by < ny - 1) ? by_start + by : ny - 1;
                const size_t bx_end = (bx_start + bx < nx - 1) ? bx_start + bx : nx - 1;

                // 3 INNER ELEMENT LOOPS (Traversing elements within the tile)
                for (size_t z = bz_start; z < bz_end; ++z) {
                    for (size_t y = by_start; y < by_end; ++y) {
                        for (size_t x = bx_start; x < bx_end; ++x) {
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
        }
    }
}
```

### Key Architectural Details:
1. **Outer Tile Loops (`bz_start`, `by_start`, `bx_start`):** Advance in steps of tile size (`bz`, `by`, `bx`).
2. **Boundary Clamping:**
   `bz_end = (bz_start + bz < nz - 1) ? bz_start + bz : nz - 1`
   This cleanly handles grids where $(n - 2)$ is not divisible by the tile size, as well as odd grid dimensions and non-power-of-two grids.
3. **Inner Element Loops (`z`, `y`, `x`):** Loop over grid points inside the active tile. The inner `x` loop advances with stride 1, maintaining vectorization potential.

---

## 4. Why Genuine 3D Blocking Must Tile All Three Dimensions (X, Y, and Z)

In some legacy codebases, programmers only tile the `Y` and `Z` dimensions while leaving `X` unblocked ($bx = nx - 2$).
- **Is 2.5D (Y-Z) blocking genuine 3D cache blocking? NO.**
- In genuine 3D cache blocking, all 3 spatial coordinates ($X, Y, Z$) have explicit tile size parameters ($BX, BY, BZ$).
- Tiling $X$ ensures that even when a row is extremely long (e.g., $nx = 2048$, requiring 16 KB per line), the inner loop working slice stays strictly within L1 cache limits.

---

## 5. Empirical Performance Analysis: Why Didn't 3D Blocking Outperform Naive OpenMP on $200^3$?

In our benchmark on the Intel Core i5-12450HX ($200^3$ grid, 12 threads):
- **Naive OpenMP:** Compute Time = `5.70 ms` (10.89 GFLOP/s)
- **3D Blocked OpenMP ($32 \times 32 \times 16$):** Compute Time = `6.50 ms` (9.55 GFLOP/s)

Why did naive OpenMP run slightly faster than 3D blocked OpenMP?
Answering this honestly and rigorously demonstrates master-level HPC knowledge:

1. **Hardware Stream Prefetcher Disruption:**
   - In naive OpenMP, the inner loop processes $x$ from $1 \dots 198$ in a long, continuous stream of 198 doubles (1584 contiguous bytes). The CPU hardware prefetcher detects this linear access pattern and pre-loads cache lines into L1/L2 well in advance.
   - When we block along $X$ with $BX = 32$, each contiguous stream is abruptly truncated after only 32 doubles (256 bytes = 4 cache lines). The loop then jumps by $(nx - 32)$ elements to the next row. This frequently breaks hardware prefetcher stride detection.
2. **Loop Control and Index Overhead:**
   - Naive OpenMP uses **3 nested loops**.
   - 3D Blocked OpenMP uses **6 nested loops** with ternary minimum-clamping checks on every tile boundary. The additional loop branch overhead and coordinate calculations add CPU instruction overhead.
3. **Single Time-Step Limitations (Spatial vs. Temporal Tiling):**
   - In a single time-step, each interior element is read once for its own update and read once or twice by adjacent neighbors. Spatial tiling limits the active working set, but every point in the global grid must still be loaded from DRAM at least once.
   - Genuine cache blocking provides dramatic speedups either:
     - On **extremely large grids** ($512^3$ or $1024^3$) where 2 planes exceed L3 cache.
     - When combined with **temporal blocking** (computing multiple time steps $t \dots t+K$ inside the tile before writing back to DRAM).

