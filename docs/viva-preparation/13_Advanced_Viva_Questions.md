# 13. Advanced Viva Questions and Deep-Dive Defense (30 Tough Questions)

This document prepares you for tough, skeptical questions from HPC professors and external examiners. Each question includes the underlying concept, the pitfall to avoid, and the exact model answer.

---

### Category 1: Hardware Architecture and NUMA Nuances (Q1 – Q8)

#### Q1: "Your project title mentions 'State-of-the-Art Multicore Architectures', but your laptop has only 1 NUMA node. Doesn't that invalidate your NUMA analysis?"
**Concept:** Architectural honesty vs. simulated/theoretical claims.  
**Trap:** Pretending your laptop has NUMA, or claiming multi-socket interconnect benefits that did not physically occur.  
**Model Answer:**  
> *"Not at all. Part of rigorous scientific computing is verifying the actual hardware rather than assuming features exist. We used the Windows system API (`GetNumaHighestNodeNumber`) to inspect the host Intel Core i5-12450HX and verified that it is a single-socket UMA system with 1 NUMA node. However, we implemented the First Touch pattern correctly using `#pragma omp parallel for schedule(static) collapse(2)` so that the code is 100% portable to multi-socket NUMA servers. Furthermore, on our UMA testbed, First Touch demonstrated a remarkable 7.4x to 8.7x speedup in array initialization because it parallelizes OS page fault handling, page zeroing, and memory write bandwidth across all CPU cores."*

#### Q2: "Why did 8 threads run faster than 12 threads on Naive OpenMP (2.80 ms vs 3.30 ms) on an Intel Core i5-12450HX?"
**Concept:** Asymmetric hybrid core architectures (Alder Lake P-cores vs. E-cores).  
**Model Answer:**  
> *"The Intel Core i5-12450HX is a heterogeneous hybrid CPU: it has 4 Performance cores (P-cores) supporting Hyper-Threading (8 logical threads) and 4 Efficient cores (E-cores) supporting single threads, giving 12 threads total. Under `schedule(static)`, OpenMP partitions the iteration space into 12 equal slices. The 4 E-core threads run at lower frequencies and IPC than the P-cores. Because OpenMP places an implicit barrier at the end of the parallel loop, the 8 fast P-core threads finish early and sit idle, waiting for the slower E-core threads. At 8 threads, execution runs predominantly on the fast P-cores, eliminating this load imbalance."*

#### Q3: "What is false sharing, and does it affect your 3D stencil implementation?"
**Concept:** Cache coherency protocols (MESI/MOESI) and cache line boundaries (64 bytes).  
**Model Answer:**  
> *"False sharing occurs when threads running on different cores modify different variables that reside on the same 64-byte cache line. The cache coherence protocol repeatedly invalidates the cache line across cores, causing severe bus traffic ('cacheline ping-pong'). In our implementation, false sharing is practically nonexistent: threads are partitioned along the outer $Z$ and $Y$ dimensions (`collapse(2)`). Each thread updates entire continuous rows of length $nx$ (where $nx \ge 64$, or 512 bytes). Threads only meet at row boundaries, which occur once every hundreds or thousands of elements, making false sharing negligible."*

#### Q4: "What is write allocation, and how does it affect memory traffic?"
**Concept:** CPU write policies (Write-Allocate with Write-Back).  
**Model Answer:**  
> *"When a CPU writes to memory address `u_out[center]`, modern x86 processors use a 'write-allocate write-back' policy: if the destination address is not currently in cache, the CPU must first read the entire 64-byte cache line from DRAM into cache, modify the word, and later write it back to DRAM. This means writing 8 bytes of data actually causes 8 bytes of read traffic plus 8 bytes of write traffic. That is why the theoretical minimum DRAM traffic for 1 write is 16 bytes rather than 8 bytes, unless non-temporal streaming stores are used."*

#### Q5: "Why did you remove non-temporal streaming stores (`_mm256_stream_pd`) from the project?"
**Concept:** Requirement compliance and code simplicity.  
**Model Answer:**  
> *"Non-temporal stores bypass the cache hierarchy to eliminate write-allocate traffic. However, our professor explicitly asked us to remove AVX2 SIMD intrinsics, prefetching, and non-temporal stores, and focus on clean standard C and OpenMP. Non-temporal stores require SIMD-aligned vector intrinsics, which obscures the code, makes it non-portable across architectures, and deviates from the core requirement of studying genuine 3D cache blocking and First Touch."*

#### Q6: "Why did you choose 64-byte alignment for memory allocation?"
**Concept:** Cache line boundary alignment.  
**Model Answer:**  
> *"Modern x86-64 processors have a 64-byte cache line size. By aligning array base addresses to 64 bytes using `aligned_alloc(64, ...)`:
> 1. We guarantee that the start of the array aligns perfectly with a physical cache line boundary.
> 2. We eliminate unaligned memory access penalties across cache line splits.
> 3. It allows the compiler's auto-vectorizer to generate aligned load/store instructions (`vmovapd`)."*

#### Q7: "How does the OS page size interact with your 3D grid?"
**Concept:** Virtual memory page sizes (4 KB standard pages vs. 2 MB Huge/Large Pages).  
**Model Answer:**  
> *"Standard virtual memory pages are 4 KB (512 double-precision floats). For our $200^3$ grid (8 million doubles, 61 MB), the array spans approximately 15,625 pages. During serial initialization, the master thread incurs 15,625 individual page faults sequentially. In our First Touch parallel implementation, those 15,625 page faults are divided across 12 threads, which explains the dramatic 8x reduction in initialization latency."*

#### Q8: "If you had access to a dual-socket server with 2 NUMA nodes, how would you configure OpenMP environment variables to maximize First Touch?"
**Concept:** Thread affinity and binding (`OMP_PROC_BIND` and `OMP_PLACES`).  
**Model Answer:**  
> *"We would set `OMP_PROC_BIND=close` or `spread` and `OMP_PLACES=cores`. This binds each OpenMP thread to a specific physical core and prevents the OS scheduler from migrating threads between sockets. Without thread affinity, if the OS migrates a thread to Socket 1 after it initialized memory on Socket 0, the NUMA locality benefit would be destroyed."*

---

### Category 2: 3D Cache Blocking Mechanics (Q9 – Q15)

#### Q9: "Why did you use 6 nested loops instead of 3 nested loops for cache blocking?"
**Concept:** Spatial loop tiling decomposition.  
**Model Answer:**  
> *"Loop tiling mathematically splits each iteration dimension into two: an outer loop that iterates over tile coordinates by step size $B$, and an inner loop that iterates over individual elements within the active tile $[B_{\text{start}}, B_{\text{end}})$. For a 3D problem with dimensions $Z, Y, X$, tiling all three dimensions naturally produces $3 \times 2 = 6$ nested loops. The 3 outer loops define the tile grid, and the 3 inner loops execute the local stencil updates."*

#### Q10: "Why did 3D spatial cache blocking achieve 9.86 GFLOP/s while naive OpenMP achieved 11.94 GFLOP/s on $200^3$? Did blocking fail?"
**Concept:** Single-step spatial vs. temporal blocking trade-offs.  
**Model Answer:**  
> *"Blocking did not fail; it behaved exactly as computer architecture theory predicts for a single-step spatial stencil.
> In a single time-step update, every point in the global array must be read from DRAM at least once to calculate the new state. Spatial blocking keeps the active working set inside L2 cache, eliminating capacity misses.
> However, on a $200^3$ grid, naive OpenMP has very long continuous rows (198 elements), which modern hardware prefetchers stream with near-100% efficiency. Blocking along $X$ with $BX=32$ truncates those streams, increasing branch overhead and prefetcher misses.
> 3D cache blocking becomes dramatically faster than naive OpenMP on massive grids ($512^3$ or $1024^3$) where 2 planes exceed L3 cache capacity, or when combined with temporal blocking across multiple time steps."*

#### Q11: "What is the difference between Z-plane unrolling and 3D cache blocking?"
**Concept:** Register reuse vs. cache tiling.  
**Model Answer:**  
> *"Z-plane unrolling is an in-register optimization where multiple consecutive planes along $Z$ are unrolled inside the loop body to hold intermediate values in CPU registers ($z-1$, $z$, $z+1$).
> 3D cache blocking is an algorithm-level memory hierarchy optimization where the global 3D space is chopped into 3D sub-volumes ($BX \times BY \times BZ$) to keep data inside L1/L2/L3 cache.
> Confusing unrolling with blocking is a common mistake; our project implements genuine 3D cache blocking with three outer tile loops and three inner loops."*

#### Q12: "How did you ensure that cells on tile boundaries get the correct neighbor values?"
**Concept:** Spatial domain decomposition vs. shared memory array indexing.  
**Model Answer:**  
> *"Because all threads share the global input array `u_in` in a single unified address space, accessing a neighbor across a tile boundary is completely seamless: the code simply evaluates `INDEX(x+1, y, z, nx, ny)` or `INDEX(x, y+1, z, nx, ny)`. The neighbor value is read directly from memory. Because `u_in` is read-only during the time step, no halo exchange or MPI-style communication buffers are needed."*

#### Q13: "What happens if the grid dimension $(nx-2)$ is not divisible by the tile size $BX$?"
**Concept:** Remainder tile handling and memory safety.  
**Model Answer:**  
> *"We protect against this using boundary clamping:
> `bx_end = (bx_start + bx < nx - 1) ? bx_start + bx : nx - 1;`
> If the remaining slice is smaller than $BX$, the inner loop safely truncates at $nx-1$. This prevents out-of-bounds memory accesses and was proven correct in our test suite on prime/odd dimensions like $67^3$."*

#### Q14: "Why did you choose $32 \times 32 \times 16$ as the default tile size?"
**Concept:** L2 cache sizing.  
**Model Answer:**  
> *"A tile of $32 \times 32 \times 16$ doubles contains 16,384 elements. At 8 bytes per double, the active tile data footprint is $\approx 131 \text{ KB}$. Even accounting for reading the surrounding halo surface and writing to `u_out`, the entire working set is under 350 KB, which fits easily inside the 1.25 MB L2 cache of the Intel P-cores while keeping $BX=32$ large enough for 256-bit SIMD vectorization."*

#### Q15: "Why not tile the time dimension as well (Temporal Blocking)?"
**Concept:** Spatial vs. temporal blocking complexity.  
**Model Answer:**  
> *"Temporal blocking requires skewed diamond or trapezoidal time-space tiling to respect causal dependencies across time steps. Our professor's requirements explicitly mandated spatial 3D cache blocking ($BX, BY, BZ$). Spatial blocking directly addresses the memory access footprint per time step with simple, robust code that is easy to explain during a viva."*

---

### Category 3: Mathematics, Finite Differences, and Correctness (Q16 – Q22)

#### Q16: "Derive the 7-point stencil coefficients from the Heat Equation."
**Concept:** Finite difference discretization.  
**Model Answer:**  
> *"Starting from $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$, we apply Forward Euler in time: $\frac{u^{t+1} - u^t}{\Delta t}$, and second-order central differences in space:
> $\frac{\partial^2 u}{\partial x^2} \approx \frac{u(x+1) - 2u(x) + u(x-1)}{\Delta x^2}$.
> Assuming uniform spacing $\Delta x = \Delta y = \Delta z = h$:
> $u^{t+1} = u^t + \frac{\alpha \Delta t}{h^2} \left[ \sum u_{\text{neighbors}} - 6 u^t \right]$.
> Setting $r = \frac{\alpha \Delta t}{h^2}$, this rearranges to:
> $u^{t+1} = (1 - 6r) u^t + r \sum u_{\text{neighbors}}$.
> Matching our code: $c_0 = (1 - 6r)$ and $c_1 = r$."*

#### Q17: "What is the numerical stability criterion for this 3D explicit heat stencil?"
**Concept:** The CFL (Courant-Friedrichs-Lewy) condition for parabolic PDEs.  
**Model Answer:**  
> *"For an explicit 3D diffusion equation with uniform grid spacing $h$, numerical stability requires:
> $r = \frac{\alpha \Delta t}{h^2} \le \frac{1}{2d} = \frac{1}{6} \approx 0.1667$.
> If $r > 1/6$, the coefficient $c_0 = (1 - 6r)$ becomes negative, and high-frequency numerical errors grow exponentially, causing the simulation to explode to infinity."*

#### Q18: "Why did you measure a Maximum Absolute Error of $0.00 \times 10^0$ across all 55 tests?"
**Concept:** Floating-point determinism and compiler flags.  
**Model Answer:**  
> *"Because all four parallel variants evaluate the neighbor additions in the exact same mathematical sequence:
> $c_0 \cdot \text{center} + c_1 \cdot (\text{east} + \text{west} + \text{north} + \text{south} + \text{front} + \text{back})$
> and we explicitly compiled with `-ffp-contract=off` to prevent the compiler from arbitrarily merging some operations into fused multiply-adds (FMA) while leaving others unfused. This ensures bit-level reproducibility between serial and parallel executions."*

#### Q19: "Why is Maximum Absolute Error preferred over Root Mean Square Error (RMSE) for correctness checking?"
**Concept:** Error masking in aggregate norms.  
**Model Answer:**  
> *"RMSE averages squared errors across all 8 million cells. If a bug corrupts just 1 or 2 boundary cells with an error of $10.0$, the square root of the mean across 8,000,000 cells would be only $\approx 0.003$, potentially masking a critical localized bug. Maximum Absolute Error ($L_\infty$ norm) guarantees that the worst single point in the entire 3D volume is detected."*

#### Q20: "What is an anisotropic grid, and why did you test $113 \times 97 \times 83$?"
**Concept:** Dimension independence and stride bug detection.  
**Model Answer:**  
> *"An anisotropic grid has unequal dimensions ($nx \neq ny \neq nz$). We specifically chose $113 \times 97 \times 83$ (three distinct prime numbers). If a developer accidentally writes `INDEX(x, y, z, ny, nx)` or swaps $nx$ and $ny$ in loop bounds or strides, a symmetric cubic grid ($100^3$) might not reveal the bug because all dimensions are identical. An anisotropic prime grid immediately triggers an error if strides are inverted."*

#### Q21: "Why did you use a deterministic sine wave for initial values rather than `rand()`?"
**Concept:** Reproducibility and benchmarking stability.  
**Model Answer:**  
> *"Standard C `rand()` produces non-reproducible pseudo-random sequences if not seeded identically, and calling `rand()` inside multi-threaded loops requires thread synchronization or `rand_r()`. A deterministic function $u(x, y, z) = \sin(idx \times 0.001)$ is purely functional, thread-safe, 100% reproducible, and generates smooth thermal gradients."*

#### Q22: "Can you run this stencil in-place with a single array?"
**Concept:** Data dependencies (anti-dependencies / WAR hazards).  
**Model Answer:**  
> *"No, not in an explicit time-stepping scheme. If cell $(x, y, z)$ is updated in-place at step $t+1$, its original value at step $t$ is lost. When its right neighbor $(x+1, y, z)$ is updated next, it would read the new $t+1$ value instead of the old $t$ value. That violates the mathematical definition of explicit finite differences. Double buffering is strictly required."*

---

### Category 4: Roofline Model and Performance Engineering (Q23 – Q30)

#### Q23: "In your Roofline analysis, why did naive OpenMP achieve 10.89 GFLOP/s if the zero-reuse ceiling was 4.02 GFLOP/s?"
**Concept:** The 'Roofline Paradox' of assuming zero cache reuse.  
**Model Answer:**  
> *"Assuming zero cache reuse (64 bytes transferred per update) yields an Arithmetic Intensity of $8/64 = 0.125 \text{ FLOP/B}$. Multiplying by our measured DRAM bandwidth ($32.13 \text{ GB/s}$) gives an apparent ceiling of $4.02 \text{ GFLOP/s}$.
> Achieving 10.89 GFLOP/s is not a violation of physics; it proves that the zero-reuse assumption is false! Modern x86 CPUs automatically load 64-byte cache lines (8 doubles). When the CPU loads point $x$, the subsequent 7 points along $X$ are already in L1 cache. The actual operational intensity is $\approx 0.34 \text{ FLOP/B}$, which gives an attainable ceiling of $\approx 11 \text{ GFLOP/s}$, perfectly matching our empirical measurement."*

#### Q24: "How does the Roofline Model help guide future optimization decisions?"
**Concept:** Actionable diagnostic insights from performance modeling.  
**Model Answer:**  
> *"The Roofline Model tells us where we are relative to the hardware ridge point. Because our stencil operates at $0.34 \text{ FLOP/B}$ (far to the left of the ridge point at $1.136 \text{ FLOP/B}$), we know with mathematical certainty that optimizing arithmetic instructions (e.g., AVX-512 FMA tuning or higher CPU clock frequencies) cannot make the application run faster. To improve performance, we must either increase memory bandwidth (e.g., run on DDR5 or GPU HBM) or increase Arithmetic Intensity via Temporal Blocking."*

#### Q25: "How did you measure the sustained DRAM memory bandwidth of your machine?"
**Concept:** STREAM Triad microbenchmarking.  
**Model Answer:**  
> *"We implemented a STREAM Triad benchmark ($A[i] = B[i] + q \cdot C[i]$) in `microbench_roofline.c`. We allocated three 32 MB arrays (96 MB total), which is 8x larger than the 12 MB L3 cache, forcing every read and write to travel across the physical DRAM bus. Using OpenMP across all cores, we measured a sustained DRAM bandwidth of **32.13 GB/s**."*

#### Q26: "Why does STREAM Triad perform 3 memory operations per iteration?"
**Concept:** Memory traffic counting in STREAM.  
**Model Answer:**  
> *"For $A[i] = B[i] + q \cdot C[i]$, the processor must:
> 1. Read $B[i]$ (8 bytes).
> 2. Read $C[i]$ (8 bytes).
> 3. Write $A[i]$ (8 bytes, plus 8 bytes write-allocate on standard caches).
> According to standard STREAM convention, 24 bytes of user data are transferred per triad iteration."*

#### Q27: "What is Amdahl's Law, and how does it apply to your measured speedups?"
**Concept:** Theoretical limits of parallel speedup.  
**Model Answer:**  
> *"Amdahl's Law states:
> $S(P) = \frac{1}{(1 - f) + \frac{f}{P}}$
> where $f$ is the parallel fraction of execution time.
> In our project, initialization took 460 ms and computation took 9.5 ms. If we had left initialization serial, $f$ would be only $\approx 2\%$, and speedup would be capped at $1.02x$ regardless of thread count! By parallelizing initialization with First Touch, we reduced init time to 53 ms, enabling real end-to-end acceleration."*

#### Q28: "Why did you use `omp_get_wtime()` instead of C library `clock()`?"
**Concept:** Wall-clock vs. CPU core-cycle time in multi-threaded programs.  
**Model Answer:**  
> *"The standard C `clock()` function measures total CPU time summed across all active threads. If 12 threads run for 1 second of wall-clock time, `clock()` returns 12 seconds of CPU time! `omp_get_wtime()` measures elapsed real wall-clock time with nanosecond resolution, which is mandatory for measuring true parallel speedup."*

#### Q29: "What compiler optimizations were enabled in your Makefile?"
**Concept:** Compiler optimization flags and tradeoffs.  
**Model Answer:**  
> *"We compiled with `-O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra`.
> - `-O3`: Enables high-level loop optimizations, auto-vectorization, and aggressive register allocation.
> - `-march=native`: Directs the compiler to utilize instruction set extensions supported by the host CPU.
> - `-ffp-contract=off`: Disables floating-point contractions to ensure deterministic arithmetic matching the serial reference.
> - `-fopenmp`: Enables the OpenMP runtime and directive parsing."*

#### Q30: "What was the most important lesson you learned from completing this HPC project?"
**Concept:** Engineering maturity and synthesis.  
**Model Answer:**  
> *"The most vital lesson was understanding that **computational throughput in modern architectures is governed by memory movement, not arithmetic operations**. Before this project, we assumed that adding more threads or writing complex loop unrolling would always speed up code. By measuring the memory hierarchy and building the Roofline model, we proved that once the 32 GB/s memory bus is saturated, the CPU is starved for data. True HPC optimization requires structuring memory access—through First Touch page placement and cache blocking—to align with the physical reality of the hardware."*

