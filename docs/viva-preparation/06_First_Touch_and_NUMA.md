# 06. First Touch Initialization and NUMA Architecture

This document provides a comprehensive examination of Non-Uniform Memory Access (NUMA) architecture, physical memory page allocation policies in modern operating systems, the "First Touch" mechanism, and an honest architectural analysis of our empirical results on the test machine.

---

## 1. Computer Memory Architectures: UMA vs. NUMA

### Uniform Memory Access (UMA)
In a **UMA (Symmetric Multiprocessing / SMP)** system, all CPU cores connect to a single, centralized system memory controller via a shared bus or crossbar.
- Memory latency and bandwidth are identical regardless of which core accesses which physical memory address.
- Most consumer desktop and laptop processors (including the Intel Core i5-12450HX in our test machine) are fundamentally UMA architectures.

```
       ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
       │ Core 0  │   │ Core 1  │   │ Core 2  │   │ Core 3  │
       └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
            └───────────────┼─────────────┘
                            ▼
                ┌───────────────────────┐
                │ Shared Memory Interc. │
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────┐
                │ Central DRAM (Pool)   │
                └───────────────────────┘
```

### Non-Uniform Memory Access (NUMA)
In high-end workstations and multi-socket servers (e.g., dual Intel Xeon or AMD EPYC servers), physical memory is divided into multiple banks, each physically wired directly to an individual CPU socket:
- **Local Memory Access:** When Core 0 on Socket 0 accesses DRAM connected directly to Socket 0's memory channels, latency is low (~60–80 ns) and bandwidth is maximum.
- **Remote Memory Access:** When Core 0 on Socket 0 needs data stored in DRAM physically attached to Socket 1, the request must travel across an inter-socket interconnect (such as Intel Ultra Path Interconnect - UPI, or AMD Infinity Fabric). This introduces higher latency (~120–160 ns) and saturates the inter-socket link bandwidth.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│          NUMA Node 0            │       │          NUMA Node 1            │
│  ┌──────────┐     ┌──────────┐  │       │  ┌──────────┐     ┌──────────┐  │
│  │ Core 0   │     │ Core 1   │  │       │  │ Core 2   │     │ Core 3   │  │
│  └────┬─────┘     └────┬─────┘  │       │  └────┬─────┘     └────┬─────┘  │
│       └────────┬───────┘        │       │       └────────┬───────┘        │
│                ▼                │       │                ▼                │
│    Local Memory Controller      │◄─────►│    Local Memory Controller      │
│                │                │ Inter-│                │                │
│                ▼                │ socket│                ▼                │
│       Local DRAM Node 0         │ link  │       Local DRAM Node 1         │
└─────────────────────────────────┘ (UPI) └─────────────────────────────────┘
```

---

## 2. Operating System Memory Allocation: The "First Touch" Policy

When a C program calls `malloc()` or `aligned_alloc()` to allocate an array:
```c
double *u = (double *)aligned_alloc(64, num_elements * sizeof(double));
```
**Does the operating system allocate physical DRAM pages immediately? NO.**

Modern operating systems (Linux, Windows) use **Demand Paging** and **Virtual Memory**:
1. `malloc()` only reserves an address range in the process's **virtual address space**. No physical DRAM frames are assigned yet. The page table entries for these virtual pages are marked as "not present" (invalid).
2. **Page Fault:** When a thread first writes to a virtual memory address in that range, the CPU hardware MMU (Memory Management Unit) triggers a **page fault exception**.
3. **The First Touch Allocation Rule:** The OS kernel page fault handler intercepts this exception and selects a physical DRAM page frame to back that virtual page.
   - On a NUMA system, the default policy of both the Linux and Windows kernels is **First Touch**: the kernel allocates the physical memory page from the NUMA node that is local to the core executing the thread that caused the fault!

---

## 3. The Problem with Serial Initialization

Consider what happens if an array is initialized serially by the main thread before starting a parallel OpenMP loop:

```c
// SERIAL INITIALIZATION (Anti-Pattern on NUMA)
for (size_t i = 0; i < total_elements; ++i) {
    u[i] = initial_value(i); // Master thread touches ALL pages!
}

// PARALLEL COMPUTATION
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < total_elements; ++i) {
    compute(u, i); // Multiple threads now access the memory
}
```

### What happens on a NUMA system?
1. The **master thread** running on Socket 0 writes to every element during initialization.
2. Therefore, **100% of the physical memory pages are allocated on NUMA Node 0's local DRAM**.
3. During parallel computation, worker threads running on Socket 1, 2, or 3 must access their chunks of array `u`.
4. Because those pages physically live on Socket 0, all remote worker threads must continuously fetch data across the inter-socket interconnect (UPI/QPI).
5. **Consequence:** Massive remote memory access penalties, severe interconnect bandwidth saturation, and poor parallel scalability.

---

## 4. The Solution: Parallel First Touch Initialization

In [src/kernels_3d_opt.c](file:///c:/Users/Vaarshini Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/src/kernels_3d_opt.c), our project implements parallel First Touch initialization:

```c
void first_touch_init_3d(double *restrict u,
                         size_t nx, size_t ny, size_t nz,
                         double (*init_func)(size_t, size_t, size_t, size_t, size_t))
{
    #pragma omp parallel for schedule(static) collapse(2) default(none) \
        shared(u, nx, ny, nz, init_func)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = INDEX(x, y, z, nx, ny);
                u[idx] = init_func ? init_func(x, y, z, nx, ny) : 0.0;
            }
        }
    }
}
```

### Critical Rules for Correct First Touch:
1. **Identical Scheduling:** Both `first_touch_init_3d` and `heat3d_omp` must use `#pragma omp parallel for schedule(static) collapse(2)`.
2. **Deterministic Thread Affinity:** With static scheduling and fixed thread counts, Thread $k$ touches the exact same sub-volume of the grid during initialization that it will later compute over during the stencil updates.
3. On a multi-socket NUMA system, each thread's assigned sub-volume is physically placed in its local socket's DRAM, ensuring zero remote memory traffic!

---

## 5. Architectural Honesty: What Actually Happened on Our Test Machine?

In academic HPC projects, a common mistake is claiming NUMA speedups on hardware that does not even have NUMA.

### The Host Architecture
Our test machine is an **Intel Core i5-12450HX laptop**:
- We queried the Windows operating system API directly:
  ```powershell
  [System.Runtime.InteropServices.Marshal]::GetNumaHighestNodeNumber([ref]$node)
  ```
- Result: **Highest NUMA Node = 0** (meaning there is exactly **1 NUMA node**).
- **Physical Reality:** This CPU is a single-socket processor with an integrated dual-channel DDR memory controller. It is a **UMA system**, not a multi-socket NUMA system.

### Why Did First Touch Show a Massive Speedup Anyway?
In our empirical benchmarks on the $200^3$ grid (61.0 MB per array, 122 MB total):
- **Serial Initialization Time:** `460.0 ms`
- **First Touch Parallel Initialization Time:** `53.0 ms - 62.0 ms`
- **Initialization Speedup:** **$\approx 7.4\times$ to $8.7\times$ faster!**

Why did parallel initialization run ~8x faster if there are no remote NUMA nodes?
1. **Parallel Page Table Fault Servicing:**
   - On Windows, a page fault requires acquiring OS kernel virtual memory manager spinlocks, zeroing physical page frames, and updating page table entries.
   - When a single thread touches 122 MB serially, 1 core does all the page fault work sequentially.
   - In parallel First Touch, all 12 hardware threads concurrently trigger page faults across disjoint virtual memory ranges, distributing kernel page zeroing and fault handling across all CPU cores.
2. **Memory Bus Saturation during Writes:**
   - Writing 122 MB of data requires memory write bandwidth. A single core cannot saturate the full 32 GB/s physical DRAM bus. 12 parallel threads drive the memory bus closer to its saturation limit during array zeroing/filling.

### Computation Time Difference:
- Naive OpenMP Compute Time: `5.70 ms`
- First Touch + OpenMP Compute Time: `5.20 ms` (~9% faster)
Because both run on a single NUMA node, the compute time difference is modest. On a real dual-socket server (e.g., 2x Intel Xeon), un-initialized memory causes up to a $2\times$ compute time penalty due to remote socket bus congestion.

