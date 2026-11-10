# 14. Three-Minute Project Presentation Script and Slide Outline

This document provides a word-for-word, timed presentation script and slide-by-slide guide designed for a 3-minute university presentation.

---

## Presentation Metadata
- **Project ID:** P13
- **Project Title:** Stencil Computation Optimization and Auto-tuning on State-of-the-Art Multicore Architectures
- **Team ID:** T09
- **Team Members:** Harinakshi S, Vaarshini R
- **Target Time:** 3 minutes (180 seconds)

---

## Slide-by-Slide Presentation Structure

```
┌────────────────────────────────────────────────────────────────────────┐
│ SLIDE 1 (0:00 - 0:30): Title, Problem Definition & Scientific Motif    │
├────────────────────────────────────────────────────────────────────────┤
│ SLIDE 2 (0:30 - 1:00): Core Methodology: OpenMP, First Touch & Blocking│
├────────────────────────────────────────────────────────────────────────┤
│ SLIDE 3 (1:00 - 1:40): Experimental Benchmarks & 55/55 Verification   │
├────────────────────────────────────────────────────────────────────────┤
│ SLIDE 4 (1:40 - 2:30): The Roofline Model: Hardware Bottleneck Proof   │
├────────────────────────────────────────────────────────────────────────┤
│ SLIDE 5 (2:30 - 3:00): Key Architectural Insights & Conclusion         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Word-for-Word Spoken Script

### Slide 1: Introduction, Problem & Scientific Motif (0:00 – 0:30)
> *"Good morning, respected professors and examiners.  
> We are Harinakshi and Vaarshini, Team T09, presenting Project P13: **Stencil Computation Optimization on State-of-the-Art Multicore Architectures**.  
> In scientific computing, stencil algorithms belong to the 'Structured Grids' motif—fundamental to computational fluid dynamics, weather forecasting, and heat transfer.  
> Our project implements a **3D 7-point heat diffusion stencil** in IEEE double precision.  
> In accordance with our professor's exact requirements, we implemented, evaluated, and verified four key components:
> 1. Serial and OpenMP implementations of the 3D 7-point stencil.
> 2. Genuine 3D spatial cache blocking.
> 3. NUMA-aware First Touch initialization.
> 4. Empirical Roofline analysis to diagnose performance bottlenecks."*

---

### Slide 2: Core Methodology and Implementation (0:30 – 1:00)
> *"Our computation represents a 3D thermal domain stored as a single contiguous, 64-byte aligned 1D array using row-major indexing: $((z \cdot ny + y) \cdot nx + x)$.  
> To parallelize computation, we used OpenMP `#pragma omp parallel for schedule(static) collapse(2)` across the $Z$ and $Y$ dimensions, preserving contiguous vector-streaming along $X$.  
> To optimize memory locality:
> - First, we implemented **parallel First Touch initialization**, distributing initial OS page faults and memory bus write bandwidth across all CPU cores.
> - Second, we implemented **genuine 3D cache blocking** using 6 nested loops with dimensions $BX, BY, BZ$, ensuring sub-volumes fit into L2 cache, with robust boundary clamping for non-divisible and odd grids."*

---

### Slide 3: Verification and Benchmark Results (1:00 – 1:40)
> *"In HPC, correctness is paramount. Before benchmarking, we validated all kernels with an automated 55-test verification suite comparing each parallel implementation against the serial reference.  
> Across powers-of-two, prime dimensions like $67^3$, anisotropic grids ($113\times 97\times 83$), and 5-step ping-pong iterations, **all 55 tests passed with zero error** ($0.00\times 10^0$).  
> On our testbed—an Intel Core i5-12450HX with 8 physical cores and 12 threads:
> - For a $200^3$ grid (61 MB per array), serial execution achieved 6.54 GFLOP/s.
> - OpenMP accelerated computation to **11.94 GFLOP/s**—a 1.83x speedup.
> - Most dramatically, parallel First Touch reduced memory initialization time from **460 ms down to 53 ms**—an **$8.7\times$ speedup** by parallelizing OS page fault zeroing."*

---

### Slide 4: Roofline Analysis & Bottleneck Identification (1:40 – 2:30)
> *"To understand why compute speedup saturated at 1.83x, we performed Roofline analysis.  
> Using our STREAM Triad microbenchmark, we measured the machine's memory hierarchy:
> - L1 bandwidth: 575 GB/s  
> - L2 bandwidth: 200 GB/s  
> - L3 bandwidth: 61 GB/s  
> - DRAM bandwidth: **32.13 GB/s**, with an empirical compute ceiling of 36.5 GFLOP/s.  
> This yields a machine Ridge Point of $1.136 \text{ FLOP/byte}$.  
> Each 3D stencil update executes 8 floating-point operations. Considering 64-byte cache lines and spatial reuse along $X$, our actual operational Arithmetic Intensity is $\approx 0.34 \text{ FLOP/byte}$.  
> Because $0.34$ is far to the left of the $1.136$ ridge point, our application is strictly **Memory Bandwidth Bound**.  
> In fact, our OpenMP + First Touch kernel achieves **74.3% of the absolute physical DRAM bandwidth ceiling** of the machine."*

---

### Slide 5: Key Takeaways & Conclusion (2:30 – 3:00)
> *"In conclusion, this project yielded three major architectural insights:
> 1. **Memory is the Bottleneck:** Stencil performance is throttled by memory bus throughput, not arithmetic CPU speed.
> 2. **First Touch is Essential:** Parallel initialization eliminated the dominant serial overhead of OS virtual memory page allocation.
> 3. **Hardware-Aware Design:** On hybrid P-core/E-core CPUs, thread load-balancing across asymmetric cores is crucial.  
> Our complete codebase is verified, documented, and fully reproducible.  
> Thank you, and we welcome your questions."*

---

## Slide Content Summary for Presentation Slides

### Slide 1: Title & Motivation
- **Title:** Stencil Computation Optimization on Multicore Architectures
- **Team T09:** Harinakshi S, Vaarshini R
- **Focus:** 3D 7-point Heat Diffusion Stencil
- **4 Key Deliverables:** Serial + OpenMP, 3D Cache Blocking, First Touch, Roofline Model

### Slide 2: Implementation Highlights
- **Data Layout:** 1D Flattened 64-byte aligned array, row-major macro indexing
- **OpenMP:** `#pragma omp parallel for schedule(static) collapse(2)`
- **First Touch:** OpenMP parallel memory initialization matching compute thread affinity
- **3D Blocking:** 6 nested loops ($BZ, BY, BX$ outer, $z, y, x$ inner) with remainder clamping

### Slide 3: Verification & Performance Table
- **Correctness:** 55 / 55 tests passed ($L_\infty = 0.00 \times 10^0$ error vs. serial)
- **Init Time:** 460 ms (Serial) $\rightarrow$ **53 ms (First Touch)** = **8.7x Speedup**
- **Compute Time:** 9.50 ms (Serial) $\rightarrow$ **5.20 ms (OpenMP+FT)** = **1.83x Speedup**
- **Throughput:** 6.54 GFLOP/s $\rightarrow$ **11.94 GFLOP/s**

### Slide 4: Roofline Model & Bottleneck Proof
- **STREAM Triad Ceilings:** L1 = 576 GB/s, L2 = 200 GB/s, L3 = 61 GB/s, DRAM = 32.13 GB/s
- **Ridge Point:** $I^* = 1.136 \text{ FLOP/byte}$
- **Operational Intensity:** $I_{\text{actual}} \approx 0.34 \text{ FLOP/byte} \ll 1.136$
- **Verdict:** Application is **100% Memory Bandwidth Bound** (operating at 74.3% of DRAM roof)

### Slide 5: Conclusions
- First Touch slashes one-time page-fault costs by 8.7x.
- OpenMP saturates available dual-channel DRAM bandwidth.
- Hybrid P-core/E-core load balancing explains 8T vs. 12T scaling.

