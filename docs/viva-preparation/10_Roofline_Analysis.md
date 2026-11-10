# 10. The Roofline Model and Bottleneck Analysis

This document provides a thorough mathematical and architectural breakdown of the **Roofline Model**, the STREAM Triad memory hierarchy microbenchmark, the derivation of Arithmetic Intensity for a 3D 7-point stencil, and the proof that our application is strictly **Memory Bandwidth Bound**.

---

## 1. What is the Roofline Model?

The **Roofline Model** is an intuitive visual performance model developed by Samuel Williams, Andrew Waterman, and David Patterson (UC Berkeley, 2009).

It models the maximum achievable floating-point performance of an application running on a specific multicore architecture as a function of its **Arithmetic Intensity**.

### The Mathematical Roofline Formula:
$$\text{Attainable Performance (GFLOP/s)} = \min \left( \text{Peak Compute Performance}, \; \text{Memory Bandwidth} \times \text{Arithmetic Intensity} \right)$$

In logarithmic coordinates, this produces a "roofline" curve with two distinct regions:
1. **Memory-Bound Ceiling (Slanted Line):** For algorithms with low Arithmetic Intensity, performance is limited by the rate at which data can be transferred from memory: $\text{Performance} \le \text{Bandwidth} \times \text{AI}$.
2. **Compute-Bound Ceiling (Flat Horizontal Line):** For algorithms with high Arithmetic Intensity, memory bandwidth is no longer the bottleneck; performance is capped by the CPU's arithmetic execution units (ALUs / FPU execution pipes): $\text{Performance} \le \text{Peak Compute}$.
3. **The Ridge Point ($I^* = \text{Peak Compute} / \text{Memory Bandwidth}$):** The inflection point separating the memory-bound regime from the compute-bound regime.

```
 Performance
  (GFLOP/s)
      ▲
      │                    Flat Compute Roof (Peak GFLOP/s)
      │             ──────────────────────────────────────
      │            /▲
      │           / │
      │          /  │  Ridge Point: Peak GFLOP/s / Bandwidth
      │         /   │
      │        /    │
      │       /     │
      │      /      │
      │     /       │   Memory-Bound Region (Left of Ridge Point)
      │    /        │
      │   /         │
      └──┴──────────┴───────────────────────────────────────►
      0             I* (Ridge Point)         Arithmetic Intensity (FLOP/byte)
```

---

## 2. Arithmetic Intensity of a 3D 7-Point Stencil

**Arithmetic Intensity ($I$)** measures the ratio of floating-point operations performed to bytes of data moved between the processor and memory:
$$I = \frac{\text{Total Floating-Point Operations (FLOP)}}{\text{Total Memory Traffic (Bytes)}}$$

### The Numerator: 8 FLOPs per Point
For every updated interior point $(x, y, z)$:
$$u_{\text{out}}(x, y, z) = c_0 u_{\text{in}}(x, y, z) + c_1 [ u_{\text{in}}(x+1) + u_{\text{in}}(x-1) + u_{\text{in}}(y+1) + u_{\text{in}}(y-1) + u_{\text{in}}(z+1) + u_{\text{in}}(z-1) ]$$
- 1 multiply ($c_0 \cdot \text{center}$)
- 5 adds (summing the 6 neighbors)
- 1 multiply ($c_1 \cdot \text{sum}$)
- 1 add (adding the two terms together)
- **Total:** $1 + 5 + 1 + 1 = \mathbf{8 \text{ FLOP}}$

### The Denominator: DRAM Memory Traffic Bounds

How many bytes must be transferred across the memory bus to update one point?

#### 1. Upper Bound: Theoretical Zero Cache Reuse (64 Bytes/point)
If the CPU cache retained zero data between accesses:
- Read 7 double-precision points from DRAM: $7 \times 8 \text{ bytes} = 56 \text{ bytes}$.
- Write 1 double-precision result to DRAM: $1 \times 8 \text{ bytes} = 8 \text{ bytes}$.
- Total traffic: $56 + 8 = \mathbf{64 \text{ bytes/point}}$.
$$I_{\text{no-reuse}} = \frac{8 \text{ FLOP}}{64 \text{ bytes}} = \mathbf{0.125 \text{ FLOP/byte}}$$

#### 2. Lower Bound: Theoretical Ideal Cache Reuse (16 Bytes/point)
With an infinitely large, fully-associative cache, each grid point is loaded from DRAM exactly once and written back to DRAM exactly once:
- Read 1 double-precision point: $8 \text{ bytes}$.
- Write 1 double-precision point: $8 \text{ bytes}$ (or 16 bytes if write-allocate fetch is counted).
- Total ideal traffic: $\mathbf{16 \text{ bytes/point}}$.
$$I_{\text{ideal}} = \frac{8 \text{ FLOP}}{16 \text{ bytes}} = \mathbf{0.500 \text{ FLOP/byte}}$$

#### 3. Realistic Operational Intensity on Modern x86-64 (~0.34 FLOP/byte)
In reality, modern CPUs have a 64-byte cache line size ($8 \times 8$-byte doubles). When the CPU loads `u_in[x, y, z]`, it automatically loads the next 7 adjacent doubles along the $X$-axis into L1 cache for free!
Thus, spatial reuse along $X$ is automatic. After factoring in write-allocate traffic ($1 \text{ read} + 1 \text{ write-allocate} + 1 \text{ write} = 24 \text{ bytes}$), the actual DRAM traffic is $\approx 23.5 \text{ bytes/point}$, yielding:
$$I_{\text{actual}} \approx \frac{8}{23.5} \approx \mathbf{0.34 \text{ FLOP/byte}}$$

---

## 3. Microbenchmarking Hardware Ceilings (`microbench_roofline.c`)

To construct an empirical Roofline model for the host machine (Intel Core i5-12450HX), [bench/microbench_roofline.c](file:///c:/Users/Vaarshini Ravikumar/OneDrive/Desktop/Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures/bench/microbench_roofline.c) benchmarks the full memory hierarchy using STREAM Triad ($A[i] = B[i] + q \cdot C[i]$) across buffer sizes fitted to each cache level:

| Memory Hierarchy Level | Buffer Size Tested | Measured Bandwidth |
|:---|:---:|:---:|
| **L1 Data Cache** | 24 KB (per core) | **575.96 GB/s** |
| **L2 Cache** | 512 KB (per core) | **199.92 GB/s** |
| **L3 Shared Cache** | 6 MB (shared) | **61.38 GB/s** |
| **Main Memory (DRAM)** | 96 MB (exceeds L3) | **32.13 GB/s** |

### Peak Floating-Point Compute Roof:
- **Measured Empirical DP FMA (OpenMP):** **36.50 GFLOP/s**
- **Theoretical Turbo Peak:** $8 \text{ physical cores} \times 2.8 \text{ GHz} \times 16 \text{ DP FLOPs/cycle} = \mathbf{380.80 \text{ GFLOP/s}}$

---

## 4. The Ridge Point Calculation

Using the measured DRAM bandwidth ($32.13 \text{ GB/s}$) and empirical compute peak ($36.50 \text{ GFLOP/s}$):
$$I^* = \frac{\text{Peak Compute}}{\text{DRAM Bandwidth}} = \frac{36.50 \text{ GFLOP/s}}{32.13 \text{ GB/s}} = \mathbf{1.136 \text{ FLOP/byte}}$$

### The Verdict:
- Our stencil's Arithmetic Intensity is in the range of **$0.125 \dots 0.500 \text{ FLOP/byte}$**.
- Because $I_{\text{actual}} \approx 0.34 \ll 1.136$, our stencil is located **deeply in the memory-bound region** on the far left side of the Roofline chart!

```
 GFLOP/s
   ▲
 40│                     Compute Roof (36.50 GF/s)
   │                     ────────────────────────────────
 30│                    /
   │                   /
 20│                  /
 16│───DRAM Capacity / Roof (16.07 GF/s at 0.50 FLOP/B)
 12│       ★ OMP+FT (11.94 GF/s)
 10│       ★ OMP Naive (10.89 GF/s)
  8│       ★ OMP Blocked (9.86 GF/s)
  6│       ● Serial (6.54 GF/s)
  4│
  0└───┴─────────────┴───────────────────────────────────►
      0.125        0.500        1.136 (Ridge)      Arithmetic Intensity (FLOP/B)
```

---

## 5. Performance Relative to the Physical DRAM Roof

The maximum possible performance any algorithm with $I \le 0.50 \text{ FLOP/byte}$ can ever achieve on this machine is:
$$\text{Max DRAM Performance} = \text{DRAM Bandwidth} \times I_{\max} = 32.13 \text{ GB/s} \times 0.50 \text{ FLOP/byte} = \mathbf{16.07 \text{ GFLOP/s}}$$

Evaluating our benchmarked kernels against this physical ceiling:

| Implementation Variant | Achieved GFLOP/s | % of Physical DRAM Roof (16.07 GF/s) | Operational Status |
|:---|:---:|:---:|:---:|
| **Serial Reference** | 6.54 GF/s | 40.7% | Single-thread memory bound |
| **Naive OpenMP** | 10.89 GF/s | 67.8% | Memory bus bandwidth saturated |
| **OpenMP + First Touch** | **11.94 GF/s** | **74.3%** | **Saturating ~75% of available DRAM bandwidth** |
| **OpenMP + 3D Blocked** | 9.55 GF/s | 59.4% | Memory bus bound with loop overhead |
| **OpenMP + FT + 3D Blocked** | 9.86 GF/s | 61.4% | Memory bus bound with loop overhead |

### Conclusion:
Our OpenMP + First Touch implementation achieves **74.3% of the absolute physical memory bandwidth ceiling** of the machine. The application is completely memory-bound; no amount of arithmetic optimization, SIMD tuning, or faster CPU clocks can make it go faster without faster DRAM.

