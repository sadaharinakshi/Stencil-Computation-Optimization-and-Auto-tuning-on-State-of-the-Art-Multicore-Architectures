# P13 HPC Project - Comprehensive Viva Preparation & Knowledge Base

**Project ID:** P13  
**Project Title:** Stencil Computation Optimization and Auto-tuning on State-of-the-Art Multicore Architectures  
**Team ID:** T09  
**Team Members:** Harinakshi S, Vaarshini R  
**Motif:** Structured Grids  
**Primary Technology:** C and OpenMP (IEEE 754 Double Precision)  
**Host Architecture:** Windows 11, Intel Core i5-12450HX (8 Cores, 12 Threads, UMA)

---

## Welcome to the P13 Viva Preparation Package

This repository contains a dedicated, 16-module university viva preparation curriculum. It is structured to take you from foundational mathematics and computer architecture up to advanced performance modeling and live defense against tough examiner questions.

---

## Recommended Study Pathways

### Pathway A: Rapid 30-Minute Cramming (Night Before Viva)
1. [16_Quick_Revision.md](16_Quick_Revision.md) — 1-page high-yield cheat sheet of all formulas, metrics, and answers.
2. [14_Three_Minute_Project_Presentation.md](14_Three_Minute_Project_Presentation.md) — Memorize the 3-minute spoken presentation script.
3. [15_Live_Demo_Instructions.md](15_Live_Demo_Instructions.md) — Memorize the exact PowerShell commands to run live during the viva.
4. [12_Viva_Questions_and_Answers.md](12_Viva_Questions_and_Answers.md) — Review the top 50 foundational viva questions.

### Pathway B: Deep Architectural Study (Full Preparation)
1. **The Fundamentals:**
   - [01_Project_Overview.md](01_Project_Overview.md) — Problem statement, requirements, scope boundaries.
   - [02_Heat_Stencil_Fundamentals.md](02_Heat_Stencil_Fundamentals.md) — 3D heat equation, finite differences, stability, 8 FLOPs derivation.
   - [03_Project_Architecture.md](03_Project_Architecture.md) — Repository structure, data layouts, build system.
   - [04_Source_Code_Explanation.md](04_Source_Code_Explanation.md) — Line-by-line breakdown of all C source and header files.
2. **Parallel & Memory Engineering:**
   - [05_OpenMP_Explained.md](05_OpenMP_Explained.md) — Pragmas, `collapse(2)`, `schedule(static)`, race-freedom.
   - [06_First_Touch_and_NUMA.md](06_First_Touch_and_NUMA.md) — Virtual memory paging, demand paging, NUMA vs UMA, page faulting.
   - [07_3D_Cache_Blocking.md](07_3D_Cache_Blocking.md) — Memory wall, cache hierarchy, 6-loop spatial blocking, remainder clamping.
3. **Verification & Benchmarks:**
   - [08_Correctness_and_Testing.md](08_Correctness_and_Testing.md) — $L_\infty$ norm, 55-test suite, odd/prime/anisotropic testing.
   - [09_Benchmarking_and_Performance.md](09_Benchmarking_and_Performance.md) — Timing methodology, separating init from compute, GFLOP/s, speedup.
   - [10_Roofline_Analysis.md](10_Roofline_Analysis.md) — Williams Roofline model, arithmetic intensity bounds, ridge point, memory-bound proof.
   - [11_Results_and_Interpretation.md](11_Results_and_Interpretation.md) — Real measured data on $128^3$ and $200^3$, hybrid core thread scaling.
4. **Examination & Defense:**
   - [12_Viva_Questions_and_Answers.md](12_Viva_Questions_and_Answers.md) — 50 comprehensive Q&As.
   - [13_Advanced_Viva_Questions.md](13_Advanced_Viva_Questions.md) — 30 tough examiner defense questions.
   - [14_Three_Minute_Project_Presentation.md](14_Three_Minute_Project_Presentation.md) — 3-minute presentation script and slide guide.
   - [15_Live_Demo_Instructions.md](15_Live_Demo_Instructions.md) — Step-by-step PowerShell demo commands.
   - [16_Quick_Revision.md](16_Quick_Revision.md) — Quick revision cheat sheet.

---

## Module Directory Index

| File | Title | Key Topics Covered |
|:---|:---|:---|
| [01_Project_Overview.md](01_Project_Overview.md) | Project Overview & Goals | Team info, 4 professor requirements, features removed |
| [02_Heat_Stencil_Fundamentals.md](02_Heat_Stencil_Fundamentals.md) | Heat Stencil Fundamentals | PDE math, central differences, CFL stability, 8 FLOPs derivation |
| [03_Project_Architecture.md](03_Project_Architecture.md) | Project Architecture | Directory map, 1D flattened array, row-major indexing |
| [04_Source_Code_Explanation.md](04_Source_Code_Explanation.md) | Source Code Walkthrough | Detailed explanation of `src/`, `include/`, and `bench/` files |
| [05_OpenMP_Explained.md](05_OpenMP_Explained.md) | OpenMP Implementation | `parallel for`, `collapse(2)`, `schedule(static)`, race-free proof |
| [06_First_Touch_and_NUMA.md](06_First_Touch_and_NUMA.md) | First Touch & NUMA | Demand paging, page faults, UMA vs NUMA, ~8x init speedup |
| [07_3D_Cache_Blocking.md](07_3D_Cache_Blocking.md) | Genuine 3D Cache Blocking | 6-loop tiling ($BX, BY, BZ$), L2 cache sizing, spatial trade-offs |
| [08_Correctness_and_Testing.md](08_Correctness_and_Testing.md) | Correctness Verification | 55 automated tests, $L_\infty$ error, prime/anisotropic tests |
| [09_Benchmarking_and_Performance.md](09_Benchmarking_and_Performance.md) | Benchmarking Methodology | Timing rules, separating init from compute, median filtering |
| [10_Roofline_Analysis.md](10_Roofline_Analysis.md) | Roofline Performance Model | STREAM Triad, Arithmetic Intensity bounds, Ridge point, proof |
| [11_Results_and_Interpretation.md](11_Results_and_Interpretation.md) | Results & Analysis | $128^3$ & $200^3$ benchmark data, 8T vs 12T hybrid core scaling |
| [12_Viva_Questions_and_Answers.md](12_Viva_Questions_and_Answers.md) | 50 Comprehensive Q&As | Foundational questions across all 6 project areas |
| [13_Advanced_Viva_Questions.md](13_Advanced_Viva_Questions.md) | 30 Advanced Defense Q&As | Deep architectural defense against skeptical examiners |
| [14_Three_Minute_Project_Presentation.md](14_Three_Minute_Project_Presentation.md) | 3-Minute Presentation | Word-for-word timed spoken script & slide outline |
| [15_Live_Demo_Instructions.md](15_Live_Demo_Instructions.md) | Live Demo Instructions | Exact PowerShell commands for compilation, testing & plotting |
| [16_Quick_Revision.md](16_Quick_Revision.md) | Quick Revision Sheet | One-page cheat sheet for the final 10 minutes |

