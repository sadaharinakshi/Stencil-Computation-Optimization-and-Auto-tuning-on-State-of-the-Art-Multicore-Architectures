# 15. Live Demonstration Instructions for Viva and Exam

This document provides step-by-step instructions for demonstrating the P13 HPC project live to examiners or professors using Windows PowerShell.

---

## 1. Quick-Start Summary of Demo Commands

Open **PowerShell** or **Command Prompt** in the project directory:
```powershell
cd "c:\Users\Vaarshini Ravikumar\OneDrive\Desktop\Stencil-Computation-Optimization-and-Auto-tuning-on-State-of-the-Art-Multicore-Architectures"
```

The recommended one-click helper script is `make.bat`:

| Objective | Single Command | Execution Time | Output Description |
|:---|:---|:---:|:---|
| **Clean Build** | `.\make.bat clean; .\make.bat all` | ~5 seconds | Recompiles all 3 binaries cleanly with GCC |
| **Verify Correctness** | `.\make.bat correctness` | ~8 seconds | Runs all 55 correctness tests |
| **Run Benchmarks** | `.\make.bat bench` | ~15 seconds | 5-way benchmark, thread scaling & tile sweep |
| **Run Roofline Model** | `.\make.bat roofline` | ~12 seconds | Measures L1/L2/L3/DRAM & computes ridge point |
| **Generate Plots** | `.\make.bat plots` | ~3 seconds | Generates PNG/PDF plots in `plots/` |

---

## 2. Detailed Live Demonstration Script

Follow this sequence during your examination:

### Step 1: Clean Compilation (`.\make.bat clean; .\make.bat all`)
**What to say to the examiner:**
> *"First, we will do a clean build of all project binaries using the Makefile. You will see that GCC compiles all code with zero warnings and zero errors under strict `-Wall -Wextra` flags."*

**Command:**
```powershell
.\make.bat clean
.\make.bat all
```

**Expected Output:**
```
gcc -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude -c src/kernels_3d_serial.c -o src/kernels_3d_serial.o
gcc -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude -c src/kernels_3d_opt.c -o src/kernels_3d_opt.o
gcc -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude -c src/kernels_3d_dispatch.c -o src/kernels_3d_dispatch.o
gcc -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude -c bench/correctness_check.c -o bench/correctness_check.o
gcc ... -o correctness_check.exe
gcc ... -o bench_3d.exe
gcc ... -o microbench_roofline.exe
[OK] Build complete.
```

---

### Step 2: Live Correctness Verification (`.\make.bat correctness`)
**What to say to the examiner:**
> *"Now we execute our automated correctness verification suite. It runs 55 test cases comparing all parallel implementations against the serial reference across powers-of-two, prime odd dimensions like $67^3$, anisotropic grids ($113\times 97\times 83$), 4 tile sizes, and 5-step time iterations."*

**Command:**
```powershell
.\make.bat correctness
```

**What to point out on screen:**
- Point to the right-hand column: `PASSED (MaxErr: 0.00e+00)`.
- Point to the final summary line:
  `VERIFICATION SUMMARY: 55 / 55 tests PASSED (0 failures)`.
- Highlight that the error is **$0.00\times 10^0$**, well below the strict $1.0\times 10^{-12}$ tolerance.

---

### Step 3: Run the 5-Way Benchmark Comparison (`.\make.bat bench`)
**What to say to the examiner:**
> *"Now we run our main benchmark. It compares all 5 configurations on $128^3$ and $200^3$ grids, measures thread scaling up to 12 threads, and sweeps 3D tile sizes. Notice that we benchmark memory initialization time completely separately from stencil compute time."*

**Command:**
```powershell
.\make.bat bench
```

**What to point out on screen:**
1. **First Touch Speedup:**
   Show the $200^3$ table:
   - Serial Init Time: ~460 ms
   - First Touch Init Time: **~53–62 ms** (~8x faster!)
   - Explain: *"This is because First Touch parallelizes the OS page fault servicing and memory bus write bandwidth across all CPU cores."*
2. **Compute Time:**
   - Serial Compute: ~9.5 ms (6.54 GFLOP/s)
   - OpenMP + FT: **~5.2 ms** (11.94 GFLOP/s, 1.83x speedup)
3. **Thread Scaling Anomaly (8T vs. 12T):**
   - Point out that 8 threads runs in **2.8 ms**, while 12 threads runs in **3.3 ms**.
   - Explain: *"This demonstrates the effect of the Intel hybrid architecture: the 8 P-core threads finish early and must wait at the OpenMP barrier for the 4 slower E-core threads."*

---

### Step 4: Roofline Microbenchmark (`.\make.bat roofline`)
**What to say to the examiner:**
> *"Now we run our STREAM Triad microbenchmark across buffer sizes corresponding to L1, L2, L3 cache and DRAM to measure the actual memory bandwidth hierarchy of this specific laptop."*

**Command:**
```powershell
.\make.bat roofline
```

**What to point out on screen:**
- L1 Data Bandwidth: **~576 GB/s**
- L2 Bandwidth: **~200 GB/s**
- L3 Bandwidth: **~61 GB/s**
- Sustained DRAM Bandwidth: **~32.13 GB/s**
- Ridge Point: **1.136 FLOP/byte**
- Explain: *"Our stencil has an operational Arithmetic Intensity of ~0.34 FLOP/byte, which is far to the left of the 1.136 ridge point. This proves mathematically that our stencil is strictly memory bandwidth bound."*

---

### Step 5: Visualizing Results (`.\make.bat plots`)
**Command:**
```powershell
.\make.bat plots
```

Open the generated images in Windows Photos:
```powershell
Invoke-Item plots/roofline_3d.png
Invoke-Item plots/speedup_3d.png
```

**What to show on `roofline_3d.png`:**
- Show where the points lie: all 5 variants sit directly under the slanted DRAM bandwidth roof, far below the horizontal compute ceiling.
- Show that OpenMP + FT achieves ~74.3% of the absolute physical DRAM roof.

---

## 3. Fallback Commands if Batch Scripts Cannot Be Executed

If the examiner asks you to run commands directly without `.bat` files:

```powershell
# Set PATH to bundled compiler
$env:PATH = "$PWD\tools\w64devkit\bin;$env:PATH"

# Compile everything
make clean
make all

# Run executables directly
.\correctness_check.exe
.\bench_3d.exe
.\microbench_roofline.exe
python scripts/plot_roofline.py
```

