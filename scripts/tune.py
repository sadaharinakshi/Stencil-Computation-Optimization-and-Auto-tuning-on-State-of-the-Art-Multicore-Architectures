#!/usr/bin/env python3
"""
HPC Auto-tuner for Stencil Computations
Explores:
- Tile sizes (TX, TY, TZ)
- Thread count (1..12)
- OpenMP schedule (static, dynamic, guided) and chunk size
- SIMD vectorization (on/off)
- Software prefetch distance (0, 8, 16, 32)
Saves results to results/tuning_*.csv and plots tile size heatmaps to plots/.
"""

import os
import sys
import subprocess
import re
import csv
import random
import time
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

BENCH_TUNE_EXE = os.path.join(".", "bench_tune.exe")

def run_trial(kernel, dim, tx, ty, tz, threads, sched, chunk, simd, prefetch):
    cmd = [
        BENCH_TUNE_EXE,
        "--kernel", str(kernel),
        "--dim", str(dim),
        "--tx", str(tx),
        "--ty", str(ty),
        "--tz", str(tz),
        "--threads", str(threads),
        "--sched", str(sched),
        "--chunk", str(chunk),
        "--simd", str(simd),
        "--prefetch", str(prefetch)
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Parse: RESULT: time_ms=2.000, gflops=35.65, gb_s=16.78
        m = re.search(r"RESULT:\s*time_ms=([\d\.]+),\s*gflops=([\d\.]+),\s*gb_s=([\d\.]+)", proc.stdout)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
    except Exception as e:
        print(f"Error running trial: {e}")
    return 999999.0, 0.0, 0.0

def tune_tile_sizes_2d(kernel="gauss5", dim=2048, threads=12):
    print(f"\n=======================================================")
    print(f" Phase 3: Exhaustive 2D Tile Size Sweep for {kernel.upper()} ({dim}x{dim})")
    print(f"=======================================================")

    tx_list = [16, 32, 64, 128, 256, 512]
    ty_list = [8, 16, 32, 64, 128, 256]

    heatmap_data = np.zeros((len(ty_list), len(tx_list)))
    csv_file = "results/tuning_2d_heatmap.csv"

    records = []
    best_time = 1e9
    best_cfg = None

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Kernel", "Dim", "TX", "TY", "Threads", "TimeMS", "GFLOPs", "GBs"])

        for i, ty in enumerate(ty_list):
            for j, tx in enumerate(tx_list):
                t_ms, gflops, gbs = run_trial(kernel, dim, tx, ty, 16, threads, 0, 0, 1, 16)
                heatmap_data[i, j] = gflops
                writer.writerow([kernel, dim, tx, ty, threads, f"{t_ms:.3f}", f"{gflops:.2f}", f"{gbs:.2f}"])
                f.flush()

                if t_ms < best_time:
                    best_time = t_ms
                    best_cfg = (tx, ty, gflops, t_ms)

                print(f"  [TX={tx:3d}, TY={ty:3d}] => Time: {t_ms:6.2f} ms | GFLOP/s: {gflops:6.2f} | GB/s: {gbs:5.2f}")

    print(f"\n>>> Best 2D Tile Configuration: TX={best_cfg[0]}, TY={best_cfg[1]} ({best_cfg[3]:.2f} ms, {best_cfg[2]:.2f} GFLOP/s)")

    # Plot 2D Heatmap
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.imshow(heatmap_data, cmap="viridis", aspect="auto", origin="lower")
    cbar = plt.colorbar(im)
    cbar.set_label("Performance (GFLOP/s)", fontsize=11)

    plt.xticks(range(len(tx_list)), [str(x) for x in tx_list], fontsize=10)
    plt.yticks(range(len(ty_list)), [str(y) for y in ty_list], fontsize=10)
    plt.xlabel("Tile Size X (TX elements)", fontsize=12)
    plt.ylabel("Tile Size Y (TY rows)", fontsize=12)
    plt.title(f"2D Stencil Optimization: Tile Size Exploration ({kernel.upper()}, {dim}x{dim})\nOptimal: TX={best_cfg[0]}, TY={best_cfg[1]} ({best_cfg[2]:.1f} GFLOP/s)", fontsize=13)

    # Annotate values inside cells
    for i in range(len(ty_list)):
        for j in range(len(tx_list)):
            val = heatmap_data[i, j]
            color = "white" if val < np.max(heatmap_data) * 0.65 else "black"
            plt.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/tuning_heatmap_2d.png", dpi=300)
    plt.savefig("plots/tuning_heatmap_2d.pdf")
    plt.close()
    print("  Saved heatmap to plots/tuning_heatmap_2d.png and .pdf")

    return best_cfg

def tune_hyperparameters_2d(kernel="gauss5", dim=2048, time_budget_sec=25):
    print(f"\n=======================================================")
    print(f" Phase 3: Multi-Parameter Search (Time Budget: {time_budget_sec}s)")
    print(f"=======================================================")

    tx_choices = [32, 64, 128, 256]
    ty_choices = [16, 32, 64, 128]
    thread_choices = [2, 4, 6, 8, 10, 12]
    sched_choices = [0, 1, 2] # static, dynamic, guided
    chunk_choices = [0, 8, 16, 32]
    simd_choices = [0, 1]
    prefetch_choices = [0, 8, 16, 32]

    # Measure untuned baseline: TX=64, TY=32, 1 thread, static, no SIMD, no prefetch
    t_base, gf_base, _ = run_trial(kernel, dim, 64, 32, 16, 1, 0, 0, 0, 0)
    print(f"Untuned Configuration (1 thread, no SIMD): {t_base:.2f} ms ({gf_base:.2f} GFLOP/s)")

    start_time = time.time()
    trials_csv = "results/tuning_2d_trials.csv"
    best_time = 1e9
    best_record = None

    with open(trials_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "TX", "TY", "Threads", "Sched", "Chunk", "SIMD", "Prefetch", "TimeMS", "GFLOPs", "GBs", "SpeedupVsUntuned"])

        trial_idx = 0
        while time.time() - start_time < time_budget_sec:
            tx = random.choice(tx_choices)
            ty = random.choice(ty_choices)
            th = random.choice(thread_choices)
            sc = random.choice(sched_choices)
            ch = random.choice(chunk_choices)
            si = random.choice(simd_choices)
            pr = random.choice(prefetch_choices)

            t_ms, gflops, gbs = run_trial(kernel, dim, tx, ty, 16, th, sc, ch, si, pr)
            speedup = t_base / t_ms if t_ms > 0 else 0.0

            writer.writerow([trial_idx, tx, ty, th, sc, ch, si, pr, f"{t_ms:.3f}", f"{gflops:.2f}", f"{gbs:.2f}", f"{speedup:.2f}"])
            f.flush()

            if t_ms < best_time:
                best_time = t_ms
                best_record = (tx, ty, th, sc, ch, si, pr, t_ms, gflops, speedup)
                print(f"  [New Best Trial {trial_idx:2d}] Time: {t_ms:5.2f} ms | GFLOP/s: {gflops:6.2f} | Speedup: {speedup:5.2f}x | Th={th}, TX={tx}, TY={ty}, SIMD={si}, Prefetch={pr}")

            trial_idx += 1

    print("\n-------------------------------------------------------")
    print(f"Tuning Completed: {trial_idx} trials in {time.time() - start_time:.1f}s")
    print(f"Best Configuration: TX={best_record[0]}, TY={best_record[1]}, Threads={best_record[2]}, Sched={best_record[3]}, SIMD={best_record[5]}, Prefetch={best_record[6]}")
    print(f"Untuned Runtime: {t_base:.2f} ms  -->  Best Tuned Runtime: {best_record[7]:.2f} ms")
    print(f"Tuning Speedup: {best_record[9]:.2f}x speedup over untuned baseline!")
    print(f"Saved trials to {trials_csv}")

def tune_3d_heat(dim=128):
    print(f"\n=======================================================")
    print(f" Phase 3: 3D Heat Stencil Tuning ({dim}x{dim}x{dim})")
    print(f"=======================================================")
    tz_list = [4, 8, 16, 32]
    ty_list = [8, 16, 32, 64]

    heatmap_3d = np.zeros((len(ty_list), len(tz_list)))
    csv_file = "results/tuning_3d_trials.csv"

    best_time = 1e9
    best_cfg = None

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dim", "TZ", "TY", "Threads", "TimeMS", "GFLOPs", "GBs"])

        for i, ty in enumerate(ty_list):
            for j, tz in enumerate(tz_list):
                t_ms, gflops, gbs = run_trial("heat3d", dim, 64, ty, tz, 12, 0, 0, 1, 8)
                heatmap_3d[i, j] = gflops
                writer.writerow([dim, tz, ty, 12, f"{t_ms:.3f}", f"{gflops:.2f}", f"{gbs:.2f}"])
                f.flush()

                if t_ms < best_time:
                    best_time = t_ms
                    best_cfg = (tz, ty, gflops, t_ms)
                print(f"  [TZ={tz:2d}, TY={ty:2d}] => Time: {t_ms:5.2f} ms | GFLOP/s: {gflops:5.2f} | GB/s: {gbs:5.2f}")

    print(f"\n>>> Best 3D Configuration: TZ={best_cfg[0]}, TY={best_cfg[1]} ({best_cfg[3]:.2f} ms, {best_cfg[2]:.2f} GFLOP/s)")

    # Plot 3D Heatmap
    plt.figure(figsize=(7, 5), dpi=300)
    im = plt.imshow(heatmap_3d, cmap="magma", aspect="auto", origin="lower")
    cbar = plt.colorbar(im)
    cbar.set_label("Performance (GFLOP/s)", fontsize=11)

    plt.xticks(range(len(tz_list)), [str(z) for z in tz_list], fontsize=10)
    plt.yticks(range(len(ty_list)), [str(y) for y in ty_list], fontsize=10)
    plt.xlabel("Tile Size Z (TZ slices)", fontsize=12)
    plt.ylabel("Tile Size Y (TY rows)", fontsize=12)
    plt.title(f"3D Heat Stencil: Blocking Exploration ({dim}^3)\nOptimal: TZ={best_cfg[0]}, TY={best_cfg[1]} ({best_cfg[2]:.1f} GFLOP/s)", fontsize=13)

    for i in range(len(ty_list)):
        for j in range(len(tz_list)):
            val = heatmap_3d[i, j]
            color = "white" if val < np.max(heatmap_3d) * 0.7 else "black"
            plt.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/tuning_heatmap_3d.png", dpi=300)
    plt.savefig("plots/tuning_heatmap_3d.pdf")
    plt.close()
    print("  Saved 3D heatmap to plots/tuning_heatmap_3d.png and .pdf")

if __name__ == "__main__":
    tune_tile_sizes_2d("gauss5", 2048, 12)
    tune_hyperparameters_2d("gauss5", 2048, 20)
    tune_3d_heat(128)
    print("\nPhase 3: All auto-tuning experiments and heatmaps completed successfully!\n")

