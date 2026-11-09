#!/usr/bin/env python3
"""
Performance Visualization Suite for HPC Stencil Project
Generates:
1. plots/speedup_variants_2d.png/.pdf: Speedup ladder across variants (v0 -> v5)
2. plots/speedup_variants_3d.png/.pdf: 3D Heat Stencil ladder (v0 -> v6)
3. plots/thread_scaling_2d.png/.pdf: Strong scaling speedup and parallel efficiency
4. plots/thread_scaling_3d.png/.pdf: 3D strong scaling
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

def plot_speedup_ladder_2d():
    if not os.path.exists("results/bench_2d_variants.csv"): return

    data = {}
    with open("results/bench_2d_variants.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Width"] == "4096":
                k = row["Kernel"]
                v = row["Variant"]
                s = float(row["Speedup"])
                if k not in data: data[k] = {}
                data[k][v] = s

    variants = ["v0_serial", "v1_omp_naive", "v2_first_touch", "v3_tiling", "v4_simd_window", "v5_prefetch_nt"]
    labels = ["v0 Serial", "v1 OMP", "v2 NUMA", "v3 Tiling", "v4 SIMD", "v5 Prefetch"]
    kernels = ["Gauss3x3", "Gauss5x5", "Sobel", "Laplacian"]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    x = np.arange(len(variants))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for i, (k, col) in enumerate(zip(kernels, colors)):
        vals = [data[k].get(v, 1.0) for v in variants]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=k, color=col, edgecolor="black", alpha=0.85)

    ax.set_ylabel("Speedup over Serial Naive Baseline (x)", fontsize=11)
    ax.set_title("Optimization Ladder Speedup (4096 x 4096 Float32, 12 Threads)\nFollowing Datta et al. SC08 Optimization Progression", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/speedup_variants_2d.png", dpi=300)
    plt.savefig("plots/speedup_variants_2d.pdf")
    plt.close()
    print("Saved plots/speedup_variants_2d.png and .pdf")

def plot_speedup_ladder_3d():
    if not os.path.exists("results/bench_3d_variants.csv"): return

    data = {}
    with open("results/bench_3d_variants.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["GridN"] == "256":
                v = row["Variant"]
                s = float(row["Speedup"])
                data[v] = s

    variants = ["v0_serial", "v1_omp_naive", "v2_first_touch", "v3_tiling", "v4_simd_window", "v5_prefetch_nt", "v6_unroll_reg"]
    labels = ["v0 Serial", "v1 OMP", "v2 NUMA", "v3 Tiling", "v4 SIMD", "v5 Prefetch", "v6 Z-Unroll"]
    vals = [data.get(v, 1.0) for v in variants]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.bar(labels, vals, color="#e377c2", edgecolor="black", alpha=0.85, width=0.55)
    ax.axhline(1.0, color="gray", linestyle="--")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Speedup over Serial Naive (x)", fontsize=11)
    ax.set_title("3D 7-Point Heat Stencil Optimization Ladder (256^3 Double, 128 MB)\nFollowing Datta et al. SC08 Section 4", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/speedup_variants_3d.png", dpi=300)
    plt.savefig("plots/speedup_variants_3d.pdf")
    plt.close()
    print("Saved plots/speedup_variants_3d.png and .pdf")

def plot_thread_scaling_2d():
    if not os.path.exists("results/bench_2d_scaling.csv"): return

    data = {}
    with open("results/bench_2d_scaling.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["Kernel"]
            v = row["Variant"]
            th = int(row["Threads"])
            sp = float(row["Speedup"])
            eff = float(row["Efficiency"])

            tag = f"{k} ({'Naive' if 'v1' in v else 'SIMD'})"
            if tag not in data: data[tag] = {"threads": [], "speedup": [], "eff": []}
            data[tag]["threads"].append(th)
            data[tag]["speedup"].append(sp)
            data[tag]["eff"].append(eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Speedup
    all_threads = [1, 2, 4, 6, 8, 10, 12]
    ax1.plot(all_threads, all_threads, 'k--', label="Ideal Linear Speedup", alpha=0.7)

    for tag, d in data.items():
        ls = '-' if 'SIMD' in tag else '--'
        ax1.plot(d["threads"], d["speedup"], marker='o', linestyle=ls, label=tag, linewidth=1.8)

    ax1.set_xlabel("OpenMP Threads", fontsize=11)
    ax1.set_ylabel("Strong Scaling Speedup (x)", fontsize=11)
    ax1.set_title("2D Stencil Strong Scaling Speedup (4096 x 4096)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.5)

    # Efficiency
    ax2.axhline(100.0, color='k', linestyle='--', label="100% Ideal Efficiency", alpha=0.7)
    for tag, d in data.items():
        ls = '-' if 'SIMD' in tag else '--'
        ax2.plot(d["threads"], d["eff"], marker='s', linestyle=ls, label=tag, linewidth=1.8)

    ax2.set_xlabel("OpenMP Threads", fontsize=11)
    ax2.set_ylabel("Parallel Efficiency (%)", fontsize=11)
    ax2.set_title("2D Stencil Parallel Efficiency (4096 x 4096)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig("plots/thread_scaling_2d.png", dpi=300)
    plt.savefig("plots/thread_scaling_2d.pdf")
    plt.close()
    print("Saved plots/thread_scaling_2d.png and .pdf")

def plot_thread_scaling_3d():
    if not os.path.exists("results/bench_3d_scaling.csv"): return

    data = {}
    with open("results/bench_3d_scaling.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row["Variant"]
            th = int(row["Threads"])
            sp = float(row["Speedup"])
            eff = float(row["Efficiency"])

            tag = "v1 Naive OpenMP" if "v1" in v else "v6 Z-Unrolled"
            if tag not in data: data[tag] = {"threads": [], "speedup": [], "eff": []}
            data[tag]["threads"].append(th)
            data[tag]["speedup"].append(sp)
            data[tag]["eff"].append(eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    threads = [1, 2, 4, 6, 8, 10, 12]
    ax1.plot(threads, threads, 'k--', label="Ideal Linear Speedup", alpha=0.7)
    for tag, d in data.items():
        ax1.plot(d["threads"], d["speedup"], marker='o', linewidth=2.0, label=tag)

    ax1.set_xlabel("OpenMP Threads", fontsize=11)
    ax1.set_ylabel("Speedup (x)", fontsize=11)
    ax1.set_title("3D Heat Stencil Strong Scaling (256^3 Grid)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.axhline(100.0, color='k', linestyle='--', label="100% Ideal Efficiency", alpha=0.7)
    for tag, d in data.items():
        ax2.plot(d["threads"], d["eff"], marker='s', linewidth=2.0, label=tag)

    ax2.set_xlabel("OpenMP Threads", fontsize=11)
    ax2.set_ylabel("Parallel Efficiency (%)", fontsize=11)
    ax2.set_title("3D Heat Stencil Parallel Efficiency (256^3 Grid)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("plots/thread_scaling_3d.png", dpi=300)
    plt.savefig("plots/thread_scaling_3d.pdf")
    plt.close()
    print("Saved plots/thread_scaling_3d.png and .pdf")

if __name__ == "__main__":
    plot_speedup_ladder_2d()
    plot_speedup_ladder_3d()
    plot_thread_scaling_2d()
    plot_thread_scaling_3d()

