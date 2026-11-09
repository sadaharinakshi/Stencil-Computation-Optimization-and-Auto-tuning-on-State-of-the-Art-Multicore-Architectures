#!/usr/bin/env python3
"""
HPC Roofline Analysis and Visualization
Generates:
1. results/roofline_summary.csv: Detailed breakdown of AI, GFLOP/s, % roof, limiting ceiling
2. plots/roofline_model.png and plots/roofline_model.pdf: Log-log roofline plot
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Analytical Arithmetic Intensities (FLOPs / byte)
# AI_ideal: assuming perfect cache reuse (1 read + 1 write per grid element)
# AI_noreuse: assuming every stencil tap fetches from DRAM
AI_DICT = {
    "Gauss3x3": {"ideal": 17.0 / 8.0,  "noreuse": 17.0 / 40.0, "flops_per_elem": 17.0},
    "Gauss5x5": {"ideal": 49.0 / 8.0,  "noreuse": 49.0 / 104.0, "flops_per_elem": 49.0},
    "Sobel":    {"ideal": 16.0 / 8.0,  "noreuse": 16.0 / 52.0, "flops_per_elem": 16.0},
    "Laplacian":{"ideal": 6.0 / 8.0,   "noreuse": 6.0 / 24.0,  "flops_per_elem": 6.0},
    "Heat7pt":  {"ideal": 8.0 / 16.0,  "noreuse": 8.0 / 64.0,  "flops_per_elem": 8.0}
}

def load_ceilings():
    ceilings = {}
    with open("results/roofline_ceilings.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ceilings[row["CeilingType"]] = float(row["Value"])
    return ceilings

def load_bench_data():
    points = []

    # 2D variants (4096 x 4096)
    if os.path.exists("results/bench_2d_variants.csv"):
        with open("results/bench_2d_variants.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Width"] == "4096":
                    points.append({
                        "domain": "2D",
                        "kernel": row["Kernel"],
                        "variant": row["Variant"],
                        "gflops": float(row["GFLOPs"]),
                        "threads": int(row["Threads"])
                    })

    # 3D variants (256^3)
    if os.path.exists("results/bench_3d_variants.csv"):
        with open("results/bench_3d_variants.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["GridN"] == "256":
                    points.append({
                        "domain": "3D",
                        "kernel": "Heat7pt",
                        "variant": row["Variant"],
                        "gflops": float(row["GFLOPs"]),
                        "threads": int(row["Threads"])
                    })
    return points

def build_summary(ceilings, points):
    dram_bw = ceilings.get("DRAMBandwidth", 20.93)
    # Use the empirical sustained FMA peak as the compute roof as requested
    sustained_comp_roof = ceilings.get("MeasuredSustainedFMAPeak", 471.68)

    summary_rows = []
    print("\n==========================================================================================")
    print("                    ROOFLINE MODEL SUMMARY & BOTTLENECK ANALYSIS                          ")
    print("==========================================================================================")
    print(f"{'Kernel':<10} {'Variant':<16} {'AI (ideal)':<11} {'AI (no-reuse)':<13} {'GFLOP/s':<10} {'Roof (GFLOP/s)':<15} {'% Roof':<8} {'Limiting Ceiling'}")
    print("------------------------------------------------------------------------------------------")

    with open("results/roofline_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Kernel", "Variant", "Domain", "AI_Ideal", "AI_NoReuse", "Measured_GFLOPs", "Attainable_Roof_GFLOPs", "Percent_Roof", "LimitingCeiling"])

        for pt in points:
            k = pt["kernel"]
            v = pt["variant"]
            gflops = pt["gflops"]
            ai_info = AI_DICT.get(k, {"ideal": 1.0, "noreuse": 0.5})
            ai_ideal = ai_info["ideal"]
            ai_noreuse = ai_info["noreuse"]

            # Attainable roof at ideal AI with DRAM bandwidth and sustained compute roof
            bw_roof = dram_bw * ai_ideal
            attainable_roof = min(sustained_comp_roof, bw_roof)
            pct = (gflops / attainable_roof) * 100.0 if attainable_roof > 0 else 0.0

            limiting = "Memory Bandwidth Bound" if bw_roof < sustained_comp_roof else "Compute Bound"

            summary_rows.append({
                "kernel": k, "variant": v, "ai_ideal": ai_ideal, "ai_noreuse": ai_noreuse,
                "gflops": gflops, "attainable_roof": attainable_roof, "pct": pct, "limiting": limiting
            })

            print(f"{k:<10} {v:<16} {ai_ideal:<11.3f} {ai_noreuse:<13.3f} {gflops:<10.2f} {attainable_roof:<15.2f} {pct:<8.1f}% {limiting}")
            writer.writerow([k, v, pt["domain"], f"{ai_ideal:.3f}", f"{ai_noreuse:.3f}", f"{gflops:.2f}", f"{attainable_roof:.2f}", f"{pct:.1f}", limiting])

    print("==========================================================================================\n")
    print("Saved summary to results/roofline_summary.csv")
    return summary_rows

def plot_roofline(ceilings, points):
    dram_bw = ceilings.get("DRAMBandwidth", 20.93)
    l3_bw = ceilings.get("L3Bandwidth", 39.82)
    l2_bw = ceilings.get("L2Bandwidth", 134.06)
    l1_bw = ceilings.get("L1Bandwidth", 774.43)

    theo_turbo = ceilings.get("TheoreticalPeakComputeTurbo", 704.0)
    theo_base  = ceilings.get("TheoreticalPeakComputeBase", 198.4)
    burst_fma  = ceilings.get("MeasuredBurstFMAPeak", 558.76)
    sustained_fma = ceilings.get("MeasuredSustainedFMAPeak", 471.68)
    scalar_peak = ceilings.get("MeasuredScalarPeak", 37.09)

    # Arithmetic intensity axis range
    ai_vals = np.logspace(-1.5, 1.5, 500)

    # Ceilings plotted relative to the sustained compute roof
    dram_roof = np.minimum(sustained_fma, dram_bw * ai_vals)
    l3_roof   = np.minimum(sustained_fma, l3_bw * ai_vals)
    l2_roof   = np.minimum(sustained_fma, l2_bw * ai_vals)
    l1_roof   = np.minimum(sustained_fma, l1_bw * ai_vals)

    plt.figure(figsize=(11, 7.5), dpi=300)

    # 1. Bandwidth Ceilings
    plt.loglog(ai_vals, dram_roof, 'r-', linewidth=2.5, label=f"DRAM Bandwidth ({dram_bw:.1f} GB/s)")
    plt.loglog(ai_vals, l3_roof, color='#8c564b', linestyle='--', linewidth=1.8, label=f"L3 Cache Bandwidth ({l3_bw:.1f} GB/s)")
    plt.loglog(ai_vals, l2_roof, color='#9467bd', linestyle='-.', linewidth=1.6, label=f"L2 Cache Bandwidth ({l2_bw:.1f} GB/s)")
    plt.loglog(ai_vals, l1_roof, color='#17becf', linestyle=':', linewidth=1.6, label=f"L1 Data Cache Bandwidth ({l1_bw:.1f} GB/s)")

    # 2. Compute Ceilings
    plt.axhline(theo_turbo, color='darkgreen', linestyle='-', linewidth=2.2, label=f"Theoretical Peak Turbo ({theo_turbo:.0f} GFLOP/s)")
    plt.axhline(sustained_fma, color='green', linestyle='--', linewidth=2.2, label=f"Measured Sustained FMA Roof ({sustained_fma:.1f} GFLOP/s)")
    plt.axhline(burst_fma, color='lightgreen', linestyle=':', linewidth=1.5, label=f"Measured Burst FMA Peak ({burst_fma:.1f} GFLOP/s)")
    plt.axhline(theo_base, color='gray', linestyle='-.', linewidth=1.5, label=f"Theoretical Base Peak ({theo_base:.1f} GFLOP/s)")
    plt.axhline(scalar_peak, color='orange', linestyle=':', linewidth=1.8, label=f"Measured Scalar Peak ({scalar_peak:.1f} GFLOP/s)")

    # Color palette and markers for kernels
    kernel_styles = {
        "Gauss3x3":  {"color": "#1f77b4", "marker": "o"},
        "Gauss5x5":  {"color": "#2ca02c", "marker": "s"},
        "Sobel":     {"color": "#d62728", "marker": "^"},
        "Laplacian": {"color": "#e377c2", "marker": "D"},
        "Heat7pt":   {"color": "#bcbd22", "marker": "p"}
    }

    variant_alpha = {
        "v0_serial": 0.35,
        "v1_omp_naive": 0.50,
        "v2_first_touch": 0.65,
        "v3_tiling": 0.80,
        "v4_simd_window": 0.95,
        "v5_prefetch_nt": 1.0,
        "v6_unroll_reg": 1.0
    }

    # Plot kernel points
    plotted_labels = set()
    for pt in points:
        k = pt["kernel"]
        v = pt["variant"]
        ai = AI_DICT.get(k, {}).get("ideal", 1.0)
        gflops = pt["gflops"]
        style = kernel_styles.get(k, {"color": "black", "marker": "o"})
        alpha = variant_alpha.get(v, 0.8)
        size = 80 if "v4" in v or "v5" in v or "v6" in v else 45

        label = k if k not in plotted_labels else None
        if label: plotted_labels.add(k)

        plt.scatter(ai, gflops, color=style["color"], marker=style["marker"], s=size, alpha=alpha, edgecolors='black', label=label, zorder=5)

        # Annotate peak variant points
        if v in ["v5_prefetch_nt", "v4_simd_window", "v6_unroll_reg"] and k in ["Gauss5x5", "Gauss3x3", "Sobel"]:
            plt.annotate(f"{k} ({v})\n{gflops:.1f} GF/s",
                         (ai, gflops),
                         textcoords="offset points",
                         xytext=(10, -5 if k=="Gauss3x3" else 10),
                         fontsize=8, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.0))

    # Annotate DRAM Ridge Point
    ridge_ai = sustained_fma / dram_bw
    plt.axvline(ridge_ai, color='gray', linestyle='-.', alpha=0.5, linewidth=1.0)
    plt.annotate(f"DRAM Knee\nAI = {ridge_ai:.1f} FLOP/B", (ridge_ai, 1.2), fontsize=8, color='dimgray', ha='center',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))

    plt.xlabel("Arithmetic Intensity (FLOPs / Byte, Ideal Cache)", fontsize=11)
    plt.ylabel("Performance (GFLOP/s)", fontsize=11)
    plt.title("Empirical Roofline Model: Intel Core i5-1235U Hybrid (2P+8E)\nSC08 Datta et al. Ladder (v0 Serial -> v6 Optimized)", fontsize=12)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    plt.xlim(0.05, 30.0)
    plt.ylim(0.5, 900.0)

    plt.tight_layout()
    plt.savefig("plots/roofline_model.png", dpi=300)
    plt.savefig("plots/roofline_model.pdf")
    plt.close()
    print("Saved updated roofline plot to plots/roofline_model.png and .pdf")

if __name__ == "__main__":
    ceilings = load_ceilings()
    points = load_bench_data()
    summary = build_summary(ceilings, points)
    plot_roofline(ceilings, points)
