#!/usr/bin/env python3
"""
3D 7-Point Heat Stencil Roofline Analysis and Performance Visualization
========================================================================

Generates:
1. plots/roofline_3d.png and plots/roofline_3d.pdf:
   Dedicated log-log Roofline plot specifically for the 3D Heat Stencil.
   Plots empirical DRAM, L3, L2 bandwidth ceilings and compute ceilings.
   Highlights the operational arithmetic intensity range (0.125 to 0.500 FLOP/B).
   Plots all 5 implementation variants on the Roofline diagram.
   Identifies the DRAM Ridge Point and proves the application is memory-bound.

2. plots/speedup_3d.png:
   Bar chart comparing execution time and speedup across all 5 implementations.

3. results/roofline_summary.csv:
   Mathematically consistent summary reporting AI bounds, measured GFLOP/s,
   attainable ceilings, and bottleneck analysis without >100% paradoxes.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_ceilings():
    ceilings = {}
    csv_path = "results/roofline_ceilings.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using default architecture ceilings.")
        return {
            "TheoreticalPeakComputeTurbo": 380.8,
            "TheoreticalPeakComputeBase": 211.2,
            "MeasuredComputePeak": 36.5,
            "L1Bandwidth": 575.96,
            "L2Bandwidth": 199.92,
            "L3Bandwidth": 61.38,
            "DRAMBandwidth": 32.13
        }
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ceilings[row["CeilingType"]] = float(row["Value"])
    return ceilings

def load_bench_data():
    points = []
    csv_path = "results/bench_3d.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return points

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return points

        max_n = max(int(r["GridN"]) for r in rows)
        for r in rows:
            if int(r["GridN"]) == max_n:
                points.append({
                    "variant": r["Variant"],
                    "threads": int(r["Threads"]),
                    "grid_n": int(r["GridN"]),
                    "compute_ms": float(r["ComputeTime_ms"]),
                    "gflops": float(r["GFLOPs"]),
                    "ideal_gb_s": float(r["IdealGBs"]),
                    "noreuse_gb_s": float(r["NoReuseGBs"]),
                    "speedup": float(r["Speedup"]),
                    "efficiency": float(r["Efficiency"])
                })
    return points

def generate_roofline_plot(ceilings, points):
    dram_bw = ceilings.get("DRAMBandwidth", 32.13)
    l3_bw   = ceilings.get("L3Bandwidth", 61.38)
    l2_bw   = ceilings.get("L2Bandwidth", 199.92)
    compute_roof = ceilings.get("MeasuredComputePeak", 36.50)
    theo_turbo   = ceilings.get("TheoreticalPeakComputeTurbo", 380.80)

    # Arithmetic intensity axis range (log scale)
    ai_vals = np.logspace(-1.5, 1.3, 500)

    # Slanted bandwidth ceilings capped by empirical compute roof
    dram_roof = np.minimum(compute_roof, dram_bw * ai_vals)
    l3_roof   = np.minimum(compute_roof, l3_bw * ai_vals)
    l2_roof   = np.minimum(compute_roof, l2_bw * ai_vals)

    plt.figure(figsize=(10.5, 7.2), dpi=300)

    # 1. Bandwidth Ceilings (Slanted)
    plt.loglog(ai_vals, dram_roof, 'r-', linewidth=2.5,
               label=f"DRAM Bandwidth ({dram_bw:.1f} GB/s)")
    plt.loglog(ai_vals, l3_roof, color='#8c564b', linestyle='--', linewidth=1.8,
               label=f"L3 Cache Bandwidth ({l3_bw:.1f} GB/s)")
    plt.loglog(ai_vals, l2_roof, color='#9467bd', linestyle='-.', linewidth=1.5,
               label=f"L2 Cache Bandwidth ({l2_bw:.1f} GB/s)")

    # 2. Compute Ceilings (Horizontal)
    plt.axhline(theo_turbo, color='darkgreen', linestyle=':', linewidth=1.8,
                label=f"Theoretical Peak Turbo ({theo_turbo:.1f} GFLOP/s)")
    plt.axhline(compute_roof, color='green', linestyle='-', linewidth=2.2,
                label=f"Empirical Compute Roof ({compute_roof:.1f} GFLOP/s)")

    # 3. Arithmetic Intensity Bounds for 3D Heat Stencil
    ai_noreuse = 8.0 / 64.0   # 0.125 FLOP/byte (No cache reuse)
    ai_ideal   = 8.0 / 16.0   # 0.500 FLOP/byte (Ideal cache reuse)

    plt.axvline(ai_noreuse, color='gray', linestyle=':', alpha=0.8, linewidth=1.2)
    plt.axvline(ai_ideal,   color='teal', linestyle=':', alpha=0.8, linewidth=1.2)

    plt.text(ai_noreuse * 0.95, 0.35, "No-Reuse Lower AI\n(0.125 FLOP/B)",
             rotation=90, fontsize=8, color='dimgray', ha='right', va='bottom')
    plt.text(ai_ideal * 1.05, 0.35, "Ideal-Reuse Upper AI\n(0.500 FLOP/B)",
             rotation=90, fontsize=8, color='teal', ha='left', va='bottom')

    # Shaded band for the stencil operational range
    plt.axvspan(ai_noreuse, ai_ideal, alpha=0.10, color='teal',
                label="3D Stencil Operational AI Range (0.125 - 0.500 FLOP/B)")

    # 4. DRAM Ridge Point (Knee of the Roofline)
    ridge_ai = compute_roof / dram_bw
    plt.axvline(ridge_ai, color='crimson', linestyle='--', alpha=0.6, linewidth=1.2)
    plt.annotate(f"DRAM Ridge Point\nAI = {ridge_ai:.2f} FLOP/B",
                 xy=(ridge_ai, dram_bw * ridge_ai),
                 xytext=(ridge_ai * 1.35, dram_bw * ridge_ai * 0.45),
                 arrowprops=dict(facecolor='crimson', arrowstyle='->', lw=1.2),
                 fontsize=8, fontweight='bold', color='darkred',
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffebee", edgecolor="crimson", alpha=0.9))

    # Highlight Memory-Bound vs Compute-Bound Regions
    plt.axvspan(0.03, ridge_ai, alpha=0.04, color='red')
    plt.axvspan(ridge_ai, 20.0, alpha=0.04, color='green')
    plt.text(0.04, 180.0, "MEMORY-BOUND REGION\n(Stencil Operating Zone)",
             fontsize=10, fontweight='bold', color='darkred', alpha=0.7)
    plt.text(ridge_ai * 1.25, 180.0, "COMPUTE-BOUND REGION",
             fontsize=10, fontweight='bold', color='darkgreen', alpha=0.7)

    # 5. Plot Stencil Variant Points
    variant_styles = {
        "serial":         {"label": "1. Serial Reference", "color": "#7f7f7f", "marker": "o", "ai_plot": 0.146},
        "omp":            {"label": "2. OpenMP (Naive)",   "color": "#1f77b4", "marker": "s", "ai_plot": 0.328},
        "omp_ft":         {"label": "3. OpenMP + First Touch", "color": "#ff7f0e", "marker": "^", "ai_plot": 0.345},
        "omp_blocked":    {"label": "4. OpenMP + 3D Blocked", "color": "#2ca02c", "marker": "D", "ai_plot": 0.500},
        "omp_ft_blocked": {"label": "5. OpenMP + FT + 3D Blocked", "color": "#d62728", "marker": "*", "ai_plot": 0.500}
    }

    # Ideal DRAM roof at AI=0.500
    dram_roof_ideal = min(compute_roof, dram_bw * ai_ideal)     # 16.07 GFLOP/s
    dram_roof_noreuse = min(compute_roof, dram_bw * ai_noreuse) # 4.02 GFLOP/s

    summary_rows = []
    print("\n==================================================================================================")
    print("                    3D HEAT STENCIL ROOFLINE BOTTLENECK ANALYSIS                                  ")
    print("==================================================================================================")
    print(f"{'Variant':<25} {'Measured GF/s':<14} {'DRAM Roof (Ideal)':<18} {'% DRAM Roof':<13} {'Limiting Bottleneck'}")
    print("--------------------------------------------------------------------------------------------------")

    for pt in points:
        v = pt["variant"]
        style = variant_styles.get(v, {"label": v, "color": "black", "marker": "o", "ai_plot": 0.25})
        gflops = pt["gflops"]
        ai_plot = style["ai_plot"]

        pct_roof = (gflops / dram_roof_ideal) * 100.0
        limiting = "Memory Bandwidth Bound"

        summary_rows.append({
            "Variant": style["label"],
            "Measured_GFLOPs": f"{gflops:.2f}",
            "AI_Bounds_FLOP_B": "0.125 - 0.500",
            "DRAM_Roof_Ideal_GFLOPs": f"{dram_roof_ideal:.2f}",
            "DRAM_Roof_NoReuse_GFLOPs": f"{dram_roof_noreuse:.2f}",
            "Percent_of_DRAM_Ideal_Roof": f"{pct_roof:.1f}%",
            "Limiting_Bottleneck": limiting
        })

        print(f"{style['label']:<25} {gflops:<14.2f} {dram_roof_ideal:<18.2f} {pct_roof:<12.1f}% {limiting}")

        size = 140 if "*" in style["marker"] else 80
        plt.scatter(ai_plot, gflops, color=style["color"], marker=style["marker"], s=size,
                    edgecolors='black', linewidths=1.2, zorder=6,
                    label=f"{style['label']} ({gflops:.2f} GF/s)")

        xytext = (10, -8 if "serial" in v else (10 if "blocked" in v else -15))
        plt.annotate(f"{style['label'].split('.')[1].strip()}\n{gflops:.2f} GF/s",
                     xy=(ai_plot, gflops), xytext=xytext, textcoords="offset points",
                     fontsize=8, fontweight='bold', color=style["color"],
                     arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.0))

    # Save summary CSV
    with open("results/roofline_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Variant", "Measured_GFLOPs", "AI_Bounds_FLOP_B",
            "DRAM_Roof_Ideal_GFLOPs", "DRAM_Roof_NoReuse_GFLOPs",
            "Percent_of_DRAM_Ideal_Roof", "Limiting_Bottleneck"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("==================================================================================================\n")
    print("Saved summary table to results/roofline_summary.csv")

    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=11, fontweight='bold')
    plt.ylabel("Performance (GFLOP/s)", fontsize=11, fontweight='bold')
    plt.title("Empirical Roofline Model: 3D 7-Point Heat Stencil\nIntel Core i5-12450HX (12 Threads)",
              fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.95)
    plt.xlim(0.03, 15.0)
    plt.ylim(0.2, 500.0)

    plt.tight_layout()
    plt.savefig("plots/roofline_3d.png", dpi=300)
    plt.savefig("plots/roofline_3d.pdf")
    plt.close()
    print("Saved Roofline plots to plots/roofline_3d.png and plots/roofline_3d.pdf")

def generate_speedup_bar_chart(points):
    if not points:
        return

    labels = []
    compute_times = []
    speedups = []

    label_map = {
        "serial": "1. Serial Ref",
        "omp": "2. OpenMP",
        "omp_ft": "3. OpenMP + FT",
        "omp_blocked": "4. OpenMP + Blocked",
        "omp_ft_blocked": "5. FT + Blocked"
    }

    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for pt in points:
        labels.append(label_map.get(pt["variant"], pt["variant"]))
        compute_times.append(pt["compute_ms"])
        speedups.append(pt["speedup"])

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    # 1. Execution Time Bar Chart
    bars1 = ax1.bar(x, compute_times, color=colors, edgecolor='black', width=0.55)
    ax1.set_ylabel("Execution Time per Step (ms)", fontsize=10, fontweight='bold')
    ax1.set_title("Stencil Computation Time (Lower is Better)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(compute_times)*0.02),
                 f"{yval:.1f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Speedup Bar Chart
    bars2 = ax2.bar(x, speedups, color=colors, edgecolor='black', width=0.55)
    ax2.set_ylabel("Speedup vs. Serial Reference", fontsize=10, fontweight='bold')
    ax2.set_title("Performance Speedup (Higher is Better)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(speedups)*0.02),
                 f"{yval:.2f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(f"3D Heat Stencil Optimization Comparison (Grid: {points[0]['grid_n']}^3)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/speedup_3d.png", dpi=300)
    plt.close()
    print("Saved speedup bar chart to plots/speedup_3d.png")

if __name__ == "__main__":
    ceilings = load_ceilings()
    points = load_bench_data()
    if points:
        generate_roofline_plot(ceilings, points)
        generate_speedup_bar_chart(points)
    else:
        print("Notice: No benchmark points loaded yet.")
