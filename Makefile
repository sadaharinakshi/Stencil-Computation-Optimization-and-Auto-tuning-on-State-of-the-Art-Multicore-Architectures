# ============================================================================
# Makefile for 3D 7-Point Heat Stencil HPC Project
# Team: T09 | Harinakshi S, Vaarshini R
# Motif: Structured Grids | Technology: C & OpenMP
# ============================================================================

SHELL := tools/w64devkit/bin/sh.exe
TOOLCHAIN_BIN := tools/w64devkit/bin/
EXE := .exe
RM := rm -f

CC := $(TOOLCHAIN_BIN)gcc$(EXE)
CFLAGS := -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude
LDFLAGS := -fopenmp -lm

# Binaries
CORRECTNESS_BIN := correctness_check$(EXE)
BENCH_3D_BIN    := bench_3d$(EXE)
ROOFLINE_BIN    := microbench_roofline$(EXE)

# Sources
STENCIL_SRCS := src/kernels_3d_serial.c src/kernels_3d_opt.c src/kernels_3d_dispatch.c

.PHONY: all bench correctness roofline plots clean

all: $(CORRECTNESS_BIN) $(BENCH_3D_BIN) $(ROOFLINE_BIN)

$(CORRECTNESS_BIN): bench/correctness_check.c $(STENCIL_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BENCH_3D_BIN): bench/benchmark_3d.c $(STENCIL_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(ROOFLINE_BIN): bench/microbench_roofline.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

correctness: $(CORRECTNESS_BIN)
	./$(CORRECTNESS_BIN)

bench: $(BENCH_3D_BIN)
	./$(BENCH_3D_BIN)

roofline: $(ROOFLINE_BIN)
	./$(ROOFLINE_BIN)
	python scripts/plot_roofline.py

plots:
	python scripts/plot_roofline.py

clean:
	$(RM) *$(EXE)
