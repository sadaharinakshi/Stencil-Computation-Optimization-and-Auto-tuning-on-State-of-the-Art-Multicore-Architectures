# Makefile for HPC Stencil Computation Project
# Stencil Computation Optimization and Auto-tuning for Multicore CPUs

SHELL := tools/w64devkit/bin/sh.exe
TOOLCHAIN_BIN := tools/w64devkit/bin/
EXE := .exe
RM := rm -f

CC := $(TOOLCHAIN_BIN)gcc$(EXE)
CFLAGS := -O3 -march=native -ffp-contract=off -fopenmp -Wall -Wextra -Iinclude
LDFLAGS := -fopenmp -lm

# Binaries
CORRECTNESS_BIN := correctness_check$(EXE)
SYNTHETIC_BIN := synthetic_gen$(EXE)
BENCH_2D_BIN := bench_2d$(EXE)
BENCH_3D_BIN := bench_3d$(EXE)
BENCH_TUNE_BIN := bench_tune$(EXE)
ROOFLINE_BIN := microbench_roofline$(EXE)

# Sources
COMMON_SRCS := src/image_io.c src/kernels_2d_serial.c src/kernels_3d_serial.c \
               src/kernels_2d_opt.c src/kernels_3d_opt.c \
               src/kernels_2d_dispatch.c src/kernels_3d_dispatch.c

.PHONY: all bench tune roofline plots clean

all: $(CORRECTNESS_BIN) $(SYNTHETIC_BIN) $(BENCH_2D_BIN) $(BENCH_3D_BIN) $(BENCH_TUNE_BIN) $(ROOFLINE_BIN)

$(BENCH_TUNE_BIN): bench/bench_tune.c $(COMMON_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(CORRECTNESS_BIN): bench/correctness_check.c $(COMMON_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(SYNTHETIC_BIN): bench/synthetic_gen.c $(COMMON_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BENCH_2D_BIN): bench/benchmark_2d.c $(COMMON_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BENCH_3D_BIN): bench/benchmark_3d.c $(COMMON_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(ROOFLINE_BIN): bench/microbench_roofline.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

bench: all
	./$(BENCH_2D_BIN)
	./$(BENCH_3D_BIN)

tune: all
	python scripts/tune.py

roofline: all
	./$(ROOFLINE_BIN)
	python scripts/plot_roofline.py

plots:
	python scripts/plot_results.py

clean:
	$(RM) *$(EXE)
