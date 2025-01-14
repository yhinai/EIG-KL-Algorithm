# Get LLVM and Eigen paths from Homebrew
LLVM_PATH := $(shell brew --prefix llvm)
EIGEN_PATH := $(shell brew --prefix eigen)

# Compiler settings
CXX = $(LLVM_PATH)/bin/clang++
CXXFLAGS = -std=c++17 -Wall -O3
OMPFLAGS = -fopenmp
EIGENFLAGS = -I$(EIGEN_PATH)/include/eigen3 -I.
LLVM_FLAGS = -L$(LLVM_PATH)/lib -Wl,-rpath,$(LLVM_PATH)/lib

# OpenMP settings for clang
OMPLIB = -L$(LLVM_PATH)/lib

all: EIG KL_single_thread KL_multi_thread_omp

EIG: cEIG.cpp
	$(CXX) $(CXXFLAGS) $(EIGENFLAGS) $(LLVM_FLAGS) $< -o $@

KL_single_thread: cKL.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $< -o $@

KL_multi_thread_omp: cKL.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(LLVM_FLAGS) $(OMPLIB) -DUSE_OPENMP $< -o $@

clean:
	rm -f EIG KL_single_thread KL_multi_thread_omp
	rm -f results/*

.PHONY: all clean
