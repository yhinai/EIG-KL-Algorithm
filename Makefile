# Compiler settings
CXX ?= g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I/usr/include/eigen3 -I/usr/local/include/spectra
LDFLAGS = -llapack -lblas -llapacke

# OpenMP settings
OMPFLAGS = -fopenmp
OMPLIBS = -lomp

# Source files
EIG_SRC = cEIG.cpp
KL_SRC = cKL.cpp

# Output executables
EIG_TARGET = EIG
KL_TARGET = KL

# Default target
all: clean directories $(EIG_TARGET) $(KL_TARGET)

# Create necessary directories
directories:
	@mkdir -p results
	@mkdir -p pre_saved_EIG

# EIG algorithm
$(EIG_TARGET): $(EIG_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPLIBS)

# KL algorithm
$(KL_TARGET): $(KL_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) $(OMPLIBS)

# Clean build files
clean:
	rm -f $(EIG_TARGET) $(KL_TARGET)

.PHONY: all clean directories