# Check if CONDA_PREFIX is set
ifndef CONDA_PREFIX
$(error CONDA_PREFIX is not set. Please activate your conda environment first: conda activate KL)
endif

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra \
           -I$(CONDA_PREFIX)/include \
           -I$(CONDA_PREFIX)/include/eigen3 \
           -fopenmp

# Libraries
LIBS = -L$(CONDA_PREFIX)/lib -llapack -lblas -llapacke

# Output directories
DIRS = results

# Targets
.PHONY: all clean check_env build

# Default target
all: clean_binary check_env build

# Check conda environment
check_env:
	@if [ -z "$$CONDA_PREFIX" ]; then \
		echo "Error: CONDA_PREFIX is not set. Please activate your conda environment."; \
		exit 1; \
	fi
	@if [ ! -d "$(CONDA_PREFIX)/include/eigen3" ]; then \
		echo "Error: Eigen3 not found. Please install required dependencies."; \
		echo "Run: conda install -c conda-forge eigen"; \
		exit 1; \
	fi
	@if [ ! -d "$(CONDA_PREFIX)/include/Spectra" ]; then \
		echo "Error: Spectra not found. Please install Spectra library."; \
		echo "See README.md for installation instructions."; \
		exit 1; \
	fi


# Build target for actual compilation
build: EIG KL
	@echo
# Compile EIG executable
EIG: cEIG.cpp
	@echo "Building EIG:" 
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)
	@echo

# Compile KL executable
KL: cKL.cpp
	@echo "Building KL:" 
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)
	@echo

# Clean target
clean_binary:
	rm -f EIG KL

clean: clean_binary
	rm -rf $(DIRS)