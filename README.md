# Circuit Partitioning using EIG+KL Hybrid Algorithm

A high-performance circuit partitioning solution combining Eigenvalue (EIG) and Kernighan-Lin (KL) algorithms to optimize ratio cuts while maintaining balanced partitions. This implementation leverages sparse matrix operations, efficient eigenvector computation, and parallel processing.

## Overview

The project implements a hybrid approach to circuit partitioning by combining:
- Eigenvalue-based initial partitioning (EIG)
- Kernighan-Lin optimization (KL)
- Sparse matrix optimizations
- OpenMP parallel processing

## Prerequisites

The following packages are required (Ubuntu):

```bash
# Install essential build tools and libraries
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libomp-dev \
    liblapack-dev \
    liblapacke-dev \
    libblas-dev \
    libeigen3-dev
    
# Install Spectra library
cd /tmp
git clone https://github.com/yixuan/spectra.git
cd spectra
mkdir build && cd build
# Configure CMake to install in the system directory
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install
```

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/yhinai/EIG-KL-Algorithm.git
cd EIG-KL-Algorithm
```

2. Build the project:
```bash
make
```

This will create two executables:
- `EIG`: Eigenvalue-based partitioning algorithm
- `KL`: Kernighan-Lin partitioning algorithm (OpenMP-enabled)

## Usage

### Running the Algorithms

1. Run EIG algorithm:
```bash
./EIG <input_file>
```

2. Run KL algorithm:
```bash
# Run KL with automatic thread detection
./KL <circuit_input_file>

# Run KL with EIG initialization
./KL <circuit_input_file> -EIG

# Run KL with specific number of threads
OMP_NUM_THREADS=4 ./KL <circuit_input_file>
```

### Output Files

The algorithms generate output files in the `results/` directory:
- `results/<input_file>_out.txt`: EIG partitioning results
- `results/<input_file>_KL_CutSize_output.txt`: KL algorithm results
- `results/<input_file>_KL_CutSize_EIG_output.txt`: Hybrid EIG+KL results

## Performance Optimization

For best performance:

1. Adjust OpenMP threads based on your system:
```bash
# Use all available cores
export OMP_NUM_THREADS=$(nproc)

# Or specify a specific number
export OMP_NUM_THREADS=4
```

2. Enable performance mode (requires root):
```bash
sudo cpupower frequency-set -g performance
```

## Implementation Details

### EIG Algorithm

- Uses Eigen and Spectra libraries for efficient sparse matrix operations
- Computes the second smallest eigenvalue and corresponding eigenvector
- Implements parallel processing through OpenMP
- Memory-efficient sparse matrix representation

### KL Algorithm

- Optimized gain calculation with parallel processing
- Dynamic cut size computation
- OpenMP parallel processing support
- Memory-efficient sparse matrix implementation
- Smart termination conditions

## Input File Format

The input file should follow this structure:
```
<number_of_nets> <number_of_nodes>
<node1> <node2> ... <nodeN>  # Net 1
<node1> <node2> ... <nodeM>  # Net 2
...
```

Example:
```
3 5     # First line list the number of nets followed by the number of nodes in this circuit 
1 2 3   # Net connecting nodes 1, 2, and 3
2 4     # Net connecting nodes 2 and 4
3 4 5   # Net connecting nodes 3, 4, and 5
```

## Troubleshooting

1. OpenMP Issues:
   - Verify installation: `echo |cpp -fopenmp -dM |grep -i open`
   - Check number of threads: `OMP_NUM_THREADS=4 ./KL <input_file>`

2. Library Issues:
   ```bash
   # Reinstall dependencies if needed
   sudo apt-get install --reinstall \
       libomp-dev liblapack-dev liblapacke-dev libblas-dev libeigen3-dev
   ```

3. Compilation Issues:
   - Clean and rebuild: `make clean && make`
   - Check compiler version: `g++ --version`
   - Verify Spectra installation: `ls /usr/local/include/Spectra`
