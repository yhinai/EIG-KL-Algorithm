# Circuit Partitioning using EIG+KL Hybrid Algorithm

A high-performance circuit partitioning solution combining Eigenvalue (EIG) and Kernighan-Lin (KL) algorithms to optimize ratio cuts while maintaining balanced partitions. This implementation leverages sparse matrix operations, efficient eigenvector computation, and parallel processing.

## Overview

The project implements a hybrid approach to circuit partitioning by combining:
- Eigenvalue-based initial partitioning (EIG)
- Kernighan-Lin optimization (KL)
- Sparse matrix optimizations
- OpenMP parallel processing

## Prerequisites

This project uses Conda for dependency management. Make sure you have Conda installed on your system. If not, you can install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Download and install the latest Miniconda (Linux AMD64)
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Initialize conda on all shells
source ~/miniconda3/bin/activate
conda init --all

```
## Environment Setup

```bash
# Request 1 GPU
salloc -G1

# Clone the repository:
git clone https://github.com/yhinai/EIG-KL-Algorithm.git
cd EIG-KL-Algorithm

# Create a new conda environment
conda create --name KL -y
conda activate KL

# Install essential build tools and libraries
conda install -c conda-forge -y \
    cmake \
    openmp \
    lapack \
    blas \
    eigen

# Clone and install Spectra
git clone https://github.com/yixuan/spectra.git
cd spectra
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make install
cd ../..
rm -rf spectra

# Add conda library path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Build the project
make

```

This will create two executables:
- `cEIG`: Eigenvalue-based partitioning algorithm
- `cKL`: Kernighan-Lin partitioning algorithm (OpenMP-enabled)

## Usage

### Running the Algorithms

1. Run EIG algorithm:
```bash
./cEIG <input_file>
```

2. Run KL algorithm:
```bash
# Run KL with automatic thread detection
./cKL <circuit_input_file>

# Run KL with EIG initialization
./cKL <circuit_input_file> -EIG

# Run KL with specific number of threads
OMP_NUM_THREADS=4 ./cKL <circuit_input_file>
```

### Building and Running gKL (CUDA Version)

#### Building gKL

```bash
# Build the CUDA version
make gKL
```

#### Running gKL on GPU Cluster (ICE.PACE)

```bash
# Request GPU allocation
salloc -G1

# Run the script
./gKL <input_file>
./gKL <input_file> [-EIG]
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


## Environment Management

To activate conda environment:
```bash
conda activate KL
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

To request GPU on ICE.PACE:
```bash
# Request 1 GPU
salloc -G1
```


To remove the environment:
```bash
conda deactivate
conda env remove -n KL
```

To save the environment configuration:
```bash
conda env export > environment.yml
```

To recreate the environment on another machine:
```bash
conda env create -f environment.yml
```


## Troubleshooting

1. OpenMP Issues:
   - Verify OpenMP is installed: `conda list openmp`
   - Check number of threads: `echo $OMP_NUM_THREADS`

2. Library Issues:
   ```bash
   # Reinstall dependencies if needed
   conda install -c conda-forge --force-reinstall \
       openmp lapack blas eigen
   ```

3. Compilation Issues:
   - Clean and rebuild: `make clean && make`
   - Check compiler version: `g++ --version`
   - Verify Spectra installation: `ls $CONDA_PREFIX/include/Spectra`
   - Make sure environment is activated: `conda activate KL`

4. Library Path Issues:
   - Verify library path: `echo $LD_LIBRARY_PATH`
   - Reset library path if needed:
     ```bash
     export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
     ```