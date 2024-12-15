# EIG+KL Hybrid Algorithm Accelerator

A high-performance circuit partitioning solution that combines Eigenvalue (EIG) and Kernighan-Lin (KL) algorithms, leveraging sparse matrix methods and efficient eigenvector computation. This hybrid approach optimizes ratio cuts while significantly reducing runtime, especially for large circuit benchmarks.

## Overview

This project implements a hybrid approach to circuit partitioning by combining:
- Eigenvalue-based initial partitioning (EIG)
- Kernighan-Lin optimization (KL)
- Sparse matrix optimizations
- OpenMP parallel processing

The implementation shows significant improvements in both cut size quality (>20% improvement) and runtime performance, particularly for large benchmarks like IBM18.

## Features

- **EIG Algorithm Implementation**:
  - Clique-based graph model of netlist using 2-clique model
  - Laplacian Matrix computation (L = D-A)
  - Efficient computation of 2nd smallest eigenvalue and corresponding eigenvector
  - Sparse matrix optimization for large circuits
  - Built-in area bound constraints

- **KL Algorithm Implementation**:
  - Optimized cut size computation
  - Efficient gain calculation for node swaps
  - Multi-threaded implementation using OpenMP
  - Memory-efficient sparse matrix representation
  - Adaptive termination conditions

- **Hybrid Approach Benefits**:
  - Improved initial partitioning from EIG
  - Further optimization through KL refinement
  - Significant runtime improvements
  - Better cut size results
  - Scalable for large benchmarks

## Requirements

- C++ Compiler with C++17 support
- Eigen library
- OpenMP support
- CMake (3.10 or higher)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EIG-KL-Hybrid-Algorithm-Accelerator.git
cd EIG-KL-Hybrid-Algorithm-Accelerator
```

2. Install dependencies (if not already installed):
```bash
# For Ubuntu/Debian
sudo apt-get install libeigen3-dev
sudo apt-get install libomp-dev

# For macOS using Homebrew
brew install eigen
brew install libomp
```

3. Create and enter the build directory:
```bash
mkdir build && cd build
```

4. Build the project:
```bash
cmake ..
make
```

## Usage

### Running the EIG Algorithm

1. Compile the EIG implementation:
```bash
g++ cEIG.cpp -std=c++17 -I eigen -o EIG
```

2. Run EIG with an input file:
```bash
./EIG <input_file>
```

### Running the KL Algorithm

#### Single-Thread Version
1. Compile:
```bash
g++ cKL.cpp -std=c++17 -I eigen -o KL_single_thread
```

2. Run:
```bash
# Standard execution
./KL_single_thread <input_file>

# Using EIG solution as initial partition
./KL_single_thread <input_file> -EIG
```

#### Multi-Thread Version
1. Compile:
```bash
g++ cKL.cpp -std=c++17 -fopenmp -o KL_multi_thread_omp
```

2. Run:
```bash
# Standard execution
./KL_multi_thread_omp <input_file>

# Using EIG solution as initial partition
./KL_multi_thread_omp <input_file> -EIG
```

### Input File Format
The input file should follow this format:
```
<number_of_nets> <number_of_nodes>
<net_1_nodes>
<net_2_nodes>
...
<net_n_nodes>
```

Example:
```
147 149
1 120
2 148
3 119
...
```

## Performance Metrics

### EIG Results

| Benchmark | Total Cells | Left Partition | Right Partition | CutSize | RatioCut | Runtime (s) |
|-----------|-------------|----------------|-----------------|---------|-----------|-------------|
| Fract     | 149         | 75            | 74             | 27.75   | 5e-3      | 0.5         |
| Industry2 | 12637       | 6319          | 6318           | 1153.63 | 2.89e-5   | 432         |
| Ibm01     | 12752       | 6376          | 6376           | 740.948 | 1.82e-5   | 31          |
| Ibm10     | 69429       | 34715         | 34714          | 5229.01 | 4.34e-6   | 3227        |
| Ibm18     | 210613      | 105307        | 105306         | 4038.83 | 3.6e-7    | 37758       |

### EIG+KL Hybrid Results

| Benchmark | Total Cells | Left Partition | Right Partition | CutSize | RatioCut | Runtime (s) |
|-----------|-------------|----------------|-----------------|---------|-----------|-------------|
| Fract     | 149         | 75            | 74             | 21.5    | 5e-3      | 0.01        |
| Industry2 | 12637       | 6319          | 6318           | 957.714 | 2.89e-5   | 24.159      |
| Ibm01     | 12752       | 6376          | 6376           | 501.365 | 1.82e-5   | 11.625      |
| Ibm10     | 69429       | 34715         | 34714          | 4048.24 | 4.34e-6   | 617.88      |
| Ibm18     | 210613      | 105307        | 105306         | 3245.44 | 3.6e-7    | 3178.55     |

## Key Findings

1. The hybrid EIG+KL approach shows significant improvements:
   - Cut size reduction of >20% across benchmarks
   - Runtime improvements up to 90% for large circuits
   - Better scalability for larger benchmarks

2. Sparse matrix optimizations proved crucial:
   - Reduced memory usage for large circuits
   - Improved computation efficiency
   - Better handling of sparse connectivity patterns

3. Performance characteristics:
   - KL heuristics improve timing for most benchmarks
   - Hybrid computation provides better results consistently
   - Large benchmark partitioning requires significant computational resources

## Output Directory Structure

After running the algorithms, results are stored in the following structure:
```
results/
├── <input_file>_out.txt                    # EIG results
├── <input_file>_KL_CutSize_output.txt      # KL results
└── <input_file>_KL_CutSize_EIG_output.txt  # Hybrid results
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Eigen library for efficient matrix computations
- OpenMP for parallel processing support
- IBM benchmark suite for testing and validation
