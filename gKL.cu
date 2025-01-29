#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>

// Debug macros
#define DEBUG_PRINT(msg) std::cout << "DEBUG: " << msg << std::endl
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// Forward declarations for CUDA kernels
__global__ void calculateNodeGainsKernel(const int* rowPtrs, const int* colIndices,
                                       const float* values, const int* leftNodes,
                                       const int* rightNodes, float* nodeGains,
                                       int leftSize, int rightSize, int nodeNum,
                                       int valuesSize);

__global__ void calculateCutSizeKernel(const int* rowPtrs, const int* colIndices,
                                     const float* values, const int* leftNodes,
                                     const int* rightNodes, float* partialSums,
                                     int leftSize, int rightSize, int nodeNum);

// Main data structures
struct GPUSparseMatrix {
    unsigned int nodeNum;
    thrust::device_vector<int> rowPtrs;      // CSR format row pointers
    thrust::device_vector<int> colIndices;   // CSR format column indices
    thrust::device_vector<float> values;     // CSR format values
    thrust::device_vector<int> split[2];     // Partitions
    thrust::device_vector<int> remain[2];    // Remaining nodes in each partition
    thrust::device_vector<float> nodeGains;  // Node gains

    explicit GPUSparseMatrix(unsigned int size);
};

struct BuildHelper {
    std::vector<std::vector<std::pair<int, float>>> adjacencyLists;
    explicit BuildHelper(unsigned int size);
};

// Function declarations
void debugPrintGPUMemory();
void createDir(const std::string& dirName);
std::string getBaseName(const std::string& path);
void InitializeGPUSparseMatrix(const std::string &filename, GPUSparseMatrix &gpuMat, BuildHelper &builder);
void shuffleGPUSparseMatrix(GPUSparseMatrix &gpuMat);
float calculateCutSize(GPUSparseMatrix& gpuMat);
void validateArraySizes(GPUSparseMatrix& gpuMat);
void KL(GPUSparseMatrix& gpuMat);

// Error checking template
template<typename T>
void check(T err, const char* const func, const char* const file, const int line);

// Global variables
bool EIG_init;
std::string EIG_file;
std::string fout_name;


// Constructor implementations
GPUSparseMatrix::GPUSparseMatrix(unsigned int size) : nodeNum(size) {
    try {
        nodeGains.resize(size, 0.0f);
    } catch (const std::runtime_error& e) {
        std::cerr << "GPU memory allocation failed: " << e.what() << std::endl;
        exit(1);
    }
}

BuildHelper::BuildHelper(unsigned int size) : adjacencyLists(size) {}

// Debug utilities
void debugPrintGPUMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory Status:\n"
              << "  Total Memory: " << (total_mem / 1024.0 / 1024.0) << " MB\n"
              << "  Free Memory:  " << (free_mem / 1024.0 / 1024.0) << " MB\n"
              << "  Used Memory:  " << ((total_mem - free_mem) / 1024.0 / 1024.0) << " MB\n";
}

// File system utilities
void createDir(const std::string& dirName) {
    std::filesystem::create_directories(dirName);
}

std::string getBaseName(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}

// Error checking template implementation
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// CUDA kernel implementations
__device__ float getEdgeWeightDevice(const int* rowPtrs, const int* colIndices, 
                                   const float* values, int node1, int node2) {
    if (node1 > node2) {
        int temp = node1;
        node1 = node2;
        node2 = temp;
    }
    
    int start = rowPtrs[node1];
    int end = rowPtrs[node1 + 1];
    
    for (int i = start; i < end; i++) {
        if (colIndices[i] == node2) {
            return values[i];
        }
    }
    return 0.0f;
}

__global__ void calculateNodeGainsKernel(const int* rowPtrs, const int* colIndices,
                                       const float* values, const int* leftNodes,
                                       const int* rightNodes, float* nodeGains,
                                       int leftSize, int rightSize, int nodeNum,
                                       int valuesSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeNum) return;

    int node = idx;
    float external = 0.0f;
    float internal = 0.0f;

    for (int j = 0; j < leftSize + rightSize; j++) {
        int neighbor = j < leftSize ? leftNodes[j] : rightNodes[j - leftSize];
        bool inLeft = j < leftSize;
        
        if (neighbor != node) {
            float weight = getEdgeWeightDevice(rowPtrs, colIndices, values, node, neighbor);
            if (weight > 0.0f) {
                if (inLeft) internal += weight;
                else external += weight;
            }
        }
    }

    nodeGains[idx] = external - internal;
}

__global__ void calculateCutSizeKernel(const int* rowPtrs, const int* colIndices, 
                                     const float* values, const int* leftNodes, 
                                     const int* rightNodes, float* partialSums,
                                     int leftSize, int rightSize, int nodeNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leftSize) return;

    int node = leftNodes[idx];
    float localSum = 0.0f;

    for (int j = 0; j < rightSize; j++) {
        int rightNode = rightNodes[j];
        float weight = getEdgeWeightDevice(rowPtrs, colIndices, values, node, rightNode);
        localSum += weight;
    }

    partialSums[idx] = localSum;
}

// Matrix initialization and utilities
void InitializeGPUSparseMatrix(const std::string& filename, GPUSparseMatrix& gpuMat, BuildHelper& builder) {
    std::cout << "\n============= Reading Input File ==============\n";
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    // Read circuit statistics
    std::string line;
    getline(fin, line);
    long int netsNum, nodesNum;
    std::stringstream(line) >> netsNum >> nodesNum;

    std::cout << "Circuit Statistics\n"
              << "  - Total Nets : " << netsNum << "\n"
              << "  - Total Nodes: " << nodesNum << "\n";

    // Initialize builder and process nets
    builder = BuildHelper(nodesNum);
    long int nonZeroElements = 0;
    std::vector<int> nodes;
    nodes.reserve(1000);

    // Process each net
    for (int i = 0; i < netsNum; i++) {
        getline(fin, line);
        std::stringstream ss(line);
        nodes.clear();

        int node;
        while (ss >> node) {
            nodes.push_back(node - 1);  // Convert to 0-based indexing
        }

        float weight = 1.0f / (nodes.size() - 1);

        // Add edges between all pairs in the net
        for (size_t j = 0; j < nodes.size(); j++) {
            for (size_t k = j + 1; k < nodes.size(); k++) {
                int node1 = nodes[j];
                int node2 = nodes[k];

                if (node1 > node2) std::swap(node1, node2);

                builder.adjacencyLists[node1].push_back({node2, weight});
                nonZeroElements++;
            }
        }
    }

    fin.close();

    // Convert to CSR format
    std::vector<int> rowPtrs(nodesNum + 1);
    std::vector<int> colIndices;
    std::vector<float> values;
    colIndices.reserve(nonZeroElements);
    values.reserve(nonZeroElements);

    rowPtrs[0] = 0;
    for (int i = 0; i < nodesNum; i++) {
        auto& adjList = builder.adjacencyLists[i];
        std::sort(adjList.begin(), adjList.end());
        rowPtrs[i + 1] = rowPtrs[i] + adjList.size();
        for (const auto& [col, val] : adjList) {
            colIndices.push_back(col);
            values.push_back(val);
        }
    }

    // Transfer to GPU
    gpuMat.nodeNum = nodesNum;
    gpuMat.rowPtrs = thrust::device_vector<int>(rowPtrs.begin(), rowPtrs.end());
    gpuMat.colIndices = thrust::device_vector<int>(colIndices.begin(), colIndices.end());
    gpuMat.values = thrust::device_vector<float>(values.begin(), values.end());
}

void shuffleGPUSparseMatrix(GPUSparseMatrix& gpuMat) {
    std::vector<int> leftNodes, rightNodes;
    
    if (EIG_init) {
        // Initialize from EIG file if specified
        std::ifstream fEIG(EIG_file);
        if (!fEIG.is_open()) {
            std::cerr << "Error: EIG file not found" << std::endl;
            exit(1);
        }
        
        std::string line;
        getline(fEIG, line);  // Skip first two lines
        getline(fEIG, line);
        
        while (getline(fEIG, line)) {
            int node, split_side;
            double weight;
            std::stringstream(line) >> node >> split_side >> weight;
            
            if (split_side == 0) leftNodes.push_back(node);
            else rightNodes.push_back(node);
        }
        fEIG.close();
    } else {
        // Random partitioning
        std::vector<int> nodes(gpuMat.nodeNum);
        std::iota(nodes.begin(), nodes.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(nodes.begin(), nodes.end(), gen);
        
        size_t mid = gpuMat.nodeNum / 2;
        leftNodes.assign(nodes.begin(), nodes.begin() + mid);
        rightNodes.assign(nodes.begin() + mid, nodes.end());
    }

    // Transfer to GPU
    gpuMat.split[0] = thrust::device_vector<int>(leftNodes.begin(), leftNodes.end());
    gpuMat.split[1] = thrust::device_vector<int>(rightNodes.begin(), rightNodes.end());
    gpuMat.remain[0] = gpuMat.split[0];
    gpuMat.remain[1] = gpuMat.split[1];
}

float calculateCutSize(GPUSparseMatrix& gpuMat) {
    int leftSize = gpuMat.remain[0].size();
    int rightSize = gpuMat.remain[1].size();
    
    // Allocate memory for partial sums
    thrust::device_vector<float> d_partialSums(leftSize);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (leftSize + threadsPerBlock - 1) / threadsPerBlock;
    
    calculateCutSizeKernel<<<numBlocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(gpuMat.rowPtrs.data()),
        thrust::raw_pointer_cast(gpuMat.colIndices.data()),
        thrust::raw_pointer_cast(gpuMat.values.data()),
        thrust::raw_pointer_cast(gpuMat.remain[0].data()),
        thrust::raw_pointer_cast(gpuMat.remain[1].data()),
        thrust::raw_pointer_cast(d_partialSums.data()),
        leftSize, rightSize, gpuMat.nodeNum
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Sum up partial results
    return thrust::reduce(d_partialSums.begin(), d_partialSums.end(), 0.0f);
}

void validateArraySizes(GPUSparseMatrix& gpuMat) {
    if (gpuMat.nodeNum == 0) {
        std::cerr << "Error: Empty matrix" << std::endl;
        exit(1);
    }
    if (gpuMat.split[0].size() + gpuMat.split[1].size() != gpuMat.nodeNum) {
        std::cerr << "Error: Invalid partition sizes" << std::endl;
        exit(1);
    }
}

// Node swapping helper
void swapNodes(thrust::device_vector<int>& remain0,
              thrust::device_vector<int>& remain1,
              thrust::device_vector<int>& split0,
              thrust::device_vector<int>& split1,
              int idx1, int idx2) {
    int node1 = remain0[idx1];
    int node2 = remain1[idx2];

    // Update remain vectors
    remain0.erase(remain0.begin() + idx1);
    remain1.erase(remain1.begin() + idx2);

    // Update split vectors
    auto it1 = thrust::find(split0.begin(), split0.end(), node1);
    auto it2 = thrust::find(split1.begin(), split1.end(), node2);
    
    if (it1 != split0.end()) *it1 = node2;
    if (it2 != split1.end()) *it2 = node1;
}

//... [Previous implementation remains the same until the KL function]

void KL(GPUSparseMatrix& gpuMat) {
    DEBUG_PRINT("Starting KL Algorithm");
    debugPrintGPUMemory();
    
    shuffleGPUSparseMatrix(gpuMat);
    validateArraySizes(gpuMat);
    
    // Open output file
    std::ofstream fout(fout_name);
    if (!fout.is_open()) {
        std::cerr << "Error: Cannot open output file" << std::endl;
        exit(1);
    }

    // Initialize algorithm variables
    int iteration = 0;
    int terminate = 0;
    int terminateLimit = static_cast<int>(log2(gpuMat.nodeNum)) + 5;
    float globalMinCutSize = std::numeric_limits<float>::max();
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    // Calculate initial cut size
    float cutSize = calculateCutSize(gpuMat);
    float minCutSize = cutSize;
    float initialCutSize = cutSize;

    // Initialize CUDA configuration
    int threadsPerBlock = 256;
    int numBlocks = (gpuMat.nodeNum + threadsPerBlock - 1) / threadsPerBlock;

    // Print initial information
    std::cout << "\nInitial Partition Information:\n"
              << "  - Left partition size: " << gpuMat.split[0].size() << "\n"
              << "  - Right partition size: " << gpuMat.split[1].size() << "\n"
              << "  - Initial cut size: " << cutSize << "\n\n";
    
    fout << "0\t" << cutSize << "\t0" << std::endl;

    // Initialize node gains
    gpuMat.nodeGains.resize(gpuMat.nodeNum);
    thrust::fill(gpuMat.nodeGains.begin(), gpuMat.nodeGains.end(), 0.0f);

    // Print iteration header
    std::cout << "\n============================== KL Iterations ==============================\n"
              << "---------------------------------------------------------------------------\n"
              << std::setw(10) << "Iteration" 
              << std::setw(15) << "Cut Size" 
              << std::setw(20) << "Gain (delta)" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Improvement" << "\n"
              << std::string(75, '-') << "\n";

    // Main KL loop
    while (!gpuMat.remain[0].empty() && !gpuMat.remain[1].empty()) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Calculate node gains
        calculateNodeGainsKernel<<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(gpuMat.rowPtrs.data()),
            thrust::raw_pointer_cast(gpuMat.colIndices.data()),
            thrust::raw_pointer_cast(gpuMat.values.data()),
            thrust::raw_pointer_cast(gpuMat.split[0].data()),
            thrust::raw_pointer_cast(gpuMat.split[1].data()),
            thrust::raw_pointer_cast(gpuMat.nodeGains.data()),
            gpuMat.split[0].size(),
            gpuMat.split[1].size(),
            gpuMat.nodeNum,
            gpuMat.values.size()
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Find maximum gain pair
        thrust::device_vector<float> maxGains(gpuMat.remain[0].size());
        thrust::device_vector<float> minGains(gpuMat.remain[1].size());

        // Transform node indices to their gains
        auto gains_ptr = thrust::raw_pointer_cast(gpuMat.nodeGains.data());
        thrust::transform(
            gpuMat.remain[0].begin(), gpuMat.remain[0].end(),
            maxGains.begin(),
            [gains_ptr] __device__ (int node) { return gains_ptr[node]; }
        );
        thrust::transform(
            gpuMat.remain[1].begin(), gpuMat.remain[1].end(),
            minGains.begin(),
            [gains_ptr] __device__ (int node) { return gains_ptr[node]; }
        );

        // Find best nodes to swap
        auto maxIt = thrust::max_element(maxGains.begin(), maxGains.end());
        auto minIt = thrust::min_element(minGains.begin(), minGains.end());
        
        int maxIdx = maxIt - maxGains.begin();
        int minIdx = minIt - minGains.begin();

        if (maxIdx >= 0 && minIdx >= 0) {
            int node1 = gpuMat.remain[0][maxIdx];
            int node2 = gpuMat.remain[1][minIdx];

            // Get edge weight between selected nodes
            thrust::host_vector<int> h_rowPtrs = gpuMat.rowPtrs;
            thrust::host_vector<int> h_colIndices = gpuMat.colIndices;
            thrust::host_vector<float> h_values = gpuMat.values;

            float edgeWeight = 0.0f;
            int start = h_rowPtrs[std::min(node1, node2)];
            int end = h_rowPtrs[std::min(node1, node2) + 1];
            
            for (int i = start; i < end; i++) {
                if (h_colIndices[i] == std::max(node1, node2)) {
                    edgeWeight = h_values[i];
                    break;
                }
            }

            // Calculate gain and update cut size
            float gain = *maxIt - *minIt - 2.0f * edgeWeight;
            cutSize -= gain;
            minCutSize = std::min(minCutSize, cutSize);

            // Perform the swap
            swapNodes(gpuMat.remain[0], gpuMat.remain[1], 
                     gpuMat.split[0], gpuMat.split[1], 
                     maxIdx, minIdx);

            // Update iteration statistics
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            iteration++;
            float improvement = 100.0f * (1.0f - cutSize/initialCutSize);

            // Print iteration results
            std::cout << std::setw(8) << iteration 
                     << std::setw(17) << std::fixed << std::setprecision(2) << cutSize 
                     << std::setw(18) << std::fixed << std::setprecision(2) << gain 
                     << std::setw(15) << duration.count()
                     << std::setw(15) << std::fixed << std::setprecision(2) << improvement << "%\n";

            fout << iteration << "\t" << cutSize << "\t" << gain << std::endl;

            // Check termination conditions
            if (gain <= 0.0f) {
                if (++terminate > terminateLimit) break;
            } else {
                terminate = 0;
            }
        } else {
            break;
        }
    }

    // Calculate and print final results
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end_time - total_start_time);
    
    globalMinCutSize = std::min(globalMinCutSize, minCutSize);

    std::cout << "\n\n=============== Final Results =================\n"
              << std::left << std::setw(24) << "Total iterations" << ": " << iteration << "\n"
              << std::left << std::setw(24) << "Initial cut size" << ": " << std::fixed << std::setprecision(2) << initialCutSize << "\n"
              << std::left << std::setw(24) << "Best cut size achieved" << ": " << globalMinCutSize << "\n"
              << std::left << std::setw(24) << "Overall improvement" << ": " 
              << std::fixed << std::setprecision(2) << 100.0f * (1.0f - globalMinCutSize/initialCutSize) << "%\n"
              << std::left << std::setw(24) << "Total runtime" << ": " << total_duration.count() << " seconds\n";

    fout.close();
}

// Main function
int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_file> [-EIG]" << std::endl;
        return 1;
    }

    try {
        DEBUG_PRINT("Program started");
        DEBUG_PRINT("Initializing CUDA device");
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        debugPrintGPUMemory();

        // Set up file paths
        std::string input_file = argv[1];
        std::string base_name = getBaseName(input_file);
        fout_name = "results/" + base_name + "_KL_CutSize_output.txt";

        if (argc == 3 && std::string(argv[2]) == "-EIG") {
            EIG_init = true;
            EIG_file = "pre_saved_EIG/" + base_name + "_out.txt";
            fout_name = "results/" + base_name + "_KL_CutSize_EIG_output.txt";
            DEBUG_PRINT("EIG mode enabled");
        }

        // Create output directories
        DEBUG_PRINT("Creating directories");
        createDir("results");
        createDir("pre_saved_EIG");

        // Initialize matrix structures
        DEBUG_PRINT("Creating initial GPU matrix");
        GPUSparseMatrix gpuMat(1);
        BuildHelper builder(1);

        // Initialize and run KL algorithm
        DEBUG_PRINT("Initializing sparse matrix");
        InitializeGPUSparseMatrix(input_file, gpuMat, builder);

        DEBUG_PRINT("Starting KL algorithm");
        KL(gpuMat);

        // Cleanup
        DEBUG_PRINT("Cleaning up GPU vectors");
        gpuMat.rowPtrs.clear();
        gpuMat.rowPtrs.shrink_to_fit();
        gpuMat.colIndices.clear();
        gpuMat.colIndices.shrink_to_fit();
        gpuMat.values.clear();
        gpuMat.values.shrink_to_fit();
        gpuMat.nodeGains.clear();
        gpuMat.nodeGains.shrink_to_fit();
        gpuMat.split[0].clear();
        gpuMat.split[0].shrink_to_fit();
        gpuMat.split[1].clear();
        gpuMat.split[1].shrink_to_fit();
        gpuMat.remain[0].clear();
        gpuMat.remain[0].shrink_to_fit();
        gpuMat.remain[1].clear();
        gpuMat.remain[1].shrink_to_fit();

        // Final cleanup
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        DEBUG_PRINT("Cleaning up CUDA resources");
        CHECK_CUDA_ERROR(cudaDeviceReset());
        DEBUG_PRINT("Program completed successfully");
        
    } catch (const thrust::system_error& e) {
        std::cerr << "Thrust error at " << __FILE__ << ":" << __LINE__ << "\n"
                  << e.what() << std::endl;
        debugPrintGPUMemory();
        return 1;
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << "\n"
                  << e.what() << std::endl;
        debugPrintGPUMemory();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error occurred at " << __FILE__ << ":" << __LINE__ << "\n"
                  << e.what() << std::endl;
        debugPrintGPUMemory();
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred at " << __FILE__ << ":" << __LINE__ << std::endl;
        debugPrintGPUMemory();
        return 1;
    }

    return 0;
}