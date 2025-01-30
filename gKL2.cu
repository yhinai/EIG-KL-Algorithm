#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <limits>
#include <chrono>
#include <sys/stat.h>
#include <filesystem>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thread>
#include <iomanip>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------
// 1. GLOBALS & CONSTANTS
// ---------------------------------------------------------------------

bool EIG_init = false;
float globalMin = std::numeric_limits<float>::max();
const float HOST_EPSILON = 1e-6f;
const int MAX_POWER_ITERATIONS = 1000;
const float POWER_CONVERGENCE = 1e-6f;

// Device-side constants
__constant__ int d_terminateLimit;
__constant__ float d_epsilon;

// ---------------------------------------------------------------------
// 2. STRUCTURE DEFINITIONS
// ---------------------------------------------------------------------

struct alignas(16) sparseMatrix {
    unsigned int nodeNum;
    
    // Original members remain the same
    std::vector<std::vector<int>> Nodes;
    std::vector<std::vector<float>> Weights;
    std::vector<int> split[2];
    std::vector<int> remain[2];
    alignas(16) std::vector<int> adjacencyOffsets;
    alignas(16) std::vector<int> adjacencyIndices;
    alignas(16) std::vector<float> adjacencyWeights;
    
    // Device pointers for original data
    int* d_adjacencyOffsets = nullptr;
    int* d_adjacencyIndices = nullptr;
    float* d_adjacencyWeights = nullptr;
    
    // New members for sparse Laplacian
    int* d_laplacianOffsets = nullptr;
    int* d_laplacianIndices = nullptr;
    float* d_laplacianValues = nullptr;
    
    // EIG vectors
    float* d_eigenvector = nullptr;
    float* d_tempVector = nullptr;
};

// Sparse matrix-vector multiplication kernel
__global__ void sparseMVKernel(
    const int* __restrict__ rowOffsets,
    const int* __restrict__ colIndices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int numRows,
    float shift
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        const int start = rowOffsets[row];
        const int end = rowOffsets[row + 1];
        float sum = 0.0f;
        
        // Compute dot product for this row
        for (int i = start; i < end; i++) {
            sum += values[i] * x[colIndices[i]];
        }
        
        // Apply shift-and-invert transformation
        y[row] = x[row] - sum / shift;
    }
}


// ---------------------------------------------------------------------
// 3. ERROR CHECKING MACRO
// ---------------------------------------------------------------------

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(err); \
    } \
}

// ---------------------------------------------------------------------
// 4. EIG CUDA KERNELS
// ---------------------------------------------------------------------

// Kernel to construct Laplacian matrix
__global__ void constructLaplacianKernel(
    const int* __restrict__ adjacencyOffsets,
    const int* __restrict__ adjacencyIndices,
    const float* __restrict__ adjacencyWeights,
    float* __restrict__ laplacianMatrix,
    int nodeNum
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nodeNum) {
        float diagSum = 0.0f;
        int start = adjacencyOffsets[row];
        int end = adjacencyOffsets[row + 1];
        
        // Initialize row to zero
        for (int col = 0; col < nodeNum; col++) {
            laplacianMatrix[row * nodeNum + col] = 0.0f;
        }
        
        // Fill off-diagonal elements and compute diagonal sum
        for (int e = start; e < end; e++) {
            int col = adjacencyIndices[e];
            float weight = -2.0f * adjacencyWeights[e] / 
                          static_cast<float>(end - start); // Normalize by degree
            laplacianMatrix[row * nodeNum + col] = weight;
            diagSum -= weight;
        }
        
        // Set diagonal element
        laplacianMatrix[row * nodeNum + row] = diagSum;
    }
}

// Compute norm kernel
__global__ void computeNormKernel(
    const float* __restrict__ vector,
    float* __restrict__ norm,
    int nodeNum
) {
    __shared__ float s_sum[256];
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    s_sum[tid] = 0.0f;
    
    // Compute partial sums
    while (idx < nodeNum) {
        s_sum[tid] += vector[idx] * vector[idx];
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(norm, s_sum[0]);
    }
}

// Modified normalize vector kernel
__global__ void normalizeVectorKernel(
    float* __restrict__ vector,
    const float* __restrict__ norm,
    int nodeNum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sqrtNorm = sqrtf(*norm);
    
    if (idx < nodeNum && sqrtNorm > 0.0f) {
        vector[idx] /= sqrtNorm;
    }
}

// Modified power iteration kernel
__global__ void powerIterationKernel(
    const float* __restrict__ laplacianMatrix,
    const float* __restrict__ vector,
    float* __restrict__ result,
    float shift,
    int nodeNum
) {
    __shared__ float s_vector[256];
    const int tid = threadIdx.x;
    const int row = blockIdx.x * blockDim.x + tid;
    
    float sum = 0.0f;
    
    // Process row in tiles
    for (int tile = 0; tile < (nodeNum + blockDim.x - 1) / blockDim.x; ++tile) {
        // Load tile of vector into shared memory
        const int tileOffset = tile * blockDim.x;
        if (tileOffset + tid < nodeNum) {
            s_vector[tid] = vector[tileOffset + tid];
        }
        __syncthreads();
        
        // Compute partial sum for this tile
        if (row < nodeNum) {
            const int tileSize = min(blockDim.x, nodeNum - tileOffset);
            for (int j = 0; j < tileSize; ++j) {
                sum += laplacianMatrix[row * nodeNum + (tileOffset + j)] * s_vector[j];
            }
        }
        __syncthreads();
    }
    
    // Write result with shift-and-invert transformation
    if (row < nodeNum) {
        result[row] = vector[row] - sum / shift;
    }
}

void writeEIGResults(const std::string& baseName,
                    const std::vector<float>& eigenvalues,
                    const std::vector<float>& eigenvector,
                    float medianValue,
                    int nodeNum) {
    std::string outfile = "pre_saved_EIG/" + baseName + "_out.txt";
    std::ofstream fout(outfile);
    if (!fout.is_open()) {
        std::cerr << "Error opening output file: " << outfile << std::endl;
        return;
    }
    
    // Write first eigenvalue
    fout << std::setprecision(12) << eigenvalues[0] << std::endl;
    
    // Write median value
    fout << std::setprecision(12) << medianValue << std::endl;
    
    // Write node assignments
    for (int i = 0; i < nodeNum; i++) {
        fout << i << "\t" << (medianValue > eigenvector[i] ? 1 : 0) << "\t"
             << std::setprecision(12) << eigenvector[i] << std::endl;
    }
    
    fout.close();
    std::cout << "EIG results written to: " << outfile << std::endl;
}

void computeEigenpartition(sparseMatrix& spMat) {
    std::cout << "\nComputing EIG partitioning...\n";
    const int nodeNum = spMat.nodeNum;
    
    // First, construct sparse Laplacian matrix in CPU
    std::vector<int> laplacianOffsets(nodeNum + 1, 0);
    std::vector<int> laplacianIndices;
    std::vector<float> laplacianValues;
    
    // Count non-zeros per row first
    #pragma omp parallel for
    for (int i = 0; i < nodeNum; i++) {
        laplacianOffsets[i + 1] = spMat.Nodes[i].size() + 1; // +1 for diagonal
    }
    
    // Compute offsets
    for (int i = 0; i < nodeNum; i++) {
        laplacianOffsets[i + 1] += laplacianOffsets[i];
    }
    
    // Allocate arrays for indices and values
    size_t totalNnz = laplacianOffsets[nodeNum];
    laplacianIndices.resize(totalNnz);
    laplacianValues.resize(totalNnz);
    
    // Fill the sparse Laplacian matrix
    #pragma omp parallel for
    for (int i = 0; i < nodeNum; i++) {
        int offset = laplacianOffsets[i];
        float diagSum = 0.0f;
        
        // Off-diagonal elements
        for (size_t j = 0; j < spMat.Nodes[i].size(); j++) {
            int col = spMat.Nodes[i][j];
            float weight = -2.0f * spMat.Weights[i][j] / 
                          static_cast<float>(spMat.Nodes[i].size());
            
            laplacianIndices[offset] = col;
            laplacianValues[offset] = weight;
            diagSum -= weight;
            offset++;
        }
        
        // Diagonal element
        laplacianIndices[offset] = i;
        laplacianValues[offset] = diagSum;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&spMat.d_laplacianOffsets, (nodeNum + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&spMat.d_laplacianIndices, totalNnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&spMat.d_laplacianValues, totalNnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spMat.d_eigenvector, nodeNum * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spMat.d_tempVector, nodeNum * sizeof(float)));
    
    // Copy Laplacian to device
    CHECK_CUDA(cudaMemcpy(spMat.d_laplacianOffsets, laplacianOffsets.data(),
                         (nodeNum + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(spMat.d_laplacianIndices, laplacianIndices.data(),
                         totalNnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(spMat.d_laplacianValues, laplacianValues.data(),
                         totalNnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize random eigenvector
    std::vector<float> initialVector(nodeNum);
    srand(42);
    for (int i = 0; i < nodeNum; i++) {
        initialVector[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    CHECK_CUDA(cudaMemcpy(spMat.d_eigenvector, initialVector.data(),
                         nodeNum * sizeof(float), cudaMemcpyHostToDevice));
    
    float* d_norm;
    CHECK_CUDA(cudaMalloc(&d_norm, sizeof(float)));
    
    dim3 blockDim(256);
    dim3 gridDim(min((nodeNum + blockDim.x - 1) / blockDim.x, 65535));
    
    float shift = 2.0f;
    float prevNorm = 0.0f;
    
    // Power iteration
    std::cout << "Starting power iteration...\n";
    for (int iter = 0; iter < MAX_POWER_ITERATIONS; iter++) {
        // Reset norm
        float zero = 0.0f;
        CHECK_CUDA(cudaMemcpy(d_norm, &zero, sizeof(float), cudaMemcpyHostToDevice));
        
        sparseMVKernel<<<gridDim, blockDim>>>(
            spMat.d_laplacianOffsets,
            spMat.d_laplacianIndices,
            spMat.d_laplacianValues,
            spMat.d_eigenvector,
            spMat.d_tempVector,
            nodeNum,
            shift
        );
        
        // Compute norm
        computeNormKernel<<<gridDim, blockDim>>>(
            spMat.d_tempVector,
            d_norm,
            nodeNum
        );
        
        // Normalize
        normalizeVectorKernel<<<gridDim, blockDim>>>(
            spMat.d_tempVector,
            d_norm,
            nodeNum
        );
        
        // Check convergence
        float norm;
        CHECK_CUDA(cudaMemcpy(&norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost));
        
        if (std::abs(norm - prevNorm) < POWER_CONVERGENCE && iter > 100) {
            std::cout << "Converged after " << iter << " iterations\n";
            break;
        }
        prevNorm = norm;
        
        // Swap pointers
        float* tempPtr = spMat.d_eigenvector;
        spMat.d_eigenvector = spMat.d_tempVector;
        spMat.d_tempVector = tempPtr;
        
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", norm diff: " 
                      << std::abs(norm - prevNorm) << std::endl;
        }
    }
    
    // Get final eigenvector
    std::vector<float> eigenvector(nodeNum);
    CHECK_CUDA(cudaMemcpy(eigenvector.data(), spMat.d_eigenvector,
                         nodeNum * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute median and partition
    std::vector<float> sortedEig = eigenvector;
    std::sort(sortedEig.begin(), sortedEig.end());
    float median = sortedEig[nodeNum / 2];
    
    // Create partitions
    spMat.split[0].clear();
    spMat.split[1].clear();
    spMat.remain[0].clear();
    spMat.remain[1].clear();
    
    for (int i = 0; i < nodeNum; i++) {
        if (median > eigenvector[i]) {
            spMat.split[0].push_back(i);
            spMat.remain[0].push_back(i);
        } else {
            spMat.split[1].push_back(i);
            spMat.remain[1].push_back(i);
        }
    }
    
    std::cout << "Initial partition sizes: " << spMat.split[0].size() 
              << " and " << spMat.split[1].size() << "\n";
    
    // Cleanup
    CHECK_CUDA(cudaFree(spMat.d_laplacianOffsets));
    CHECK_CUDA(cudaFree(spMat.d_laplacianIndices));
    CHECK_CUDA(cudaFree(spMat.d_laplacianValues));
    CHECK_CUDA(cudaFree(spMat.d_eigenvector));
    CHECK_CUDA(cudaFree(spMat.d_tempVector));
    CHECK_CUDA(cudaFree(d_norm));
    
    spMat.d_laplacianOffsets = nullptr;
    spMat.d_laplacianIndices = nullptr;
    spMat.d_laplacianValues = nullptr;
    spMat.d_eigenvector = nullptr;
    spMat.d_tempVector = nullptr;
}

void shuffleSparceMatrix(sparseMatrix& spMat) {
    spMat.split[0].clear();
    spMat.split[1].clear();
    spMat.remain[0].clear();
    spMat.remain[1].clear();
    
    if (EIG_init) {
        computeEigenpartition(spMat);
        return;
    }
    
    // Random partitioning if not using EIG
    std::vector<int> all;
    all.reserve(spMat.nodeNum);
    for (unsigned int i = 0; i < spMat.nodeNum; i++) {
        all.push_back(i);
    }
    std::random_shuffle(all.begin(), all.end());
    
    unsigned int half = spMat.nodeNum / 2;
    for (unsigned int i = 0; i < half; i++) {
        spMat.split[0].push_back(all[i]);
        spMat.remain[0].push_back(all[i]);
    }
    for (unsigned int i = half; i < spMat.nodeNum; i++) {
        spMat.split[1].push_back(all[i]);
        spMat.remain[1].push_back(all[i]);
    }
}

// ---------------------------------------------------------------------
// 5. GPU KERNEL: connectionsKernel
//    out[i] = E - I for remain[i],
//    E = external (sum of edges to other partition)
//    I = internal (sum of edges to own partition)
// ---------------------------------------------------------------------

// Define the kernels for the KL Algorithm on GPU
template<int BLOCK_SIZE>
__global__ void connectionsKernel(
    const int* __restrict__ adjacencyOffsets,
    const int* __restrict__ adjacencyIndices,
    const float* __restrict__ adjacencyWeights,
    const int* __restrict__ membership,
    const int* __restrict__ d_remain,
    float* __restrict__ d_out,
    int remainSize
) {
    // Only keeping the memory we actually use
    __shared__ int s_membership[BLOCK_SIZE];  
    
    auto block = cg::this_thread_block();
    const int tid = block.thread_rank();
    const int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx < remainSize) {
        const int node = d_remain[idx];
        const int start = adjacencyOffsets[node];
        const int end = adjacencyOffsets[node + 1];
        
        s_membership[tid] = membership[node];
        block.sync();
        
        float E = 0.f, I = 0.f;
        const int mySide = s_membership[tid];
        
        #pragma unroll 4
        for (int e = start; e < end; e++) {
            const int neigh = adjacencyIndices[e];
            const float w = adjacencyWeights[e];
            if (membership[neigh] == mySide) {
                I += w;
            } else {
                E += w;
            }
        }
        
        d_out[idx] = (E - I);
    }
}

// Copy the adjacency matrix to the device
void copyAdjacencyToDevice(sparseMatrix& spMat, cudaStream_t stream) {
    const size_t offsetSize = (spMat.nodeNum + 1) * sizeof(int);
    const size_t totalEdges = spMat.adjacencyOffsets[spMat.nodeNum];
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&spMat.d_adjacencyOffsets, offsetSize));
    CHECK_CUDA(cudaMalloc((void**)&spMat.d_adjacencyIndices, totalEdges * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&spMat.d_adjacencyWeights, totalEdges * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(spMat.d_adjacencyOffsets, spMat.adjacencyOffsets.data(),
                         offsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(spMat.d_adjacencyIndices, spMat.adjacencyIndices.data(),
                         totalEdges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(spMat.d_adjacencyWeights, spMat.adjacencyWeights.data(),
                         totalEdges * sizeof(float), cudaMemcpyHostToDevice));
}

void freeDeviceAdjacency(sparseMatrix& spMat, cudaStream_t stream) {
    if (spMat.d_adjacencyOffsets) {
        CHECK_CUDA(cudaFree(spMat.d_adjacencyOffsets));
        spMat.d_adjacencyOffsets = nullptr;
    }
    if (spMat.d_adjacencyIndices) {
        CHECK_CUDA(cudaFree(spMat.d_adjacencyIndices));
        spMat.d_adjacencyIndices = nullptr;
    }
    if (spMat.d_adjacencyWeights) {
        CHECK_CUDA(cudaFree(spMat.d_adjacencyWeights));
        spMat.d_adjacencyWeights = nullptr;
    }
}


// ---------------------------------------------------------------------
// 6. GPU CONNECTIONS: single call for remain[] nodes
//    This version reuses allocated buffers d_remain, d_out, d_membership
//    so we only do cudaMemcpy each iteration, not cudaMalloc/cudaFree.
// ---------------------------------------------------------------------

void gpuConnections(
    const sparseMatrix& spMat,
    int* d_remain,
    int* d_membership,
    float* d_out,
    const std::vector<int>& remain,
    const std::vector<int>& membershipHost,
    std::vector<float>& out,
    cudaStream_t stream
) {
    const int remainSize = static_cast<int>(remain.size());
    if (remainSize == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    const int gridSize = (remainSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Copy data (remain, membership) to device
    CHECK_CUDA(cudaMemcpyAsync(d_remain, remain.data(),
                              remainSize * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_membership, membershipHost.data(),
                              spMat.nodeNum * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
    
    // Launch the kernel
    connectionsKernel<BLOCK_SIZE><<<gridSize, BLOCK_SIZE, 0, stream>>>(
        spMat.d_adjacencyOffsets,
        spMat.d_adjacencyIndices,
        spMat.d_adjacencyWeights,
        d_membership,
        d_remain,
        d_out,
        remainSize
    );
    
    // Copy the results back to host
    CHECK_CUDA(cudaMemcpyAsync(out.data(), d_out,
                              remainSize * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
}

// ---------------------------------------------------------------------
// 7. HELPER FUNCTIONS
// ---------------------------------------------------------------------

// Helper function to create a directory
void createDir(const std::string& dirName) {
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {
        mkdir(dirName.c_str(), 0755);
    }
}

std::string getBaseName(const std::string& path) {
    std::filesystem::path fp(path);
    return fp.filename().string();
}


// Helper function to build the flattened adjacency list
void buildFlattenedAdjacency(sparseMatrix& spMat) {
    spMat.adjacencyOffsets.resize(spMat.nodeNum + 1, 0);
    
    #pragma omp parallel for
    for (unsigned int i = 0; i < spMat.nodeNum; i++) {
        spMat.adjacencyOffsets[i + 1] = spMat.adjacencyOffsets[i] + spMat.Nodes[i].size();
    }
    
    size_t totalEdges = spMat.adjacencyOffsets[spMat.nodeNum];
    spMat.adjacencyIndices.resize(totalEdges);
    spMat.adjacencyWeights.resize(totalEdges);
    
    #pragma omp parallel for
    for (unsigned int i = 0; i < spMat.nodeNum; i++) {
        int start = spMat.adjacencyOffsets[i];
        for (unsigned int j = 0; j < spMat.Nodes[i].size(); j++) {
            spMat.adjacencyIndices[start + j] = spMat.Nodes[i][j];
            spMat.adjacencyWeights[start + j] = spMat.Weights[i][j];
        }
    }
}

float computeCutSize(const sparseMatrix& spMat, const std::vector<int>& membership) {
    float E = 0.f;
    #pragma omp parallel for reduction(+:E)
    for (auto node0 : spMat.split[0]) {
        const auto& nbrs = spMat.Nodes[node0];
        const auto& wts = spMat.Weights[node0];
        for (size_t j = 0; j < nbrs.size(); j++) {
            if (membership[nbrs[j]] == 1) {
                E += wts[j];
            }
        }
    }
    return E;
}

float nodeConnection(const sparseMatrix& spMat, int a, int b) {
    const auto& nbrs = spMat.Nodes[a];
    const auto& wts = spMat.Weights[a];
    for (size_t i = 0; i < nbrs.size(); i++) {
        if (nbrs[i] == b) return wts[i];
    }
    return 0.f;
}

void swip(sparseMatrix& spMat, std::vector<int>& membership, int num1, int num2) {
    membership[num1] = 1;
    membership[num2] = 0;
    
    auto it1 = std::find(spMat.remain[0].begin(), spMat.remain[0].end(), num1);
    if (it1 != spMat.remain[0].end()) spMat.remain[0].erase(it1);
    
    auto it2 = std::find(spMat.remain[1].begin(), spMat.remain[1].end(), num2);
    if (it2 != spMat.remain[1].end()) spMat.remain[1].erase(it2);
    
    auto s1 = std::find(spMat.split[0].begin(), spMat.split[0].end(), num1);
    if (s1 != spMat.split[0].end()) *s1 = num2;
    
    auto s2 = std::find(spMat.split[1].begin(), spMat.split[1].end(), num2);
    if (s2 != spMat.split[1].end()) *s2 = num1;
}

// ---------------------------------------------------------------------
// 8. MAIN KL IMPLEMENTATION
// ---------------------------------------------------------------------

// Add this helper function for periodic verification
float verifyAndCorrectCutSize(const sparseMatrix& spMat, 
                            const std::vector<int>& membership, 
                            float& currentCutSize,
                            int iteration) {
    // Verify every N iterations (e.g., every 10 iterations)
    const int VERIFY_INTERVAL = 10;
    if (iteration % VERIFY_INTERVAL == 0) {
        float verifiedCutSize = computeCutSize(spMat, membership);
        if (std::abs(verifiedCutSize - currentCutSize) > HOST_EPSILON) {
            // If difference detected, correct it
            currentCutSize = verifiedCutSize;
        }
    }
    return currentCutSize;
}

float calculateGain(const sparseMatrix& spMat,
                   const std::vector<int>& membership,
                   int node1, int node2,
                   float maxGain1, float maxGain2) {
    // Calculate edge weight between nodes
    float edgeWeight = nodeConnection(spMat, node1, node2);
    
    // Use Kahan summation for better numerical stability
    float sum = 0.0f;
    float c = 0.0f;  // compensation
    
    // Add maxGain1
    float y = maxGain1 - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
    
    // Add maxGain2
    y = maxGain2 - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
    
    // Subtract edge weight contribution
    y = -2.0f * edgeWeight - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
    
    return sum;
}


void KL(sparseMatrix& spMat) {
    std::cout << "\n========== Starting Optimized KL Algorithm ==========\n";
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    shuffleSparceMatrix(spMat);
    std::vector<int> membership(spMat.nodeNum, -1);
    for (auto n : spMat.split[0]) membership[n] = 0;
    for (auto n : spMat.split[1]) membership[n] = 1;
    
    float cutSize = computeCutSize(spMat, membership);
    const float initialCutSize = cutSize;
    float bestCut = cutSize;
    
    const int maxRem = std::max(spMat.remain[0].size(), spMat.remain[1].size());
    int* d_remain;
    float* d_out;
    int* d_membership;
    
    CHECK_CUDA(cudaMallocManaged(&d_remain, maxRem * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&d_membership, spMat.nodeNum * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&d_out, maxRem * sizeof(float)));
    
    int iteration = 0;
    int terminate = 0;
    const int terminateLimit = static_cast<int>(std::log2(spMat.nodeNum)) + 5;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nIteration Progress:\n";
    std::cout << std::setw(10) << "Iteration" 
              << std::setw(15) << "Cut Size" 
              << std::setw(18) << "Gain"
              << std::setw(18) << "Time (ms)"
              << std::setw(16) << "Improvement\n";
    
    while (!spMat.remain[0].empty() && !spMat.remain[1].empty()) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        std::vector<float> con_1(spMat.remain[0].size());
        std::vector<float> con_2(spMat.remain[1].size());
        
        gpuConnections(spMat, d_remain, d_membership, d_out,
                    spMat.remain[0], membership, con_1, stream);
        gpuConnections(spMat, d_remain, d_membership, d_out,
                    spMat.remain[1], membership, con_2, stream);
        
        float maxGain1 = -std::numeric_limits<float>::infinity();
        float maxGain2 = -std::numeric_limits<float>::infinity();
        int node1 = -1, node2 = -1;
        
        // Find optimal nodes with stable summation
        for (size_t i = 0; i < con_1.size(); i++) {
            if (con_1[i] > maxGain1) {
                maxGain1 = con_1[i];
                node1 = spMat.remain[0][i];
            }
        }
        
        for (size_t i = 0; i < con_2.size(); i++) {
            if (con_2[i] > maxGain2) {
                maxGain2 = con_2[i];
                node2 = spMat.remain[1][i];
            }
        }
        
        // Calculate gain with improved numerical stability
        float gain = calculateGain(spMat, membership, node1, node2, maxGain1, maxGain2);
        
        // Update cut size with periodic verification
        cutSize -= gain;
        // cutSize = verifyAndCorrectCutSize(spMat, membership, cutSize, iteration);
        bestCut = std::min(bestCut, cutSize);
        
        // Perform swap
        swip(spMat, membership, node1, node2);
        
        // Update termination condition
        if (gain <= HOST_EPSILON) {  // Using host-side epsilon instead of device constant
            if (++terminate > terminateLimit) break;
        } else {
            terminate = 0;
        }
        
        // Update global minimum
        globalMin = std::min(globalMin, cutSize);
        
        if (iteration % 100 == 0) {
            // Print iteration progress every 100 iterations
            auto iterEnd = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(iterEnd - iterStart);
            float improvement = 100.0f * (1.0f - cutSize / initialCutSize);
            
            std::cout << std::setw(8) << iteration 
                  << std::setw(17) << std::fixed << std::setprecision(2) << cutSize 
                  << std::setw(18) << std::fixed << std::setprecision(2) << gain 
                  << std::setw(15) << duration.count()
                  << std::setw(15) << std::fixed << std::setprecision(2) << improvement << "%\n";
        }
        
        iteration++;

    }
    
    // Cleanup and synchronize
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Final verification
    float finalCheck = computeCutSize(spMat, membership);
    if (std::abs(finalCheck - cutSize) > HOST_EPSILON) {  // Using host-side epsilon
        std::cout << "\nWarning: Cut size verification difference detected.\n"
                  << "Incremental: " << cutSize << ", From-scratch: " << finalCheck << std::endl;
        cutSize = finalCheck;
    }
    
    // Print final results
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\n=============== Final Results =================\n";
    std::cout << "Total iterations: " << iteration << "\n";
    std::cout << "Initial cut size: " << std::fixed << std::setprecision(2) << initialCutSize << "\n";
    std::cout << "Best cut size   : " << bestCut << "\n";
    std::cout << "Improvement     : " << std::fixed << std::setprecision(2) 
              << 100.0f * (1.0f - bestCut/initialCutSize) << "%\n";
    std::cout << "Total runtime   : " << totalDuration.count() << " seconds\n";
    
    // Ensure all operations are complete then free memory
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_remain));
    CHECK_CUDA(cudaFree(d_membership));
    CHECK_CUDA(cudaFree(d_out));
}

// ---------------------------------------------------------------------
// 9. GPU INFO AND INITIALIZATION
// ---------------------------------------------------------------------

void printGPUInfo() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "\n================= GPU Info ===================\n";
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Number of multiprocessors: " << prop.multiProcessorCount << "\n";
        
    }
}

void InitializeSparsMatrix(const std::string& filename, sparseMatrix& spMat) {
    std::cout << "\n============= Reading Input File ==============\n";
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    std::getline(fin, line);
    int netsNum = 0, nodesNum = 0;
    {
        std::stringstream ss(line);
        ss >> netsNum >> nodesNum;
    }
    
    std::cout << "Circuit Statistics\n";
    std::cout << "  - Total Nets : " << netsNum << "\n";
    std::cout << "  - Total Nodes: " << nodesNum << "\n";
    
    spMat.nodeNum = nodesNum;
    spMat.Nodes.resize(nodesNum);
    spMat.Weights.resize(nodesNum);
    
    for (auto& node : spMat.Nodes) {
        node.reserve(32);  // Typical average degree
    }
    for (auto& weight : spMat.Weights) {
        weight.reserve(32);
    }
    
    long int nonZeroElements = nodesNum;
    long int numEdges = 0;
    
    #pragma omp parallel for reduction(+:nonZeroElements,numEdges)
    for (int i = 0; i < netsNum; i++) {
        std::string localLine;
        #pragma omp critical
        {
            std::getline(fin, localLine);
        }
        
        std::stringstream ss(localLine);
        std::vector<int> nodes;
        int nd;
        while (ss >> nd) {
            nodes.push_back(nd);
        }
        
        if (nodes.size() < 2) continue;
        float weight = 1.f / float(nodes.size() - 1);
        
        for (size_t j = 0; j < nodes.size(); j++) {
            for (size_t k = j + 1; k < nodes.size(); k++) {
                numEdges++;
                int a = nodes[j] - 1, b = nodes[k] - 1;
                
                #pragma omp critical
                {
                    auto it = std::find(spMat.Nodes[a].begin(), spMat.Nodes[a].end(), b);
                    if (it == spMat.Nodes[a].end()) {
                        spMat.Nodes[a].push_back(b);
                        spMat.Weights[a].push_back(weight);
                        spMat.Nodes[b].push_back(a);
                        spMat.Weights[b].push_back(weight);
                        nonZeroElements += 2;
                    } else {
                        int idxA = it - spMat.Nodes[a].begin();
                        spMat.Weights[a][idxA] += weight;
                        auto it2 = std::find(spMat.Nodes[b].begin(), spMat.Nodes[b].end(), a);
                        int idxB = it2 - spMat.Nodes[b].begin();
                        spMat.Weights[b][idxB] += weight;
                    }
                }
            }
        }
    }
    fin.close();
    
    buildFlattenedAdjacency(spMat);
    
    float fullMatrixSize = static_cast<float>(nodesNum * nodesNum * sizeof(float)) / (1024.0f * 1024.0f);
    float sparseMatrixSize = static_cast<float>(nonZeroElements * (sizeof(float) + 2 * sizeof(int))) / (1024.0f * 1024.0f);
    
    std::cout << "\n============= Matrix Statistics ===============\n";
    std::cout << "Matrix Dimensions\n";
    std::cout << "  - Full matrix: " << nodesNum << " x " << nodesNum << "\n";
    std::cout << "  - Non-zero   : " << nonZeroElements << "\n";
    std::cout << "  - Density    : " << std::fixed << std::setprecision(3)
              << (100.0f * nonZeroElements / (static_cast<uint64_t>(nodesNum) * nodesNum)) << "%\n";
    std::cout << "\nMemory Usage\n";
    std::cout << "  - Full matrix  : " << std::fixed << std::setprecision(3) << fullMatrixSize << " MB\n";
    std::cout << "  - Sparse matrix: " << std::fixed << std::setprecision(3) << sparseMatrixSize << " MB\n";
}


// ---------------------------------------------------------------------
// 7. MAIN FUNCTION
// ---------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <inputFile> [-EIG]\n";
        return 1;
    }
    
    createDir("results");
    createDir("pre_saved_EIG");
    
    std::string inputFile = argv[1];
    std::string baseName = getBaseName(inputFile);
    
    if (argc == 3 && std::string(argv[2]) == "-EIG") {
        EIG_init = true;
    }
    
    std::string foutName = "results/" + baseName;
    foutName += EIG_init ? "_KL_CutSize_EIG_output.txt" : "_KL_CutSize_output.txt";
    
    printGPUInfo();
    
    sparseMatrix spMat;
    InitializeSparsMatrix(inputFile, spMat);
    
    cudaStream_t mainStream;
    CHECK_CUDA(cudaStreamCreate(&mainStream));
    
    copyAdjacencyToDevice(spMat, mainStream);
    
    if (EIG_init) {
        computeEigenpartition(spMat);
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    KL(spMat);
    
    freeDeviceAdjacency(spMat, mainStream);
    CHECK_CUDA(cudaStreamDestroy(mainStream));
    
    std::cout << "\nGlobal Minimum Cut across all runs: " << std::fixed 
              << std::setprecision(2) << globalMin << std::endl;
    
    return 0;
}