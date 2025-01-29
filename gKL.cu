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
#include <thread>

// ---------------------------------------------------------------------
// 1. GLOBALS & STRUCTS
// ---------------------------------------------------------------------

// Add at the top with other globals
bool EIG_init = false;
std::string EIG_file;
float gloableMin = std::numeric_limits<float>::max();

struct sparseMatrix {
    unsigned int nodeNum;

    // CPU adjacency
    std::vector<std::vector<int>>   Nodes;    // for each node i, Nodes[i] = adjacency list
    std::vector<std::vector<float>> Weights;  // same shape, Weights[i] = edge weights

    // Two partitions
    std::vector<int> split[2];   // each partition's node IDs
    std::vector<int> remain[2];  // "remaining" nodes that can still be swapped

    // Flattened adjacency for GPU
    std::vector<int>   adjacencyOffsets; // [nodeNum+1]
    std::vector<int>   adjacencyIndices; // total adjacency size
    std::vector<float> adjacencyWeights; // total adjacency size

    // Device pointers
    int   *d_adjacencyOffsets = nullptr;
    int   *d_adjacencyIndices = nullptr;
    float *d_adjacencyWeights = nullptr;

    // For membership-based approach, we store node -> partition
    // membership[node] = 0 or 1
    // We'll keep a CPU vector<int> membershipOfNode, but also
    // allocate once on the GPU
};

// Error checking macro
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// ---------------------------------------------------------------------
// 2. BUILD & COPY FLATTENED ADJACENCY
// ---------------------------------------------------------------------

void buildFlattenedAdjacency(sparseMatrix &spMat)
{
    spMat.adjacencyOffsets.resize(spMat.nodeNum+1, 0);
    // adjacencyOffsets[i+1] = adjacencyOffsets[i] + degree(i)
    for (unsigned int i = 0; i < spMat.nodeNum; i++) {
        spMat.adjacencyOffsets[i+1] = spMat.adjacencyOffsets[i] + spMat.Nodes[i].size();
    }
    size_t totalEdges = spMat.adjacencyOffsets[spMat.nodeNum];
    spMat.adjacencyIndices.resize(totalEdges);
    spMat.adjacencyWeights.resize(totalEdges);

    for (unsigned int i = 0; i < spMat.nodeNum; i++) {
        int start = spMat.adjacencyOffsets[i];
        for (unsigned int j = 0; j < spMat.Nodes[i].size(); j++){
            spMat.adjacencyIndices[start + j] = spMat.Nodes[i][j];
            spMat.adjacencyWeights [start + j] = spMat.Weights[i][j];
        }
    }
}

void copyAdjacencyToDevice(sparseMatrix &spMat)
{
    // d_adjacencyOffsets
    gpuErrchk(cudaMalloc((void**)&spMat.d_adjacencyOffsets, (spMat.nodeNum+1)*sizeof(int)));
    gpuErrchk(cudaMemcpy(spMat.d_adjacencyOffsets,
                         spMat.adjacencyOffsets.data(),
                         (spMat.nodeNum+1)*sizeof(int),
                         cudaMemcpyHostToDevice));
    // d_adjacencyIndices, d_adjacencyWeights
    size_t totalEdges = spMat.adjacencyOffsets[spMat.nodeNum];
    gpuErrchk(cudaMalloc((void**)&spMat.d_adjacencyIndices,  totalEdges*sizeof(int)));
    gpuErrchk(cudaMemcpy(spMat.d_adjacencyIndices,
                         spMat.adjacencyIndices.data(),
                         totalEdges*sizeof(int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**)&spMat.d_adjacencyWeights,  totalEdges*sizeof(float)));
    gpuErrchk(cudaMemcpy(spMat.d_adjacencyWeights,
                         spMat.adjacencyWeights.data(),
                         totalEdges*sizeof(float),
                         cudaMemcpyHostToDevice));
}

void freeDeviceAdjacency(sparseMatrix &spMat)
{
    if(spMat.d_adjacencyOffsets) {
        gpuErrchk(cudaFree(spMat.d_adjacencyOffsets));
        spMat.d_adjacencyOffsets = nullptr;
    }
    if(spMat.d_adjacencyIndices) {
        gpuErrchk(cudaFree(spMat.d_adjacencyIndices));
        spMat.d_adjacencyIndices = nullptr;
    }
    if(spMat.d_adjacencyWeights) {
        gpuErrchk(cudaFree(spMat.d_adjacencyWeights));
        spMat.d_adjacencyWeights = nullptr;
    }
}

// ---------------------------------------------------------------------
// 3. GPU KERNEL: connectionsKernel
//    out[i] = E - I for remain[i],
//    E = external (sum of edges to other partition)
//    I = internal (sum of edges to own partition)
// ---------------------------------------------------------------------
__global__
void connectionsKernel(const int* __restrict__ adjacencyOffsets,
                       const int* __restrict__ adjacencyIndices,
                       const float* __restrict__ adjacencyWeights,
                       const int* __restrict__ membership, // membership[node] = 0 or 1
                       const int* __restrict__ d_remain,   // list of nodes we want connections for
                       float* __restrict__ d_out,
                       int remainSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < remainSize) {
        int node = d_remain[idx];
        int start = adjacencyOffsets[node];
        int end   = adjacencyOffsets[node+1];

        float E = 0.f, I = 0.f;
        int mySide = membership[node];
        for(int e = start; e < end; e++){
            int neigh = adjacencyIndices[e];
            float w   = adjacencyWeights[e];
            if (membership[neigh] == mySide) {
                I += w;
            } else {
                E += w;
            }
        }
        d_out[idx] = (E - I);
    }
}

// ---------------------------------------------------------------------
// 4. GPU CONNECTIONS: single call for remain[] nodes
//    This version reuses allocated buffers d_remain, d_out, d_membership
//    so we only do cudaMemcpy each iteration, not cudaMalloc/cudaFree.
// ---------------------------------------------------------------------
void gpuConnections(
    const sparseMatrix &spMat,
    int *d_remain,     // allocated once outside
    int *d_membership, // allocated once
    float *d_out,
    std::vector<int> &remain,          // host remain
    std::vector<int> &membershipHost,  // host membership
    std::vector<float> &out            // host output
)
{
    int remainSize = (int)remain.size();
    if(remainSize == 0) return;

    // 1) copy remain[] to device
    gpuErrchk(cudaMemcpy(d_remain, remain.data(),
                         remainSize*sizeof(int),
                         cudaMemcpyHostToDevice));

    // 2) copy membership to device
    gpuErrchk(cudaMemcpy(d_membership, membershipHost.data(),
                         spMat.nodeNum*sizeof(int),
                         cudaMemcpyHostToDevice));

    // 3) run kernel
    int blockSize = 256;
    int gridSize  = (remainSize + blockSize - 1)/blockSize;
    connectionsKernel<<<gridSize, blockSize>>>(
        spMat.d_adjacencyOffsets,
        spMat.d_adjacencyIndices,
        spMat.d_adjacencyWeights,
        d_membership,
        d_remain,
        d_out,
        remainSize
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 4) copy d_out back
    gpuErrchk(cudaMemcpy(out.data(), d_out,
                         remainSize*sizeof(float),
                         cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------
// 5. HELPER: INITIALIZE SPARSE MATRIX from input
// ---------------------------------------------------------------------
void InitializeSparsMatrix(const std::string &filename, sparseMatrix & spMat)
{
    std::cout << "\n============= Reading Input File ==============\n";
    std::ifstream fin(filename.c_str());
    if(!fin.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    std::getline(fin, line);
    int netsNum=0, nodesNum=0;
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

    // read each net line
    long int nonZeroElements = nodesNum;
    long int numEdges = 0;
    for(int i = 0; i < netsNum; i++){
        if(!std::getline(fin, line)) break;
        std::stringstream ss(line);
        std::vector<int> nodes;
        int nd;
        while(ss >> nd){
            nodes.push_back(nd);
        }
        // assign fractional weight
        if(nodes.size() < 2) continue;  // ignore nets with 0 or 1 node
        float weight = 1.f / float(nodes.size()-1);

        for(unsigned int j = 0; j < nodes.size(); j++){
            for(unsigned int k = j+1; k < nodes.size(); k++){
                numEdges++;
                int a = nodes[j]-1, b = nodes[k]-1;
                // see if b is in adjacency of a
                auto it = std::find(spMat.Nodes[a].begin(), spMat.Nodes[a].end(), b);
                if(it == spMat.Nodes[a].end()) {
                    // not found
                    spMat.Nodes[a].push_back(b);
                    spMat.Weights[a].push_back(weight);
                    spMat.Nodes[b].push_back(a);
                    spMat.Weights[b].push_back(weight);
                    nonZeroElements += 2;
                } else {
                    // found => increment weights
                    int idxA = (int)(it - spMat.Nodes[a].begin());
                    spMat.Weights[a][idxA] += weight;
                    // find a in adjacency of b
                    auto it2 = std::find(spMat.Nodes[b].begin(), spMat.Nodes[b].end(), a);
                    int idxB = (int)(it2 - spMat.Nodes[b].begin());
                    spMat.Weights[b][idxB] += weight;
                }
            }
        }
    }
    fin.close();

    // Flatten adjacency for GPU
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
// 6. SHUFFLE SPARSE MATRIX => create initial partition
// ---------------------------------------------------------------------
void shuffleSparceMatrix(sparseMatrix & spMat)
{
    // clear old
    spMat.split[0].clear(); spMat.split[1].clear();
    spMat.remain[0].clear(); spMat.remain[1].clear();

    if (EIG_init) {
        std::ifstream fEIG(EIG_file.c_str());
        if(!fEIG.is_open()){
            std::cerr << "Error: EIG file not found: " << EIG_file << std::endl;
            exit(1);
        }
        // skip first two lines
        std::string line;
        std::getline(fEIG, line);
        std::getline(fEIG, line);
        
        while(std::getline(fEIG, line)){
            std::stringstream ss(line);
            int i, side;
            double w;
            ss >> i >> side >> w;
            if(side == 0){
                spMat.split[0].push_back(i);
                spMat.remain[0].push_back(i);
            } else {
                spMat.split[1].push_back(i);
                spMat.remain[1].push_back(i);
            }
        }
        fEIG.close();
        std::cout << "Using EIG init: split[0].size=" << spMat.split[0].size()
                  << ", split[1].size=" << spMat.split[1].size() << std::endl;
        return;
    }

    // random partition if not using EIG
    std::vector<int> all;
    all.reserve(spMat.nodeNum);
    for(unsigned int i=0; i<spMat.nodeNum; i++){
        all.push_back(i);
    }
    std::random_shuffle(all.begin(), all.end());
    // half => partition 0, half => partition 1
    unsigned int half = spMat.nodeNum/2;
    for(unsigned int i=0; i<half; i++){
        spMat.split[0].push_back(all[i]);
        spMat.remain[0].push_back(all[i]);
    }
    for(unsigned int i=half; i<spMat.nodeNum; i++){
        spMat.split[1].push_back(all[i]);
        spMat.remain[1].push_back(all[i]);
    }
    std::cout << "shuffle => part0.size=" << spMat.split[0].size()
              << ", part1.size=" << spMat.split[1].size() << std::endl;
}

// ---------------------------------------------------------------------
// 7. (OPTIONAL) from-scratch cut size check for debugging/final
//    Uses membership array for O(1) checks
// ---------------------------------------------------------------------
float computeCutSize(const sparseMatrix &spMat, const std::vector<int> &membership)
{
    float E = 0.f;
    // sum edges from partition0 => partition1
    for(auto node0 : spMat.split[0]){
        const auto &nbrs   = spMat.Nodes[node0];
        const auto &wts    = spMat.Weights[node0];
        for(size_t j=0; j<nbrs.size(); j++){
            if(membership[nbrs[j]] == 1){
                E += wts[j];
            }
        }
    }
    return E;
}

// ---------------------------------------------------------------------
// 8. nodeConnection for edge (a,b), also membership-based check optional
// ---------------------------------------------------------------------
float nodeConnection(const sparseMatrix &spMat, int a, int b)
{
    const auto &nbrs = spMat.Nodes[a];
    const auto &wts  = spMat.Weights[a];
    for(size_t i=0; i<nbrs.size(); i++){
        if(nbrs[i] == b) {
            return wts[i];
        }
    }
    return 0.f;
}

// ---------------------------------------------------------------------
// 9. SWAP function
// ---------------------------------------------------------------------
void swip(sparseMatrix &spMat, std::vector<int> &membership, int num1, int num2)
{
    // membership changes
    membership[num1] = 1;  // now in partition 1
    membership[num2] = 0;  // now in partition 0

    // remove from remain sets
    auto it1 = std::find(spMat.remain[0].begin(), spMat.remain[0].end(), num1);
    if(it1 != spMat.remain[0].end()) spMat.remain[0].erase(it1);
    auto it2 = std::find(spMat.remain[1].begin(), spMat.remain[1].end(), num2);
    if(it2 != spMat.remain[1].end()) spMat.remain[1].erase(it2);

    // swap inside split arrays
    auto s1 = std::find(spMat.split[0].begin(), spMat.split[0].end(), num1);
    if(s1 != spMat.split[0].end()) *s1 = num2;
    auto s2 = std::find(spMat.split[1].begin(), spMat.split[1].end(), num2);
    if(s2 != spMat.split[1].end()) *s2 = num1;
}

// ---------------------------------------------------------------------
// 10. KL with Incremental Cut Size & GPU-based connections
//     (One-time GPU buffers for remain, membership, out arrays.)
// ---------------------------------------------------------------------
void KL(sparseMatrix &spMat)
{
    std::cout << "\n=========== Starting KL Algorithm =============\n";
    shuffleSparceMatrix(spMat);

    // Build membership array
    std::vector<int> membership(spMat.nodeNum, -1);
    for(auto n : spMat.split[0]) membership[n] = 0;
    for(auto n : spMat.split[1]) membership[n] = 1;

    float cutSize = computeCutSize(spMat, membership);
    float initialCutSize = cutSize;
    float bestCut = cutSize;

    std::cout << "\nInitial Partition Information:\n";
    std::cout << "  - Left partition size: " << spMat.split[0].size() << "\n";
    std::cout << "  - Right partition size: " << spMat.split[1].size() << "\n";
    std::cout << "  - Initial cut size: " << std::fixed << std::setprecision(3) << cutSize << "\n\n";

    // 3) Allocate big enough GPU arrays for remain & membership & out
    //    We'll reuse them every iteration
    int maxRem = (int)std::max(spMat.remain[0].size(), spMat.remain[1].size());
    int *d_remain = nullptr;
    float *d_out = nullptr;
    int *d_membership = nullptr;
    gpuErrchk(cudaMalloc((void**)&d_remain,     maxRem * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_membership, spMat.nodeNum*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_out,        maxRem * sizeof(float)));

    std::vector<float> con_1, con_2;

    std::cout << "============================== KL Iterations ==============================\n";
    std::cout << "---------------------------------------------------------------------------\n";
    std::cout << std::setw(10) << "Iteration" 
              << std::setw(15) << "Cut Size" 
              << std::setw(20) << "Gain (delta)" 
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Improvement" << "\n";
    std::cout << "---------------------------------------------------------------------------\n";

    int count = 0;
    int terminate = 0;
    int terminateLimit = (int)(std::log2((double)spMat.nodeNum)) + 5;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Main KL loop
    while(!spMat.remain[0].empty() && !spMat.remain[1].empty())
    {
        auto iter_start = std::chrono::high_resolution_clock::now();

        // 4) GPU connections for partition 0 remain
        con_1.resize(spMat.remain[0].size());
        gpuConnections(spMat, d_remain, d_membership, d_out,
                       spMat.remain[0], membership, con_1);

        // 5) GPU connections for partition 1 remain
        con_2.resize(spMat.remain[1].size());
        gpuConnections(spMat, d_remain, d_membership, d_out,
                       spMat.remain[1], membership, con_2);

        // 6) Find global max in con_1 => node1, global min in con_2 => node2
        float global_max_1 = -std::numeric_limits<float>::infinity();
        int   global_max_idx_1 = -1;
        for(int i=0; i<(int)con_1.size(); i++){
	    if(con_1[i] > global_max_1){
                global_max_1 = con_1[i];
                global_max_idx_1 = i;
            }
        }
        float global_max_2 = -std::numeric_limits<float>::infinity();
        int   global_max_idx_2 = -1;
        for(int i=0; i<(int)con_2.size(); i++){
            if(con_2[i] > global_max_2){
                global_max_2 = con_2[i];
                global_max_idx_2 = i;
            }
        }

        // The chosen nodes to swap
        int node1 = spMat.remain[0][global_max_idx_1];
        int node2 = spMat.remain[1][global_max_idx_2];

        // 7) Incremental gain
        //    gain = (global_max - global_min) - 2*w(node1, node2)
        float edgeW = nodeConnection(spMat, node1, node2);
        float gain = (global_max_1 + global_max_2) - 2.0f * edgeW;

        // 8) Update cut size
        cutSize = cutSize - gain;  // if logic is correct, won't go negative

        if(cutSize < bestCut){
            bestCut = cutSize;
        }

        // 9) swap
        swip(spMat, membership, node1, node2);

        // 10) iteration stats
        count++;
        auto iter_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start);
        float improvement = 100.0f * (1.0f - cutSize/initialCutSize);
        
        std::cout << std::setw(8) << count 
                  << std::setw(17) << std::fixed << std::setprecision(2) << cutSize 
                  << std::setw(18) << std::fixed << std::setprecision(2) << gain 
                  << std::setw(15) << duration.count()
                  << std::setw(15) << std::fixed << std::setprecision(2) << improvement << "%\n";
        
        // 10) update global min
	    if(cutSize < gloableMin) gloableMin = cutSize;

        // 11) Temperature: keep the search while no gain improvement
        if(gain <= 0.f){
            terminate++;
            if(terminate > terminateLimit){
                break;
            }
        } else {
            terminate = 0;
        }
    } // end while

    // 12) free the GPU buffers
    gpuErrchk(cudaFree(d_remain));
    gpuErrchk(cudaFree(d_membership));
    gpuErrchk(cudaFree(d_out));

    // Optionally do a final from-scratch check (to confirm correctness)
    {
        float finalCheck = computeCutSize(spMat, membership);
        std::cout << "[FinalCheck] cutSize from-scratch=" << finalCheck
                  << ", incrementalCut=" << cutSize << std::endl;
        cutSize = finalCheck; // force them consistent
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end_time - total_start_time);

    // Final results output
    std::cout << "\n=============== Final Results =================\n";
    std::cout << std::left << std::setw(24) << "Total iterations" << ": " << count << "\n";
    std::cout << std::left << std::setw(24) << "Initial cut size" << ": " << std::fixed << std::setprecision(2) << initialCutSize << "\n";
    std::cout << std::left << std::setw(24) << "Best cut size achieved" << ": " << bestCut << "\n";
    std::cout << std::left << std::setw(24) << "Overall improvement" << ": " 
              << std::fixed << std::setprecision(2) << 100.0f * (1.0f - bestCut/initialCutSize) << "%\n";
    std::cout << std::left << std::setw(24) << "Total runtime" << ": " << total_duration.count() << " seconds\n";
}


// ---------------------------------------------------------------------
// 11. HELPER: get base name from full path
// ---------------------------------------------------------------------

void createDir(const std::string& dirName) {
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {
        #ifdef _WIN32
            _mkdir(dirName.c_str());
        #else
            mkdir(dirName.c_str(), 0755);
        #endif
    }
}

std::string getBaseName(const std::string& path) {
    std::filesystem::path fp(path);
    return fp.filename().string();
}

void printGPUInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "\n================= GPU Info ===================\n";
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Number of multiprocessors: " << prop.multiProcessorCount << "\n";
    }
}

// ---------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <inputFile> [ -EIG ]\n";
        return 1;
    }

    // Create necessary directories
    createDir("results");
    createDir("pre_saved_EIG");

    std::string inputFile = argv[1];
    std::string baseName = getBaseName(inputFile);
    
    // Handle EIG flag
    if(argc == 3 && std::string(argv[2]) == "-EIG"){
        EIG_init = true;
        EIG_file = "pre_saved_EIG/" + baseName + "_out.txt";
    }

    // Output file name for results
    std::string foutName = "results/" + baseName;
    foutName += EIG_init ? "_KL_CutSize_EIG_output.txt" : "_KL_CutSize_output.txt";

    // 1) Print GPU info
    printGPUInfo();

    // 2) Build adjacency from input
    sparseMatrix spMat;
    InitializeSparsMatrix(inputFile, spMat);

    // 3) Copy adjacency to GPU (one time)
    copyAdjacencyToDevice(spMat);

    // sleep for 3 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 4) Run KL
    KL(spMat);

    // 5) Cleanup
    freeDeviceAdjacency(spMat);

    std::cout << "Global Min Cut over runs: " << gloableMin << std::endl;
    return 0;
}