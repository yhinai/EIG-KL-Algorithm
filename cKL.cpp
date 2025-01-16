#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <memory>
#include <sys/stat.h>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <iomanip>  // For setw and setprecision

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_num_procs() 1
#endif

using namespace std;

// Global variables
bool EIG_init = false;
string EIG_file;
string fout_name;

// Optimized sparse matrix structure
struct sparseMatrix {
    unsigned int nodeNum;
    vector<unordered_map<int, float>> adjacencyList;  // Store edges only once: if i < j then edge is in adjacencyList[i][j]
    vector<int> split[2];
    vector<int> remain[2];
    vector<float> nodeGains;  // Cache for node gains
    unordered_map<int, vector<int>> nodeConnections;  // Cache for node-to-node connections
    
    explicit sparseMatrix(unsigned int size) : nodeNum(size) {
        try {
            adjacencyList.resize(size);
            nodeGains.resize(size, 0.0f);
        } catch (const std::bad_alloc& e) {
            cerr << "Memory allocation failed: " << e.what() << endl;
            exit(1);
        }
    }
    
    // Initialize node connections cache
    void initNodeConnections() {
        #pragma omp parallel for schedule(dynamic)
        for (unsigned int i = 0; i < nodeNum; i++) {
            vector<int> connected;
            connected.reserve(adjacencyList[i].size() * 2);  // Reserve space for both directions
            
            // Add direct connections where i is smaller
            for (const auto& [j, _] : adjacencyList[i]) {
                connected.push_back(j);
            }
            
            // Add reverse connections where i is larger
            for (unsigned int j = 0; j < i; j++) {
                if (adjacencyList[j].find(i) != adjacencyList[j].end()) {
                    connected.push_back(j);
                }
            }
            
            #pragma omp critical
            nodeConnections[i] = std::move(connected);
        }
    }
};

// Helper function to safely get edge weight
float getEdgeWeight(const sparseMatrix& spMat, int node1, int node2) {
    if (node1 > node2) {
        swap(node1, node2);
    }
    const auto& neighbors = spMat.adjacencyList[node1];
    auto it = neighbors.find(node2);
    return (it != neighbors.end()) ? it->second : 0.0f;
}

void InitializeSparsMatrix(const string& filename, sparseMatrix& spMat) {
    cout << "\n============= Reading Input File =============\n";
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    getline(fin, line);
    long int netsNum, nodesNum;
    stringstream(line) >> netsNum >> nodesNum;
    
    cout << "Network Statistics:\n";
    cout << "  - Total Nets: " << netsNum << "\n";
    cout << "  - Total Nodes: " << nodesNum << "\n";
    
    // Reinitialize sparse matrix with correct size
    spMat = sparseMatrix(nodesNum);
    
    long int nonZeroElements = 0;
    vector<int> nodes;
    nodes.reserve(1000);  // Pre-allocate space
    
    cout << "\nProcessing nets and building sparse matrix...\n";
    
    // Process each net
    for (int i = 0; i < netsNum; i++) {
        getline(fin, line);
        stringstream ss(line);
        nodes.clear();
        
        int node;
        while (ss >> node) {
            nodes.push_back(node - 1);  // Convert to 0-based indexing
        }
        
        float weight = 1.0f / (nodes.size() - 1);
        
        // Add edges between all pairs in the net (store only one direction)
        for (size_t j = 0; j < nodes.size(); j++) {
            for (size_t k = j + 1; k < nodes.size(); k++) {
                int node1 = nodes[j];
                int node2 = nodes[k];
                
                // Ensure node1 < node2 for consistent storage
                if (node1 > node2) {
                    swap(node1, node2);
                }
                
                spMat.adjacencyList[node1][node2] += weight;
                nonZeroElements++;
            }
        }
    }
    
    // Calculate memory usage
    double fullMatrixSize = (double)(nodesNum * nodesNum * sizeof(float)) / (1024 * 1024);  // Size in MB
    double sparseMatrixSize = (double)(nonZeroElements * (sizeof(float) + 2 * sizeof(int))) / (1024 * 1024);  // Size in MB

    cout << "\n============= Matrix Statistics =============\n";
    cout << "Matrix Dimensions:\n";
    cout << "  - Full matrix size: " << nodesNum << " x " << nodesNum << "\n";
    cout << "  - Non-zero elements: " << nonZeroElements << "\n";
    
    cout << "\nMemory Usage:\n";
    cout << "  - Full matrix: " << fixed << setprecision(2) << fullMatrixSize << " MB\n";
    cout << "  - Sparse matrix: " << fixed << setprecision(2) << sparseMatrixSize << " MB\n";
    cout << "  - Memory saved: " << fixed << setprecision(2) << (fullMatrixSize - sparseMatrixSize) << " MB";
    cout << " (" << fixed << setprecision(2) << (100.0 * (fullMatrixSize - sparseMatrixSize) / fullMatrixSize) << "%)\n";
    
    cout << "\nSparsity Analysis:\n";
    cout << "  - Density: " << fixed << setprecision(4) 
         << (100.0 * nonZeroElements / (nodesNum * nodesNum)) << "%\n";
    
    cout << "==========================================\n\n";
         
    fin.close();
}

void shuffleSparceMatrix(sparseMatrix& spMat) {
    // Clear existing partitions
    for (auto& partition : spMat.split) partition.clear();
    for (auto& partition : spMat.remain) partition.clear();
    
    if (EIG_init) {
        ifstream fEIG(EIG_file);
        if (!fEIG.is_open()) {
            cerr << "Error: EIG file not found" << endl;
            exit(1);
        }
        
        string line;
        getline(fEIG, line);  // Skip first two lines
        getline(fEIG, line);
        
        while (getline(fEIG, line)) {
            int node, split_side;
            double weight;
            stringstream(line) >> node >> split_side >> weight;
            
            spMat.split[split_side].push_back(node);
            spMat.remain[split_side].push_back(node);
        }
        fEIG.close();
    } else {
        // Random partitioning
        vector<int> nodes(spMat.nodeNum);
        iota(nodes.begin(), nodes.end(), 0);
        
        random_device rd;
        mt19937 gen(rd());
        shuffle(nodes.begin(), nodes.end(), gen);
        
        size_t mid = spMat.nodeNum / 2;
        spMat.split[0].reserve(mid);
        spMat.remain[0].reserve(mid);
        spMat.split[1].reserve(spMat.nodeNum - mid);
        spMat.remain[1].reserve(spMat.nodeNum - mid);
        
        copy(nodes.begin(), nodes.begin() + mid, back_inserter(spMat.split[0]));
        copy(nodes.begin(), nodes.begin() + mid, back_inserter(spMat.remain[0]));
        copy(nodes.begin() + mid, nodes.end(), back_inserter(spMat.split[1]));
        copy(nodes.begin() + mid, nodes.end(), back_inserter(spMat.remain[1]));
    }
    
    cout << "Partition sizes - Left: " << spMat.split[0].size() 
         << " Right: " << spMat.split[1].size() << endl;
}

float calCutSize(const sparseMatrix& spMat) {
    float cutSize = 0.0f;
    unordered_set<int> rightNodes(spMat.remain[1].begin(), spMat.remain[1].end());
    
    #pragma omp parallel for reduction(+:cutSize) schedule(dynamic)
    for (size_t i = 0; i < spMat.remain[0].size(); i++) {
        int node = spMat.remain[0][i];
        // Check forward connections
        for (const auto& [neighbor, weight] : spMat.adjacencyList[node]) {
            if (rightNodes.count(neighbor)) {
                cutSize += weight;
            }
        }
        // Check backward connections
        for (int neighbor : rightNodes) {
            if (neighbor < node) {
                auto it = spMat.adjacencyList[neighbor].find(node);
                if (it != spMat.adjacencyList[neighbor].end()) {
                    cutSize += it->second;
                }
            }
        }
    }
    return cutSize;
}

float connections(const sparseMatrix& spMat, int node) {
    float external = 0.0f, internal = 0.0f;
    unordered_set<int> leftNodes(spMat.split[0].begin(), spMat.split[0].end());
    
    // Process forward edges
    for (const auto& [neighbor, weight] : spMat.adjacencyList[node]) {
        if (leftNodes.count(neighbor)) {
            internal += weight;
        } else {
            external += weight;
        }
    }
    
    // Process backward edges
    for (int i = 0; i < node; i++) {
        auto it = spMat.adjacencyList[i].find(node);
        if (it != spMat.adjacencyList[i].end()) {
            if (leftNodes.count(i)) {
                internal += it->second;
            } else {
                external += it->second;
            }
        }
    }
    
    return external - internal;
}

void updateAffectedNodeGains(sparseMatrix& spMat, int node1, int node2) {
    // Use vector instead of unordered_set for OpenMP compatibility
    vector<int> affectedNodes;
    affectedNodes.reserve(spMat.nodeConnections[node1].size() + spMat.nodeConnections[node2].size());
    
    // Collect all nodes connected to either swapped node
    for (int node : spMat.nodeConnections[node1]) {
        affectedNodes.push_back(node);
    }
    for (int node : spMat.nodeConnections[node2]) {
        affectedNodes.push_back(node);
    }
    
    // Remove duplicates
    sort(affectedNodes.begin(), affectedNodes.end());
    affectedNodes.erase(unique(affectedNodes.begin(), affectedNodes.end()), affectedNodes.end());
    
    // Update gains only for affected nodes using standard index-based loop
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < affectedNodes.size(); i++) {
        int node = affectedNodes[i];
        spMat.nodeGains[node] = connections(spMat, node);
    }
}

void swip(sparseMatrix& spMat, int num1, int num2) {
    // Remove from remain vectors
    auto it1 = find(spMat.remain[0].begin(), spMat.remain[0].end(), num1);
    auto it2 = find(spMat.remain[1].begin(), spMat.remain[1].end(), num2);
    
    if (it1 != spMat.remain[0].end()) spMat.remain[0].erase(it1);
    if (it2 != spMat.remain[1].end()) spMat.remain[1].erase(it2);
    
    // Update split vectors
    auto split1 = find(spMat.split[0].begin(), spMat.split[0].end(), num1);
    auto split2 = find(spMat.split[1].begin(), spMat.split[1].end(), num2);
    
    if (split1 != spMat.split[0].end()) *split1 = num2;
    if (split2 != spMat.split[1].end()) *split2 = num1;
}

void KL(sparseMatrix& spMat) {
    cout << "\n============= Starting KL Algorithm =============\n";
    shuffleSparceMatrix(spMat);
    
    // Initialize node connections cache
    cout << "Initializing node connections cache...\n";
    spMat.initNodeConnections();
    
    ofstream fout(fout_name);
    if (!fout.is_open()) {
        cerr << "Error: Cannot open output file" << endl;
        exit(1);
    }
    
    int iteration = 0;
    int terminate = 0;
    int terminateLimit = log2(spMat.nodeNum) + 5;
    float globalMinCutSize = numeric_limits<float>::max();
    
    float cutSize = calCutSize(spMat);
    float minCutSize = cutSize;
    float initialCutSize = cutSize;  // Store initial cut size for improvement calculation
    
    cout << "\nInitial Partition Information:\n";
    cout << "  - Left partition size: " << spMat.split[0].size() << "\n";
    cout << "  - Right partition size: " << spMat.split[1].size() << "\n";
    cout << "  - Initial cut size: " << cutSize << "\n\n";
    
    fout << "0\t" << cutSize << "\t0" << endl;
    
    cout << "Initializing node gains...\n";
    // Initialize all node gains
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < spMat.nodeNum; i++) {
        spMat.nodeGains[i] = connections(spMat, i);
    }
    
    cout << "\n============= KL Iterations =============\n";
    cout << setw(10) << "Iteration" 
         << setw(15) << "Cut Size" 
         << setw(15) << "Gain" 
         << setw(15) << "Time (ms)" 
         << setw(15) << "Improvement" << "\n";
    cout << string(70, '-') << "\n";
    
    auto total_start_time = chrono::high_resolution_clock::now();
    
    while (!spMat.remain[0].empty() && !spMat.remain[1].empty()) {
        auto start_time = chrono::high_resolution_clock::now();
        
        float maxGain = -numeric_limits<float>::max();
        float minGain = numeric_limits<float>::max();
        int maxIdx = -1, minIdx = -1;
        
        // Find maximum gain in partition 0
        for (size_t i = 0; i < spMat.remain[0].size(); i++) {
            int node = spMat.remain[0][i];
            if (spMat.nodeGains[node] > maxGain) {
                maxGain = spMat.nodeGains[node];
                maxIdx = i;
            }
        }
        
        // Find minimum gain in partition 1
        for (size_t i = 0; i < spMat.remain[1].size(); i++) {
            int node = spMat.remain[1][i];
            if (spMat.nodeGains[node] < minGain) {
                minGain = spMat.nodeGains[node];
                minIdx = i;
            }
        }
        
        if (maxIdx >= 0 && minIdx >= 0) {
            int node1 = spMat.remain[0][maxIdx];
            int node2 = spMat.remain[1][minIdx];
            float gain = maxGain - minGain - 2 * getEdgeWeight(spMat, node1, node2);
            
            cutSize -= gain;
            minCutSize = min(minCutSize, cutSize);
            
            // Swap nodes and update affected nodes
            swip(spMat, node1, node2);
            updateAffectedNodeGains(spMat, node1, node2);
            
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            
            iteration++;
            float improvement = 100.0f * (1.0f - cutSize/initialCutSize);
            
            cout << setw(10) << iteration 
                 << setw(15) << fixed << setprecision(2) << cutSize 
                 << setw(15) << fixed << setprecision(2) << gain 
                 << setw(15) << duration.count()
                 << setw(15) << fixed << setprecision(2) << improvement << "%\n";
                 
            fout << iteration << "\t" << cutSize << "\t" << gain << endl;
            
            if (gain <= 0) {
                if (++terminate > terminateLimit) break;
            } else {
                terminate = 0;
            }
        } else {
            break;
        }
    }
    
    auto total_end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::seconds>(total_end_time - total_start_time);
    
    globalMinCutSize = min(globalMinCutSize, minCutSize);
    
    cout << "\n============= Final Results =============\n";
    cout << "  - Total iterations: " << iteration << "\n";
    cout << "  - Initial cut size: " << initialCutSize << "\n";
    cout << "  - Best cut size achieved: " << globalMinCutSize << "\n";
    cout << "  - Overall improvement: " << fixed << setprecision(2) 
         << 100.0 * (1.0 - globalMinCutSize/initialCutSize) << "%\n";
    cout << "  - Total runtime: " << total_duration.count() << " seconds\n";
    cout << "=========================================\n\n";
    
    fout.close();
}

void createDir(const string& dirName) {
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {  // Directory doesn't exist
        #ifdef _WIN32
            _mkdir(dirName.c_str());
        #else
            mkdir(dirName.c_str(), 0755);  // Read/write for owner, read for others
        #endif
    }
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);  // Optimize I/O operations
    cin.tie(nullptr);
    
    createDir("results");
    createDir("pre_saved_EIG");

    if (argc != 2 && argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> [-EIG]" << endl;
        return 1;
    }

    string input_file = argv[1];
    fout_name = "results/" + input_file + "_KL_CutSize_output.txt";

    if (argc == 3 && strcmp(argv[2], "-EIG") == 0) {
        EIG_init = true;
        EIG_file = "pre_saved_EIG/" + input_file + "_out.txt";
        fout_name = "results/" + input_file + "_KL_CutSize_EIG_output.txt";
    }

    try {
        // Initialize with minimum size first to avoid large memory allocation
        sparseMatrix spMat(1);
        
        // Read file and properly initialize the matrix
        InitializeSparsMatrix(input_file, spMat);
        
        // Set number of threads for OpenMP
        #ifdef _OPENMP
            int num_threads = omp_get_num_procs();
            omp_set_num_threads(num_threads);
            cout << "Using " << num_threads << " threads" << endl;
        #endif

        // Run KL algorithm
        KL(spMat);
        
    } catch (const std::exception& e) {
        cerr << "Error occurred: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }

    return 0;
}