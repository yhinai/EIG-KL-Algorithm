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
#include <iomanip>
#include <filesystem>

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
    size_t nodeNum;
    vector<unordered_map<uint32_t, float>> adjacencyList;
    vector<uint32_t> split[2];
    vector<uint32_t> remain[2];
    vector<float> nodeGains;
    unordered_map<uint32_t, vector<uint32_t>> nodeConnections;
    
    explicit sparseMatrix(size_t size) : nodeNum(size) {
        try {
            adjacencyList.resize(size);
            nodeGains.resize(size, 0.0f);
        } catch (const std::bad_alloc& e) {
            cerr << "Memory allocation failed: " << e.what() << endl;
            exit(1);
        }
    }
    
    void initNodeConnections() {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nodeNum; i++) {
            vector<uint32_t> connected;
            connected.reserve(adjacencyList[i].size() * 2);
            
            for (const auto& [j, _] : adjacencyList[i]) {
                connected.push_back(j);
            }
            
            for (size_t j = 0; j < i; j++) {
                if (adjacencyList[j].find(i) != adjacencyList[j].end()) {
                    connected.push_back(static_cast<uint32_t>(j));
                }
            }
            
            #pragma omp critical
            nodeConnections[i] = std::move(connected);
        }
    }
};

float getEdgeWeight(const sparseMatrix& spMat, uint32_t node1, uint32_t node2) {
    if (node1 > node2) {
        swap(node1, node2);
    }
    const auto& neighbors = spMat.adjacencyList[node1];
    auto it = neighbors.find(node2);
    return (it != neighbors.end()) ? it->second : 0.0f;
}

void InitializeSparsMatrix(const string& filename, sparseMatrix& spMat) {
    cout << "\n============= Reading Input File ==============\n";
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    getline(fin, line);
    uint32_t netsNum, nodesNum;
    stringstream(line) >> netsNum >> nodesNum;
    
    cout << "Circuit Statistics\n";
    cout << "  - Total Nets : " << netsNum << "\n";
    cout << "  - Total Nodes: " << nodesNum << "\n";
    
    spMat = sparseMatrix(nodesNum);
    
    uint64_t nonZeroElements = 0;
    vector<uint32_t> nodes;
    nodes.reserve(1000);
        
    for (uint32_t i = 0; i < netsNum; i++) {
        getline(fin, line);
        stringstream ss(line);
        nodes.clear();
        
        uint32_t node;
        while (ss >> node) {
            nodes.push_back(node - 1);
        }
        
        float weight = 1.0f / (nodes.size() - 1);
        
        for (size_t j = 0; j < nodes.size(); j++) {
            for (size_t k = j + 1; k < nodes.size(); k++) {
                uint32_t node1 = nodes[j];
                uint32_t node2 = nodes[k];
                
                if (node1 > node2) {
                    swap(node1, node2);
                }
                
                spMat.adjacencyList[node1][node2] += weight;
                nonZeroElements++;
            }
        }
    }
    
    float fullMatrixSize = static_cast<float>(nodesNum * nodesNum * sizeof(float)) / (1024.0f * 1024.0f);
    float sparseMatrixSize = static_cast<float>(nonZeroElements * (sizeof(float) + 2 * sizeof(uint32_t))) / (1024.0f * 1024.0f);

    cout << "\n\n============= Matrix Statistics ===============\n";
    cout << "Matrix Dimensions\n";
    cout << "  - Full matrix: " << nodesNum << " x " << nodesNum << "\n";
    cout << "  - Non-zero   : " << nonZeroElements << "\n";
    cout << "  - Density    : " << fixed << setprecision(3) 
         << (100.0f * nonZeroElements / (static_cast<uint64_t>(nodesNum) * nodesNum)) << "%\n";

    cout << "\nMemory Usage\n";
    cout << "  - Full matrix  : " << fixed << setprecision(3) << fullMatrixSize << " MB\n";
    cout << "  - Sparse matrix: " << fixed << setprecision(3) << sparseMatrixSize << " MB\n";
                 
    fin.close();
}

void shuffleSparceMatrix(sparseMatrix& spMat) {
    for (auto& partition : spMat.split) partition.clear();
    for (auto& partition : spMat.remain) partition.clear();
    
    if (EIG_init) {
        ifstream fEIG(EIG_file);
        if (!fEIG.is_open()) {
            cerr << "Error: EIG file not found" << endl;
            exit(1);
        }
        
        string line;
        getline(fEIG, line);
        getline(fEIG, line);
        
        while (getline(fEIG, line)) {
            uint32_t node, split_side;
            float weight;
            stringstream(line) >> node >> split_side >> weight;
            
            spMat.split[split_side].push_back(node);
            spMat.remain[split_side].push_back(node);
        }
        fEIG.close();
    } else {
        vector<uint32_t> nodes(spMat.nodeNum);
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
    unordered_set<uint32_t> rightNodes(spMat.remain[1].begin(), spMat.remain[1].end());
    
    #pragma omp parallel for reduction(+:cutSize) schedule(dynamic)
    for (size_t i = 0; i < spMat.remain[0].size(); i++) {
        uint32_t node = spMat.remain[0][i];
        // Check forward connections
        for (const auto& [neighbor, weight] : spMat.adjacencyList[node]) {
            if (rightNodes.count(neighbor)) {
                cutSize += weight;
            }
        }
        // Check backward connections
        for (uint32_t neighbor : rightNodes) {
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

float connections(const sparseMatrix& spMat, uint32_t node) {
    float external = 0.0f, internal = 0.0f;
    unordered_set<uint32_t> leftNodes(spMat.split[0].begin(), spMat.split[0].end());
    
    // Process forward edges
    for (const auto& [neighbor, weight] : spMat.adjacencyList[node]) {
        if (leftNodes.count(neighbor)) {
            internal += weight;
        } else {
            external += weight;
        }
    }
    
    // Process backward edges
    for (size_t i = 0; i < node; i++) {
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

void updateAffectedNodeGains(sparseMatrix& spMat, uint32_t node1, uint32_t node2) {
    vector<uint32_t> affectedNodes;
    affectedNodes.reserve(spMat.nodeConnections[node1].size() + spMat.nodeConnections[node2].size());
    
    for (uint32_t node : spMat.nodeConnections[node1]) {
        affectedNodes.push_back(node);
    }
    for (uint32_t node : spMat.nodeConnections[node2]) {
        affectedNodes.push_back(node);
    }
    
    sort(affectedNodes.begin(), affectedNodes.end());
    affectedNodes.erase(unique(affectedNodes.begin(), affectedNodes.end()), affectedNodes.end());
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < affectedNodes.size(); i++) {
        uint32_t node = affectedNodes[i];
        spMat.nodeGains[node] = connections(spMat, node);
    }
}

void swip(sparseMatrix& spMat, uint32_t num1, uint32_t num2) {
    auto it1 = find(spMat.remain[0].begin(), spMat.remain[0].end(), num1);
    auto it2 = find(spMat.remain[1].begin(), spMat.remain[1].end(), num2);
    
    if (it1 != spMat.remain[0].end()) spMat.remain[0].erase(it1);
    if (it2 != spMat.remain[1].end()) spMat.remain[1].erase(it2);
    
    auto split1 = find(spMat.split[0].begin(), spMat.split[0].end(), num1);
    auto split2 = find(spMat.split[1].begin(), spMat.split[1].end(), num2);
    
    if (split1 != spMat.split[0].end()) *split1 = num2;
    if (split2 != spMat.split[1].end()) *split2 = num1;
}

void KL(sparseMatrix& spMat) {
    cout << "\n\n=========== Starting KL Algorithm =============\n";
    shuffleSparceMatrix(spMat);
    
    cout << "Initializing node connections cache...\n";
    spMat.initNodeConnections();
    
    ofstream fout(fout_name);
    if (!fout.is_open()) {
        cerr << "Error: Cannot open output file" << endl;
        exit(1);
    }
    
    uint32_t iteration = 0;
    uint32_t terminate = 0;
    uint32_t terminateLimit = static_cast<uint32_t>(log2(spMat.nodeNum)) + 5;
    float globalMinCutSize = numeric_limits<float>::max();
    
    float cutSize = calCutSize(spMat);
    float minCutSize = cutSize;
    float initialCutSize = cutSize;
    
    cout << "\nInitial Partition Information:\n";
    cout << "  - Left partition size: " << spMat.split[0].size() << "\n";
    cout << "  - Right partition size: " << spMat.split[1].size() << "\n";
    cout << "  - Initial cut size: " << cutSize << "\n\n";
    
    fout << "0\t" << cutSize << "\t0" << endl;
    
    cout << "Initializing node gains...\n";
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < spMat.nodeNum; i++) {
        spMat.nodeGains[i] = connections(spMat, static_cast<uint32_t>(i));
    }
    
    cout << "\n============================== KL Iterations ==============================";
    cout << "\n---------------------------------------------------------------------------" << "\n";
    cout << setw(10) << "Iteration" 
         << setw(15) << "Cut Size" 
         << setw(20) << "Gain (delta)" 
         << setw(15) << "Time (ms)" 
         << setw(15) << "Improvement" << "\n";
    cout << string(75, '-') << "\n";
    
    auto total_start_time = chrono::high_resolution_clock::now();
    
    while (!spMat.remain[0].empty() && !spMat.remain[1].empty()) {
        auto start_time = chrono::high_resolution_clock::now();
        
        float maxGain = -numeric_limits<float>::max();
        float minGain = numeric_limits<float>::max();
        size_t maxIdx = SIZE_MAX, minIdx = SIZE_MAX;
        
        for (size_t i = 0; i < spMat.remain[0].size(); i++) {
            uint32_t node = spMat.remain[0][i];
            if (spMat.nodeGains[node] > maxGain) {
                maxGain = spMat.nodeGains[node];
                maxIdx = i;
            }
        }
        
        for (size_t i = 0; i < spMat.remain[1].size(); i++) {
            uint32_t node = spMat.remain[1][i];
            if (spMat.nodeGains[node] < minGain) {
                minGain = spMat.nodeGains[node];
                minIdx = i;
            }
        }
        
        if (maxIdx != SIZE_MAX && minIdx != SIZE_MAX) {
            uint32_t node1 = spMat.remain[0][maxIdx];
            uint32_t node2 = spMat.remain[1][minIdx];
            float gain = maxGain - minGain - 2.0f * getEdgeWeight(spMat, node1, node2);
            
            cutSize -= gain;
            minCutSize = min(minCutSize, cutSize);
            
            swip(spMat, node1, node2);
            updateAffectedNodeGains(spMat, node1, node2);
            
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            
            iteration++;
            float improvement = 100.0f * (1.0f - cutSize/initialCutSize);
            
            cout << setw(8) << iteration 
                 << setw(17) << fixed << setprecision(2) << cutSize 
                 << setw(18) << fixed << setprecision(2) << gain 
                 << setw(15) << duration.count()
                 << setw(15) << fixed << setprecision(2) << improvement << "%\n";
                 
            fout << iteration << "\t" << cutSize << "\t" << gain << endl;
            
            if (gain <= 0.0f) {
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

    cout << "\n\n=============== Final Results =================\n";
    cout << left << setw(24) << "Total iterations" << ": " << iteration << "\n";
    cout << left << setw(24) << "Initial cut size" << ": " << fixed << setprecision(2) << initialCutSize << "\n";
    cout << left << setw(24) << "Best cut size achieved" << ": " << globalMinCutSize << "\n";
    cout << left << setw(24) << "Overall improvement" << ": " 
         << fixed << setprecision(2) << 100.0f * (1.0f - globalMinCutSize/initialCutSize) << "%\n";
    cout << left << setw(24) << "Total runtime" << ": " << total_duration.count() << " seconds\n";
    
    fout.close();
}

void createDir(const string& dirName) {
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {
        #ifdef _WIN32
            _mkdir(dirName.c_str());
        #else
            mkdir(dirName.c_str(), 0755);
        #endif
    }
}

string getBaseName(const string& path) {
    filesystem::path fp(path);
    return fp.filename().string();
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    createDir("results");
    createDir("pre_saved_EIG");

    if (argc != 2 && argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> [-EIG]" << endl;
        return 1;
    }

    string input_file = argv[1];
    string base_name = getBaseName(input_file);
    fout_name = "results/" + base_name + "_KL_CutSize_output.txt";

    if (argc == 3 && strcmp(argv[2], "-EIG") == 0) {
        EIG_init = true;
        EIG_file = "pre_saved_EIG/" + base_name + "_out.txt";
        fout_name = "results/" + base_name + "_KL_CutSize_EIG_output.txt";
    }

    try {
        sparseMatrix spMat(1);
        InitializeSparsMatrix(input_file, spMat);
        
        #ifdef _OPENMP
            uint32_t num_threads = omp_get_num_procs();
            omp_set_num_threads(num_threads);
            cout << "\n\n=============== OpenMP Thread =================\n";
            cout << "Number of cores: " << num_threads << endl;
        #endif

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
