#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <system_error>
#include <omp.h>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <mutex>

// Eigen and Spectra Headers with Warning Suppressions
#ifdef __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-but-set-variable"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#ifdef __clang__
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

// Namespace and Type Definitions
using namespace std;
using namespace Eigen;
using namespace Spectra;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;


// Utility Functions
// Calculate median value - kept single-threaded for consistency
double median(const VectorXd& arr, int size) {
    if (size <= 0) return 0.0;
    
    VectorXd sorted = arr;
    std::sort(sorted.data(), sorted.data() + size);
    
    if (size % 2 != 0) {
        return sorted[size/2];
    }
    return (sorted[(size-1)/2] + sorted[size/2]) / 2.0;
}

// Create directory with error handling
void createDir(const std::string& dirName) {
    try {
        filesystem::create_directories(dirName);
    } catch (const filesystem::filesystem_error& e) {
        cerr << "Error creating directory " << dirName << ": " << e.what() << endl;
        throw;
    }
}

// Extract base filename from path
string getBaseName(const string& path) {
    return filesystem::path(path).filename().string();
}




// Initialize sparse matrix with controlled parallelism
SpMat initializeMatrix(ifstream& fin, int nodes, int nets) {
    vector<T> triplets;
    triplets.reserve(nets * 10);
    
    // Read all nets first to maintain correct order
    vector<vector<int>> netLists(nets);
    string line;
    
    for(int i = 0; i < nets; i++) {
        getline(fin, line);
        stringstream ss(line);
        int num;
        while(ss >> num) {
            netLists[i].push_back(num - 1);  // Convert to 0-based indexing
        }
    }
    
    // Process nets in parallel with controlled access
    mutex triplets_mutex;
    #pragma omp parallel for schedule(dynamic, 32)
    for(int i = 0; i < nets; i++) {
        vector<T> local_triplets;
        const auto& list = netLists[i];
        
        double weight = 2.0 / static_cast<double>(list.size());
        
        for(size_t j = 0; j < list.size() - 1; j++) {
            for(size_t k = j + 1; k < list.size(); k++) {
                local_triplets.emplace_back(list[j], list[k], -weight);
                local_triplets.emplace_back(list[k], list[j], -weight);
            }
        }
        
        lock_guard<mutex> lock(triplets_mutex);
        triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
    }
    
    SpMat mat(nodes, nodes);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    
    // Calculate diagonal elements sequentially for consistency
    for(int i = 0; i < nodes; i++) {
        double sum = mat.row(i).sum();
        mat.coeffRef(i, i) = -sum;
    }
    
    return mat;
}

//-----------------------------------------------------------------------------
// Main Function
//-----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    auto start_time = chrono::high_resolution_clock::now();
    
    try {
        // Argument validation
        if (argc != 2) {
            throw runtime_error("Usage: ./EIG <input_file>");
        }
        
        // Create output directories
        createDir("results");
        createDir("pre_saved_EIG");
        
        // Configure thread settings
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        Eigen::setNbThreads(1);  // Keep Eigen single-threaded for consistency
        
        // Print initialization information
        cout << "\n============= Initialization =============\n";
        cout << "OpenMP threads: " << num_threads << endl;
        cout << "Eigen threads: " << Eigen::nbThreads() << endl;
        
        // Setup file paths
        string filename = argv[1];
        string base_name = getBaseName(filename);
        string outfile = "pre_saved_EIG/" + base_name + "_out.txt";
        
        // Open input and output files
        ifstream fin(filename);
        ofstream fout(outfile);
        
        if (!fin.is_open()) {
            throw runtime_error("Error opening input file: " + filename);
        }
        if (!fout.is_open()) {
            throw runtime_error("Error opening output file: " + outfile);
        }
        
        // Read problem dimensions
        string line;
        getline(fin, line);
        stringstream ss(line);
        int nets, nodes;
        ss >> nets >> nodes;
        
        cout << "\nProblem Size:\n";
        cout << "  - Nets: " << nets << "\n";
        cout << "  - Nodes: " << nodes << "\n";
        
        // Initialize and compute matrix
        cout << "\nInitializing sparse matrix...\n";
        SpMat mat = initializeMatrix(fin, nodes, nets);
        
        // Compute eigenvalues
        cout << "Computing eigenvalues...\n";
        SparseSymMatProd<double> op(mat);
        SymEigsSolver<SparseSymMatProd<double>> eigs(op, 2, min(100, nodes/2));
        
        eigs.init();
        eigs.compute(SortRule::SmallestAlge);
        
        if(eigs.info() != CompInfo::Successful) {
            throw runtime_error("Eigenvalue computation failed");
        }
        
        // Extract results
        VectorXd evalues = eigs.eigenvalues();
        MatrixXd evecs = eigs.eigenvectors();
        VectorXd firstEigenvector = evecs.col(0);
        
        double median_val = median(firstEigenvector, nodes);
        
        // Write results
        cout << "\nWriting results...\n";
        fout << std::setprecision(12) << evalues(0) << endl;
        fout << std::setprecision(12) << median_val << endl;
        
        // Write node assignments sequentially
        for (int i = 0; i < nodes; i++) {
            fout << i << "\t" << (median_val > firstEigenvector[i]) << "\t" 
                 << std::setprecision(12) << firstEigenvector[i] << endl;
        }
        
        // Print summary
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        
        cout << "\n============= Summary =============\n";
        cout << "Execution time: " << duration.count() / 1000.0 << " seconds\n";
        cout << "Results written to: " << outfile << "\n";
        cout << "================================\n\n";
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}