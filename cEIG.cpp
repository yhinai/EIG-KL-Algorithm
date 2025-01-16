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

using namespace std;
using namespace Eigen;
using namespace Spectra;

typedef Eigen::SparseMatrix<double> SpMat;

double median(VectorXd arr, int size) {
    sort(arr.data(), arr.data()+size);
    if (size % 2 != 0) return (double)arr[size/2];
    return (double)(arr[(size-1)/2] + arr[size/2])/2.0;
}

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

int main(int argc, char *argv[]) {
    createDir("results");
    createDir("pre_saved_EIG");

    if (argc != 2) {
        cout << "Usage: ./EIG <input_file>" << endl;
        return 0;
    }

    cout << "Reading input file " << argv[1] << endl;

    Eigen::initParallel();
    int n = Eigen::nbThreads();
    cout << "Number of threads: " << n << endl;
    
    string filename = argv[1];
    string outfile = "results/" + filename + "_out.txt";

    ifstream fin(filename.c_str());
    ofstream fout(outfile.c_str());

    if (!fin.is_open() || !fout.is_open()) {
        cout << "Error opening file" << endl;
        return -1;
    }

    string line;
    getline(fin, line);
    stringstream ss(line);
    int nets, nodes;
    ss >> nets >> nodes;

    cout << "Initialize matrix" << endl;
    SpMat mat(nodes, nodes);

    for(int i = 0; i < nets; i++) {
        getline(fin, line);
        stringstream ss(line);
        vector<int> list;
        int num;

        while(ss >> num) {
            list.push_back(num);
        }

        float weight = 2.0f / static_cast<float>(list.size());
        
        for(size_t j = 0; j < list.size() - 1; j++) {
            for(size_t k = j + 1; k < list.size(); k++) {
                mat.coeffRef(list[j]-1, list[k]-1) -= weight;
                mat.coeffRef(list[k]-1, list[j]-1) -= weight;
            }
        }
    }

    for(int i = 0; i < nodes; i++) {
        double sum = mat.row(i).sum();
        mat.coeffRef(i,i) = -sum;
    }

    SparseSymMatProd<double> op(mat);
    SymEigsSolver<SparseSymMatProd<double>> eigs(op, 2, 100);

    eigs.init();
    eigs.compute(SortRule::SmallestAlge);

    Eigen::VectorXd evalues;
    Eigen::MatrixXd evecs;
    if(eigs.info() == CompInfo::Successful) {
        evalues = eigs.eigenvalues();
        evecs = eigs.eigenvectors();
    }
    
    double median_eigenvalue = median(evecs.col(0), nodes);
    cout << "median eigenvalue: " << median_eigenvalue << endl;

    fout << evalues(0) << endl;
    fout << median_eigenvalue << endl;
    for (int i = 0; i < nodes; i++) {
        fout << i << "\t" << (median_eigenvalue > evecs.col(0)[i]) << "\t" << evecs.col(0)[i] << endl;
    }

    fout.close();
    return 0;
}