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

// Eigen includes
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include "eigen/GenEigsSolver.h"
#include "eigen/SymEigsSolver.h"
#include "eigen/MatOp/SparseGenMatProd.h"
#include "eigen/MatOp/SparseSymMatProd.h"
#pragma clang diagnostic pop

using namespace std;
using namespace Eigen;
using namespace Spectra;

typedef Eigen::SparseMatrix<double> SpMat;

double median(VectorXd arr, int size){

    sort(arr.data(), arr.data()+size);

   if (size % 2 != 0) return (double)arr[size/2];
   
   return (double)(arr[(size-1)/2] + arr[size/2])/2.0;
}

void createDir(const std::string& dirName) {
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {  // Directory doesn't exist
        #ifdef _WIN32
            _mkdir(dirName.c_str());
        #else
            mkdir(dirName.c_str(), 0755);  // Read/write for owner, read for others
        #endif
    }
}

string line;

int main(int argc, char *argv[]){
    
    createDir("results");
    createDir("pre_saved_EIG");

    if (argc != 2){
        cout << "Usage: ./cEIG <input_file>" << endl;
        return 0;
    }
    else
    {
        cout << "Reading input file " << argv[1] << endl;
    }

    Eigen::initParallel();

    int n = Eigen::nbThreads( );

    cout << "Number of threads: " << n << endl;

    int nets, nodes;
    
    cout << "read from file" << endl;
    
    string filename = argv[1];
    string outfile = "results/" + filename + "_out.txt";

    ifstream fin(filename.c_str());
    ofstream fout(outfile.c_str());


    if (!fin.is_open() || !fout.is_open()) {
        cout << "Error opening file" << endl;
        return -1;
    }


    getline(fin, line);
    stringstream ss(line);
    ss >> nets >> nodes;

    
    //Initialize matrix
    cout << "Initialize matrix" << endl;
    SpMat mat(nodes, nodes);


    //Read the nets and add the edges to the matrix
    for(int i = 0; i < nets; i++){
        getline(fin, line);
        stringstream ss(line);

        vector<int> list;
        int num;

        while(ss >> num){
            list.push_back(num);
        }

        float weight = 2.0 / (list.size());
        // cout << "line: " << line << endl;
        // cout << list.size() << endl;
        for(int j = 0; j < list.size() - 1; j++){
            for(int k = j+1; k < list.size(); k++){
                mat.coeffRef(list[j]-1, list[k]-1) -= weight;
                mat.coeffRef(list[k]-1, list[j]-1) -= weight;

            }
        }
    }


    //sum the rows and subtract from the diagonal element of the matrix mat
    for(int i = 0; i < nodes; i++){
        double sum = mat.row(i).sum();
        mat.coeffRef(i,i) = -sum;
    }

    //print diagonal elements
    
    //SparseGenMatProd<double> op(mat);
    SparseSymMatProd<double> op(mat);

    SymEigsSolver<SparseSymMatProd<double>> eigs(op, 2, 100);

    eigs.init();
    eigs.compute(SortRule::SmallestAlge);  // Remove nconv since we check success with info()

    Eigen::VectorXd evalues;
    Eigen::MatrixXd evecs;
    if(eigs.info() == CompInfo::Successful){
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
