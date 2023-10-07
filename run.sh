scl enable devtoolset-9 bash
ulimit -s unlimited

g++ cEIG.cpp -std=c++17 -I eigen -o EIG

g++ cKL.cpp -std=c++17 -o KL_single_thread

g++ cKL.cpp -std=c++17 -fopenmp -o KL_multi_thread_omp

