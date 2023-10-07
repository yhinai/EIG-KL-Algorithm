# EIG & KL Hybrid Algorithm Accelerator
 Circuit Partitioning Accelerator utilizing EIG and KL algorithms, leveraging sparse matrix methods and eigenvector computation. Optimize ratio cuts and reduce runtime for large circuits like IBM18. Create a hybrid EIG+KL solution for performance comparison.


# How to Run the Code
1. Clone this repository and run the following commands in the terminal:
```
    cd EIG-KL-Hybrid-Algorithm-Accelerator
    ./run.sh
```

2. For Eigen algorithm:
Compile the code:
```
    g++ cEIG.cpp -std=c++17 -I eigen -o EIG
```
Run the code:
```
    ./EIG <dataset_input_file>
```

3. For KL algorithm Single Thread:
Compile the code:
```
    g++ cKL.cpp -std=c++17 -I eigen -o KL
```
Run the code:
```
    ./KL_single_thread <dataset_input_file>         # produce the Cutsize for each iteration
    ./KL_single_thread <dataset_input_file> -EIG    # produces the Cutsize for each iteration with Eigen soltion as the initial split
```

4. For KL algorithm Multi Thread:
Compile the code:
```
    g++ cKL.cpp -std=c++17 -o KL_multi_thread_omp
```
Run the code:
```
    ./KL_multi_thread_omp <dataset_input_file>         # produce the Cutsize for each iteration
    ./KL_multi_thread_omp <dataset_input_file> -EIG    # produces the Cutsize for each iteration with Eigen soltion as the initial split
```

5. The resutls are saved in the folder (results/) for each implementation 