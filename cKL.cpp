#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <bits/stdc++.h>
#include <chrono>

#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_num_procs() 1
#endif


using namespace std;


bool EIG_init = false;
string EIG_file;
string fout_name;


//build a struct type of an array of two vectors
struct sparseMatrix{
    unsigned int nodeNum;
    vector<vector<int> > Nodes;
    vector<vector<float> > Weights;
    vector<int> split[2];
    vector<int> remain[2];
};

string line;

void InitializeSparsMatrix(string filename, sparseMatrix & spMat){
    long int netsNum, nodesNum;

    ifstream fin;
    fin.open(filename.c_str());
    if (!fin.is_open()) {
        cout << "Error opening file" << endl;
        exit(0);
        
    }
    else{
        cout << "read from file " << filename.c_str() << endl;
    }
    getline(fin, line);
    stringstream ss(line);
    ss >> netsNum >> nodesNum;

    cout << "nets Numbers: " << netsNum << " nodes Numbers: " << nodesNum << endl;

    spMat.nodeNum = nodesNum;


    cout << "Initializing Sparse Matrix" << endl;
    for(unsigned int i = 0; i < nodesNum; i++){
        vector<int> temp;
        vector<float> temp2;
        spMat.Nodes.push_back(temp);
        spMat.Weights.push_back(temp2);
    }
    long int nonZeroElements = nodesNum;
    long int numEdges = 0;
    cout << "Add nodes and weights to Sparse Matrix" << endl << endl;
    
    //Read the nets and add the edges weight to the sparse matrix spMat
    for(unsigned int i = 0; i < netsNum; i++){
        getline(fin, line);
        stringstream ss(line);
        int node;
        vector<int> nodes;
        vector<float> weights;

        while(ss >> node){
            nodes.push_back(node);
        }

        float weight = 1.0 / (nodes.size()-1);

        for(unsigned int j = 0; j < nodes.size(); j++){
            for(unsigned int k = j + 1; k < nodes.size(); k++){
                //find element nodes[k]-1 in SpMat.Nodes[nodes[j]-1] and add to its Weights if it exists, otherwise add the element to SpMat.Nodes[nodes[j]-1] and add the weight to SpMat.Weights[nodes[j]-1]
                //and the same for SpMat.Nodes[nodes[k]-1] and SpMat.Weights[nodes[k]-1]
                numEdges++;
                unsigned int index1 = find(spMat.Nodes[nodes[j]-1].begin(), spMat.Nodes[nodes[j]-1].end(), nodes[k]-1) - spMat.Nodes[nodes[j]-1].begin();
                

                if(index1 == spMat.Nodes[nodes[j]-1].size())
                {
                    // cout << "add " << nodes[k] << " to " << nodes[j] << endl;
                    spMat.Nodes[nodes[j]-1].push_back(nodes[k]-1);
                    spMat.Nodes[nodes[k]-1].push_back(nodes[j]-1);

                    spMat.Weights[nodes[j]-1].push_back(weight);
                    spMat.Weights[nodes[k]-1].push_back(weight);

                    nonZeroElements += 2;
                    

                }
                else
                {
                    int index2 = find(spMat.Nodes[nodes[k]-1].begin(), spMat.Nodes[nodes[k]-1].end(), nodes[j]-1) - spMat.Nodes[nodes[k]-1].begin();
                    spMat.Weights[nodes[j]-1][index1] += weight;
                    spMat.Weights[nodes[k]-1][index2] += weight;
                }

                // cout << endl;
            }
        }
    }

    cout << "Size of matrix: " << (long int)(nodesNum*nodesNum) << endl;
    cout << "Non-Zero Elements: " << nonZeroElements << endl;
    cout << "Ratio: " << (double) 100.0 * nonZeroElements / (nodesNum*nodesNum) << "%"<< endl;
    cout << "node-to-node Edges: " << numEdges << endl << endl;

}

void shuffleSparceMatrix(sparseMatrix & spMat){
    
    //clear the split and remain vectors
    for(unsigned int i = 0; i < 2; i++){
        spMat.split[i].clear();
        spMat.remain[i].clear();
    }

    if (EIG_init){
        ifstream fEIG;

        fEIG.open(EIG_file);

        if (!fEIG.is_open()){
            cout << "Error: EIG file not found" << endl;
            exit(0);
        }

        string line;
        
        //read the first two lines of the EIG file
        getline(fEIG, line);
        getline(fEIG, line);

        //read the next lines in interations, eachline has two ints in it and a double element at the end
        while(getline(fEIG, line)){
            stringstream ss(line);
            int i, Split_side;
            double weight;
            ss >> i >> Split_side >> weight;

            // if the node2 is zero then it is in split[0] else if one then it is in split[1]            
            if(Split_side == 0){
                spMat.split[0].push_back(i);
                spMat.remain[0].push_back(i);
            }
            else{
                spMat.split[1].push_back(i);
                spMat.remain[1].push_back(i);
            }
        }

        fEIG.close();

        //print out size of the split and remain vectors
        cout << "split[0]: " << spMat.split[0].size() << endl;
        cout << "split[1]: " << spMat.split[1].size() << endl;


        return;
    }


    // shuffle random numbers from 0 to nodes-1 and assgin half of it to one vector and the other half to another vector
    vector<int> random;
    for(unsigned int i = 0; i < spMat.nodeNum; i++){
        random.push_back(i);
    }

    random_shuffle(random.begin(), random.end());



    for(unsigned int i = 0; i < spMat.nodeNum/2; i++){
        spMat.split [0].push_back(random[i]);
        spMat.remain[0].push_back(random[i]);
    }
    for(unsigned int i = spMat.nodeNum/2; i < spMat.nodeNum; i++){
        spMat.split [1].push_back(random[i]);
        spMat.remain[1].push_back(random[i]);
    }

    //print out size of the split and remain vectors
    cout << "split size: " << spMat.split[0].size() << endl;
    cout << "remain size: " << spMat.remain[0].size() << endl;


}



float calCutSize(sparseMatrix &spMat){
    float E = 0;

    #pragma omp parallel for reduction(+:E)
    for(unsigned int i = 0; i < spMat.remain[0].size(); i++){
        int rightIdx = spMat.remain[0][i];
        vector<int> node = spMat.Nodes[rightIdx];
        vector<float> weight = spMat.Weights[rightIdx];


        for(unsigned int j = 0; j < node.size(); j++){
            if(find(spMat.remain[1].begin(), spMat.remain[1].end(), node[j]) != spMat.remain[1].end()){
                E += weight[j];
            }
        }
    }

    return E;
}



float connections(sparseMatrix &spMat, int a){
    float E = 0;
    float I = 0;
    vector<int> node = spMat.Nodes[a];
    vector<float> weight = spMat.Weights[a];
    // cout << "node: [" << a << "] " << endl;

    for(unsigned int i = 0; i < node.size(); i++){
        if(find(spMat.split[0].begin(), spMat.split[0].end(), node[i]) != spMat.split[0].end()){
            I += weight[i];
            // cout << "I: " << node[i] << "\t" << weight[i] <<  endl;
        }
        else{
            E += weight[i];
            // cout << "E: " << node[i] << "\t" << weight[i] << endl;
        }
    }

    return E - I;
}



float nodeConnection(sparseMatrix &spMat, int a, int b){
    vector<int> node = spMat.Nodes[a];
    vector<float> weight = spMat.Weights[a];

    for(unsigned int i = 0; i < node.size(); i++){
        if(node[i] == b){
            return weight[i];
        }
    }

    return 0.0;
}




void swip(sparseMatrix &spMat, int num1, int num2){
    //remove num1 and num2 from remainNodes[0] and remainNodes[1]
    spMat.remain[0].erase(find(spMat.remain[0].begin(), spMat.remain[0].end(), num1));
    spMat.remain[1].erase(find(spMat.remain[1].begin(), spMat.remain[1].end(), num2));

    //find the index of num1 and num2 in splitNodes[0] and splitNodes[1]
    int idx_1 = find(spMat.split[0].begin(), spMat.split[0].end(), num1) - spMat.split[0].begin();
    int idx_2 = find(spMat.split[1].begin(), spMat.split[1].end(), num2) - spMat.split[1].begin();

    //swip num1 and num2 in splitNodes[0] and splitNodes[1]
    spMat.split[0][idx_1] = num2;
    spMat.split[1][idx_2] = num1;
}



float gloableMin = numeric_limits<float>::max();


void KL(sparseMatrix &spMat)
{
    cout << "Starting KL" << endl;
    shuffleSparceMatrix(spMat);

    //open file to write
    ofstream fout(fout_name.c_str());
    
    if(!fout.is_open()){
        cout << "Error: can not open file to write" << endl;
        exit(1);
    }
    

    int count = 0;
    int terminate = 0;
    int terminateLimit = log2(spMat.Nodes.size()) + 5;
    
    int num_cores = omp_get_num_procs();

    float cutSize = calCutSize(spMat);
    float minCutSize = cutSize;
    fout << 0 << "\t" << cutSize << "\t" << 0 << endl;
    cout << "Starting cutSize: " << cutSize << endl;

    while(spMat.remain[0].size() != 0 && spMat.remain[1].size() != 0)
    {
        // calcualte the time of each iteration
        clock_t start = clock();

        // calcualte the real time of each iteration
        auto t_start = std::chrono::high_resolution_clock::now();


        //save connections(spMat, spMat.remain[0][j]) into a vector of size num_cores each element is a partal size connections(spMat, spMat.remain[0][j]) 
        vector<vector<float> > con_1_partition(num_cores);
        vector<vector<float> > con_2_partition(num_cores);

        int part_size_1 = spMat.remain[0].size() / num_cores;
        int part_size_2 = spMat.remain[1].size() / num_cores;

        #pragma omp parallel for 
        for(int i = 0; i < num_cores; i++){

            int start_1 = i * part_size_1;
            //if end_1 is larger than the size of con_1, then set it to the size of con_1
            int end_1 = start_1 + part_size_1;
            if(end_1 > spMat.remain[0].size()) end_1 = spMat.remain[0].size();
        

            int start_2 = i * part_size_2;
            //if end_2 is larger than the size of con_2, then set it to the size of con_2
            int end_2 = start_2 + part_size_2;
            if(end_2 > spMat.remain[1].size()) end_2 = spMat.remain[1].size();


            for(int j = start_1; j < end_1; j++){
                con_1_partition[i].push_back(connections(spMat, spMat.remain[0][j]));
            }

            for(int j = start_2; j < end_2; j++){
                con_2_partition[i].push_back(connections(spMat, spMat.remain[1][j]));
            }

        }


        //calculate the total connections of each partition
        float local_max[num_cores];
        int local_max_idx[num_cores];

        float local_min[num_cores];
        int local_min_idx[num_cores];

        
        #pragma omp parallel for
        for(int i = 0; i < num_cores; i++){
            
            local_max[i] = con_1_partition[i][0];
            local_max_idx[i] = 0;
            local_min[i] = con_2_partition[i][0];
            local_min_idx[i] = 0;

            for(int j = 1; j < con_1_partition[i].size(); j++){
                if(con_1_partition[i][j] > local_max[i]){
                    local_max[i] = con_1_partition[i][j];
                    local_max_idx[i] = part_size_2 * i + j;
                }
            }

            for(int j = 1; j < con_2_partition[i].size(); j++){
                if(con_2_partition[i][j] < local_min[i]){
                    local_min[i] = con_2_partition[i][j];
                    local_min_idx[i] = part_size_2 * i + j;
                }
            }
        }


        float global_max = local_max[0];
        int global_max_idx = local_max_idx[0];

        float global_min = local_min[0];
        int global_min_idx = local_min_idx[0];


        for(int i = 1; i < num_cores; i++){
            if(local_max[i] > global_max){
                global_max = local_max[i];
                global_max_idx = local_max_idx[i];
            }
            if(local_min[i] < global_min){
                global_min = local_min[i];
                global_min_idx = local_min_idx[i];
            }
        }        

        //swip the node with the maximum connection with the node with the minimum connection
        float gain = global_max - global_min - 2*nodeConnection(spMat, spMat.remain[0][global_max_idx], spMat.remain[1][global_min_idx]);
        

        cutSize = cutSize - gain;
        if(cutSize < minCutSize){
            minCutSize = cutSize;
        }

        swip(spMat, spMat.remain[0][global_max_idx], spMat.remain[1][global_min_idx]);


        //end of iteration
        clock_t end = clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        
        double elapsed_secs = double(end - start) / CLOCKS_PER_SEC; 
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0;
            

        cout << "iteration: " << ++count << "\t cutSize: " << cutSize << "\t gain: " << gain << endl;
        fout << count << "\t" << cutSize << "\t" << gain << endl;
        cout << "CPU time: " << elapsed_secs << " sec \treal time: " << elapsed_time_ms << " sec" << endl << endl;


        if(gain <= 0){
            terminate++;
            if(terminate > terminateLimit){
                break;
            }
        }
        else{
            terminate = 0;
        }

    }

    if (minCutSize < gloableMin) gloableMin = minCutSize;

    fout.close();
}



int main(int argc, char *argv[]){

    srand(time(NULL));


    if(argc != 2 && argc != 3){
        cout << "Usage: ./main <input file> <optional -EIG>" << endl;
        return 0;
    }
    string input_file = argv[1];
    fout_name = "results/" + input_file + "_KL_CutSize_output.txt";


    if (argc == 3){
        if(strcmp(argv[2], "-EIG") == 0){
            EIG_init = true;
            EIG_file = "pre_saved_EIG/" + input_file + "_out.txt";
            fout_name = "results/" + input_file + "_KL_CutSize_EIG_output.txt";
        }
        
    }


    sparseMatrix spMat;
    InitializeSparsMatrix(input_file, spMat);

    //preform KL 1 time
    for(int i = 0; i < 1; i++){
        KL(spMat);
        cout << "gloableMin: " << gloableMin << endl;
    }

    
    return 0;
}
