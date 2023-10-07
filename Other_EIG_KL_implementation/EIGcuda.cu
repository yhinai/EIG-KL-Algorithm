#include <stdio.h>
#include <iostream>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>


using namespace std;

const int num_mat = 1; // total number of matrices = total number of threads
// const int N = 149;   // square symmetric matrix dimension
const int nTPB = 1024;  // threads per block

// test symmetric matrices

  // float a1[N*N] = {
  //     4.0,  -30.0,    60.0,   -35.0, 
  //   -30.0,  300.0,  -675.0,   420.0, 
  //    60.0, -675.0,  1620.0, -1050.0, 
  //   -35.0,  420.0, -1050.0,   700.0 };

  // float a2[N*N] = {
  //   4.0, 0.0, 0.0, 0.0, 
  //   0.0, 1.0, 0.0, 0.0, 
  //   0.0, 0.0, 3.0, 0.0, 
  //   0.0, 0.0, 0.0, 2.0 };

  // float a3[N*N] = {
  //   -2.0,   1.0,   0.0,   0.0,
  //    1.0,  -2.0,   1.0,   0.0,
  //    0.0,   1.0,  -2.0,   1.0,
  //    0.0,   0.0,   1.0,  -2.0 }; 


/* ---------------------------------------------------------------- */
//
// the following functions come from here:
//
// https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_eigenvalue/jacobi_eigenvalue.cpp
//
// attributed to j. burkardt, FSU
// they are unmodified except to add __host__ __device__ decorations
//
//****************************************************************************80
__host__ __device__
void r8mat_diag_get_vector ( int n, float a[], float v[] )

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_DIAG_GET_VECTOR gets the value of the diagonal of an R8MAT.
//
//  Discussion:
//
//    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
//    in column-major order.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    15 July 2013
//
//  Author:
//
//    John Burkardt
//
//  Input:
//
//    int N, the number of rows and columns of the matrix.
//
//    float A[N*N], the N by N matrix.
//
//  Output:
//
//    float V[N], the diagonal entries
//    of the matrix.
//
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    v[i] = a[i+i*n];
  }

  return;
}
/* PASTE IN THE CODE HERE, FROM THE ABOVE LINK, FOR THIS FUNCTION */
//****************************************************************************80
__host__ __device__
void r8mat_identity ( int n, float a[] )

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_IDENTITY sets the square matrix A to the identity.
//
//  Discussion:
//
//    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
//    in column-major order.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    01 December 2011
//
//  Author:
//
//    John Burkardt
//
//  Input:
//
//    int N, the order of A.
//
//  Output:
//
//    float A[N*N], the N by N identity matrix.
//
{
  int i;
  int j;
  int k;

  k = 0;
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      if ( i == j )
      {
        a[k] = 1.0;
      }
      else
      {
        a[k] = 0.0;
      }
      k = k + 1;
    }
  }

  return;
}
//****************************************************************************80
__host__ __device__
void jacobi_eigenvalue ( int n, float a[], int it_max, float v[], 
  float d[], int &it_num, int &rot_num )

//****************************************************************************80
//
//  Purpose:
//
//    JACOBI_EIGENVALUE carries out the Jacobi eigenvalue iteration.
//
//  Discussion:
//
//    This function computes the eigenvalues and eigenvectors of a
//    real symmetric matrix, using Rutishauser's modfications of the classical
//    Jacobi rotation method with threshold pivoting. 
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 September 2013
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Gene Golub, Charles VanLoan,
//    Matrix Computations,
//    Third Edition,
//    Johns Hopkins, 1996,
//    ISBN: 0-8018-4513-X,
//    LC: QA188.G65.
//
//  Input:
//
//    int N, the order of the matrix.
//
//    float A[N*N], the matrix, which must be square, real,
//    and symmetric.
//
//    int IT_MAX, the maximum number of iterations.
//
//  Output:
//
//    float V[N*N], the matrix of eigenvectors.
//
//    float D[N], the eigenvalues, in descending order.
//
//    int &IT_NUM, the total number of iterations.
//
//    int &ROT_NUM, the total number of rotations.
//
{
  float *bw;
  float c;
  float g;
  float gapq;
  float h;
  int i;
  int j;
  int k;
  int l;
  int m;
  int p;
  int q;
  float s;
  float t;
  float tau;
  float term;
  float termp;
  float termq;
  float theta;
  float thresh;
  float w;
  float *zw;

  r8mat_identity ( n, v );

  r8mat_diag_get_vector ( n, a, d );

  bw = new float[n];
  zw = new float[n];

  for ( i = 0; i < n; i++ )
  {
    bw[i] = d[i];
    zw[i] = 0.0;
  }
  it_num = 0;
  rot_num = 0;

  while ( it_num < it_max )
  {
    it_num = it_num + 1;
//
//  The convergence threshold is based on the size of the elements in
//  the strict upper triangle of the matrix.
//
    thresh = 0.0;
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < j; i++ )
      {
        thresh = thresh + a[i+j*n] * a[i+j*n];
      }
    }

    thresh = sqrt ( thresh ) / ( float ) ( 4 * n );

    if ( thresh == 0.0 )
    {
      break;
    }

    for ( p = 0; p < n; p++ )
    {
      for ( q = p + 1; q < n; q++ )
      {
        gapq = 10.0 * fabs ( a[p+q*n] );
        termp = gapq + fabs ( d[p] );
        termq = gapq + fabs ( d[q] );
//
//  Annihilate tiny offdiagonal elements.
//
        if ( 4 < it_num &&
             termp == fabs ( d[p] ) &&
             termq == fabs ( d[q] ) )
        {
          a[p+q*n] = 0.0;
        }
//
//  Otherwise, apply a rotation.
//
        else if ( thresh <= fabs ( a[p+q*n] ) )
        {
          h = d[q] - d[p];
          term = fabs ( h ) + gapq;

          if ( term == fabs ( h ) )
          {
            t = a[p+q*n] / h;
          }
          else
          {
            theta = 0.5 * h / a[p+q*n];
            t = 1.0 / ( fabs ( theta ) + sqrt ( 1.0 + theta * theta ) );
            if ( theta < 0.0 )
            {
              t = - t;
            }
          }
          c = 1.0 / sqrt ( 1.0 + t * t );
          s = t * c;
          tau = s / ( 1.0 + c );
          h = t * a[p+q*n];
//
//  Accumulate corrections to diagonal elements.
//
          zw[p] = zw[p] - h;                 
          zw[q] = zw[q] + h;
          d[p] = d[p] - h;
          d[q] = d[q] + h;

          a[p+q*n] = 0.0;
//
//  Rotate, using information from the upper triangle of A only.
//
          for ( j = 0; j < p; j++ )
          {
            g = a[j+p*n];
            h = a[j+q*n];
            a[j+p*n] = g - s * ( h + g * tau );
            a[j+q*n] = h + s * ( g - h * tau );
          }

          for ( j = p + 1; j < q; j++ )
          {
            g = a[p+j*n];
            h = a[j+q*n];
            a[p+j*n] = g - s * ( h + g * tau );
            a[j+q*n] = h + s * ( g - h * tau );
          }

          for ( j = q + 1; j < n; j++ )
          {
            g = a[p+j*n];
            h = a[q+j*n];
            a[p+j*n] = g - s * ( h + g * tau );
            a[q+j*n] = h + s * ( g - h * tau );
          }
//
//  Accumulate information in the eigenvector matrix.
//
          for ( j = 0; j < n; j++ )
          {
            g = v[j+p*n];
            h = v[j+q*n];
            v[j+p*n] = g - s * ( h + g * tau );
            v[j+q*n] = h + s * ( g - h * tau );
          }
          rot_num = rot_num + 1;
        }
      }
    }

    for ( i = 0; i < n; i++ )
    {
      bw[i] = bw[i] + zw[i];
      d[i] = bw[i];
      zw[i] = 0.0;
    }
  }
//
//  Restore upper triangle of input matrix.
//
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < j; i++ )
    {
      a[i+j*n] = a[j+i*n];
    }
  }
//
//  Ascending sort the eigenvalues and eigenvectors.
//
  for ( k = 0; k < n - 1; k++ )
  {
    m = k;
    for ( l = k + 1; l < n; l++ )
    {
      if ( d[l] < d[m] )
      {
        m = l;
      }
    }

    if ( m != k )
    {
      t    = d[m];
      d[m] = d[k];
      d[k] = t;
      for ( i = 0; i < n; i++ )
      {
        w        = v[i+m*n];
        v[i+m*n] = v[i+k*n];
        v[i+k*n] = w;
      }
    }
  }

  delete [] bw;
  delete [] zw;

  return;
}

__global__ void je(int num_matr, int n, float *a, int it_max, float *v, float *d){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  int it_num;
  int rot_num;
  if (idx < num_matr){
    jacobi_eigenvalue(n, a+(idx*n*n), it_max, v+(idx*n*n), d+(idx*n), it_num, rot_num);
  }
}

void initialize_matrix(int mat_id, int n, float *mat, float *v){

  for (int i = 0; i < n*n; i++) *(v+(mat_id*n*n)+i) = mat[i];
}

void print_vec(int vec_id, int n, float *d){

  std::cout << "matrix " << vec_id << " eigenvalues: " << std::endl;
  for (int i = 0; i < n; i++) std::cout << i << ": " << *(d+(n*vec_id)+i) << std::endl;
  std::cout << std::endl;
}

string line;

int main(int argc, char *argv[]){
// make sure device heap has enough space for in-kernel new allocations
  // const int heapsize = N*sizeof(float)*2;
  // const int chunks = heapsize/(8192*1024) + 1;

  // cudaError_t cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, (8192*1024) * chunks);
  // if (cudaStatus != cudaSuccess) {
  //       fprintf(stderr, "set device heap limit failed!");
  //   }
    
  int nets, nodes;

  cout << "read from file" << endl;
  ifstream fin;
  
  if (argc > 1){
    fin.open(argv[1]);
  }
  else{
    fin.open("fract.hgr");
  }

  if (!fin.is_open()) {
      cout << "Error opening file" << endl;
      return -1;
  }

  getline(fin, line);
  stringstream ss(line);
  ss >> nets >> nodes;

  const int N = nodes;
  const int max_iter = N*N;

  float *h_a, *d_a, *h_v, *d_v, *h_d, *d_d;
  h_a = (float *)malloc(num_mat*N*N*sizeof(float));
  h_v = (float *)malloc(num_mat*N*N*sizeof(float));
  h_d = (float *)malloc(num_mat*  N*sizeof(float));

  cudaMalloc(&d_a, num_mat*N*N*sizeof(float));
  cudaMalloc(&d_v, num_mat*N*N*sizeof(float));
  cudaMalloc(&d_d, num_mat*  N*sizeof(float));

  memset(h_a, 0, num_mat*N*N*sizeof(float));
  memset(h_v, 0, num_mat*N*N*sizeof(float));
  memset(h_d, 0, num_mat*  N*sizeof(float));


    cout << "nets: " << nets << " nodes: " << nodes << endl;
    //Initialize matrix
    cout << "Initialize matrix" << endl;

    float a1[nodes * nodes];


    //initialize matrix a1 elements to 0
    for (int i = 0; i < nodes * nodes; i++) {
        a1[i] = 0;
    }

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
                a1[ (list[j]-1) * nodes + (list[k]-1) ] -= weight;
                a1[ (list[k]-1) * nodes + (list[j]-1) ] -= weight;
            }
        }
    }
    


    //sum the rows and subtract from the diagonal element of the matrix a1
    for(int i = 0; i < nodes; i++){
        float sum = 0;
        for(int j = 0; j < nodes; j++){
            sum += a1[i * nodes + j];
        }
        a1[i * nodes + i] -= sum;
    }


    

  initialize_matrix(0, N*N, a1, h_a);
  
  // initialize_matrix(1, N, a2, h_a);
  // initialize_matrix(2, N, a3, h_a);

  cudaMemcpy(d_a, h_a, num_mat*N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, num_mat*N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, h_d, num_mat*  N*sizeof(float), cudaMemcpyHostToDevice);

  je<<<(num_mat+nTPB-1)/nTPB, nTPB>>>(num_mat, N, d_a, max_iter, d_v, d_d);

  cudaMemcpy(h_d, d_d, num_mat*N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_v, d_v, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  
  //find the smallest element value in h_d and its index in the array. then print them.
  // float min_val = h_d[0];
  // int min_idx = 0;
  // for (int i = 0; i < num_mat; i++){
  //   for (int j = 0; j < N; j++){
  //     if (h_d[i*N+j] < min_val){
  //       min_val = h_d[i*N+j];
  //       min_idx = i;
  //     }
  //   }
  // }

  cout << h_d[1] << endl;
  //print the eigenvector corresponding to the smallest eigenvalue
  print_vec(1, N, h_v);
  


  return 0;
}
