#include "fused_distance_nn/l2_exp.cuh"
#include "fused_distance_nn/cutlass_base_customized.cuh"
#include "header.cuh"
#include <cuda_runtime.h>
#include <iostream>

using namespace my_test;
// to be modified!!!!!
int main(){
    using IdxT = int;
    using DataT = float;
    using L2Op                  = l2_exp_cutlass_op<DataT, DataT>;
    using OutT = KeyValuePair<IdxT, DataT>;
    using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
    using MinAndDistanceReduceOp = MinAndDistanceReduceOpImpl<IdxT, DataT>;
    using KVPMinReduce = KVPMinReduceImpl<IdxT, DataT>;

    OutT* min, *dmin;
    DataT* x, *dx;
    DataT* y, *dy;
    DataT* xn, *dxn;
    DataT* yn, *dyn;
    IdxT m = 32768;
    IdxT n = 128;
    IdxT k = 128;
    int* workspace, *dworkspace;
    MinAndDistanceReduceOp redOp;
    KVPMinReduce pairRedOp;
    bool sqrt = false;
    cudaStream_t stream;
    cudaStreamCreate ( &stream) ;

 
    int lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;
    kvp_cg_min_reduce_op_ cg_reduce_op;
    L2Op L2_dist_op(sqrt);

    cudaMalloc((void**)&dx, sizeof(DataT) * m * k);
    cudaMalloc((void**)&dy, sizeof(DataT) * n * k);
    cudaMalloc((void**)&dxn, sizeof(DataT) * m);
    cudaMalloc((void**)&dyn, sizeof(DataT) * n);
    cudaMalloc((void**)&dmin, sizeof(OutT) * m);
    cudaMalloc((void**)&dworkspace, sizeof(IdxT));
    // Sample
    x = (DataT*)malloc(sizeof(DataT) * m * k);
    // Centroids
    y = (DataT*)malloc(sizeof(DataT) * n * k);
    // X-norm
    xn = (DataT*)malloc(sizeof(DataT) * m);
    // Y-norm
    yn = (DataT*)malloc(sizeof(DataT) * n);
    // Best-fit Centroids
    min = (OutT*)malloc(sizeof(OutT) * m); 
    workspace = (IdxT*)malloc(sizeof(IdxT));
    *workspace = 0;
    
    for(int i = 0; i < m * k / 2; ++i)     x[i] = 1;
    for(int i = 0; i < n * k / 2; ++i)     y[i] = 1; 
    for(int i = m * k / 2; i < m * k; ++i) x[i] = 2;
    for(int i = n * k / 2; i < n * k; ++i) y[i] = 2; 
    for(int i = 0; i < m / 2; ++i)     xn[i] = 1 * k;
    for(int i = 0; i < n / 2; ++i)     yn[i] = 1 * k; 
    for(int i = m / 2; i < m; ++i) xn[i] = 4 * k;
    for(int i = n / 2; i < n; ++i) yn[i] = 4 * k; 
    for(int i = 0; i < m; ++i) min[i].value = std::numeric_limits<float>::max(); 
    for(int i = 0; i < m; ++i) min[i].key = -1; 

    // for(int i = 0; i < m; ++i) printf("xn, id = %d, norm = %f\n", i, xn[i]);
    // for(int i = 0; i < n; ++i) printf("yn, id = %d, norm = %f\n", i, yn[i]);
    // for(int i = 0; i < m * k; ++i) printf("x, id = %d, val = %f\n", int(i / k), x[i]);
    // for(int i = 0; i < n * k; ++i) printf("y, id = %d, val = %f\n", int(i / k), y[i]);
  
    cudaMemcpy((void*)dx, (void*)x, sizeof(DataT) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dy, (void*)y, sizeof(DataT) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dxn, (void*)xn, sizeof(DataT) * m, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dyn, (void*)yn, sizeof(DataT) * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dmin, (void*)min, sizeof(OutT) * m, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dworkspace, (void*)workspace, sizeof(IdxT), cudaMemcpyHostToDevice);
  
    cudaDeviceSynchronize();
				cudaEvent_t beg, end;
    			cudaEventCreate(&beg);
    			cudaEventCreate(&end);
    			float elapsed;
		cudaEventRecord(beg);
    cutlassFusedDistanceNN_codegen<DataT,
                           DataT,
                           OutT,
                           IdxT,
                           16 / sizeof(DataT),
                           kvp_cg_min_reduce_op_,
                           L2Op, 
                           MinAndDistanceReduceOp,
                           KVPMinReduce>(dx,
                                         dy,
                                         dxn,
                                         dyn,
                                         m, 
                                         n,
                                         k, 
                                         lda,
                                         ldb,
                                         ldd,
                                         dmin,
                                         dworkspace,
                                         cg_reduce_op,
                                         L2_dist_op,
                                         redOp,
                                         pairRedOp,
                                         stream);
  cudaDeviceSynchronize();
	cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, beg, end);
  double gflops = (double(2 * 1 * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
  //printf("%d, %d, %d, %f, %f\n", m, n, k, elapsed, gflops);
  cudaMemcpy((void*)x, (void*)dx, sizeof(DataT) * m * k, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)y, (void*)dy, sizeof(DataT) * n * k, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)xn, (void*)dxn, sizeof(DataT) * m, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)yn, (void*)dyn, sizeof(DataT) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)min, (void*)dmin, sizeof(OutT) * m, cudaMemcpyDeviceToHost);

  // for(int i = 0; i < m; ++i){
  //   printf("id=%d, class_id = %d, dis = %f\n", i, min[i].key, min[i].value);
  // }
  for (int i = 0; i < m/2; i++)
    if (min[i].value > 0.1 ) {
      printf("Wrong!");
      return 0;
    }

    // cudaError_t cudaError = cudaGetLastError();
    // if (cudaError != cudaSuccess) {
    //     std::cerr << "CUDA Error (cudaMalloc): " << cudaGetErrorString(cudaError) << std::endl;
    //     return 1;
    // }
  
  // update_centroids

  // for(int i = 0; i < m; ++i) printf("xn, id = %d, norm = %f\n", i, xn[i]);
  // for(int i = 0; i < n; ++i) printf("yn, id = %d, norm = %f\n", i, yn[i]);
  // for(int i = 0; i < m * k; ++i) printf("x, id = %d, val = %f\n", i / k, x[i]);
  // for(int i = 0; i < n * k; ++i) printf("y, id = %d, val = %f\n", i / k, y[i]);
  return 0;
}