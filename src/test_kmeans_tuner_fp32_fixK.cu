#include <cuda_runtime.h>
#include <stdio.h>
#include "fused_distance_nn/l2_exp.cuh"
#include "fused_distance_nn/cutlass_base_customized.cuh"
#include "header.cuh"
#include "helper.h"
#include "cstdlib"

#include <time.h>
#include <sys/time.h>
#include <cstdint>
using namespace my_test;
using DataT = float;
using IdxT = int;
using OutT = KeyValuePair<IdxT, DataT>;
using L2Op                  = l2_exp_cutlass_op<DataT, DataT>;
using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
using MinAndDistanceReduceOp = MinAndDistanceReduceOpImpl<IdxT, DataT>;
using KVPMinReduce = KVPMinReduceImpl<IdxT, DataT>;
const int num_tests = 10;

// double getTime(){
//   struct timeval tp;
//   gettimeofday(&tp,NULL);
//   return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
// }

void DistanceNN(int num, int m, int n, int k, double &Gflops, double &Elapsed, int &max_num,
                OutT* min[num_tests], OutT* min1[num_tests], OutT* dmin[num_tests], 
                DataT* x[num_tests], DataT* dx[num_tests],
                DataT* y[num_tests], DataT* dy[num_tests],
                DataT* xn[num_tests], DataT* dxn[num_tests],
                DataT* yn[num_tests], DataT* dyn[num_tests],
                int* workspace[num_tests], int* dworkspace[num_tests],
                cudaStream_t stream){
    MinAndDistanceReduceOp redOp;
    KVPMinReduce pairRedOp;
    bool sqrt = false;

    int lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;
    kvp_cg_min_reduce_op_ cg_reduce_op;
    L2Op L2_dist_op(sqrt);

    for (int i = 0; i < num_tests; i ++){
        cudaMemcpy((void*)dmin[i], (void*)min[i], sizeof(OutT) * m, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)dworkspace[i], (void*)workspace[i], sizeof(IdxT), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
	cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    // for (int i = 0; i < 1; i++) {
    // if (num ==0) cutlassFusedDistanceNN_0<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 3) cutlassFusedDistanceNN_3<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 9) cutlassFusedDistanceNN_9<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 13) cutlassFusedDistanceNN_13<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 22) cutlassFusedDistanceNN_22<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 66) cutlassFusedDistanceNN_66<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 69) cutlassFusedDistanceNN_69<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 81) cutlassFusedDistanceNN_81<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 83) cutlassFusedDistanceNN_83<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 85) cutlassFusedDistanceNN_85<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // if (num == 88) cutlassFusedDistanceNN_88<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // cudaDeviceSynchronize();
    // }
    cudaEventRecord(beg);

    for (int i = 0; i < num_tests; i++) {
    // double st = getTime();
    if (num ==0) cutlassFusedDistanceNN_0<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    //start of injection

    // if (num == 1) cutlassFusedDistanceNN_1<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 2) cutlassFusedDistanceNN_2<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 3) cutlassFusedDistanceNN_3<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 4) cutlassFusedDistanceNN_4<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 5) cutlassFusedDistanceNN_5<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 6) cutlassFusedDistanceNN_6<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 7) cutlassFusedDistanceNN_7<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 8) cutlassFusedDistanceNN_8<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 9) cutlassFusedDistanceNN_9<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 10) cutlassFusedDistanceNN_10<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 11) cutlassFusedDistanceNN_11<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 12) cutlassFusedDistanceNN_12<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 13) cutlassFusedDistanceNN_13<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 14) cutlassFusedDistanceNN_14<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 15) cutlassFusedDistanceNN_15<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 16) cutlassFusedDistanceNN_16<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 17) cutlassFusedDistanceNN_17<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 18) cutlassFusedDistanceNN_18<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 19) cutlassFusedDistanceNN_19<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 20) cutlassFusedDistanceNN_20<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 21) cutlassFusedDistanceNN_21<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 22) cutlassFusedDistanceNN_22<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 23) cutlassFusedDistanceNN_23<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 24) cutlassFusedDistanceNN_24<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 25) cutlassFusedDistanceNN_25<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 26) cutlassFusedDistanceNN_26<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 27) cutlassFusedDistanceNN_27<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 28) cutlassFusedDistanceNN_28<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 29) cutlassFusedDistanceNN_29<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 30) cutlassFusedDistanceNN_30<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 31) cutlassFusedDistanceNN_31<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 32) cutlassFusedDistanceNN_32<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 33) cutlassFusedDistanceNN_33<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 34) cutlassFusedDistanceNN_34<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 35) cutlassFusedDistanceNN_35<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 36) cutlassFusedDistanceNN_36<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 37) cutlassFusedDistanceNN_37<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 38) cutlassFusedDistanceNN_38<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 39) cutlassFusedDistanceNN_39<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 40) cutlassFusedDistanceNN_40<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 41) cutlassFusedDistanceNN_41<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 42) cutlassFusedDistanceNN_42<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 43) cutlassFusedDistanceNN_43<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 44) cutlassFusedDistanceNN_44<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 45) cutlassFusedDistanceNN_45<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 46) cutlassFusedDistanceNN_46<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 47) cutlassFusedDistanceNN_47<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 48) cutlassFusedDistanceNN_48<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 49) cutlassFusedDistanceNN_49<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 50) cutlassFusedDistanceNN_50<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 51) cutlassFusedDistanceNN_51<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 52) cutlassFusedDistanceNN_52<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 53) cutlassFusedDistanceNN_53<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 54) cutlassFusedDistanceNN_54<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 55) cutlassFusedDistanceNN_55<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 56) cutlassFusedDistanceNN_56<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 57) cutlassFusedDistanceNN_57<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 58) cutlassFusedDistanceNN_58<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 59) cutlassFusedDistanceNN_59<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 60) cutlassFusedDistanceNN_60<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 61) cutlassFusedDistanceNN_61<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 62) cutlassFusedDistanceNN_62<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 63) cutlassFusedDistanceNN_63<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 64) cutlassFusedDistanceNN_64<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 65) cutlassFusedDistanceNN_65<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 66) cutlassFusedDistanceNN_66<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 67) cutlassFusedDistanceNN_67<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 68) cutlassFusedDistanceNN_68<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 69) cutlassFusedDistanceNN_69<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 70) cutlassFusedDistanceNN_70<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 71) cutlassFusedDistanceNN_71<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 72) cutlassFusedDistanceNN_72<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 73) cutlassFusedDistanceNN_73<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 74) cutlassFusedDistanceNN_74<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 75) cutlassFusedDistanceNN_75<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 76) cutlassFusedDistanceNN_76<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 77) cutlassFusedDistanceNN_77<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 78) cutlassFusedDistanceNN_78<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 79) cutlassFusedDistanceNN_79<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 80) cutlassFusedDistanceNN_80<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 81) cutlassFusedDistanceNN_81<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 82) cutlassFusedDistanceNN_82<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 83) cutlassFusedDistanceNN_83<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 84) cutlassFusedDistanceNN_84<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 85) cutlassFusedDistanceNN_85<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 86) cutlassFusedDistanceNN_86<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 87) cutlassFusedDistanceNN_87<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 88) cutlassFusedDistanceNN_88<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 89) cutlassFusedDistanceNN_89<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 90) cutlassFusedDistanceNN_90<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 91) cutlassFusedDistanceNN_91<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 92) cutlassFusedDistanceNN_92<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 93) cutlassFusedDistanceNN_93<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 94) cutlassFusedDistanceNN_94<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 95) cutlassFusedDistanceNN_95<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 96) cutlassFusedDistanceNN_96<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 97) cutlassFusedDistanceNN_97<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 98) cutlassFusedDistanceNN_98<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 99) cutlassFusedDistanceNN_99<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 100) cutlassFusedDistanceNN_100<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 101) cutlassFusedDistanceNN_101<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 102) cutlassFusedDistanceNN_102<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 103) cutlassFusedDistanceNN_103<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 104) cutlassFusedDistanceNN_104<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 105) cutlassFusedDistanceNN_105<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 106) cutlassFusedDistanceNN_106<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 107) cutlassFusedDistanceNN_107<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 108) cutlassFusedDistanceNN_108<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 109) cutlassFusedDistanceNN_109<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 110) cutlassFusedDistanceNN_110<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 111) cutlassFusedDistanceNN_111<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 112) cutlassFusedDistanceNN_112<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 113) cutlassFusedDistanceNN_113<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 114) cutlassFusedDistanceNN_114<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 115) cutlassFusedDistanceNN_115<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 116) cutlassFusedDistanceNN_116<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 117) cutlassFusedDistanceNN_117<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 118) cutlassFusedDistanceNN_118<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 119) cutlassFusedDistanceNN_119<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 120) cutlassFusedDistanceNN_120<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // //end of injection
    cudaDeviceSynchronize();
    // double ed = getTime();
    // printf("Round %d: %f\n", i, 1000 * (ed - st));
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    double gflops = (2 * num_tests * double(double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);

    bool check = 1;
    // for (int j = 0; j < num_tests; j++) {
    //     cudaMemcpy((void*)min1[j], (void*)dmin[j], sizeof(OutT) * m, cudaMemcpyDeviceToHost);
    //     for (int i = 0; i < m/2; i ++) {
    //         // printf("%d\n", min[j][i].key);
    //         if (min1[j][i].key!=0) check = 0;
    //     }
    //     for (int i = m/2; i < m; i ++) {
    //         // printf("%d\n", min[j][i].key);
    //         if (min1[j][i].key!=min1[j][m/2].key) check = 0;
    //     }
    // }
    //if (check) printf("num=%d: %lf %lf\n",num, gflops, elapsed); else printf("num=%d: Wrong Answer!\n", num);

    if (num == 0) {
        if (check) {
            Gflops = gflops;
            Elapsed = elapsed;
        } else {
            Gflops = -1.0;
            Elapsed = -1.0;
        }
    } else {
        if (gflops > Gflops && check) {
            Gflops = gflops;
            Elapsed = elapsed;
            max_num = num;
        }
    }
}

int main(int argc, char** argv){
    int bestnum[10] = {85, 9, 13, 3, 88, 69, 83, 66, 22, 81};
    int m = 131072, n = 128, k = 512;
    OutT* min[num_tests], *min1[num_tests], *dmin[num_tests];
    DataT* x[num_tests], *dx[num_tests];
    DataT* y[num_tests], *dy[num_tests];
    DataT* xn[num_tests], *dxn[num_tests];
    DataT* yn[num_tests], *dyn[num_tests];
    int* workspace[num_tests], *dworkspace[num_tests];
    cudaStream_t stream;
    cudaStreamCreate ( &stream) ;

    // OutT* min_r[num_tests], *min1_r[num_tests], *dmin_r[num_tests];
    // DataT* x_r[num_tests], *dx_r[num_tests];
    // DataT* y_r[num_tests], *dy_r[num_tests];
    // DataT* xn_r[num_tests], *dxn_r[num_tests];
    // DataT* yn_r[num_tests], *dyn_r[num_tests];
    // int* workspace_r[num_tests], *dworkspace_r[num_tests];
    // cudaStream_t stream_r;
    // cudaStreamCreate ( &stream_r) ;

    for (int i = 0; i < num_tests; i ++) {
        cudaMalloc((void**)&dx[i], sizeof(DataT) * m * k);
        cudaMalloc((void**)&dy[i], sizeof(DataT) * n * k);
        cudaMalloc((void**)&dxn[i], sizeof(DataT) * m);
        cudaMalloc((void**)&dyn[i], sizeof(DataT) * n);
        cudaMalloc((void**)&dmin[i], sizeof(OutT) * m);
        cudaMalloc((void**)&dworkspace[i], sizeof(IdxT));

        x[i] = (DataT*)malloc(sizeof(DataT) * m * k);
        y[i] = (DataT*)malloc(sizeof(DataT) * n * k);
        xn[i] = (DataT*)malloc(sizeof(DataT) * m);
        yn[i] = (DataT*)malloc(sizeof(DataT) * n);
        min[i] = (OutT*)malloc(sizeof(OutT) * m); 
        min1[i] = (OutT*)malloc(sizeof(OutT) * m); 
        workspace[i] = (IdxT*)malloc(sizeof(IdxT));
        *workspace[i] = 0;

        // cudaMalloc((void**)&dx_r[i], sizeof(DataT) * m * k);
        // cudaMalloc((void**)&dy_r[i], sizeof(DataT) * n * k);
        // cudaMalloc((void**)&dxn_r[i], sizeof(DataT) * m);
        // cudaMalloc((void**)&dyn_r[i], sizeof(DataT) * n);
        // cudaMalloc((void**)&dmin_r[i], sizeof(OutT) * m);
        // cudaMalloc((void**)&dworkspace_r[i], sizeof(IdxT));

        // x_r[i] = (DataT*)malloc(sizeof(DataT) * m * k);
        // y_r[i] = (DataT*)malloc(sizeof(DataT) * n * k);
        // xn_r[i] = (DataT*)malloc(sizeof(DataT) * m);
        // yn_r[i] = (DataT*)malloc(sizeof(DataT) * n);
        // min_r[i] = (OutT*)malloc(sizeof(OutT) * m); 
        // min1_r[i] = (OutT*)malloc(sizeof(OutT) * m); 
        // workspace_r[i] = (IdxT*)malloc(sizeof(IdxT));
        // *workspace_r[i] = 0;
    }
    //start of injection
    IdxT num = 120;
    //IdxT num = 2;


    //chosen parameters:
    //Parameter1. 42
    //Parameter2. 18
    //When K is fixed:
    //  for (IdxT k=8; k<=128; k*=16)
    //      for (IdxT n = 8; n <= 128; n+= 8)
    //
    //When N is fixed:
    //  for (IdxT n=8; n<=128; n*=16)
    //      for (IdxT k = 8; k <= 128; k += 8)
    //
    //The loop for heatmap
    // for(IdxT n = 8; n <= 128; n += 8)
    //     for(IdxT k = 32; k <= 512; k += 32)
    int nums = 0;
    double ave_speedup = 0.0;
    for(IdxT m = (1<<17); m <= (1<<17); m *= 2){
        // Heat Map
        // for(IdxT n = 8; n <= 128; n += 8){
        //     for(IdxT k = 32; k <= 512; k += 32){
        // Fix K
        for (IdxT k=8; k<=128; k*=16){
         for (IdxT n = 8; n <= 128; n+= 8){
        // Fix N
        // for (IdxT n=8; n<=128; n*=16){
        //  for (IdxT k = 8; k <= 128; k += 8){
                for (int p = 0; p < num_tests; p ++) {
                    for(int i = 0; i < m * k / 2; ++i)     x[p][i] = 1 ; // + rand() / RAND_MAX * 0.01;
                    for(int i = 0; i < n * k / 2; ++i)     y[p][i] = 1 ; //+ rand() / RAND_MAX * 0.01; 
                    for(int i = m * k / 2; i < m * k; ++i) x[p][i] = 2 ; //+ rand() / RAND_MAX * 0.01;
                    for(int i = n * k / 2; i < n * k; ++i) y[p][i] = 2 ; //+ rand() / RAND_MAX * 0.01;
                    for(int i = 0; i < m ; ++i) {
                        xn[p][i] = 0.0;
                        for (int j = 0; j < k; ++j)
                            xn[p][i] += x[p][i * k + j] * x[p][i * k + j];
                    }
                    for(int i = 0; i < n ; ++i) {
                        yn[p][i] = 0.0;
                        for (int j = 0; j < k; ++j)
                            yn[p][i] += y[p][i * k + j] * y[p][i * k + j];
                    }
                    for(int i = 0; i < m; ++i) min[p][i].value = std::numeric_limits<float>::max(); 
                    for(int i = 0; i < m; ++i) min[p][i].key = -1;
                    *workspace[p] = 0;
                }
                for (int i = 0; i < num_tests; i ++){
                    cudaMemcpy((void*)dx[i], (void*)x[i], sizeof(DataT) * m * k, cudaMemcpyHostToDevice);
                    cudaMemcpy((void*)dy[i], (void*)y[i], sizeof(DataT) * n * k, cudaMemcpyHostToDevice);
                    cudaMemcpy((void*)dxn[i], (void*)xn[i], sizeof(DataT) * m, cudaMemcpyHostToDevice);
                    cudaMemcpy((void*)dyn[i], (void*)yn[i], sizeof(DataT) * n, cudaMemcpyHostToDevice);
                }
                double max_gflops=0.0, min_elapsed=0.0, raft_gflops=0.0, raft_elapsed=0.0;
                double max_gflops_r=0.0, min_elapsed_r=0.0;
                int max_num = 0, max_num1 = 0;
                DistanceNN(0, m, n, k, raft_gflops, raft_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);

                for (int i = 1; i <= 10; i++) {
                    DistanceNN(bestnum[i-1], m, n, k, max_gflops, min_elapsed, max_num, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                }
                double temp1_gflops = 0.0, temp1_elapsed = 0.0;
                double temp2_gflops = 0.0, temp2_elapsed = 0.0;
                DistanceNN(42, m, n, k, temp1_gflops, temp1_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                DistanceNN(18, m, n, k, temp2_gflops, temp2_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                DistanceNN(0, m, n, k, raft_gflops, raft_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                //printf("M,N,K=%d %d %d :%lf %lf %lf %d\n", m, n, k, max_gflops, raft_gflops, speedup, max_num);
                printf("%d, %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %d\n", m, n, k, max_gflops, temp1_gflops, temp2_gflops, raft_gflops, min_elapsed, temp1_elapsed, temp2_elapsed, raft_elapsed, max_gflops / raft_gflops, temp1_gflops / raft_gflops, temp2_gflops / raft_gflops, max_num);
                ave_speedup += max_gflops / raft_gflops;
                nums ++;
            }
        }
    }
    printf("Average speed up is: %lf\n", ave_speedup / nums);
    for (int i=0; i < num_tests; i++){
        free(x[i]);
        free(y[i]);
        free(xn[i]);
        free(yn[i]);
        free(min[i]);
        free(workspace[i]);
        cudaFree(dx[i]);
        cudaFree(dy[i]);
        cudaFree(dxn[i]);
        cudaFree(dyn[i]);
        cudaFree(dmin[i]);
        cudaFree(dworkspace[i]);
    }
    cudaStreamDestroy(stream);
    return 0;
}
