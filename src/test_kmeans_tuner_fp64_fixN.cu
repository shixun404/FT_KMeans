#include <cuda_runtime.h>
#include <stdio.h>
#include "fused_distance_nn/l2_exp.cuh"
#include "fused_distance_nn/cutlass_base_customized.cuh"
#include "header.cuh"
#include "helper.h"
#include "cstdlib"
using namespace my_test;
using DataT = double;
using IdxT = int;
using OutT = KeyValuePair<IdxT, DataT>;
using L2Op                  = l2_exp_cutlass_op<DataT, DataT>;
using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
using MinAndDistanceReduceOp = MinAndDistanceReduceOpImpl<IdxT, DataT>;
using KVPMinReduce = KVPMinReduceImpl<IdxT, DataT>;
const int num_tests = 10;

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
	cudaEventRecord(beg);
    for (int i = 0; i < num_tests; i++) {
    if (num ==0) cutlassFusedDistanceNN_double_0<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    //double start of injection

    // if (num == 1) cutlassFusedDistanceNN_double_1<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 2) cutlassFusedDistanceNN_double_2<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 3) cutlassFusedDistanceNN_double_3<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 4) cutlassFusedDistanceNN_double_4<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 5) cutlassFusedDistanceNN_double_5<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 6) cutlassFusedDistanceNN_double_6<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 7) cutlassFusedDistanceNN_double_7<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 8) cutlassFusedDistanceNN_double_8<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 9) cutlassFusedDistanceNN_double_9<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 10) cutlassFusedDistanceNN_double_10<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 11) cutlassFusedDistanceNN_double_11<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 12) cutlassFusedDistanceNN_double_12<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 13) cutlassFusedDistanceNN_double_13<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 14) cutlassFusedDistanceNN_double_14<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 15) cutlassFusedDistanceNN_double_15<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 16) cutlassFusedDistanceNN_double_16<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 17) cutlassFusedDistanceNN_double_17<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 18) cutlassFusedDistanceNN_double_18<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 19) cutlassFusedDistanceNN_double_19<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 20) cutlassFusedDistanceNN_double_20<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 21) cutlassFusedDistanceNN_double_21<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 22) cutlassFusedDistanceNN_double_22<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    //if (num == 23) cutlassFusedDistanceNN_double_23<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 24) cutlassFusedDistanceNN_double_24<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 25) cutlassFusedDistanceNN_double_25<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 26) cutlassFusedDistanceNN_double_26<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 27) cutlassFusedDistanceNN_double_27<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 28) cutlassFusedDistanceNN_double_28<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 29) cutlassFusedDistanceNN_double_29<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 30) cutlassFusedDistanceNN_double_30<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 31) cutlassFusedDistanceNN_double_31<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 32) cutlassFusedDistanceNN_double_32<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 33) cutlassFusedDistanceNN_double_33<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 34) cutlassFusedDistanceNN_double_34<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 35) cutlassFusedDistanceNN_double_35<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 36) cutlassFusedDistanceNN_double_36<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    if (num == 37) cutlassFusedDistanceNN_double_37<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 38) cutlassFusedDistanceNN_double_38<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 39) cutlassFusedDistanceNN_double_39<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 40) cutlassFusedDistanceNN_double_40<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 41) cutlassFusedDistanceNN_double_41<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 42) cutlassFusedDistanceNN_double_42<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 43) cutlassFusedDistanceNN_double_43<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 44) cutlassFusedDistanceNN_double_44<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 45) cutlassFusedDistanceNN_double_45<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 46) cutlassFusedDistanceNN_double_46<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 47) cutlassFusedDistanceNN_double_47<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 48) cutlassFusedDistanceNN_double_48<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 49) cutlassFusedDistanceNN_double_49<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 50) cutlassFusedDistanceNN_double_50<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 51) cutlassFusedDistanceNN_double_51<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 52) cutlassFusedDistanceNN_double_52<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 53) cutlassFusedDistanceNN_double_53<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 54) cutlassFusedDistanceNN_double_54<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 55) cutlassFusedDistanceNN_double_55<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 56) cutlassFusedDistanceNN_double_56<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 57) cutlassFusedDistanceNN_double_57<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 58) cutlassFusedDistanceNN_double_58<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 59) cutlassFusedDistanceNN_double_59<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 60) cutlassFusedDistanceNN_double_60<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 61) cutlassFusedDistanceNN_double_61<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 62) cutlassFusedDistanceNN_double_62<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 63) cutlassFusedDistanceNN_double_63<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 64) cutlassFusedDistanceNN_double_64<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 65) cutlassFusedDistanceNN_double_65<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 66) cutlassFusedDistanceNN_double_66<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 67) cutlassFusedDistanceNN_double_67<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 68) cutlassFusedDistanceNN_double_68<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 69) cutlassFusedDistanceNN_double_69<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 70) cutlassFusedDistanceNN_double_70<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 71) cutlassFusedDistanceNN_double_71<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 72) cutlassFusedDistanceNN_double_72<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 73) cutlassFusedDistanceNN_double_73<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 74) cutlassFusedDistanceNN_double_74<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 75) cutlassFusedDistanceNN_double_75<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 76) cutlassFusedDistanceNN_double_76<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 77) cutlassFusedDistanceNN_double_77<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 78) cutlassFusedDistanceNN_double_78<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 79) cutlassFusedDistanceNN_double_79<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);

    // if (num == 80) cutlassFusedDistanceNN_double_80<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx[i],dy[i],dxn[i],dyn[i],m,n,k,lda,ldb,ldd,dmin[i],dworkspace[i],cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
    // // //double end of injection
    cudaDeviceSynchronize();
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
        Gflops = gflops;
        Elapsed = elapsed;
    } else {
        if (gflops > Gflops && check) {
            Gflops = gflops;
            Elapsed = elapsed;
            max_num = num;
        }
    }
}

int main(int argc, char** argv){
    int bestnum[4] = {19, 21, 12, 18};
    //start of injection
    IdxT num = 80;
    int nums = 0;
    double ave_speedup = 0.0;
    int m = 131072, n = 128, k = 512; //k = 512!!!
    OutT* min[num_tests], *min1[num_tests], *dmin[num_tests];
    DataT* x[num_tests], *dx[num_tests];
    DataT* y[num_tests], *dy[num_tests];
    DataT* xn[num_tests], *dxn[num_tests];
    DataT* yn[num_tests], *dyn[num_tests];
    int* workspace[num_tests], *dworkspace[num_tests];
    cudaStream_t stream;
    cudaStreamCreate ( &stream) ;

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
    }

    //chosen parameters:
    //Parameter1. 37
    //Parameter2. 22
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

    for(IdxT m = (1<<17); m <= (1<<17); m *= 2){
              // Heat Map
        // for(IdxT n = 8; n <= 128; n += 8){
        //     for(IdxT k = 32; k <= 512; k += 32){
        // Fix K
        // for (IdxT k=8; k<=128; k*=16){
        //  for (IdxT n = 8; n <= 128; n+= 8){
        // Fix N
        for (IdxT n=8; n<=128; n*=16){
         for (IdxT k = 8; k <= 128; k += 8){
                for (int p = 0; p < num_tests; p ++) {
                    for(int i = 0; i < m * k / 2; ++i)     x[p][i] = 1;
                    for(int i = 0; i < n * k / 2; ++i)     y[p][i] = 1; 
                    for(int i = m * k / 2; i < m * k; ++i) x[p][i] = 2;
                    for(int i = n * k / 2; i < n * k; ++i) y[p][i] = 2; 
                    for(int i = 0; i < m / 2; ++i)     xn[p][i] = 1 * k;
                    for(int i = 0; i < n / 2; ++i)     yn[p][i] = 1 * k; 
                    for(int i = m / 2; i < m; ++i) xn[p][i] = 4 * k;
                    for(int i = n / 2; i < n; ++i) yn[p][i] = 4 * k; 
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
                int max_num = 0, max_num1 = 0;
                DistanceNN(0, m, n, k, raft_gflops, raft_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                for (int i = 1; i <= 4; i++) {
                    DistanceNN(bestnum[i-1], m, n, k, max_gflops, min_elapsed, max_num, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                }
                double temp1_gflops = 0.0, temp1_elapsed = 0.0;
                double temp2_gflops = -1.0, temp2_elapsed = 0.0;
                DistanceNN(22, m, n, k, temp1_gflops, temp1_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                DistanceNN(37, m, n, k, temp2_gflops, temp2_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                DistanceNN(0, m, n, k, raft_gflops, raft_elapsed, max_num1, min, min1, dmin, x, dx, y, dy, xn, dxn, yn, dyn, workspace, dworkspace, stream);
                //printf("M,N,K=%d %d %d :%lf %lf %lf %d\n", m, n, k, max_gflops, raft_gflops, speedup, max_num);
                //printf("%d, %d, %d, %lf, %lf, %lf, %d\n", m, n, k, max_gflops, raft_gflops, speedup, max_num);
                printf("%d, %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %d\n", m, n, k, max_gflops, temp1_gflops, temp2_gflops, raft_gflops, min_elapsed, temp1_elapsed, temp2_elapsed, raft_elapsed, max_gflops / raft_gflops, temp1_gflops / raft_gflops, temp2_gflops / raft_gflops, max_num);
                ave_speedup += max_gflops / raft_gflops;
                nums ++;
           }
       }
    }
    printf("Average speed up is: %lf\n", ave_speedup / nums);
    return 0;
}