/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wtautological-compare"


#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
// #include <rmm/device_uvector.hpp>

#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_view.h>

#include "epilogue_elementwise.cuh"  // FusedDistanceNNEpilogueElementwise
#include "gemm_customized.h"                    // FusedDistanceNNGemm
// #include <raft/util/cudart_utils.hpp>   // getMultiProcessorCount
// #include <raft/util/cutlass_utils.cuh>  // RAFT_CUTLASS_TRY
#include <type_traits>

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_codegen(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_tester<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}


template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_codegen_double(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_tester<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_0(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_0<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_0(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_0<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

//start of injection

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_1(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_1<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_2(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_2<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_3(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_3<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_4(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_4<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_5(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_5<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_6(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_6<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_7(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_7<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_8(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_8<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_9(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_9<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_10(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_10<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_11(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_11<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_12(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_12<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_13(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_13<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_14(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_14<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_15(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_15<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_16(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_16<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_17(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_17<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_18(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_18<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_19(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_19<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_20(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_20<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_21(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_21<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_22(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_22<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_23(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_23<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_24(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_24<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_25(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_25<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_26(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_26<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_27(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_27<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_28(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_28<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_29(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_29<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_30(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_30<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_31(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_31<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_32(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_32<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_33(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_33<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_34(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_34<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_35(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_35<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_36(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_36<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_37(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_37<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_38(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_38<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_39(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_39<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_40(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_40<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_41(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_41<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_42(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_42<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_43(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_43<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_44(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_44<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_45(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_45<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_46(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_46<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_47(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_47<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_48(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_48<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_49(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_49<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_50(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_50<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_51(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_51<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_52(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_52<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_53(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_53<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_54(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_54<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_55(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_55<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_56(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_56<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_57(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_57<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_58(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_58<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_59(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_59<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_60(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_60<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_61(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_61<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_62(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_62<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_63(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_63<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_64(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_64<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_65(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_65<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_66(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_66<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_67(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_67<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_68(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_68<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_69(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_69<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_70(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_70<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_71(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_71<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_72(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_72<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_73(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_73<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_74(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_74<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_75(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_75<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_76(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_76<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_77(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_77<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_78(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_78<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_79(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_79<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_80(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_80<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_81(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_81<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_82(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_82<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_83(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_83<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_84(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_84<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_85(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_85<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_86(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_86<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_87(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_87<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_88(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_88<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_89(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_89<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_90(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_90<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_91(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_91<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_92(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_92<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_93(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_93<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_94(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_94<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_95(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_95<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_96(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_96<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_97(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_97<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_98(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_98<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_99(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_99<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_100(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_100<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_101(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_101<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_102(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_102<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_103(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_103<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_104(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_104<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_105(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_105<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_106(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_106<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_107(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_107<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_108(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_108<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_109(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_109<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_110(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_110<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_111(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_111<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_112(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_112<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_113(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_113<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_114(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_114<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_115(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_115<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_116(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_116<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_117(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_117<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_118(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_118<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_119(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_119<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_120(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_120<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}
//end of injection 

//double start of injection

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_1(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_1<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_2(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_2<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_3(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_3<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_4(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_4<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_5(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_5<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_6(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_6<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_7(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_7<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_8(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_8<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_9(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_9<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_10(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_10<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_11(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_11<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_12(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_12<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_13(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_13<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_14(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_14<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_15(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_15<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_16(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_16<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_17(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_17<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_18(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_18<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_19(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_0<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_20(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_20<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_21(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_21<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_22(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_22<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_23(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_23<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_24(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_24<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_25(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_25<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_26(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_26<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_27(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_27<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_28(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_28<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_29(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_29<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_30(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_30<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_31(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_31<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_32(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_32<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_33(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_33<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_34(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_34<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_35(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_35<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_36(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_36<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_37(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_37<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_38(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_38<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_39(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_39<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_40(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_40<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_41(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_41<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_42(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_42<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_43(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_43<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_44(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_44<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_45(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_45<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_46(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_46<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_47(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_47<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_48(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_48<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_49(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_49<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_50(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_50<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_51(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_51<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_52(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_52<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_53(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_53<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_54(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_54<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_55(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_55<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_56(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_56<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_57(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_57<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_58(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_58<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_59(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_59<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_60(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_60<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_61(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_61<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_62(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_62<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_63(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_63<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_64(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_64<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_65(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_65<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_66(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_66<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_67(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_67<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_68(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_68<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_69(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_69<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_70(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_70<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_71(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_71<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_72(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_72<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_73(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_73<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_74(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_74<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_75(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_75<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_76(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_76<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_77(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_77<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_78(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_78<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_79(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_79<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}

template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_double_80(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_double_80<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}
//double end of injection

#pragma GCC diagnostic pop
