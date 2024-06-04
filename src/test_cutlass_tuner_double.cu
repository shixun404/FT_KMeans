#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "cstdlib"
using DataT = double;

void test_cublas(int m, int n, int k, int num_tests, cublasHandle_t &handle, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    for(int i = 0; i < num_tests; ++i){
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);    cudaEventSynchronize(beg);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    printf("cublas %d %d %d %f %f\n", m, n, k, elapsed, gflops);
}

bool raftSgemmNN(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

void test_raft(int m, int n, int k, int num_tests, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    bool ret = raftSgemmNN(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (ret == 0) return;
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    for(int i = 0; i < num_tests; ++i){
        raftSgemmNN(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    printf("raft %d %d %d %f %f\n", m, n, k, elapsed, gflops);
}
//start injection header

bool CutlassDgemmNN_1(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_2(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_3(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_4(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_5(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_6(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_7(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_8(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_9(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_10(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_11(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_12(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_13(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_14(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_15(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_16(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_17(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_18(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_19(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_20(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_21(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_22(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_23(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_24(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_25(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_26(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_27(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_28(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_29(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_30(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_31(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_32(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_33(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_34(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_35(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_36(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_37(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_38(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_39(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_40(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_41(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_42(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_43(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_44(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_45(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_46(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_47(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_48(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_49(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_50(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_51(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_52(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_53(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_54(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_55(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_56(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_57(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_58(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_59(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_60(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_61(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_62(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_63(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_64(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_65(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_66(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_67(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_68(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_69(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_70(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_71(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_72(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_73(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_74(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_75(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_76(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_77(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_78(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_79(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_80(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_81(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_82(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_83(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_84(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_85(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_86(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_87(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_88(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_89(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_90(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_91(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_92(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_93(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_94(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_95(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_96(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_97(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_98(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_99(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_100(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_101(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_102(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_103(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_104(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_105(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_106(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_107(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_108(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_109(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_110(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_111(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_112(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_113(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_114(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_115(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_116(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_117(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_118(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_119(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_120(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_121(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_122(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_123(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_124(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_125(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_126(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_127(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_128(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_129(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_130(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_131(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_132(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_133(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_134(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_135(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_136(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_137(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_138(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_139(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_140(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_141(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_142(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_143(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_144(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

bool CutlassDgemmNN_145(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
//end injection header

void test_cutlass(int m, int n, int k, int num_tests, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    typedef bool (*Func)(int,int,int,DataT,DataT const*,int,DataT const*,int,DataT,DataT*,int);
    //start injection arr
    Func Arr[] = {CutlassDgemmNN_1,CutlassDgemmNN_2,CutlassDgemmNN_3,CutlassDgemmNN_4,CutlassDgemmNN_5,CutlassDgemmNN_6,CutlassDgemmNN_7,CutlassDgemmNN_8,CutlassDgemmNN_9,CutlassDgemmNN_10,CutlassDgemmNN_11,CutlassDgemmNN_12,CutlassDgemmNN_13,CutlassDgemmNN_14,CutlassDgemmNN_15,CutlassDgemmNN_16,CutlassDgemmNN_17,CutlassDgemmNN_18,CutlassDgemmNN_19,CutlassDgemmNN_20,CutlassDgemmNN_21,CutlassDgemmNN_22,CutlassDgemmNN_23,CutlassDgemmNN_24,CutlassDgemmNN_25,CutlassDgemmNN_26,CutlassDgemmNN_27,CutlassDgemmNN_28,CutlassDgemmNN_29,CutlassDgemmNN_30,CutlassDgemmNN_31,CutlassDgemmNN_32,CutlassDgemmNN_33,CutlassDgemmNN_34,CutlassDgemmNN_35,CutlassDgemmNN_36,CutlassDgemmNN_37,CutlassDgemmNN_38,CutlassDgemmNN_39,CutlassDgemmNN_40,CutlassDgemmNN_41,CutlassDgemmNN_42,CutlassDgemmNN_43,CutlassDgemmNN_44,CutlassDgemmNN_45,CutlassDgemmNN_46,CutlassDgemmNN_47,CutlassDgemmNN_48,CutlassDgemmNN_49,CutlassDgemmNN_50,CutlassDgemmNN_51,CutlassDgemmNN_52,CutlassDgemmNN_53,CutlassDgemmNN_54,CutlassDgemmNN_55,CutlassDgemmNN_56,CutlassDgemmNN_57,CutlassDgemmNN_58,CutlassDgemmNN_59,CutlassDgemmNN_60,CutlassDgemmNN_61,CutlassDgemmNN_62,CutlassDgemmNN_63,CutlassDgemmNN_64,CutlassDgemmNN_65,CutlassDgemmNN_66,CutlassDgemmNN_67,CutlassDgemmNN_68,CutlassDgemmNN_69,CutlassDgemmNN_70,CutlassDgemmNN_71,CutlassDgemmNN_72,CutlassDgemmNN_73,CutlassDgemmNN_74,CutlassDgemmNN_75,CutlassDgemmNN_76,CutlassDgemmNN_77,CutlassDgemmNN_78,CutlassDgemmNN_79,CutlassDgemmNN_80,CutlassDgemmNN_81,CutlassDgemmNN_82,CutlassDgemmNN_83,CutlassDgemmNN_84,CutlassDgemmNN_85,CutlassDgemmNN_86,CutlassDgemmNN_87,CutlassDgemmNN_88,CutlassDgemmNN_89,CutlassDgemmNN_90,CutlassDgemmNN_91,CutlassDgemmNN_92,CutlassDgemmNN_93,CutlassDgemmNN_94,CutlassDgemmNN_95,CutlassDgemmNN_96,CutlassDgemmNN_97,CutlassDgemmNN_98,CutlassDgemmNN_99,CutlassDgemmNN_100,CutlassDgemmNN_101,CutlassDgemmNN_102,CutlassDgemmNN_103,CutlassDgemmNN_104,CutlassDgemmNN_105,CutlassDgemmNN_106,CutlassDgemmNN_107,CutlassDgemmNN_108,CutlassDgemmNN_109,CutlassDgemmNN_110,CutlassDgemmNN_111,CutlassDgemmNN_112,CutlassDgemmNN_113,CutlassDgemmNN_114,CutlassDgemmNN_115,CutlassDgemmNN_116,CutlassDgemmNN_117,CutlassDgemmNN_118,CutlassDgemmNN_119,CutlassDgemmNN_120,CutlassDgemmNN_121,CutlassDgemmNN_122,CutlassDgemmNN_123,CutlassDgemmNN_124,CutlassDgemmNN_125,CutlassDgemmNN_126,CutlassDgemmNN_127,CutlassDgemmNN_128,CutlassDgemmNN_129,CutlassDgemmNN_130,CutlassDgemmNN_131,CutlassDgemmNN_132,CutlassDgemmNN_133,CutlassDgemmNN_134,CutlassDgemmNN_135,CutlassDgemmNN_136,CutlassDgemmNN_137,CutlassDgemmNN_138,CutlassDgemmNN_139,CutlassDgemmNN_140,CutlassDgemmNN_141,CutlassDgemmNN_142,CutlassDgemmNN_143,CutlassDgemmNN_144,CutlassDgemmNN_145};
    //end injection arr
    double max_gflops = -1.0, max_elapsed = 0.0;
    int num = 0, maxid = -1;
    for (auto fptr : Arr) {
      cudaEvent_t beg, end;
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      float elapsed;
      bool ret = fptr(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
      cudaDeviceSynchronize();
      //std::cerr << ret << std::endl;
      if (ret == 0) { num ++; continue;}
      cudaEventRecord(beg);
      for(int i = 0; i < num_tests; ++i){
        fptr(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
        cudaDeviceSynchronize();
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed, beg, end);
      double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
      if (gflops > max_gflops) {
        max_gflops = gflops;
        max_elapsed = elapsed;
        maxid = num;
      }
      num ++;
    }
    //std::cerr << "Case: " << m << " " << n << " " << k <<" " << max_gflops <<" " << maxid << std::endl;
    //printf("%d %d %d %.3lf %d\n", m, n, k, max_gflops, maxid);
    printf("cutlass %d %d %d %f %f %d\n", m, n, k, max_elapsed, max_gflops, maxid);
}

int main(int argc, char** argv){
    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);
    DataT *A, *dA, *B, *dB, *C, *dC;
    int M = 131072, N = 128, K = 512;
    long long int A_size = M * K;
    long long int B_size = N * K;
    long long int C_size = M * N;
    A = (DataT*)malloc(sizeof(DataT) * A_size);
    B = (DataT*)malloc(sizeof(DataT) * B_size);
    C = (DataT*)malloc(sizeof(DataT) * C_size);

    cudaMalloc((void**)&dA, sizeof(DataT) * A_size);
    cudaMalloc((void**)&dB, sizeof(DataT) * B_size);
    cudaMalloc((void**)&dC, sizeof(DataT) * C_size);

    for(long long int i = 0; i < A_size; ++i) A[i] = DataT(rand() % 5) + (rand() % 5) * 0.01;
    for(long long int i = 0; i < B_size; ++i) B[i] = DataT(rand() % 5) + (rand() % 5) * 0.01;
    for(long long int i = 0; i < C_size; ++i) C[i] = DataT(rand() % 5) + (rand() % 5) * 0.01;

    cudaMemcpy(dA, A, sizeof(DataT) * A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);

    DataT alpha = 0.0001, beta = -0.0001; 

    int num_tests = 1;

    // test_cutlass(M, 32, 256, num_tests, alpha, beta, dA, dB, dC);
    
    for(int m = (1<<17); m <= (1<<17); m *= 2){
        for(int k = 8; k <= 512; k *= 2){
            for(int n = 8; n <= 128; n *= 2){
                test_cutlass(m, n, k, num_tests, alpha, beta, dA, dB, dC);
                test_raft(m, n, k, num_tests, alpha, beta, dA, dB, dC);
                test_cublas(m, n, k, num_tests, handle, alpha, beta, dA, dB, dC);
            }
        }
    }
    return 0;
}

//start injection func

bool CutlassDgemmNN_1(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_2(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_3(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_4(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_5(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_6(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_7(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_8(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_9(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_10(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_11(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_12(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_13(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_14(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_15(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_16(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_17(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_18(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_19(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_20(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_21(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_22(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_23(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_24(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_25(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_26(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_27(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_28(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_29(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_30(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_31(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_32(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_33(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_34(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_35(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_36(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_37(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_38(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_39(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_40(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_41(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_42(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_43(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_44(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_45(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_46(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_47(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_48(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_49(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 16, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_50(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_51(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 16, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_52(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_53(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_54(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_55(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_56(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_57(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_58(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_59(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_60(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_61(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_62(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_63(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_64(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_65(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_66(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_67(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 512, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_68(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 16, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_69(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_70(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_71(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_72(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 16, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_73(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 32, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_74(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_75(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_76(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_77(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16, 1024, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_78(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_79(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_80(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_81(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_82(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_83(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_84(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_85(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_86(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_87(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_88(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_89(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_90(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_91(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_92(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_93(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_94(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_95(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 1024, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_96(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_97(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_98(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_99(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_100(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_101(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_102(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_103(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_104(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_105(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_106(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_107(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_108(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_109(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_110(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_111(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_112(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_113(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_114(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_115(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_116(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_117(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_118(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 1024, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_119(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_120(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_121(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_122(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_123(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 16, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_124(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_125(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 16, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_126(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_127(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_128(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_129(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_130(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_131(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_132(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_133(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_134(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_135(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_136(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_137(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_138(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_139(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_140(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_141(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 512, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_142(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_143(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_144(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}

bool CutlassDgemmNN_145(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}
//end injection func