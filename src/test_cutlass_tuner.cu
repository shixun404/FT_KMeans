#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "cstdlib"
using DataT = float;

void test_cublas(int m, int n, int k, int num_tests, cublasHandle_t &handle, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    for(int i = 0; i < num_tests; ++i){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16,8,4>;
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

bool CutlassSgemmNN_0(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_1(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_2(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_3(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_4(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_5(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_6(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_7(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_8(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_9(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_10(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_11(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_12(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_13(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_14(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_15(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_16(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_17(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_18(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_19(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_20(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_21(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_22(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_23(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_24(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_25(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_26(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_27(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_28(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_29(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_30(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_31(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_32(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_33(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_34(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_35(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_36(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_37(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_38(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_39(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_40(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_41(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_42(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_43(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_44(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_45(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_46(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_47(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_48(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_49(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_50(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_51(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_52(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_53(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_54(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_55(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_56(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_57(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_58(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_59(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_60(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_61(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_62(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_63(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_64(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_65(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_66(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_67(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_68(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_69(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_70(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_71(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_72(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_73(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_74(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_75(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_76(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_77(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_78(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_79(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_80(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_81(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_82(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_83(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_84(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_85(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_86(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_87(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_88(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_89(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_90(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_91(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_92(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_93(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_94(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_95(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_96(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_97(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_98(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_99(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_100(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_101(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_102(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_103(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_104(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_105(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_106(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_107(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_108(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_109(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_110(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_111(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_112(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_113(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_114(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_115(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_116(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_117(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_118(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_119(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_120(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_121(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_122(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_123(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_124(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_125(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_126(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_127(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_128(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_129(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_130(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_131(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_132(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_133(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_134(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_135(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_136(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_137(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_138(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_139(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_140(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_141(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_142(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_143(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_144(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_145(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_146(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_147(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_148(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_149(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_150(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_151(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_152(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_153(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_154(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_155(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_156(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
bool CutlassSgemmNN_157(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

void test_cutlass(int m, int n, int k, int num_tests, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    typedef bool (*Func)(int,int,int,DataT,DataT const*,int,DataT const*,int,DataT,DataT*,int);
    Func Arr[] = {CutlassSgemmNN_0, CutlassSgemmNN_1, CutlassSgemmNN_2, CutlassSgemmNN_3, CutlassSgemmNN_4, CutlassSgemmNN_5, CutlassSgemmNN_6, CutlassSgemmNN_7, CutlassSgemmNN_8, CutlassSgemmNN_9, CutlassSgemmNN_10, CutlassSgemmNN_11, CutlassSgemmNN_12, CutlassSgemmNN_13, CutlassSgemmNN_14, CutlassSgemmNN_15, CutlassSgemmNN_16, CutlassSgemmNN_17, CutlassSgemmNN_18, CutlassSgemmNN_19, CutlassSgemmNN_20, CutlassSgemmNN_21, CutlassSgemmNN_22, CutlassSgemmNN_23, CutlassSgemmNN_24, CutlassSgemmNN_25, CutlassSgemmNN_26, CutlassSgemmNN_27, CutlassSgemmNN_28, CutlassSgemmNN_29, CutlassSgemmNN_30, CutlassSgemmNN_31, CutlassSgemmNN_32, CutlassSgemmNN_33, CutlassSgemmNN_34, CutlassSgemmNN_35, CutlassSgemmNN_36, CutlassSgemmNN_37, CutlassSgemmNN_38, CutlassSgemmNN_39, CutlassSgemmNN_40, CutlassSgemmNN_41, CutlassSgemmNN_42, CutlassSgemmNN_43, CutlassSgemmNN_44, CutlassSgemmNN_45, CutlassSgemmNN_46, CutlassSgemmNN_47, CutlassSgemmNN_48, CutlassSgemmNN_49, CutlassSgemmNN_50, CutlassSgemmNN_51, CutlassSgemmNN_52, CutlassSgemmNN_53, CutlassSgemmNN_54, CutlassSgemmNN_55, CutlassSgemmNN_56, CutlassSgemmNN_57, CutlassSgemmNN_58, CutlassSgemmNN_59, CutlassSgemmNN_60, CutlassSgemmNN_61, CutlassSgemmNN_62, CutlassSgemmNN_63, CutlassSgemmNN_64, CutlassSgemmNN_65, CutlassSgemmNN_66, CutlassSgemmNN_67, CutlassSgemmNN_68, CutlassSgemmNN_69, CutlassSgemmNN_70, CutlassSgemmNN_71, CutlassSgemmNN_72, CutlassSgemmNN_73, CutlassSgemmNN_74, CutlassSgemmNN_75, CutlassSgemmNN_76, CutlassSgemmNN_77, CutlassSgemmNN_78, CutlassSgemmNN_79, CutlassSgemmNN_80, CutlassSgemmNN_81, CutlassSgemmNN_82, CutlassSgemmNN_83, CutlassSgemmNN_84, CutlassSgemmNN_85, CutlassSgemmNN_86, CutlassSgemmNN_87, CutlassSgemmNN_88, CutlassSgemmNN_89, CutlassSgemmNN_90, CutlassSgemmNN_91, CutlassSgemmNN_92, CutlassSgemmNN_93, CutlassSgemmNN_94, CutlassSgemmNN_95, CutlassSgemmNN_96, CutlassSgemmNN_97, CutlassSgemmNN_98, CutlassSgemmNN_99, CutlassSgemmNN_100, CutlassSgemmNN_101, CutlassSgemmNN_102, CutlassSgemmNN_103, CutlassSgemmNN_104, CutlassSgemmNN_105, CutlassSgemmNN_106, CutlassSgemmNN_107, CutlassSgemmNN_108, CutlassSgemmNN_109, CutlassSgemmNN_110, CutlassSgemmNN_111, CutlassSgemmNN_112, CutlassSgemmNN_113, CutlassSgemmNN_114, CutlassSgemmNN_115, CutlassSgemmNN_116, CutlassSgemmNN_117, CutlassSgemmNN_118, CutlassSgemmNN_119, CutlassSgemmNN_120, CutlassSgemmNN_121, CutlassSgemmNN_122, CutlassSgemmNN_123, CutlassSgemmNN_124, CutlassSgemmNN_125, CutlassSgemmNN_126, CutlassSgemmNN_127, CutlassSgemmNN_128, CutlassSgemmNN_129, CutlassSgemmNN_130, CutlassSgemmNN_131, CutlassSgemmNN_132, CutlassSgemmNN_133, CutlassSgemmNN_134, CutlassSgemmNN_135, CutlassSgemmNN_136, CutlassSgemmNN_137, CutlassSgemmNN_138, CutlassSgemmNN_139, CutlassSgemmNN_140, CutlassSgemmNN_141, CutlassSgemmNN_142, CutlassSgemmNN_143, CutlassSgemmNN_144, CutlassSgemmNN_145, CutlassSgemmNN_146, CutlassSgemmNN_147, CutlassSgemmNN_148, CutlassSgemmNN_149, CutlassSgemmNN_150, CutlassSgemmNN_151, CutlassSgemmNN_152, CutlassSgemmNN_153, CutlassSgemmNN_154, CutlassSgemmNN_155, CutlassSgemmNN_156, CutlassSgemmNN_157};
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


bool CutlassSgemmNN_0(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_1(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_2(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_3(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_4(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_5(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_6(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_7(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_8(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_9(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_10(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_11(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_12(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_13(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_14(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_15(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_16(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_17(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_18(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_19(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_20(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_21(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_22(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_23(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_24(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_25(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_26(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_27(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_28(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_29(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_30(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_31(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_32(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_33(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_34(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_35(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_36(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_37(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_38(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_39(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_40(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_41(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_42(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,32,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_43(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_44(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_45(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_46(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,32,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_47(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_48(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_49(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,256,8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_50(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_51(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_52(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_53(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_54(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_55(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_56(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_57(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_58(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_59(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_60(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_61(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_62(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_63(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_64(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<16,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_65(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_66(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_67(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_68(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_69(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_70(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_71(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_72(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_73(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_74(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_75(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_76(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_77(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_78(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_79(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_80(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_81(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_82(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_83(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_84(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_85(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_86(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_87(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_88(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_89(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_90(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_91(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_92(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_93(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_94(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_95(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_96(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_97(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_98(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_99(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_100(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_101(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_102(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_103(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_104(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_105(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_106(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_107(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_108(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_109(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_110(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_111(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_112(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_113(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_114(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_115(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_116(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_117(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_118(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_119(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_120(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_121(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_122(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_123(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_124(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_125(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_126(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_127(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_128(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_129(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_130(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_131(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_132(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_133(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_134(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_135(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_136(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_137(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_138(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_139(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_140(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_141(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_142(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_143(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,8,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_144(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,8,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_145(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_146(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_147(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_148(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,256,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_149(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_150(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_151(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,512,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_152(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,32,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_153(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,64,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_154(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_155(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_156(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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

bool CutlassSgemmNN_157(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
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
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,128,16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<256,16,16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 4>;
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
