#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "helper.h"
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
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    printf("cublas %d %d %d %f %f\n", m, n, k, elapsed, gflops);
}

cudaError_t CutlassSgemmNN_0(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_1(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_2(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_3(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_4(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_5(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_6(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_7(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_8(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_9(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_10(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_11(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_12(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_13(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_14(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_15(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_16(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_17(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_18(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_19(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_20(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_21(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_22(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_23(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_24(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_25(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_26(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_27(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_28(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_29(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_30(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_31(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_32(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_33(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);
cudaError_t CutlassSgemmNN_34(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);

cudaError_t CutlassSgemmNN_tuned(int m, int n, int k, DataT alpha, DataT const *dA, int lda, DataT const *dB, int ldb, DataT beta, DataT *dC,int ldc){
    if (n == 8 && k == 8 ) return CutlassSgemmNN_0(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 8 ) return CutlassSgemmNN_1(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 8 ) return CutlassSgemmNN_2(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 8 ) return CutlassSgemmNN_3(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 8 ) return CutlassSgemmNN_4(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 8 && k == 16 ) return CutlassSgemmNN_5(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 16 ) return CutlassSgemmNN_6(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 16 ) return CutlassSgemmNN_7(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 16 ) return CutlassSgemmNN_8(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 16 ) return CutlassSgemmNN_9(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 8 && k == 32 ) return CutlassSgemmNN_10(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 32 ) return CutlassSgemmNN_11(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 32 ) return CutlassSgemmNN_12(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 32 ) return CutlassSgemmNN_13(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 32 ) return CutlassSgemmNN_14(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 8 && k == 64 ) return CutlassSgemmNN_15(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 64 ) return CutlassSgemmNN_16(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 64 ) return CutlassSgemmNN_17(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 64 ) return CutlassSgemmNN_18(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 64 ) return CutlassSgemmNN_19(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 8 && k == 128 ) return CutlassSgemmNN_20(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 128 ) return CutlassSgemmNN_21(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 128 ) return CutlassSgemmNN_22(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 128 ) return CutlassSgemmNN_23(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 128 ) return CutlassSgemmNN_24(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 8 && k == 256 ) return CutlassSgemmNN_25(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 256 ) return CutlassSgemmNN_26(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 256 ) return CutlassSgemmNN_27(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 256 ) return CutlassSgemmNN_28(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 256 ) return CutlassSgemmNN_29(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 8 && k == 512 ) return CutlassSgemmNN_30(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 16 && k == 512 ) return CutlassSgemmNN_31(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 32 && k == 512 ) return CutlassSgemmNN_32(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 64 && k == 512 ) return CutlassSgemmNN_33(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (n == 128 && k == 512 ) return CutlassSgemmNN_34(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    exit(-1);
}

void test_cutlass(int m, int n, int k, int num_tests, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    CutlassSgemmNN_tuned(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    for(int i = 0; i < num_tests; ++i){
        CutlassSgemmNN_tuned(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    printf("cutlass %d %d %d %f %f\n", m, n, k, elapsed, gflops);
}

int main(int argc, char** argv){
    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed: " << cublasStatus << std::endl;
        return 1;
    }

    DataT *A, *dA, *B, *dB, *C, *dC;
    int M = 131072, N = 128, K = 512;
    // freopen("input.txt", "r", stdin);
    // scanf("%d%d%d", &M, &N, &K);
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

    int num_tests = 10;
    
    for(int m = (1<<17); m <= (1<<17); m *= 2){
        for(int k = 8; k <= 512; k *= 2){
            for(int n = 8; n <= 128; n *= 2){
                test_cutlass(m, n, k, num_tests, alpha, beta, dA, dB, dC);
                test_cublas(m, n, k, num_tests, handle, alpha, beta, dA, dB, dC);
            }
        }
    }
    // // test_cutlass(10240, 10240, 10240, num_tests, alpha, beta, dA, dB, dC);
    
    return 0;
}

//all possible functions down here

cudaError_t CutlassSgemmNN_0(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_1(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_2(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_3(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_4(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_5(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_6(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,256,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_7(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_8(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_9(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_10(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512,64,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_11(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_12(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_13(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32,256,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16,256,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_14(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_15(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_16(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32,128,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_17(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_18(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_19(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_20(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_21(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_22(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_23(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256,64,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_24(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_25(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64,512,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_26(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128,32,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_27(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_28(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64,64,32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_29(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_30(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_31(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_32(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_33(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_34(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
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
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 

  constexpr int NumStages = 2;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  Gemm::Arguments arguments({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);
  return cudaSuccess;
}
