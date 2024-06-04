#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "helper.h"
using DataT = double;

bool CutlassDgemmNN(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  //injection here
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<512, 32, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 16, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue>; 

  constexpr int NumStages = 3;

  using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                           LayoutInputA,
                                           ElementInputB,
                                           LayoutInputB,
                                           ElementOutput,
                                           LayoutOutput,
                                           ElementAccumulator,
                                           MMAOp,
                                           SmArch,
                                           ShapeMMAThreadBlock,
                                           ShapeMMAWarp,
                                           ShapeMMAOp,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           NumStages>;

  Gemm gemm_op;

  Gemm::Arguments arguments({M, N, K},
                            {A, lda},
                            {B, ldb},
                            {C, ldc},
                            {C, ldc},
                            {alpha, beta});
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  status = gemm_op();
  //CUTLASS_CHECK(status);
  if (status != cutlass::Status::kSuccess)
    return 0;
  return 1;
}

void test_cutlass(int m, int n, int k, int num_tests, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    bool ret = CutlassDgemmNN(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    if (ret == 0) return;
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    for(int i = 0; i < num_tests; ++i){
        CutlassDgemmNN(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    printf("%d, %d, %d, %f, %f\n", m, n, k, elapsed, gflops);
}

int main(int argc, char** argv){
    DataT *A, *dA, *B, *dB, *C, *dC;
    int M = 65536, N = 128, K = 128;
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

    test_cutlass(M, N, K, num_tests, alpha, beta, dA, dB, dC);
    
    // for(int m = 256; m <= (1<<17); m *= 2){
    //     for(int k = 2; k <= 128; k *= 2){
    //         for(int n = 2; n <= 128; n *= 2){
    //             test_cutlass(m, n, k, num_tests, alpha, beta, dA, dB, dC);
                
    //         }
    //     }
    // }
    // // test_cutlass(10240, 10240, 10240, num_tests, alpha, beta, dA, dB, dC);
    
    return 0;
}