#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
// using DataT = float;
using DataT = double;
void test_cublas(int m, int n, int k, int num_tests, cublasHandle_t &handle, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    for(int i = 0; i < num_tests; ++i){
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    printf("%d, %d, %d, %f, %f\n", m, n, k, elapsed, gflops);
}

int main(){
    cublasHandle_t handle;

    cublasStatus_t cublasStatus = cublasCreate(&handle);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed: " << cublasStatus << std::endl;
        // 处理错误，可能需要释放资源或退出应用程序
        return 1;
    }
    
    DataT *A, *dA, *B, *dB, *C, *dC;
    long long int A_size = (1 << 17) * 512;
    long long int B_size = 128 * 512;
    long long int C_size = (1<<17) * 128;
    A = (DataT*)malloc(sizeof(DataT) * A_size);
    B = (DataT*)malloc(sizeof(DataT) * B_size);
    C = (DataT*)malloc(sizeof(DataT) * C_size);

    cudaMalloc((void**)&dA, sizeof(DataT) * A_size);
    cudaMalloc((void**)&dB, sizeof(DataT) * B_size);
    cudaMalloc((void**)&dC, sizeof(DataT) * C_size);

    for(long long int i = 0; i < A_size; ++i) A[i] = float(rand() % 5) + (rand() % 5) * 0.01;
    for(long long int i = 0; i < B_size; ++i) B[i] = float(rand() % 5) + (rand() % 5) * 0.01;
    for(long long int i = 0; i < C_size; ++i) C[i] = float(rand() % 5) + (rand() % 5) * 0.01;

    cudaMemcpy(dA, A, sizeof(DataT) * A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // 处理错误，可能需要释放资源或退出应用程序
        return 1;
    }

    DataT alpha = 0.0001, beta = -0.0001; 

    int num_tests = 10;

    // freopen("input.txt", "r", stdin);
    int m, n, k;
    // scanf("%d%d%d",&m,&n,&k);
    // fclose(stdin);
    // test_cublas(m, n, k, num_tests, handle, alpha, beta, dA, dB, dC);
    
    for(int m = 1 << 17; m < (1 << 18); m *= 2){
        printf("fix N\n");
        for(int k = 8; k <= 128; k += 120){
            for(int n = 8; n <= 128; n += 8){
                test_cublas(m, n, k, num_tests, handle, alpha, beta, dA, dB, dC);
                
            }
        }

        printf("fix K\n");
        
        for(int n = 8; n <= 128; n += 120){
            for(int k = 8; k <= 128; k += 8){
                test_cublas(m, n, k, num_tests, handle, alpha, beta, dA, dB, dC);
                
            }
        }
    }
    
    return 0;
}
