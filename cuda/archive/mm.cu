#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/common.cuh"
#include "../include/tensor.h"

__global__ void matmul_kernel_naive(
    float *A, float *B, float *C,
    int M, int N, int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void matmul_forward_pass(const Tensor *A, const Tensor *B, Tensor *C)
{
    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, A->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, C->size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A->data, A->size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B->data, B->size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C->data, d_C, C->size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmul_backward_A_kernel(
    float *dC, float *B, float *dA,
    int M, int N, int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K)
    {
        float sum = 0.0f;

        for (int j = 0; j < N; j++)
        {
            sum += dC[row * N + j] * B[col * N + j];
        }

        dA[row * K + col] = sum;
    }
}

__global__ void matmul_backward_B_kernel(
    float *A, float *dC, float *dB,
    int M, int N, int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < N)
    {
        float sum = 0.0f;

        for (int i = 0; i < M; i++)
        {
            sum += A[i * K + row] * dC[i * N + col];
        }

        dB[row * N + col] = sum;
    }
}


void matmul_backward_pass(
    const Tensor *A,
    const Tensor *B,
    const Tensor *dC,
    Tensor *dA,
    Tensor *dB
)
{
    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    float *d_A, *d_B, *d_dC, *d_dA, *d_dB;

    CUDA_CHECK(cudaMalloc(&d_A, A->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dC, dC->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA, dA->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dB, dB->size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A->data, A->size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B->data, B->size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dC, dC->data, dC->size * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_dA, 0, dA->size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dB, 0, dB->size * sizeof(float)));
    dim3 blockDim(16, 16);
    
    // dA = dC x transpose(B)
    dim3 gridA((K + 15)/16, (M + 15)/16);
    matmul_backward_A_kernel<<<gridA, blockDim>>>(d_dC, d_B, d_dA, M, N, K);

    // dB = Transpose(A) × dC
    dim3 gridB((N + 15)/16, (K + 15)/16);
    matmul_backward_B_kernel<<<gridB, blockDim>>>(d_A, d_dC, d_dB, M, N, K);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(dA->data, d_dA, dA->size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dB->data, d_dB, dB->size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_dC);
    cudaFree(d_dA);
    cudaFree(d_dB);
}
