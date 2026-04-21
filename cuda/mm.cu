#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_SIZE 32
#define COARSE_FACTOR 2

__global__ void matmul_kernel(
    float *A, float *B, float *C,
    int M, int N, int K
)
{
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE * COARSE_FACTOR];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_start = blockIdx.x * (BLOCK_SIZE * COARSE_FACTOR) + threadIdx.x;

    float accum[COARSE_FACTOR];
    #pragma unroll                    //time to spam unroll... 2 CF ain't gonna hurt
    for (int i = 0; i < COARSE_FACTOR; i++)
        accum[i] = 0.0f;

    for (int tile = 0; tile < cdiv(K, BLOCK_SIZE); tile++)
    {
        int a_col = tile * BLOCK_SIZE + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = tile * BLOCK_SIZE + threadIdx.y;
        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
        {
            int b_col = col_start + i * BLOCK_SIZE;
            s_B[threadIdx.y][threadIdx.x + i * BLOCK_SIZE] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            float a_val = s_A[threadIdx.y][k];
            #pragma unroll
            for (int i = 0; i < COARSE_FACTOR; i++)
                accum[i] += a_val * s_B[k][threadIdx.x + i * BLOCK_SIZE];
        }

        __syncthreads();
    }

    if (row < M)
    {
        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
        {
            int col = col_start + i * BLOCK_SIZE;
            if (col < N)
                C[row * N + col] = accum[i];
        }
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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(cdiv(N, BLOCK_SIZE * COARSE_FACTOR), cdiv(M, BLOCK_SIZE));

    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C->data, d_C, C->size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// dA = dC x Bt ... tiling over N to fix strided B access from naive imp
__global__ void matmul_backward_A_kernel(
    float *dC, float *B, float *dA,
    int M, int N, int K
)
{
    __shared__ float s_dC[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE * COARSE_FACTOR];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_start = blockIdx.x * (BLOCK_SIZE * COARSE_FACTOR) + threadIdx.x;

    float accum[COARSE_FACTOR];
    #pragma unroll
    for (int i = 0; i < COARSE_FACTOR; i++)
        accum[i] = 0.0f;

    for (int tile = 0; tile < cdiv(N, BLOCK_SIZE); tile++)
    {
        int n_idx = tile * BLOCK_SIZE + threadIdx.x;
        s_dC[threadIdx.y][threadIdx.x] = (row < M && n_idx < N) ? dC[row * N + n_idx] : 0.0f;

        int b_row_load = tile * BLOCK_SIZE + threadIdx.y;
        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
        {
            int col = col_start + i * BLOCK_SIZE;
            s_B[threadIdx.y][threadIdx.x + i * BLOCK_SIZE] = (col < K && b_row_load < N) ? B[col * N + b_row_load] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int n = 0; n < BLOCK_SIZE; n++)
        {
            float dc_val = s_dC[threadIdx.y][n];
            #pragma unroll
            for (int i = 0; i < COARSE_FACTOR; i++)
                accum[i] += dc_val * s_B[n][threadIdx.x + i * BLOCK_SIZE];
        }

        __syncthreads();
    }

    if (row < M)
    {
        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
        {
            int col = col_start + i * BLOCK_SIZE;
            if (col < K)
                dA[row * K + col] = accum[i];
        }
    }
}

// dB = At x dC ... tiling over M to fix strided A access from naive imp
__global__ void matmul_backward_B_kernel(
    float *A, float *dC, float *dB,
    int M, int N, int K
)
{
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_dC[BLOCK_SIZE][BLOCK_SIZE * COARSE_FACTOR];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;   // k-index
    int col_start = blockIdx.x * (BLOCK_SIZE * COARSE_FACTOR) + threadIdx.x;

    float accum[COARSE_FACTOR];
    #pragma unroll
    for (int i = 0; i < COARSE_FACTOR; i++)
        accum[i] = 0.0f;

    for (int tile = 0; tile < cdiv(M, BLOCK_SIZE); tile++)
    {
        int m_load = tile * BLOCK_SIZE + threadIdx.y;

        // loads A[m_load][k] where k spans this block's K-slice
        int k_idx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] = (m_load < M && k_idx < K) ? A[m_load * K + k_idx] : 0.0f;

        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
        {
            int col = col_start + i * BLOCK_SIZE;
            s_dC[threadIdx.y][threadIdx.x + i * BLOCK_SIZE] = (m_load < M && col < N) ? dC[m_load * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int m = 0; m < BLOCK_SIZE; m++)
        {
            float a_val = s_A[m][threadIdx.y];  // A[m][row] ... threadIdx.y is the k-row
            #pragma unroll
            for (int i = 0; i < COARSE_FACTOR; i++)
                accum[i] += a_val * s_dC[m][threadIdx.x + i * BLOCK_SIZE];
        }

        __syncthreads();
    }

    if (row < K)
    {
        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
        {
            int col = col_start + i * BLOCK_SIZE;
            if (col < N)
                dB[row * N + col] = accum[i];
        }
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
    CUDA_CHECK(cudaMalloc(&d_A,  A->size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,  B->size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dC, dC->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA, dA->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dB, dB->size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A,  A->data,  A->size  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,  B->data,  B->size  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dC, dC->data, dC->size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridA(cdiv(K, BLOCK_SIZE * COARSE_FACTOR), cdiv(M, BLOCK_SIZE));
    matmul_backward_A_kernel<<<gridA, blockDim>>>(d_dC, d_B, d_dA, M, N, K);

    dim3 gridB(cdiv(N, BLOCK_SIZE * COARSE_FACTOR), cdiv(K, BLOCK_SIZE));
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
