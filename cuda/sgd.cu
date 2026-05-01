#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_DIM 256

__global__ void sgd_step_kernel(float *param, const float *grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) param[i] -= lr * grad[i];
}

extern "C" void sgd_step_pass(Tensor *param, float lr) {
    int n = param->size;
    sgd_step_kernel<<<cdiv(n, BLOCK_DIM), BLOCK_DIM>>>(param->data, param->grad->data, lr, n);
    CUDA_CHECK(cudaGetLastError());
}
