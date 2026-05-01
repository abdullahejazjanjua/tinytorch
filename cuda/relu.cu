#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_DIM 256

__global__ void relu_forward_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

__global__ void relu_backward_kernel(const float *in, const float *dout, float *grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (in[i] > 0.0f) ? dout[i] : 0.0f;
}

void relu_forward_pass(const Tensor *input, Tensor *output) {
    int n = input->size;
    relu_forward_kernel<<<cdiv(n, BLOCK_DIM), BLOCK_DIM>>>(input->data, output->data, n);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward_pass(const Tensor *input, const Tensor *dout, Tensor *grad_in) {
    int n = input->size;
    relu_backward_kernel<<<cdiv(n, BLOCK_DIM), BLOCK_DIM>>>(input->data, dout->data, grad_in->data, n);
    CUDA_CHECK(cudaGetLastError());
}
