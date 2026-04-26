#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_DIM 256
#define BLOCK_SIZE 32
#define COARSE_FACTOR_FORWARD 4
#define COARSE_FACTOR_BACKWARD 2


__global__ void global_pooling_forward_kernel(float *data, int batch_size, int num_channels, int height_width, float *out) {
    __shared__ float input_s[BLOCK_DIM];

    int channel_groups = cdiv(num_channels, COARSE_FACTOR_FORWARD);
    int start_channel_idx = (blockIdx.x % channel_groups) * COARSE_FACTOR_FORWARD;
    int start_batch_idx  = (blockIdx.x / channel_groups) * COARSE_FACTOR_FORWARD;

    unsigned int segment = COARSE_FACTOR_FORWARD * 2 * blockDim.x * blockIdx.y;
    unsigned int i = segment + threadIdx.x;

    #pragma unroll
    for (int bs = 0; bs < COARSE_FACTOR_FORWARD; ++bs) {
        int batch_idx = start_batch_idx + bs;
        if (batch_idx >= batch_size) continue;

        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR_FORWARD; ++c) {
            int channel_idx = start_channel_idx + c;
            if (channel_idx >= num_channels) continue;

            float sum = 0.0f;
            for(unsigned int tile = 0; tile < COARSE_FACTOR_FORWARD * 2; ++tile) {
                unsigned int idx = i + tile * BLOCK_DIM;
                if (idx < height_width) {
                    sum += data[batch_idx * (num_channels * height_width) + 
                                channel_idx * height_width + 
                                idx
                            ];
                }
            }
            input_s[threadIdx.x] = sum;

            for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
                __syncthreads();
                if (threadIdx.x < stride) {
                    input_s[threadIdx.x] += input_s[threadIdx.x + stride];
                }
            }
            __syncthreads();
            float val = 0.0f;
            if (threadIdx.x < 32) {
                val += input_s[threadIdx.x] + input_s[threadIdx.x + 32];
                #pragma unroll
                for (int offset = 16; offset >= 1; offset /= 2) {
                    val += __shfl_down_sync(0xffffffff, val, offset); // find val at register_i + offset position
                }
            }
            __syncthreads(); // process all elements in the current channel
            if (threadIdx.x == 0) {
                atomicAdd(&out[batch_idx * num_channels + channel_idx], val / height_width);
            }
            __syncthreads(); // process all the elements in the current batch
        }
    }
}

__global__ void global_pooling_backward_kernel(float *dout, int batch_size, int num_channels, int height, int width, float *grad_data) {
    int channel_groups = cdiv(num_channels, COARSE_FACTOR_BACKWARD);
    int start_channel_idx = (blockIdx.x % channel_groups) * COARSE_FACTOR_BACKWARD;
    int start_batch_idx  = (blockIdx.x / channel_groups) * COARSE_FACTOR_BACKWARD;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int bs = 0; bs < COARSE_FACTOR_BACKWARD; bs++) {
        int batch_idx = start_batch_idx + bs;
        if (batch_idx >= batch_size) continue;

        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR_BACKWARD; c++) {
            int channel_idx = start_channel_idx + c;
            if (channel_idx >= num_channels) continue;

            float global_pool_val = dout[batch_idx * (num_channels) + channel_idx];
            if (row < height && col < width)
                grad_data[batch_idx * (num_channels * height * width) + 
                        channel_idx * (height * width) + 
                        row * (width) + 
                        col
                        ] = global_pool_val / (height * width);
        }
    }
}

void global_pooling_forward_pass(Tensor *input, Tensor *output) {
    int batch_size = input->shape[0];
    int num_channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];
    int height_width = height * width;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, input->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, output->size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input->data, input->size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, output->size * sizeof(float)));

    int channel_groups = cdiv(num_channels, COARSE_FACTOR_FORWARD);
    int batch_groups = cdiv(batch_size, COARSE_FACTOR_FORWARD);

    dim3 dimBlock(BLOCK_DIM, 1, 1);
    dim3 dimGrid(batch_groups * channel_groups, cdiv(height_width, COARSE_FACTOR_FORWARD * 2 * BLOCK_DIM), 1);

    global_pooling_forward_kernel<<<dimGrid, dimBlock>>>(d_input, batch_size, num_channels, height_width, d_output);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(output->data, d_output, output->size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


void global_pooling_backward_pass(Tensor *dout, Tensor *grad_input) {
    int batch_size = grad_input->shape[0];
    int num_channels = grad_input->shape[1];
    int height = grad_input->shape[2];
    int width = grad_input->shape[3];

    float *d_dout, *d_grad_input;
    CUDA_CHECK(cudaMalloc((void **)&d_dout, dout->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_grad_input, grad_input->size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dout, dout->data, dout->size * sizeof(float), cudaMemcpyHostToDevice));

    int channel_groups = cdiv(num_channels, COARSE_FACTOR_BACKWARD);
    int batch_groups = cdiv(batch_size, COARSE_FACTOR_BACKWARD);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(batch_groups * channel_groups, cdiv(height, BLOCK_SIZE), cdiv(width, BLOCK_SIZE));

    global_pooling_backward_kernel<<<dimGrid, dimBlock>>>(d_dout, batch_size, num_channels, height, width, d_grad_input);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(grad_input->data, d_grad_input, grad_input->size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dout));
    CUDA_CHECK(cudaFree(d_grad_input));
}