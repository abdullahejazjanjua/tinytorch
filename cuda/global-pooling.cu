#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_DIM 256
#define COARSE_FACTOR 1


__global__ void global_pooling_forward_kernel(float *data, int batch_size, int num_channels, int height_width, float *out) {
    __shared__ float input_s[BLOCK_DIM];

    int channel_groups = cdiv(num_channels, COARSE_FACTOR);
    int start_channel_idx = (blockIdx.x % channel_groups) * COARSE_FACTOR;
    int start_batch_idx  = (blockIdx.x / channel_groups) * COARSE_FACTOR;

    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.y;
    unsigned int i = segment + threadIdx.x;

    for (int bs = 0; bs < COARSE_FACTOR; ++bs) {
        int batch_idx = start_batch_idx + bs;
        if (batch_idx >= batch_size) continue;

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int channel_idx = start_channel_idx + c;
            if (channel_idx >= num_channels) continue;

            float sum = 0.0f;
            for(unsigned int tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
                unsigned int idx = i + tile * BLOCK_DIM;
                if (idx < height_width) {
                    sum += data[batch_idx * (num_channels * height_width) + 
                                channel_idx * height_width + 
                                idx
                            ];
                }
            }
            input_s[threadIdx.x] = sum;

            for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
                __syncthreads();
                if (threadIdx.x < stride) {
                    input_s[threadIdx.x] += input_s[threadIdx.x + stride];
                }
            }
            __syncthreads(); // process all elements in the current channel
            if (threadIdx.x == 0) {
                atomicAdd(&out[batch_idx * num_channels + channel_idx], input_s[0]);
            }
            __syncthreads(); // process all the elements in the current batch
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

    int channel_groups = cdiv(num_channels, COARSE_FACTOR);
    int batch_groups = cdiv(batch_size, COARSE_FACTOR);

    dim3 dimBlock(BLOCK_DIM, 1, 1);
    dim3 dimGrid(batch_groups * channel_groups, cdiv(height_width, COARSE_FACTOR * 2 * BLOCK_DIM), 1);

    global_pooling_forward_kernel<<<dimGrid, dimBlock>>>(d_input, batch_size, num_channels, height_width, d_output);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(output->data, d_output, output->size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}