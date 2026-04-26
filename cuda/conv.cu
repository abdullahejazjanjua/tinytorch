#include <assert.h>
#include <stdio.h>

#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_SIZE_FORWARD 32
#define BLOCK_SIZE_BACKWARD_W 16 // this is done to help occupancy due to local KxK array in backward_w
#define BLOCK_SIZE_BACKWARD_INPUT 32
#define COARSE_FACTOR 2
#define MAX_FILTER_SIZE 16384

// __constant__ float c_filter[MAX_FILTER_SIZE];

__global__ void conv2d_forward_kernel(float *in,
                                      float *filters,
                                      int batch_size,
                                      int input_height, int input_width,
                                      int output_height, int output_width,
                                      int in_channels, int out_channels,
                                      int kernel_size,
                                      int pad_h, int pad_w,
                                      float *out)
{
    // grid_width is out_channels / COARSE_FACTOR
    int filter_groups = cdiv(out_channels, COARSE_FACTOR);
    int start_filter_idx = (blockIdx.z % filter_groups) * COARSE_FACTOR;
    int start_batch_idx  = (blockIdx.z / filter_groups) * COARSE_FACTOR;

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_input_tile[];
    int tile_dim = blockDim.x + kernel_size - 1;

    // Accumulators for the filter coarsening dimension
    float accumulators[COARSE_FACTOR];

    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        int bs = start_batch_idx + i;
        if (bs >= batch_size)
            continue;

        for (int v = 0; v < COARSE_FACTOR; v++)
            accumulators[v] = 0.0f;

        for (int c = 0; c < in_channels; c++)
        {
            // co-operative load of input tile
            for (int t = (threadIdx.y * blockDim.x + threadIdx.x); t < (tile_dim * tile_dim); t += (blockDim.x * blockDim.y))
            {

                int load_y = t / tile_dim;
                int load_x = t % tile_dim;

                int in_y = (blockIdx.y * blockDim.y) + load_y - pad_h;
                int in_x = (blockIdx.x * blockDim.x) + load_x - pad_w;

                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width)
                    s_input_tile[load_y * tile_dim + load_x] = in[bs * (in_channels * input_height * input_width) +
                                                                  c * (input_height * input_width) + in_y * input_width +
                                                                  in_x];
                else
                    s_input_tile[load_y * tile_dim + load_x] = 0.0f;
            }
            __syncthreads();

            if (out_y < output_height && out_x < output_width)
            {
                for (int j = 0; j < COARSE_FACTOR; j++)
                {
                    int filter_k = start_filter_idx + j;
                    if (filter_k < out_channels)
                    {
                        #pragma unroll
                        for (int f_i = 0; f_i < kernel_size; f_i++)
                        {
                            #pragma unroll
                            for (int f_j = 0; f_j < kernel_size; f_j++)
                            {
                                accumulators[j] += s_input_tile[(threadIdx.y + f_i) * tile_dim + (threadIdx.x + f_j)] *
                                                   filters[filter_k * (in_channels * kernel_size * kernel_size) +
                                                            c * (kernel_size * kernel_size) + (f_i * kernel_size) +
                                                            f_j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (out_y < output_height && out_x < output_width)
        {
            for (int j = 0; j < COARSE_FACTOR; j++)
            {
                int filter_k = start_filter_idx + j;
                if (filter_k < out_channels)
                {
                    out[bs * (out_channels * output_height * output_width) + filter_k * (output_height * output_width) + out_y * output_width + out_x] = accumulators[j];
                }
            }
        }
    }
}

template <int K>
__global__ void conv2d_backward_weight_kernel(float *in,
                                              float *dout,
                                              int batch_size,
                                              int input_height, int input_width,
                                              int output_height, int output_width,
                                              int in_channels, int out_channels,
                                              int kernel_size,
                                              int pad_h, int pad_w,
                                              float *grad_w)
{
    int in_channel_idx = (blockIdx.z % in_channels);
    int out_channel_idx = (blockIdx.z / in_channels);

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    int tile_dim = blockDim.x + kernel_size - 1;
    __shared__ float s_input_tile[(BLOCK_SIZE_BACKWARD_W + K - 1) * (BLOCK_SIZE_BACKWARD_W + K - 1)];

    // for block-level reduction
    __shared__ float acc_reduction[K][K];
    if (threadIdx.y < K && threadIdx.x < K)
        acc_reduction[threadIdx.y][threadIdx.x] = 0.0f;

    float acc[K][K];
    #pragma unroll
    for (int ky = 0; ky < K; ky++) {
        #pragma unroll
        for (int kx = 0; kx < K; kx++) {
            acc[ky][kx] = 0.0f;
        }
    }

    for (int bs = 0; bs < batch_size; bs++)
    {
        __syncthreads();
        // co-operatively, load kxk window into shared_mem
        for (int t = (threadIdx.y * blockDim.x + threadIdx.x); t < (tile_dim * tile_dim); t += (blockDim.x * blockDim.y))
        {

            int load_y = t / tile_dim;
            int load_x = t % tile_dim;

            int in_y = (blockIdx.y * blockDim.y) + load_y - pad_h;
            int in_x = (blockIdx.x * blockDim.x) + load_x - pad_w;

            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width)
                s_input_tile[load_y * tile_dim + load_x] = in[bs * (in_channels * input_height * input_width) +
                                                              in_channel_idx * (input_height * input_width) +
                                                              in_y * input_width +
                                                              in_x];
            else
                s_input_tile[load_y * tile_dim + load_x] = 0.0f;
        }
        __syncthreads();
        
        if (out_y < output_height && out_x < output_width) {
            float dout_i = dout[bs * (out_channels * output_height * output_width) + 
                                out_channel_idx * (output_height * output_width)   + 
                                out_y * (output_width) + 
                                out_x];
            #pragma unroll
            for (int f_i = 0; f_i < K; f_i++)
            {
                #pragma unroll
                for (int f_j = 0; f_j < K; f_j++)
                {
                    acc[f_i][f_j] += s_input_tile[(threadIdx.y + f_i) * tile_dim + (threadIdx.x + f_j)] 
                                     * dout_i;
                                
                }
            }
        }
    }

    // Perform reduction in shared mem (block-level)
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            atomicAdd(&acc_reduction[ky][kx],acc[ky][kx]);
        }
    }

    __syncthreads();

    // Perform reduction in global memory
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                atomicAdd(&grad_w[out_channel_idx * (in_channels * kernel_size * kernel_size) + 
                        in_channel_idx * (kernel_size * kernel_size) + 
                        ky * (kernel_size) + 
                        kx], acc_reduction[ky][kx]);
            }
        }
    }
}

__global__ void conv2d_backward_input_kernel(float *dout,
                                             float *filters,
                                              int batch_size,
                                              int input_height, int input_width,
                                              int output_height, int output_width,
                                              int in_channels, int out_channels,
                                              int kernel_size,
                                              int pad_h, int pad_w,
                                              float *grad_in) // grad_in set to 0.0f;
{
    int in_channel_idx = (blockIdx.z % in_channels);
    int batch_idx  = (blockIdx.z / in_channels);

    int din_y = blockIdx.y * blockDim.y + threadIdx.y;
    int din_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float s_dout_tile[];
    int tile_dim = blockDim.x + kernel_size - 1;

    // shift by kernel_size to the left to account for the fact that multiple douts with multiple ws 
    // are requried for grad_in (see derivation in notes)
    int start_dout_y = (blockIdx.y * blockDim.y) + pad_h - kernel_size + 1;
    int start_dout_x = (blockIdx.x * blockDim.x) + pad_w - kernel_size + 1;

    float grad = 0.0f;
    for (int out_c = 0; out_c < out_channels; out_c++) {
        __syncthreads();
         // co-operatively load of dout tile
        for (int t = (threadIdx.y * blockDim.x + threadIdx.x); t < (tile_dim * tile_dim); t += (blockDim.x * blockDim.y)) {
            int load_y = t / tile_dim;
            int load_x = t % tile_dim;

            int dout_y = start_dout_y + load_y;
            int dout_x = start_dout_x + load_x;

            if (dout_y >= 0 && dout_y < output_height && dout_x >= 0 && dout_x < output_width)
                s_dout_tile[load_y * tile_dim + load_x] = dout[batch_idx * (out_channels * output_height * output_width) +
                                                                out_c * (output_height * output_width) +
                                                                dout_y * output_width +
                                                                dout_x];
            else
                s_dout_tile[load_y * tile_dim + load_x] = 0.0f;
        }
        __syncthreads();

        if (din_y < input_height && din_x < input_width) {
            // we reverse the loops here (see notes dL/dx6 derivation)
            for (int ky = kernel_size - 1; ky >= 0; ky--) {
                for (int kx = kernel_size - 1; kx >= 0; kx--)
                {
                    grad += filters[out_c * (in_channels * kernel_size * kernel_size) + 
                                in_channel_idx * (kernel_size * kernel_size) + 
                                (kernel_size - 1 - ky) * (kernel_size) + 
                                (kernel_size - 1 - kx)] * 
                                s_dout_tile[(threadIdx.y + ky) * tile_dim + (threadIdx.x + kx)];
                }
            }
        }
    }

    if (din_y < input_height && din_x < input_width)
        grad_in[batch_idx * (in_channels * input_height * input_width) +
                in_channel_idx * (input_height * input_width) +
                din_y * input_width +
                din_x] = grad;
}

void conv2d_forward_pass(const Tensor *input, const Tensor *filters, int padding, Tensor *output)
{
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    int out_channels = filters->shape[0];
    int kernel_size = filters->shape[2];

    int pad_h = 0, pad_w = 0;
    if (padding)
    {
        output->shape[2] = input_height;
        output->shape[3] = input_width;

        pad_h = (kernel_size - 1) / 2;
        pad_w = (kernel_size - 1) / 2;
    }
    else
    {
        output->shape[2] = input_height - kernel_size + 1;
        output->shape[3] = input_width - kernel_size + 1;
        assert(output->shape[2] > 0 && output->shape[3] > 0);
    }

    output->shape[0] = batch_size;
    output->shape[1] = out_channels;

    int output_height = output->shape[2];
    int output_width = output->shape[3];

    output->size = batch_size * out_channels * output_height * output_width;

    dim3 dimBlock(BLOCK_SIZE_FORWARD, BLOCK_SIZE_FORWARD, 1);
    dim3 dimGrid(cdiv(output_width, BLOCK_SIZE_FORWARD), cdiv(output_height, BLOCK_SIZE_FORWARD), cdiv(batch_size, COARSE_FACTOR) * cdiv(out_channels, COARSE_FACTOR));

    int tile_dim = BLOCK_SIZE_FORWARD + kernel_size - 1;
    size_t dynamic_shared_bytes = tile_dim * tile_dim * sizeof(float);

    conv2d_forward_kernel<<<dimGrid, dimBlock, dynamic_shared_bytes>>>(input->data, filters->data, batch_size, input_height,
                                                                       input_width, output_height, output_width,
                                                                       in_channels, out_channels, kernel_size,
                                                                       pad_h, pad_w, output->data);
    CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_pass_weight(const Tensor *input, const Tensor *dout, int padding, Tensor *grad_w)
{
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    int out_channels = dout->shape[1];
    int output_height = dout->shape[2];
    int output_width = dout->shape[3];

    int kernel_size = grad_w->shape[2];

    int pad_h = 0;
    int pad_w = 0;

    if (padding)
    {
        pad_h = (kernel_size - 1) / 2;
        pad_w = (kernel_size - 1) / 2;
    }
    CUDA_CHECK(cudaMemset(grad_w->data, 0, grad_w->size * sizeof(float)));

    dim3 dimBlock(BLOCK_SIZE_BACKWARD_W, BLOCK_SIZE_BACKWARD_W, 1);
    dim3 dimGrid(cdiv(output_width, BLOCK_SIZE_BACKWARD_W), cdiv(output_height, BLOCK_SIZE_BACKWARD_W), out_channels * in_channels);

    switch (kernel_size) {
        case 1: 
            conv2d_backward_weight_kernel<1><<<dimGrid, dimBlock>>>(input->data, dout->data, batch_size, input_height,
                                                                            input_width, output_height, output_width,
                                                                            in_channels, out_channels, kernel_size,
                                                                            pad_h, pad_w, grad_w->data);
            break;
        case 3: 
            conv2d_backward_weight_kernel<3><<<dimGrid, dimBlock>>>(input->data, dout->data, batch_size, input_height,
                                                                            input_width, output_height, output_width,
                                                                            in_channels, out_channels, kernel_size,
                                                                            pad_h, pad_w, grad_w->data);
            break;
        case 5: 
            conv2d_backward_weight_kernel<5><<<dimGrid, dimBlock>>>(input->data, dout->data, batch_size, input_height,
                                                                            input_width, output_height, output_width,
                                                                            in_channels, out_channels, kernel_size,
                                                                            pad_h, pad_w, grad_w->data);
            break;
        case 7: 
            conv2d_backward_weight_kernel<7><<<dimGrid, dimBlock>>>(input->data, dout->data, batch_size, input_height,
                                                                            input_width, output_height, output_width,
                                                                            in_channels, out_channels, kernel_size,
                                                                            pad_h, pad_w, grad_w->data);
            break;
        default:
            fprintf(stderr, "[%s:%d] %d not supported. Only 3x3, 5x5, 7x7 are supported, so be good boy\n", __FILE__, __LINE__, kernel_size);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_pass_input(const Tensor *filters, const Tensor *dout, int padding, Tensor *grad_input)
{
    int batch_size = grad_input->shape[0];
    int in_channels = grad_input->shape[1];
    int input_height = grad_input->shape[2];
    int input_width = grad_input->shape[3];

    int out_channels = dout->shape[1];
    int output_height = dout->shape[2];
    int output_width = dout->shape[3];

    int kernel_size = filters->shape[2];

    int pad_h = 0;
    int pad_w = 0;

    if (padding)
    {
        pad_h = (kernel_size - 1) / 2;
        pad_w = (kernel_size - 1) / 2;
    }

    dim3 dimBlock(BLOCK_SIZE_BACKWARD_INPUT, BLOCK_SIZE_BACKWARD_INPUT, 1);
    dim3 dimGrid(cdiv(input_width, BLOCK_SIZE_BACKWARD_INPUT), cdiv(input_height, BLOCK_SIZE_BACKWARD_INPUT), batch_size * in_channels);
    
    int tile_dim = BLOCK_SIZE_BACKWARD_INPUT + kernel_size - 1;
    size_t dynamic_shared_bytes = tile_dim * tile_dim * sizeof(float);
    
    conv2d_backward_input_kernel<<<dimGrid, dimBlock, dynamic_shared_bytes>>>(dout->data, filters->data, batch_size, input_height, input_width, output_height, output_width, in_channels, out_channels, kernel_size, pad_h, pad_w, grad_input->data);
    CUDA_CHECK(cudaGetLastError());
}