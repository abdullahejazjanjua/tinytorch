/*
    This file contains previous versions of conv kernel
*/


#include <assert.h>

#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_SIZE 32
#define COARSE_FACTOR 2
#define MAX_FILTER_SIZE 16384

__constant__ float c_filter[MAX_FILTER_SIZE];

__global__ void conv2d_kernel(float *in,
                              float *out,
                              int batch_size, int input_height, int input_width, int output_height, int output_width, int in_channels,
                              int out_channels, int kernel_size,
                              int pad_h, int pad_w)
{
    int filter_k = blockIdx.z * blockDim.z + threadIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (filter_k < out_channels && out_y < output_height && out_x < output_width)
    {
        for (int bs = 0; bs < batch_size; bs++)
        {
            float val = 0.0f;
            for (int c = 0; c < in_channels; c++)
            {
                for (int f_i = 0; f_i < kernel_size; f_i++)
                {
                    for (int f_j = 0; f_j < kernel_size; f_j++)
                    {
                        int in_y = out_y + f_i - pad_h;
                        int in_x = out_x + f_j - pad_w;
                        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width)
                            val += in[bs * (in_channels * input_height * input_width) +
                                      c * (input_height * input_width) + in_y * input_width +
                                      in_x] *
                                   c_filter[filter_k * (in_channels * kernel_size * kernel_size) +
                                            c * (kernel_size * kernel_size) + (f_i * kernel_size) +
                                            f_j];
                    }
                }
            }
            out[bs * (out_channels * output_height * output_width) + filter_k * (output_height * output_width) + out_y * output_width + out_x] = val;
        }
    }
}

__global__ void conv2d_kernelv2(float *in,
                                float *out,
                                int batch_size, int input_height, int input_width, int output_height, int output_width, int in_channels,
                                int out_channels, int kernel_size,
                                int pad_h, int pad_w)
{

    int filter_k = blockIdx.z * blockDim.z + threadIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_input_tile[BLOCK_SIZE][BLOCK_SIZE];

    for (int bs = 0; bs < batch_size; bs++)
    {
        float val = 0.0f;
        for (int c = 0; c < in_channels; c++)
        {
            if (out_y < input_height && out_x < input_width)
                s_input_tile[threadIdx.y][threadIdx.x] = in[bs * (in_channels * input_height * input_width) +
                                                            c * (input_height * input_width) +
                                                            out_y * (input_width) +
                                                            out_x];
            else
                s_input_tile[threadIdx.y][threadIdx.x] = 0.0f;
            __syncthreads();

            if (filter_k < out_channels && out_y < output_height && out_x < output_width)
            {
                for (int f_i = 0; f_i < kernel_size; f_i++)
                {
                    for (int f_j = 0; f_j < kernel_size; f_j++)
                    {
                        if ((threadIdx.y + f_i - pad_h) < BLOCK_SIZE &&
                            (threadIdx.x + f_j - pad_w) < BLOCK_SIZE)
                        {
                            val += s_input_tile[threadIdx.y + f_i - pad_h][threadIdx.x + f_j - pad_w] *
                                   c_filter[filter_k * (in_channels * kernel_size * kernel_size) +
                                            c * (kernel_size * kernel_size) + (f_i * kernel_size) +
                                            f_j];
                        }
                        else if ((out_y + f_i - pad_h) >= 0 && (out_y + f_i - pad_h) < input_height &&
                                 (out_x + f_j - pad_w) >= 0 && (out_x + f_j - pad_w) < input_width)
                        {
                            int in_y = out_y + f_i - pad_h;
                            int in_x = out_x + f_j - pad_w;
                            // bring that fker from cache (hopefully :D)
                            val += in[bs * (in_channels * input_height * input_width) +
                                      c * (input_height * input_width) +
                                      in_y * (input_width) +
                                      in_x] *
                                   c_filter[filter_k * (in_channels * kernel_size * kernel_size) +
                                            c * (kernel_size * kernel_size) + (f_i * kernel_size) +
                                            f_j];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (filter_k < out_channels && out_y < output_height && out_x < output_width)
            out[bs * (out_channels * output_height * output_width) + filter_k * (output_height * output_width) + out_y * output_width + out_x] = val;
    }
}

// one might be curious as to why I implemented the above version, if I find out, I will tell you (but how?)
__global__ void conv2d_kernelv3(float *in,
                                float *out,
                                int batch_size, int input_height, int input_width, int output_height, int output_width, int in_channels,
                                int out_channels, int kernel_size,
                                int pad_h, int pad_w)
{

    // here we say that batch_size is our rows and out_channels is our column
    // we always div and mod by innermost dimension, in this case out_channels.
    // intuitively, we can say first do batch 0 (as div changes slowly) and then move to the next.
    // interesting optimization, this is still slower than torch's implementation by 15-17x, but I think this much
    // optimization is enough
    int filter_k = blockIdx.z % out_channels;
    int bs = blockIdx.z / out_channels;

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_input_tile[];

    int tile_dim = blockDim.x + kernel_size - 1;

    float val = 0.0f;
    for (int c = 0; c < in_channels; c++)
    {
        // flattened thread_id -> (elements_to_load: num_threads)
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x); i < (tile_dim * tile_dim); i += (blockDim.x * blockDim.y))
        {
            /*
                We flattened our input data and threads, and map the block onto input data to contagiously
                load num_threads elements, in this case there is control diveregence at the 2 edges.
                If we used 2D nested loop, then there would be control divergence at each boundary.
            */

            // compute the 2D index
            int load_y = i / tile_dim;
            int load_x = i % tile_dim;

            // offset into the block using load_y & load_x and shift by pad_h & pad_w to handle padding.
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

        if (filter_k < out_channels && out_y < output_height && out_x < output_width)
        {
            for (int f_i = 0; f_i < kernel_size; f_i++)
            {
                for (int f_j = 0; f_j < kernel_size; f_j++)
                {
                    val += s_input_tile[(threadIdx.y + f_i) * tile_dim + (threadIdx.x + f_j)] *
                           c_filter[filter_k * (in_channels * kernel_size * kernel_size) +
                                    c * (kernel_size * kernel_size) + (f_i * kernel_size) +
                                    f_j];
                }
            }
        }
        __syncthreads();
    }
    if (filter_k < out_channels && out_y < output_height && out_x < output_width)
        out[bs * (out_channels * output_height * output_width) + filter_k * (output_height * output_width) + out_y * output_width + out_x] = val;
}
