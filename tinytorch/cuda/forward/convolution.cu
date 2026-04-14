#include <assert.h>

#include "../common.h"

#define BLOCK_SIZE 32
#define COARSE_FACTOR 2
#define MAX_FILTER_SIZE 16384

__constant__ float c_filter[MAX_FILTER_SIZE];

/*
    float *in:          Input data with shape [batch_size, in_channels, input_height, input_width]
    float *out:         Output data with shape [batch_size, out_channels, output_height, output_width]
    int batch_size:     Number of images in the input batch
    int input_height, input_width:     Height and width of the input images
    int output_height, output_width:   Height and width of the resulting output images
    int in_channels:   Number of input feature maps (e.g., 3 for RGB)
    int out_channels:    Number of output feature maps (kernels)
    int kernel_size:    Spatial dimensions of the square kernel
    int pad_h, pad_w:   Vertical and horizontal zero-padding applied to the input,
                        interestingly this equivalent to filter_radius
*/

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
                        if ((threadIdx.y + f_i - pad_h) >= 0 && (threadIdx.y + f_i - pad_h) < BLOCK_SIZE &&
                            (threadIdx.x + f_j - pad_w) >= 0 && (threadIdx.x + f_j - pad_w) < BLOCK_SIZE)
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

// COARSING
__global__ void conv2d_kernelv4(float *in,
                                float *out,
                                int batch_size, 
                                int input_height, int input_width, 
                                int output_height, int output_width, 
                                int in_channels, int out_channels, 
                                int kernel_size,
                                int pad_h, int pad_w)
{

    // here we say that batch_size is our rows and out_channels is our column
    // we always div and mod by innermost dimension, in this case out_channels.
    // intuitively, we can say first do batch 0 (as div changes slowly) and then move to the next.
    // interesting optimization, this is still slower than torch's implementation by 15-17x, but I think this much
    // optimization is enough
    int start_filter_idx = (blockIdx.z % (out_channels / COARSE_FACTOR)) * COARSE_FACTOR; // now grid_width is out_channels / COARSE_FACTOR
    int start_batch_idx = (blockIdx.z / (out_channels / COARSE_FACTOR)) * COARSE_FACTOR;

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
                        for (int f_i = 0; f_i < kernel_size; f_i++)
                        {
                            for (int f_j = 0; f_j < kernel_size; f_j++)
                            {
                                accumulators[j] += s_input_tile[(threadIdx.y + f_i) * tile_dim + (threadIdx.x + f_j)] *
                                           c_filter[filter_k * (in_channels * kernel_size * kernel_size) +
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

void conv2d_forward_pass(float *h_input,
                         float *h_filters,
                         float *out_h,
                         int batch_size, int input_height, int input_width, int in_channels,
                         int out_channels, int kernel_size,
                         int padding)
{
    float *d_input, *d_output, *filter_d;
    int output_height, output_width, pad_h, pad_w;
    if (!padding) // no padding
    {
        output_height = floor(input_height - kernel_size) + 1;
        output_width = floor(input_width - kernel_size) + 1;

        pad_h = 0;
        pad_w = 0;

        assert(output_height > 0 && output_width > 0);
    }
    else // padding i.e output_height == input_height & output_width == input_width
    {

        output_height = input_height;
        output_width = input_width;

        pad_h = (kernel_size - 1) / 2;
        pad_w = (kernel_size - 1) / 2;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_input, (batch_size * in_channels * input_height * input_width * sizeof(float))));
    CUDA_CHECK(cudaMalloc((void **)&d_output, (batch_size * out_channels * output_height * output_width * sizeof(float))));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, (batch_size * in_channels * input_height * input_width * sizeof(float)), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_filters, (out_channels * in_channels * kernel_size * kernel_size * sizeof(float))));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(cdiv(output_width, BLOCK_SIZE), cdiv(output_height, BLOCK_SIZE), cdiv(batch_size * out_channels, COARSE_FACTOR));

    int tile_dim = BLOCK_SIZE + kernel_size - 1;
    size_t dynamic_shared_bytes = tile_dim * tile_dim * sizeof(float);

    conv2d_kernelv4<<<dimGrid, dimBlock, dynamic_shared_bytes>>>(d_input, d_output, batch_size, input_height, input_width, output_height, output_width, in_channels, out_channels, kernel_size, pad_h, pad_w);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out_h, d_output, (batch_size * out_channels * output_height * output_width * sizeof(float)), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    int batch_size = 2;
    int input_height = 5;
    int input_width = 5;
    int in_channels = 3;
    int out_channels = 2;
    int kernel_size = 3;

    float *h_input = (float *)malloc(batch_size * in_channels * input_height * input_width * sizeof(float));
    float *h_filters = (float *)malloc(out_channels * in_channels * kernel_size * kernel_size * sizeof(float));

    for (int i = 0; i < batch_size * in_channels * input_height * input_width; i++)
    {
        h_input[i] = 1.0f;
    }

    for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; i++)
    {
        h_filters[i] = 1.0f;
    }

    printf("Input Data:\n");
    for (int bs = 0; bs < batch_size; bs++)
    {
        printf("Batch %d:\n", bs);
        for (int c = 0; c < in_channels; c++)
        {
            printf("  Channel %d:\n", c);
            for (int i = 0; i < input_height; i++)
            {
                for (int j = 0; j < input_width; j++)
                {
                    printf("%f ", h_input[bs * (in_channels * input_height * input_width) + c * (input_height * input_width) + i * input_width + j]);
                }
                printf("\n");
            }
        }
    }

    printf("\nFilter Data:\n");
    for (int k = 0; k < out_channels; k++)
    {
        printf("Filter %d:\n", k);
        for (int c = 0; c < in_channels; c++)
        {
            printf("  Channel %d:\n", c);
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    printf("%f ", h_filters[k * (in_channels * kernel_size * kernel_size) + c * (kernel_size * kernel_size) + i * kernel_size + j]);
                }
                printf("\n");
            }
        }
    }

    // --- Test Case 1: Padding = 0 (Valid) ---
    int padding_valid = 0;
    int h_out_valid = input_height - kernel_size + 1;
    int w_out_valid = input_width - kernel_size + 1;
    float *out_h_valid = (float *)malloc(batch_size * out_channels * h_out_valid * w_out_valid * sizeof(float));

    conv2d_forward_pass(h_input, h_filters, out_h_valid, batch_size, input_height, input_width, in_channels, out_channels, kernel_size, padding_valid);

    printf("\nOutput Data (Padding = 0):\n");
    for (int bs = 0; bs < batch_size; bs++)
    {
        printf("Batch %d:\n", bs);
        for (int k = 0; k < out_channels; k++)
        {
            printf("  Filter %d:\n", k);
            for (int i = 0; i < h_out_valid; i++)
            {
                for (int j = 0; j < w_out_valid; j++)
                {
                    printf("%f ", out_h_valid[bs * (out_channels * h_out_valid * w_out_valid) + k * (h_out_valid * w_out_valid) + i * w_out_valid + j]);
                }
                printf("\n");
            }
        }
    }
    free(out_h_valid);

    // --- Test Case 2: Padding = 1 (Same) ---
    int padding_same = 1;
    int h_out_same = input_height;
    int w_out_same = input_width;
    float *out_h_same = (float *)malloc(batch_size * out_channels * h_out_same * w_out_same * sizeof(float));

    conv2d_forward_pass(h_input, h_filters, out_h_same, batch_size, input_height, input_width, in_channels, out_channels, kernel_size, padding_same);

    printf("\nOutput Data (Padding = 1):\n");
    for (int bs = 0; bs < batch_size; bs++)
    {
        printf("Batch %d:\n", bs);
        for (int k = 0; k < out_channels; k++)
        {
            printf("  Filter %d:\n", k);
            for (int i = 0; i < h_out_same; i++)
            {
                for (int j = 0; j < w_out_same; j++)
                {
                    printf("%f ", out_h_same[bs * (out_channels * h_out_same * w_out_same) + k * (h_out_same * w_out_same) + i * w_out_same + j]);
                }
                printf("\n");
            }
        }
    }
    free(out_h_same);

    free(h_input);
    free(h_filters);

    return 0;
}
