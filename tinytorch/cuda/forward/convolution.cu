#include <assert.h>

#include "../common.h"

#define BLOCK_SIZE 32
#define COARSE_FACTOR 2
#define MAX_FILTER_SIZE 16384

__constant__ float c_filter[MAX_FILTER_SIZE];

/*
    float *in:          Input data with shape [batch_size, num_channels, h_in, w_in]
    float *out:         Output data with shape [batch_size, num_filters, h_out, w_out]
    int batch_size:     Number of images in the input batch
    int h_in, w_in:     Height and width of the input images
    int h_out, w_out:   Height and width of the resulting output images
    int num_channels:   Number of input feature maps (e.g., 3 for RGB)
    int num_filters:    Number of output feature maps (kernels)
    int filter_size:    Spatial dimensions of the square kernel
    int pad_h, pad_w:   Vertical and horizontal zero-padding applied to the input,
                        interestingly this equivalent to filter_radius
*/

__global__ void conv2d_kernel(float *in,
                              float *out,
                              int batch_size, int h_in, int w_in, int h_out, int w_out, int num_channels,
                              int num_filters, int filter_size,
                              int pad_h, int pad_w)
{
    int filter_k = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (filter_k < num_filters && row < h_out && col < w_out)
    {
        for (int bs = 0; bs < batch_size; bs++)
        {
            float val = 0.0f;
            for (int c = 0; c < num_channels; c++)
            {
                for (int f_i = 0; f_i < filter_size; f_i++)
                {
                    for (int f_j = 0; f_j < filter_size; f_j++)
                    {
                        int row_hat = row + f_i - pad_h;
                        int col_hat = col + f_j - pad_w;
                        if (row_hat >= 0 && row_hat < h_in && col_hat >= 0 && col_hat < w_in)
                            val += in[bs * (num_channels * h_in * w_in) +
                                      c * (h_in * w_in) + row_hat * w_in +
                                      col_hat] *
                                   c_filter[filter_k * (num_channels * filter_size * filter_size) +
                                            c * (filter_size * filter_size) + (f_i * filter_size) +
                                            f_j];
                    }
                }
            }
            out[bs * (num_filters * h_out * w_out) + filter_k * (h_out * w_out) + row * w_out + col] = val;
        }
    }
}

__global__ void conv2d_kernelv2(float *in,
                                float *out,
                                int batch_size, int h_in, int w_in, int h_out, int w_out, int num_channels,
                                int num_filters, int filter_size,
                                int pad_h, int pad_w)
{

    int filter_k = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float IN_DATA_CHUNK[BLOCK_SIZE][BLOCK_SIZE];

    for (int bs = 0; bs < batch_size; bs++)
    {
        float val = 0.0f;
        for (int c = 0; c < num_channels; c++)
        {
            if (row < h_in && col < w_in)
                IN_DATA_CHUNK[threadIdx.y][threadIdx.x] = in[bs * (num_channels * h_in * w_in) +
                                                             c * (h_in * w_in) +
                                                             row * (w_in) +
                                                             col];
            else
                IN_DATA_CHUNK[threadIdx.y][threadIdx.x] = 0.0f;
            __syncthreads();

            if (filter_k < num_filters && row < h_out && col < w_out)
            {
                for (int f_i = 0; f_i < filter_size; f_i++)
                {
                    for (int f_j = 0; f_j < filter_size; f_j++)
                    {
                        if ((threadIdx.y + f_i - pad_h) >= 0 && (threadIdx.y + f_i - pad_h) < BLOCK_SIZE &&
                            (threadIdx.x + f_j - pad_w) >= 0 && (threadIdx.x + f_j - pad_w) < BLOCK_SIZE)
                        {
                            val += IN_DATA_CHUNK[threadIdx.y + f_i - pad_h][threadIdx.x + f_j - pad_w] *
                                   c_filter[filter_k * (num_channels * filter_size * filter_size) +
                                            c * (filter_size * filter_size) + (f_i * filter_size) +
                                            f_j];
                        }
                        else if ((row + f_i - pad_h) >= 0 && (row + f_i - pad_h) < h_in &&
                                 (col + f_j - pad_w) >= 0 && (col + f_j - pad_w) < w_in)
                        {
                            int row_hat = row + f_i - pad_h;
                            int col_hat = col + f_j - pad_w;
                            // bring that fker from cache (hopefully :D)
                            val += in[bs * (num_channels * h_in * w_in) +
                                      c * (h_in * w_in) +
                                      row_hat * (w_in) +
                                      col_hat] *
                                   c_filter[filter_k * (num_channels * filter_size * filter_size) +
                                            c * (filter_size * filter_size) + (f_i * filter_size) +
                                            f_j];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (filter_k < num_filters && row < h_out && col < w_out)
            out[bs * (num_filters * h_out * w_out) + filter_k * (h_out * w_out) + row * w_out + col] = val;
    }
}

// one might be curious as to why I implemented the above version, if I find out, I will tell you (but how?)
__global__ void conv2d_kernelv3(float *in,
                                float *out,
                                int batch_size, int h_in, int w_in, int h_out, int w_out, int num_channels,
                                int num_filters, int filter_size,
                                int pad_h, int pad_w)
{

    // here we say that batch_size is our rows and num_filters is our column
    // we always div and mod by innermost dimension, in this case num_filters.
    // intuitively, we can say first do batch 0 (as div changes slowly) and then move to the next.
    // interesting optimization, this is still slower than torch's implementation by 15-17x, but I think this much
    // optimization is enough
    int filter_k = blockIdx.z % num_filters;
    int bs = blockIdx.z / num_filters;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float IN_DATA_CHUNK[];

    int IN_DIM = blockDim.x + filter_size - 1;

    float val = 0.0f;
    for (int c = 0; c < num_channels; c++)
    {
        // flattened thread_id -> (elements_to_load: num_threads)
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x); i < (IN_DIM * IN_DIM); i += (blockDim.x * blockDim.y))
        {
            /*
                We flattened our input data and threads, and map the block onto input data to contagiously
                load num_threads elements, in this case there is control diveregence at the 2 edges.
                If we used 2D nested loop, then there would be control divergence at each boundary.
            */

            // compute the 2D index
            int load_y = i / IN_DIM;
            int load_x = i % IN_DIM;

            // offset into the block using load_y & load_x and shift by pad_h & pad_w to handle padding.
            int row_hat = (blockIdx.y * blockDim.y) + load_y - pad_h;
            int col_hat = (blockIdx.x * blockDim.x) + load_x - pad_w;

            if (row_hat >= 0 && row_hat < h_in && col_hat >= 0 && col_hat < w_in)
                IN_DATA_CHUNK[load_y * IN_DIM + load_x] = in[bs * (num_channels * h_in * w_in) +
                                                             c * (h_in * w_in) + row_hat * w_in +
                                                             col_hat];
            else
                IN_DATA_CHUNK[load_y * IN_DIM + load_x] = 0.0f;
        }
        __syncthreads();

        if (filter_k < num_filters && row < h_out && col < w_out)
        {
            for (int f_i = 0; f_i < filter_size; f_i++)
            {
                for (int f_j = 0; f_j < filter_size; f_j++)
                {
                    val += IN_DATA_CHUNK[(threadIdx.y + f_i) * IN_DIM + (threadIdx.x + f_j)] *
                           c_filter[filter_k * (num_channels * filter_size * filter_size) +
                                    c * (filter_size * filter_size) + (f_i * filter_size) +
                                    f_j];
                }
            }
        }
        __syncthreads();
    }
    if (filter_k < num_filters && row < h_out && col < w_out)
        out[bs * (num_filters * h_out * w_out) + filter_k * (h_out * w_out) + row * w_out + col] = val;
}

// COARSING
__global__ void conv2d_kernelv4(float *in,
                                float *out,
                                int batch_size, int h_in, int w_in, int h_out, int w_out, int num_channels,
                                int num_filters, int filter_size,
                                int pad_h, int pad_w)
{

    // here we say that batch_size is our rows and num_filters is our column
    // we always div and mod by innermost dimension, in this case num_filters.
    // intuitively, we can say first do batch 0 (as div changes slowly) and then move to the next.
    // interesting optimization, this is still slower than torch's implementation by 15-17x, but I think this much
    // optimization is enough
    int start_filter_idx = (blockIdx.z % (num_filters / COARSE_FACTOR)) * COARSE_FACTOR; // now grid_width is num_filters / COARSE_FACTOR
    int start_batch_idx = (blockIdx.z / (num_filters / COARSE_FACTOR)) * COARSE_FACTOR;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float IN_DATA_CHUNK[];
    int IN_DIM = blockDim.x + filter_size - 1;

    // Accumulators for the filter coarsening dimension
    float vals[COARSE_FACTOR];

    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        int bs = start_batch_idx + i;
        if (bs >= batch_size)
            continue;

        for (int v = 0; v < COARSE_FACTOR; v++)
            vals[v] = 0.0f;

        for (int c = 0; c < num_channels; c++)
        {
            // co-operative load of input tile
            for (int t = (threadIdx.y * blockDim.x + threadIdx.x); t < (IN_DIM * IN_DIM); t += (blockDim.x * blockDim.y))
            {

                int load_y = t / IN_DIM;
                int load_x = t % IN_DIM;

                int row_hat = (blockIdx.y * blockDim.y) + load_y - pad_h;
                int col_hat = (blockIdx.x * blockDim.x) + load_x - pad_w;

                if (row_hat >= 0 && row_hat < h_in && col_hat >= 0 && col_hat < w_in)
                    IN_DATA_CHUNK[load_y * IN_DIM + load_x] = in[bs * (num_channels * h_in * w_in) +
                                                                 c * (h_in * w_in) + row_hat * w_in +
                                                                 col_hat];
                else
                    IN_DATA_CHUNK[load_y * IN_DIM + load_x] = 0.0f;
            }
            __syncthreads();

            if (row < h_out && col < w_out)
            {
                for (int j = 0; j < COARSE_FACTOR; j++)
                {
                    int filter_k = start_filter_idx + j;
                    if (filter_k < num_filters)
                    {
                        for (int f_i = 0; f_i < filter_size; f_i++)
                        {
                            for (int f_j = 0; f_j < filter_size; f_j++)
                            {
                                vals[j] += IN_DATA_CHUNK[(threadIdx.y + f_i) * IN_DIM + (threadIdx.x + f_j)] *
                                           c_filter[filter_k * (num_channels * filter_size * filter_size) +
                                                    c * (filter_size * filter_size) + (f_i * filter_size) +
                                                    f_j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (row < h_out && col < w_out)
        {
            for (int j = 0; j < COARSE_FACTOR; j++)
            {
                int filter_k = start_filter_idx + j;
                if (filter_k < num_filters)
                {
                    out[bs * (num_filters * h_out * w_out) + filter_k * (h_out * w_out) + row * w_out + col] = vals[j];
                }
            }
        }
    }
}

void conv2d_forward_pass(float *in_h,
                         float *filter_h,
                         float *out_h,
                         int batch_size, int h_in, int w_in, int num_channels,
                         int num_filters, int filter_size,
                         int padding)
{
    float *in_d, *out_d, *filter_d;
    int h_out, w_out, pad_h, pad_w;
    if (!padding) // no padding
    {
        h_out = floor(h_in - filter_size) + 1;
        w_out = floor(w_in - filter_size) + 1;

        pad_h = 0;
        pad_w = 0;

        assert(h_out > 0 && w_out > 0);
    }
    else // padding i.e h_out == H_in & W_out == W_in
    {

        h_out = h_in;
        w_out = w_in;

        pad_h = (filter_size - 1) / 2;
        pad_w = (filter_size - 1) / 2;
    }

    CUDA_CHECK(cudaMalloc((void **)&in_d, (batch_size * num_channels * h_in * w_in * sizeof(float))));
    CUDA_CHECK(cudaMalloc((void **)&out_d, (batch_size * num_filters * h_out * w_out * sizeof(float))));

    CUDA_CHECK(cudaMemcpy(in_d, in_h, (batch_size * num_channels * h_in * w_in * sizeof(float)), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, filter_h, (num_filters * num_channels * filter_size * filter_size * sizeof(float))));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(cdiv(w_out, BLOCK_SIZE), cdiv(h_out, BLOCK_SIZE), cdiv(batch_size * num_filters, COARSE_FACTOR));

    int IN_DIM = BLOCK_SIZE + filter_size - 1;
    size_t dynamic_shared_bytes = IN_DIM * IN_DIM * sizeof(float);

    conv2d_kernelv4<<<dimGrid, dimBlock, dynamic_shared_bytes>>>(in_d, out_d, batch_size, h_in, w_in, h_out, w_out, num_channels, num_filters, filter_size, pad_h, pad_w);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out_h, out_d, (batch_size * num_filters * h_out * w_out * sizeof(float)), cudaMemcpyDeviceToHost));

    cudaFree(in_d);
    cudaFree(out_d);
}

int main()
{
    int batch_size = 2;
    int h_in = 5;
    int w_in = 5;
    int num_channels = 3;
    int num_filters = 2;
    int filter_size = 3;

    float *in_h = (float *)malloc(batch_size * num_channels * h_in * w_in * sizeof(float));
    float *filter_h = (float *)malloc(num_filters * num_channels * filter_size * filter_size * sizeof(float));

    for (int i = 0; i < batch_size * num_channels * h_in * w_in; i++)
    {
        in_h[i] = 1.0f;
    }

    for (int i = 0; i < num_filters * num_channels * filter_size * filter_size; i++)
    {
        filter_h[i] = 1.0f;
    }

    printf("Input Data:\n");
    for (int bs = 0; bs < batch_size; bs++)
    {
        printf("Batch %d:\n", bs);
        for (int c = 0; c < num_channels; c++)
        {
            printf("  Channel %d:\n", c);
            for (int i = 0; i < h_in; i++)
            {
                for (int j = 0; j < w_in; j++)
                {
                    printf("%f ", in_h[bs * (num_channels * h_in * w_in) + c * (h_in * w_in) + i * w_in + j]);
                }
                printf("\n");
            }
        }
    }

    printf("\nFilter Data:\n");
    for (int k = 0; k < num_filters; k++)
    {
        printf("Filter %d:\n", k);
        for (int c = 0; c < num_channels; c++)
        {
            printf("  Channel %d:\n", c);
            for (int i = 0; i < filter_size; i++)
            {
                for (int j = 0; j < filter_size; j++)
                {
                    printf("%f ", filter_h[k * (num_channels * filter_size * filter_size) + c * (filter_size * filter_size) + i * filter_size + j]);
                }
                printf("\n");
            }
        }
    }

    // --- Test Case 1: Padding = 0 (Valid) ---
    int padding_valid = 0;
    int h_out_valid = h_in - filter_size + 1;
    int w_out_valid = w_in - filter_size + 1;
    float *out_h_valid = (float *)malloc(batch_size * num_filters * h_out_valid * w_out_valid * sizeof(float));

    conv2d_forward_pass(in_h, filter_h, out_h_valid, batch_size, h_in, w_in, num_channels, num_filters, filter_size, padding_valid);

    printf("\nOutput Data (Padding = 0):\n");
    for (int bs = 0; bs < batch_size; bs++)
    {
        printf("Batch %d:\n", bs);
        for (int k = 0; k < num_filters; k++)
        {
            printf("  Filter %d:\n", k);
            for (int i = 0; i < h_out_valid; i++)
            {
                for (int j = 0; j < w_out_valid; j++)
                {
                    printf("%f ", out_h_valid[bs * (num_filters * h_out_valid * w_out_valid) + k * (h_out_valid * w_out_valid) + i * w_out_valid + j]);
                }
                printf("\n");
            }
        }
    }
    free(out_h_valid);

    // --- Test Case 2: Padding = 1 (Same) ---
    int padding_same = 1;
    int h_out_same = h_in;
    int w_out_same = w_in;
    float *out_h_same = (float *)malloc(batch_size * num_filters * h_out_same * w_out_same * sizeof(float));

    conv2d_forward_pass(in_h, filter_h, out_h_same, batch_size, h_in, w_in, num_channels, num_filters, filter_size, padding_same);

    printf("\nOutput Data (Padding = 1):\n");
    for (int bs = 0; bs < batch_size; bs++)
    {
        printf("Batch %d:\n", bs);
        for (int k = 0; k < num_filters; k++)
        {
            printf("  Filter %d:\n", k);
            for (int i = 0; i < h_out_same; i++)
            {
                for (int j = 0; j < w_out_same; j++)
                {
                    printf("%f ", out_h_same[bs * (num_filters * h_out_same * w_out_same) + k * (h_out_same * w_out_same) + i * w_out_same + j]);
                }
                printf("\n");
            }
        }
    }
    free(out_h_same);

    free(in_h);
    free(filter_h);

    return 0;
}
