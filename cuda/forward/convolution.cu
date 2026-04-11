#include <assert.h>

#include "../common.h"

#define BLOCK_SIZE 32

/*
    float *in: input data with shape [batch_size, channels, height, width]
    float *out: resulting output data with shape [batch_size, channels', height', width']
    float filter: Kernel to slid over the in, has shape [num_kernels, num_channels, filter_size, filter_size]
    int height: represents height of in
    int width: represents width of in
*/

__global__ void conv2d_kernel(float *in, 
                              float *filter, 
                              float *out, 
                              int batch_size, int h_in, int w_in, int h_out, int w_out, int num_channels,
                              int num_filters, int filter_size,
                              int pad_h, int pad_w, int stride
                            ) 
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
                            val += in[
                                    bs * (num_channels * h_in * w_in) + 
                                    c * (h_in * w_in) + row_hat * w_in + 
                                    col_hat
                                ] 
                                     * 
                                filter[
                                    filter_k * (num_channels * filter_size * filter_size) + 
                                    c * (filter_size * filter_size) + (f_i * filter_size) + 
                                    f_j
                                ];
                    }
                }
            }
            out[bs * (num_filters * h_out * w_out) + filter_k * (h_out * w_out) + row * w_out + col] = val;
        }
    }
}

__global__ void conv2d_kernelv2(float *in, 
                              float *filter, 
                              float *out, 
                              int batch_size, int h_in, int w_in, int h_out, int w_out, int num_channels,
                              int num_filters, int filter_size,
                              int pad_h, int pad_w, int stride
                            ) 
{

    int filter_k = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float IN_DATA_CHUNK[BLOCK_SIZE][BLOCK_SIZE];

    {
        for (int bs = 0; bs < batch_size; bs++)
        {
            float val = 0.0f;
            for (int c = 0; c < num_channels; c++)
            {
                if (row < h_in && col < w_in)
                    IN_DATA_CHUNK[threadIdx.y][threadIdx.x] = in[
                                                            bs * (num_channels * h_in * w_in) + 
                                                            c * (h_in * w_in)                 + 
                                                            row * (w_in)                      + 
                                                            col
                                                     ];
                else
                    IN_DATA_CHUNK[threadIdx.y][threadIdx.x] = 0.0f;
                __syncthreads();

                if (filter_k < num_filters && row < h_out && col < w_out)
                {
                    for (int f_i = 0; f_i < filter_size; f_i++)
                    {
                        for (int f_j = 0; f_j < filter_size; f_j++)
                        {
                            if ((threadIdx.y + f_i - pad_h) >= 0 && (threadIdx.y + f_i - pad_h) < BLOCK_SIZE
                                                            &&
                                (threadIdx.x + f_j - pad_w) >= 0 && (threadIdx.x + f_j - pad_w) < BLOCK_SIZE)
                                {
                                    val += IN_DATA_CHUNK[threadIdx.y + f_i - pad_h][threadIdx.x + f_j - pad_w] *
                                            filter[
                                                filter_k * (num_channels * filter_size * filter_size) + 
                                                c * (filter_size * filter_size) + (f_i * filter_size) + 
                                                f_j
                                            ];
                                }
                            else if ((row + f_i - pad_h) >= 0 && (row + f_i - pad_h) < h_in 
                                                            &&
                                    (col + f_j - pad_w) >= 0 && (col + f_j - pad_w) < w_in)
                                {
                                    int row_hat = row + f_i - pad_h;
                                    int col_hat = col + f_j - pad_w;
                                    // bring that fker from cache (hopefully :D)
                                    val += in[
                                                bs * (num_channels * h_in * w_in) + 
                                                c * (h_in * w_in)                 + 
                                                row_hat * (w_in)                  + 
                                                col_hat
                                            ] *
                                            filter[
                                                filter_k * (num_channels * filter_size * filter_size) + 
                                                c * (filter_size * filter_size) + (f_i * filter_size) + 
                                                f_j
                                            ];
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
}

void conv2d_forward_pass(float *in_h, 
                        float *filter_h, 
                        float *out_h, 
                        int batch_size, int h_in, int w_in, int num_channels,
                        int num_filters, int filter_size, 
                        int padding, int stride
                    )
{
    float *in_d, *out_d, *filter_d;
    int h_out, w_out, pad_h, pad_w;
    if (!padding) // no padding
    {
        h_out = floor((h_in - filter_size)/(float) stride) + 1;
        w_out = floor((w_in - filter_size)/(float) stride) + 1;
        
        pad_h = 0;
        pad_w = 0;
        
        assert (h_out > 0 && w_out > 0);
    }
    else // padding i.e h_out == H_in & W_out == W_in    
    {
        assert (stride == 1);        
        h_out = h_in;
        w_out = w_in;
        
        pad_h = (filter_size - 1) / 2;
        pad_w = (filter_size - 1) / 2;
    }

    CUDA_CHECK( cudaMalloc((void**)&in_d, (batch_size * num_channels * h_in * w_in * sizeof(float))) );
    CUDA_CHECK( cudaMalloc((void**)&out_d, (batch_size * num_filters * h_out * w_out * sizeof(float))) );
    CUDA_CHECK( cudaMalloc((void**)&filter_d, (num_filters * num_channels * filter_size * filter_size * sizeof(float)) ) );

    CUDA_CHECK( cudaMemcpy(in_d, in_h, (batch_size * num_channels * h_in * w_in * sizeof(float)), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(filter_d, filter_h, (num_filters * num_channels * filter_size * filter_size * sizeof(float)), cudaMemcpyHostToDevice) );

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(cdiv(h_out, BLOCK_SIZE), cdiv(w_out, BLOCK_SIZE), num_filters);

    conv2d_kernelv2<<<dimGrid, dimBlock>>>(in_d, filter_d, out_d, batch_size, h_in, w_in, h_out, w_out, num_channels, num_filters, filter_size, pad_h, pad_w, stride);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(out_h, out_d, (batch_size * num_filters * h_out * w_out * sizeof(float)), cudaMemcpyDeviceToHost));
    
    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(filter_d);
}


int main() 
{
    int batch_size = 2;
    int h_in = 5;
    int w_in = 5;
    int num_channels = 3;
    int num_filters = 2; 
    int filter_size = 3;
    int stride = 1;

    float *in_h = (float*)malloc(batch_size * num_channels * h_in * w_in * sizeof(float));
    float *filter_h = (float*)malloc(num_filters * num_channels * filter_size * filter_size * sizeof(float));

    for (int i = 0; i < batch_size * num_channels * h_in * w_in; i++) {
        in_h[i] = 1.0f;
    }

    for (int i = 0; i < num_filters * num_channels * filter_size * filter_size; i++) {
        filter_h[i] = 1.0f;
    }

    printf("Input Data:\n");
    for (int bs = 0; bs < batch_size; bs++) {
        printf("Batch %d:\n", bs);
        for (int c = 0; c < num_channels; c++) {
            printf("  Channel %d:\n", c);
            for (int i = 0; i < h_in; i++) {
                for (int j = 0; j < w_in; j++) {
                    printf("%f ", in_h[bs * (num_channels * h_in * w_in) + c * (h_in * w_in) + i * w_in + j]);
                }
                printf("\n");
            }
        }
    }

    printf("\nFilter Data:\n");
    for (int k = 0; k < num_filters; k++) {
        printf("Filter %d:\n", k);
        for (int c = 0; c < num_channels; c++) {
            printf("  Channel %d:\n", c);
            for (int i = 0; i < filter_size; i++) {
                for (int j = 0; j < filter_size; j++) {
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
    float *out_h_valid = (float*)malloc(batch_size * num_filters * h_out_valid * w_out_valid * sizeof(float));

    conv2d_forward_pass(in_h, filter_h, out_h_valid, batch_size, h_in, w_in, num_channels, num_filters, filter_size, padding_valid, stride);
    
    printf("\nOutput Data (Padding = 0):\n");
    for (int bs = 0; bs < batch_size; bs++) {
        printf("Batch %d:\n", bs);
        for (int k = 0; k < num_filters; k++) {
            printf("  Filter %d:\n", k);
            for (int i = 0; i < h_out_valid; i++) {
                for (int j = 0; j < w_out_valid; j++) {
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
    float *out_h_same = (float*)malloc(batch_size * num_filters * h_out_same * w_out_same * sizeof(float));

    conv2d_forward_pass(in_h, filter_h, out_h_same, batch_size, h_in, w_in, num_channels, num_filters, filter_size, padding_same, stride);
    
    printf("\nOutput Data (Padding = 1):\n");
    for (int bs = 0; bs < batch_size; bs++) {
        printf("Batch %d:\n", bs);
        for (int k = 0; k < num_filters; k++) {
            printf("  Filter %d:\n", k);
            for (int i = 0; i < h_out_same; i++) {
                for (int j = 0; j < w_out_same; j++) {
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