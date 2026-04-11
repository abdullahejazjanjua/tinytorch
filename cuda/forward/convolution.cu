#include <assert.h>

#include "../common.h"

#define FILTER_RADIUS(filter_size)(((filter_size - 1)/2))


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
                              int h_in, int w_in, int h_out, int w_out, int num_channels,
                              int num_filters, int filter_size,
                              int pad_h, int pad_w, int stride
                            ) 
{
    int filter_k = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (filter_k < num_filters && row < h_out && col < w_out)
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
                        val += in[c * (h_in * w_in) + row_hat * w_in + col_hat] * 
                            filter[filter_k * (num_channels * filter_size * filter_size) + 
                                c * (filter_size * filter_size) + (f_i * filter_size) + f_j];
                }
            }
        }
        out[filter_k * (h_out * w_out) + row * w_out + col] = val;
    }
}

void conv2d_forward_pass(float *in_h, 
                        float *filter_h, 
                        float *out_h, 
                        int h_in, int w_in, int num_channels,
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

    CUDA_CHECK( cudaMalloc((void**)&in_d, (num_channels * h_in * w_in * sizeof(float))) );
    CUDA_CHECK( cudaMalloc((void**)&out_d, (num_filters * h_out * w_out * sizeof(float))) );
    CUDA_CHECK( cudaMalloc((void**)&filter_d, (num_filters * num_channels * filter_size * filter_size * sizeof(float)) ) );

    CUDA_CHECK( cudaMemcpy(in_d, in_h, (num_channels * h_in * w_in * sizeof(float)), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(filter_d, filter_h, (num_filters * num_channels * filter_size * filter_size * sizeof(float)), cudaMemcpyHostToDevice) );

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(cdiv(h_out, 32), cdiv(w_out, 32), num_filters);

    conv2d_kernel<<<dimGrid, dimBlock>>>(in_d, filter_d, out_d, h_in, w_in, h_out, w_out, num_channels, num_filters, filter_size, pad_h, pad_w, stride);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(out_h, out_d, (num_filters * h_out * w_out * sizeof(float)), cudaMemcpyDeviceToHost));
    
    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(filter_d);
}
int main() {
    int h_in = 5;
    int w_in = 5;
    int num_channels = 3;
    int num_filters = 8;
    int filter_size = 3;
    int stride = 1;

    float *in_h = (float*)malloc(num_channels * h_in * w_in * sizeof(float));
    float *filter_h = (float*)malloc(num_filters * num_channels * filter_size * filter_size * sizeof(float));

    for (int i = 0; i < num_channels * h_in * w_in; i++) {
        in_h[i] = 1.0f;
    }

    for (int i = 0; i < num_filters * num_channels * filter_size * filter_size; i++) {
        filter_h[i] = 1.0f;
    }

    printf("Input Data:\n");
    for (int c = 0; c < num_channels; c++) {
        printf("Channel %d:\n", c);
        for (int i = 0; i < h_in; i++) {
            for (int j = 0; j < w_in; j++) {
                printf("%f ", in_h[c * (h_in * w_in) + i * w_in + j]);
            }
            printf("\n");
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
    float *out_h_valid = (float*)malloc(num_filters * h_out_valid * w_out_valid * sizeof(float));

    conv2d_forward_pass(in_h, filter_h, out_h_valid, h_in, w_in, num_channels, num_filters, filter_size, padding_valid, stride);
    
    printf("\nOutput Data (Padding = 0):\n");
    for (int k = 0; k < num_filters; k++) {
        printf("Filter %d:\n", k);
        for (int i = 0; i < h_out_valid; i++) {
            for (int j = 0; j < w_out_valid; j++) {
                printf("%f ", out_h_valid[k * (h_out_valid * w_out_valid) + i * w_out_valid + j]);
            }
            printf("\n");
        }
    }
    free(out_h_valid);

    // --- Test Case 2: Padding = 1 (Same) ---
    int padding_same = 1;
    int h_out_same = h_in;
    int w_out_same = w_in;
    float *out_h_same = (float*)malloc(num_filters * h_out_same * w_out_same * sizeof(float));

    conv2d_forward_pass(in_h, filter_h, out_h_same, h_in, w_in, num_channels, num_filters, filter_size, padding_same, stride);
    
    printf("\nOutput Data (Padding = 1):\n");
    for (int k = 0; k < num_filters; k++) {
        printf("Filter %d:\n", k);
        for (int i = 0; i < h_out_same; i++) {
            for (int j = 0; j < w_out_same; j++) {
                printf("%f ", out_h_same[k * (h_out_same * w_out_same) + i * w_out_same + j]);
            }
            printf("\n");
        }
    }
    free(out_h_same);

    free(in_h);
    free(filter_h);

    return 0;
}