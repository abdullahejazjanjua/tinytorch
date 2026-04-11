#include <assert.h>

#include "../common.h"

#define FILTER_RADIUS(filter_size)(((filter_size - 1)/2))


/*
    float *in: input data with shape [batch_size, channels, height, width]
    float *out: resulting output data with shape [batch_size, channels', height', width']
    float filter: Kernel to slid over the in, has shape [filter_size, filter_size]
    int height: represents height of in
    int width: represents width of in
*/

__global__ void conv2d_kernel(float *in, 
                              float *filter, 
                              float *out, 
                              int h_in, int w_in, int h_out, int w_out,
                              int filter_size,
                              int pad_h, int pad_w, int stride
                            ) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < h_out && col < w_out)
    {
        float val = 0.0f;
        for (int f_i = 0; f_i < filter_size; f_i++)
        {
            for (int f_j = 0; f_j < filter_size; f_j++)
            {
                int row_hat = row + f_i - pad_h;
                int col_hat = col + f_j - pad_w;
                if (row_hat >= 0 && row_hat < h_in && col_hat >= 0 && col_hat < w_in)
                    val += in[row_hat * w_in + col_hat] * filter[f_i * filter_size + f_j];
            }
        }
        out[row * w_out + col] = val;
    }
}

void conv2d_forward_pass(float *in_h, 
                        float *filter_h, 
                        float *out_h, 
                        int h_in, int w_in, 
                        int filter_size, int padding, int stride
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


    CUDA_CHECK( cudaMalloc((void**)&in_d, h_in * w_in * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&out_d, h_out * w_out * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&filter_d, filter_size * filter_size * sizeof(float)) );

    CUDA_CHECK( cudaMemcpy(in_d, in_h, h_in * w_in * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(in_d, in_h, h_out * w_out * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(filter_d, filter_h, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice) );

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(cdiv(h_out, 32), cdiv(w_out, 32));

    conv2d_kernel<<<dimGrid, dimBlock>>>(in_d, filter_h, out_d, h_in, w_in, h_out, w_out, filter_size, pad_h, pad_w, stride);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(out_h, out_d, (h_out * w_out * sizeof(float)), cudaMemcpyDeviceToHost));
    
    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(filter_d);
}

int main() {
    int h_in = 5;
    int w_in = 5;
    int filter_size = 3;
    int padding = 0;
    int stride = 1;

    int h_out = h_in - filter_size + 1;
    int w_out = w_in - filter_size + 1;

    float *in_h = (float*)malloc(h_in * w_in * sizeof(float));
    float *filter_h = (float*)malloc(filter_size * filter_size * sizeof(float));
    float *out_h = (float*)malloc(h_out * w_out * sizeof(float));

    for (int i = 0; i < h_in * w_in; i++) {
        in_h[i] = 1.0f;
    }

    for (int i = 0; i < filter_size * filter_size; i++) {
        filter_h[i] = 1.0f;
    }

    conv2d_forward_pass(in_h, filter_h, out_h, h_in, w_in, filter_size, padding, stride);

    free(in_h);
    free(filter_h);
    free(out_h);

    return 0;
}