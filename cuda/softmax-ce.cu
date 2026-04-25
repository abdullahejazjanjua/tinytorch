#include "../include/common.cuh"
#include "../include/tensor.h"

/*
    float *input: [batch_size, num_classes] (logits)
    float *out: [batch_size, num_classes] (probability distribution fter applying softmax)
        - For this we need, to perform exp(val_i) reduction
        - For each val_i, exp(val_i) / reduction output
        - Plan1:
            - Kernel1 -> do reduciton
            - Kernel2 -> elementwise divide
            * This will be slow, another issue is that num_classes is often quite small, as such the overhead
            of the Kernel1 will be quite large.
            * Infact, the kernel2 (i.e element-wise divide) will also be slow as it would be memory-bound highly.
        - Plan2 (fused operation):
            - Perform reduction on the exp of input so that you are left with sum_exp[batch_size]
            - Index into the logit[y[i]], where y[i] is the true class, and then use the below formula:
                -  ln(sum_exp[i]) - logits[y[bs, i]]
            * How do I launch this plan2?
                - We could use a simple reduciton kernel and then add syncthread

*/
__global__ void softmax_ce_forward(float *logits, int batch_size, int num_classes, float *softmax_denominator, float *loss) {
    unsigned int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int class_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    if (batch_idx < batch_size) {
        if (class_idx < num_classes) val += expf(logits[batch_idx * batch_size + class_idx]);
        if (class_idx + 32 < num_classes) val += expf(logits[batch_idx * batch_size + (class_idx + 32)]);

        for (int offset = 16; offset >= 1; offset =/ 2) {
            val += expf(__shfl_down_sync(0xffffffff, val, offset));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&softmax_denominator, val)
    }
    
}


__global__ void softmax_ce_backward(float *logits, float *softmax_denominator, float* max_logit, float *grad_logits) {
    unsigned int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int tid = batch_idx * 
}