#include <float.h>

#include "../include/common.cuh"
#include "../include/tensor.h"

#define BLOCK_SIZE 32

__global__ void softmax_ce_forward_kernel(float *logits, float *y, int batch_size, int num_classes, float *loss) {
    unsigned int batch_idx = blockIdx.x;
    unsigned int class_idx = threadIdx.x;
    
    float exp_sum = 0.0f;
    float max_val = -FLT_MAX;
    if (batch_idx < batch_size) {
        // find max_val
        if (threadIdx.x < num_classes) max_val = fmaxf(max_val, logits[batch_idx * num_classes + class_idx]);
        if (threadIdx.x < num_classes && class_idx + 32 < num_classes) max_val = fmaxf(max_val, logits[batch_idx * num_classes + (class_idx + 32)]);
        for (int offset = 16; offset >= 1; offset /= 2) { 
            float max_offset = __shfl_down_sync(0xffffffff, max_val, offset);
            max_val = fmaxf(max_val, max_offset);
        }
        // update each thread's max to 0th thread's max
        max_val = __shfl_sync(0xffffffff, max_val, 0);

        // find sum of exp - max_val
        if (threadIdx.x < num_classes) exp_sum = expf(logits[batch_idx * num_classes + class_idx] - max_val);
        if (threadIdx.x < num_classes && class_idx + 32 < num_classes) exp_sum += expf(logits[batch_idx * num_classes + (class_idx + 32)] - max_val);
        for (int offset = 16; offset >= 1; offset /= 2) { 
            float exp_offset = __shfl_down_sync(0xffffffff, exp_sum, offset);
            exp_sum +=  exp_offset;
        }
    }

    if (threadIdx.x == 0) {
        int target_class = y[batch_idx];
        float target_logit = logits[batch_idx * num_classes + target_class];
        float row_loss = logf(exp_sum) + max_val - target_logit;
        atomicAdd(loss, row_loss / batch_size);
    }
}

__global__ void softmax_ce_backward_kernel(float *logits, float *y, int batch_size, int num_classes, float *grad_logits) {
   unsigned int batch_idx = blockIdx.x;
   unsigned int class_idx = threadIdx.x;
    
    float exp_sum = 0.0f;
    float max_val = -FLT_MAX;
    if (batch_idx < batch_size) {
        // this is same as forward pass code but for we will do this again to avoid having extra field 
        // in Tensor struct that caches stuff that will needed later
        // find max_val
        if (threadIdx.x < num_classes) max_val = fmaxf(max_val, logits[batch_idx * num_classes + class_idx]);
        if (threadIdx.x < num_classes && class_idx + 32 < num_classes) max_val = fmaxf(max_val, logits[batch_idx * num_classes + (class_idx + 32)]);
        for (int offset = 16; offset >= 1; offset /= 2) { 
            float max_offset = __shfl_down_sync(0xffffffff, max_val, offset);
            max_val = fmaxf(max_val, max_offset);
        }
        // broadcast max_val to all threads
        max_val = __shfl_sync(0xffffffff, max_val, 0);

        // find sum of exp - max_val
        if (threadIdx.x < num_classes) exp_sum = expf(logits[batch_idx * num_classes + class_idx] - max_val);
        if (threadIdx.x < num_classes && class_idx + 32 < num_classes) exp_sum += expf(logits[batch_idx * num_classes + (class_idx + 32)] - max_val);
        for (int offset = 16; offset >= 1; offset /= 2) { 
            float exp_offset = __shfl_down_sync(0xffffffff, exp_sum, offset);
            exp_sum +=  exp_offset;
        }
    }
    exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);
    // threads 0-31
    if (threadIdx.x < num_classes) {
        // get register max_val and exp_sum from thread 0 into current threads registers
        float prob = (expf(logits[batch_idx * num_classes + class_idx] - max_val)) / exp_sum;
        if (threadIdx.x == y[batch_idx]) grad_logits[batch_idx * num_classes + class_idx] = (prob - 1.0f) / batch_size;
        else grad_logits[batch_idx * num_classes + class_idx] = prob / batch_size;
    }
    // threads 32-63
    if (threadIdx.x + 32 < num_classes) {
        // get register max_val and exp_sum from thread 0 into current threads registers
        float prob = expf(logits[batch_idx * num_classes + (class_idx + 32)] - max_val) / exp_sum;
        if (threadIdx.x + 32 == y[batch_idx]) grad_logits[batch_idx * num_classes + (class_idx + 32)] = (prob - 1.0f) / batch_size;
        else grad_logits[batch_idx * num_classes + (class_idx + 32)] = prob / batch_size;
    }
}

void softmax_ce_forward(Tensor *logits, Tensor *labels, Tensor *loss) {
    int batch_size = logits->shape[0];
    int num_classes = logits->shape[1];

    softmax_ce_forward_kernel<<<batch_size, BLOCK_SIZE>>>(
        logits->data,
        labels->data, 
        batch_size,
        num_classes,
        loss->data
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void softmax_ce_backward(Tensor *logits, Tensor *labels, Tensor *grad_logits) {
    int batch_size = logits->shape[0];
    int num_classes = logits->shape[1];

    softmax_ce_backward_kernel<<<batch_size, BLOCK_SIZE>>>(
        logits->data,
        labels->data, 
        batch_size,
        num_classes,
        grad_logits->data
    );
    
    CUDA_CHECK(cudaGetLastError());
}