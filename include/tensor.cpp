#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "tensor.h"
#include "common.cuh"

Tensor* tensor_create(int ndim, int *shape, int requires_grad)
{
    Tensor *t = (Tensor*) malloc(sizeof(Tensor));
    if (t == nullptr) return nullptr;

    t->ndim = ndim;
    t->shape = (int*) malloc(ndim * sizeof(int));
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        total_size *= shape[i];
    }

    t->size = total_size;
    t->requires_grad = requires_grad;
    t->on_gpu = 0;
    t->grad = nullptr;

    t->data = (float*) malloc(total_size * sizeof(float));
    if (t->data == nullptr) {
        free(t->shape);
        free(t);
        return nullptr;
    }

    if (requires_grad) {
        t->grad = tensor_create(ndim, shape, 0);
    }
        
    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        if (t->grad) {
            tensor_free(t->grad);
        }
        if (t->data) {
            if (t->on_gpu)
                cudaFree(t->data);
            else 
                free(t->data);
        }
        if (t->shape) {
            free(t->shape);
        }
        free(t);
    }
}

void tensor_to_gpu(Tensor *t) {
    if (!t || !t->data || t->on_gpu) return;

    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, t->size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, t->data, t->size * sizeof(float), cudaMemcpyHostToDevice));

    if (t->grad && t->grad->data) {
        float *d_grad = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_grad, t->grad->size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_grad, t->grad->data, t->grad->size * sizeof(float), cudaMemcpyHostToDevice));
        
        free(t->grad->data);
        t->grad->data = d_grad;
        t->grad->on_gpu = 1; 
    }

    free(t->data);
    t->data = d_data;
    t->on_gpu = 1; 
}