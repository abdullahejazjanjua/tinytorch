#include <iostream>
#include <cstdlib>
#include "tensor.h"
#include "common.cuh"

Tensor* tensor_create(int ndim, int *shape, int requires_grad)
{
    Tensor *t = (Tensor*) malloc(sizeof(Tensor));
    if (t == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Failed to allocate struct Tensor of shape: ";
        for (int i = 0; i < ndim; i++) {
            std::cerr << shape[i] << " ";
        }
        std::cerr << std::endl;
        return nullptr;
    }

    t->ndim = ndim;
    t->shape = (int*) malloc(ndim * sizeof(int));
    if (t->shape == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Failed to allocate shapes array" << std::endl;
        free(t);
        return nullptr;
    }

    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        total_size *= shape[i];
    }

    t->size = total_size;
    t->requires_grad = requires_grad;
    t->grad = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&t->data, total_size * sizeof(float)));
    if (requires_grad)
        t->grad = tensor_create(ndim, shape, 0);
        
    
    if (t->data == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Failed to allocate data array on GPU" << std::endl;
        free(t->shape);
        free(t);
        return nullptr;
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        if (t->data) {
            cudaFree(t->data);
        }
        if (t->grad) {
            tensor_free(t->grad);
        }
        if (t->shape) {
            free(t->shape);
        }
        free(t);
    }
}