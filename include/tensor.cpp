#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "tensor.h"
#include "common.cuh"

void inline Malloc(float *f, int size, const char *msg) {
    f = (float*) malloc(size * sizeof(float));
    if (f == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: " << msg << "\n";
        return;
    }
}

Tensor* tensor_create(int ndim, int *shape, int requires_grad, int on_gpu)
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
    t->on_gpu = on_gpu;
    t->grad = nullptr;
    t->data = nullptr;
    t->prev = nullptr;
    
    if (on_gpu) {
        CUDA_CHECK( cudaMalloc((void**)&t->data, total_size * sizeof(float)) );
    } else {
        t->data = (float*) malloc(total_size * sizeof(float));
        if (t->data == nullptr) {
            free(t->shape);
            free(t);
            return nullptr;
        }
    }

    if (requires_grad) {
        t->grad = tensor_create(ndim, shape, 0, on_gpu);
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
    if (t == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Tensor is empty\n";
        return;
    }
    if (t->on_gpu) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Tensor is already on GPU\n";
        return;
    }
    if (t->data == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Data field in Tensor is empty\n";
        return;
    }
    
    float *d_data = nullptr, *d_grad = nullptr;
    // allocate space on GPU for data and move it there
    CUDA_CHECK(cudaMalloc((void**)&d_data, t->size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, t->data, t->size * sizeof(float), cudaMemcpyHostToDevice));

    // allocate space on GPU for grad and move it there
    if (t->grad && t->grad->data) {
        CUDA_CHECK(cudaMalloc((void**)&d_grad, t->grad->size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_grad, t->grad->data, t->grad->size * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Now, that enough space has been allocate, switch pointers and free the ones on CPU
    free(t->data);
    t->data = d_data;
    t->on_gpu = 1; 

    if (t->grad && t->grad->data) {
        free(t->grad->data);
        t->grad->data = d_grad;
        t->grad->on_gpu = 1;
    }
}

void tensor_to_cpu(Tensor *t) {
    if (t == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Tensor is empty\n";
        return;
    }
    if (!t->on_gpu) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Tensor is already on CPU\n";
        return;
    }
    if (t->data == nullptr) {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: Data field in Tensor is empty\n";
        return;
    }

    float *data = nullptr, *grad_data = nullptr;
    // Allocate space on CPU and move data there
    Malloc(data, t->size, "Couldn't allocate data field on CPU");
    CUDA_CHECK( cudaMemcpy(data, t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost) );

    if (t->grad && t->grad->data) {
        // Allocate space on CPU and move grad_data there
        Malloc(grad_data, t->grad->size, "Couldn't allocate grad->data field on CPU");
        CUDA_CHECK( cudaMemcpy(grad_data, t->grad->data, t->grad->size * sizeof(float), cudaMemcpyDeviceToHost) );
    }

    // Switch pointers and free the ones on GPU
    cudaFree(t->data);
    t->data = data;
    t->on_gpu = 0;

    if (t->grad && t->grad->data) {
        cudaFree(t->grad->data);
        t->grad->data = grad_data;
        t->grad->on_gpu = 0;
    }
}