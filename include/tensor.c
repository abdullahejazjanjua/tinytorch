#include "stdio.h"
#include "stdlib.h"
#include "tensor.h"

Tensor* tensor_create(int ndim, int *shape)
{
    Tensor *t = (Tensor*) malloc(sizeof(Tensor));
    if (t == NULL) {
        fprintf(stderr, "[%s:%d] Error: Failed to allocate struct Tensor of shape: ", __FILE__, __LINE__);
        for (int i = 0; i < ndim; i++) {
            fprintf(stderr, "%d ", shape[i]);
        }
        fprintf(stderr, "\n");
        return NULL;
    }

    t->ndim = ndim;
    t->shape = (int*) malloc(ndim * sizeof(int));
    if (t->shape == NULL) {
        fprintf(stderr, "[%s:%d] Error: Failed to allocate shapes array\n", __FILE__, __LINE__);
        free(t);
        return NULL;
    }

    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        total_size *= shape[i];
    }

    t->size = total_size;
    t->data = (float*)malloc(total_size * sizeof(float));
    if (t->data == NULL) {
        fprintf(stderr, "[%s:%d] Error: Failed to allocate data array in Tensor (size: %d)\n", __FILE__, __LINE__, total_size);
        free(t->shape);
        free(t);
        return NULL;
    }

    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        free(t->data);
        free(t->shape);
        free(t);
    }
}