#include <cstdlib>

#include "../../include/nn.h"
#include "../../include/tensor.h"
#include "../../include/functional.h"

GlobalPooling::GlobalPooling(int requires_grad) {
    this->requires_grad = requires_grad;
}

Tensor* GlobalPooling::forward(Tensor* input) {
    // (N, C, H, W) -> (N, C)
    int ndim = 2;
    int* out_shape = (int*)malloc(ndim * sizeof(int));
    out_shape[0] = input->shape[0];
    out_shape[1] = input->shape[1];

    /* 
        this seperation is done for two reaons:
            - It will easier to bind to python
            - We can call this same layer on multiple inputs and as such there will be a seperate node for each 
            input. If we had kept a node in main class then each time we would have called this layer, 
            the previous node would be overwritten i.e it's reference that would have been stored as class member
    */
    Tensor* output = global_pooling_functional_forward(input, ndim, out_shape, this->requires_grad);

    return output;
}