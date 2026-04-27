#include <cstdlib>

#include "../../include/nn.h"
#include "../../include/tensor.h"
#include "../../include/functional.h"

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int padding, int requires_grad) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = kernel_size;
    this->padding = padding;
    this->requires_grad = requires_grad;

    int wshape[] = {out_channels, in_channels, kernel_size, kernel_size};
    this->weights = tensor_create(4, wshape, this->requires_grad, 1);
    
    normal_xavier_init(this->weights, in_channels * kernel_size * kernel_size, out_channels);
}

Tensor* Conv2D::forward(Tensor* input) {
    return conv2d_functional_forward(
        input, 
        this->weights, 
        this->padding, 
        (input->requires_grad || this->requires_grad) // logical-or of all inputs
    );
}