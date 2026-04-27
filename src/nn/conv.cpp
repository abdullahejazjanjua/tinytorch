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

    int ndim = 4;
    int wshape[ndim] = {out_channels, in_channels, kernel_size, kernel_size};
    this->weights = tensor_create(ndim, wshape, this->requires_grad, 1);
}

Tensor* Conv2D::forward(Tensor* input) {
    // (N, Cin, Hin, Win) -> (N, Cout, Hout, Wout)
    int ndim = 4;
    int* out_shape = (int*)malloc(ndim * sizeof(int));
    out_shape[0] = input->shape[0];
    out_shape[1] = input->shape[1];

    Tensor* output = conv2d_functional_forward(input, weights, padding, ndim, out_shape, this->requires_grad);

    return output;
}