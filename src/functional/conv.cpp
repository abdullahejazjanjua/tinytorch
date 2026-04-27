#include <iostream>
#include <cstdlib>
#include "../../include/tensor.h"
#include "../../include/autograd.h"

void conv2d_functional_backward(Node *node, Tensor *dout) {
    Tensor *input = node->inputs[0];
    Tensor *weights = node->inputs[1];
    int padding = (int)(intptr_t)node->ctx[0];

    if (input->requires_grad) {
        conv2d_backward_pass_input(weights, dout, padding, input->grad);
    }
    if (weights->requires_grad) {
        conv2d_backward_pass_weight(input, dout, padding, weights->grad);
    }
}


Tensor* conv2d_functional_forward(Tensor *input, Tensor *weights, int padding, int requires_grad) {

    int batch_size = input->shape[0];
    int out_channels = weights->shape[0];
    int kernel_size = weights->shape[2];
    int in_h = input->shape[2];
    int in_w = input->shape[3];

    int out_h, out_w;
    if (padding) {
        out_h = in_h;
        out_w = in_w;
    } else {
        out_h = in_h - kernel_size + 1;
        out_w = in_w - kernel_size + 1;
    }

    int expected_shape[] = {batch_size, out_channels, out_h, out_w};
    
    Tensor *output = tensor_create(4, expected_shape, requires_grad, 1);

    conv2d_forward_pass(input, weights, padding, output);

    if (output->requires_grad) {
        Node *_prev = (Node *) malloc(sizeof(Node));
        if (_prev == nullptr) {
            std::cerr << "Error: Node allocation failed\n";
            return nullptr;
        }

        _prev->inputs = (Tensor**) malloc(2 * sizeof(Tensor*));
        _prev->inputs[0] = input;
        _prev->inputs[1] = weights;
        _prev->num_inputs = 2;

        _prev->ctx = (void**) malloc(sizeof(void*));
        _prev->ctx[0] = (void*) (intptr_t) padding;
        _prev->num_ctx = 1;

        _prev->backward = conv2d_functional_backward;
        output->prev = _prev;
    }
    
    return output;
}