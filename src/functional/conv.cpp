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


Tensor* conv2d_functional_forward(Tensor *input, Tensor *weights, int padding, int ndim, int *expected_shape, int requires_grad) {
    Tensor *output = tensor_create(ndim, expected_shape, requires_grad, 1); // create on gpu

    // call the forward function
    conv2d_forward_pass(input, weights, padding, output);

    if (output->requires_grad) {
        Node *_prev = (Node *) malloc(sizeof(Node));
        if (_prev == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for Node\n";
            return nullptr;
        }

        _prev->inputs = (Tensor**) malloc(2 * sizeof(Tensor*));
        if (_prev->inputs == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for pointer to inputs\n";
            return nullptr;
        }
        _prev->inputs[0] = input;
        _prev->inputs[1] = weights;
        _prev->num_inputs = 2;

        _prev->ctx = (void**) malloc(sizeof(void*));
        if (_prev->ctx == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for pointer to ctx\n";
            return nullptr;
        }
        _prev->ctx[0] = (void*) (intptr_t) padding;
        _prev->num_ctx = 1;

        _prev->backward = conv2d_functional_backward;
        output->prev = _prev;
    }
    return output;
}