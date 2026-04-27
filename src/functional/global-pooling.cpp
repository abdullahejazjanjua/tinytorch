#include <iostream>
#include <cstdlib>
#include "../../include/tensor.h"
#include "../../include/autograd.h"

void global_pooling_functional_backward(Node *node, Tensor *dout) {
    Tensor *input = node->inputs[0];
    global_pooling_backward_pass(dout, input->grad);
}

Tensor* global_pooling_functional_forward(Tensor *input, int ndim, int *expected_shape, int requires_grad) {
    Tensor *output = tensor_create(ndim, expected_shape, requires_grad, 1); // create on gpu

    // call the forward function
    global_pooling_forward_pass(input, output);

    if (output->requires_grad) {
        Node *_prev = (Node *) malloc(sizeof(Node));
        if (_prev == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for Node\n";
            return nullptr;
        }

        _prev->inputs = (Tensor**) malloc(sizeof(Tensor*));
        if (_prev->inputs == nullptr) {
            std :: cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for pointer to inputs\n";
            return nullptr;
        }
        _prev->inputs[0] = input;
        _prev->num_inputs = 1;

        _prev->ctx = nullptr; // no ctx needed for backprop
        _prev->ctx = 0;

        _prev->backward = global_pooling_functional_backward;

        output->prev = _prev;
    }

    return output;
}