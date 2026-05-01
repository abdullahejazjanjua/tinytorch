#include <iostream>
#include <cstdlib>
#include "../../include/tensor.h"
#include "../../include/autograd.h"

void relu_functional_backward(Node *node, Tensor *dout) {
    Tensor *input = node->inputs[0];
    if (input->requires_grad) {
        relu_backward_pass(input, dout, input->grad);
    }
}

Tensor* relu_functional_forward(Tensor *input, int requires_grad) {
    Tensor *output = tensor_create(input->ndim, input->shape, requires_grad, 1);

    relu_forward_pass(input, output);

    if (output->requires_grad) {
        Node *_prev = (Node *) malloc(sizeof(Node));
        if (_prev == nullptr) {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for Node\n";
            return nullptr;
        }

        _prev->inputs = (Tensor**) malloc(sizeof(Tensor*));
        if (_prev->inputs == nullptr) {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: couldn't allocate space for pointer to inputs\n";
            return nullptr;
        }
        _prev->inputs[0] = input;
        _prev->num_inputs = 1;

        _prev->ctx = nullptr;
        _prev->num_ctx = 0;

        _prev->backward = relu_functional_backward;
        output->prev = _prev;
    }

    return output;
}
