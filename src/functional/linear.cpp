#include <iostream>
#include <cstdlib>
#include "../../include/tensor.h"
#include "../../include/autograd.h"

void linear_functional_backward(Node *node, Tensor *dout) {
    Tensor *input = node->inputs[0];
    Tensor *weights = node->inputs[1];

    if (input->requires_grad) {
        matmul_backward_pass_A(input, weights, dout, input->grad);
    }
    if (weights->requires_grad) {
        matmul_backward_pass_B(input, weights, dout, weights->grad);
    }
}

Tensor* linear_functional_forward(Tensor *input, Tensor *weights, int requires_grad) {

    int batch_size = input->shape[0];
    int in_features = weights->shape[0];
    int out_features = weights->shape[1];
    

    int expected_shape[] = {batch_size, out_features};
    
    Tensor *output = tensor_create(2, expected_shape, requires_grad, 1);

    matmul_forward_pass(input, weights, nullptr, output);

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

        _prev->ctx = nullptr;
        _prev->num_ctx = 0;

        _prev->backward = linear_functional_backward;
        output->prev = _prev;
    }
    
    return output;
}