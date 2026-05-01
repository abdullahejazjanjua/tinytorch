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
    if (node->num_inputs == 3) {
        Tensor *bias = node->inputs[2];
        if (bias && bias->requires_grad) {
            matmul_backward_pass_bias(dout, bias->grad);
        }
    }
}

Tensor* linear_functional_forward(Tensor *input, Tensor *weights, Tensor *bias, int requires_grad) {

    int batch_size = input->shape[0];
    int in_features = weights->shape[0];
    int out_features = weights->shape[1];
    

    int expected_shape[] = {batch_size, out_features};
    
    Tensor *output = tensor_create(2, expected_shape, requires_grad, 1);

    matmul_forward_pass(input, weights, bias, output);

    if (output->requires_grad) {
        Node *_prev = (Node *) malloc(sizeof(Node));
        if (_prev == nullptr) {
            std::cerr << "Error: Node allocation failed\n";
            return nullptr;
        }

        int num_inputs = (bias != nullptr) ? 3 : 2;
        _prev->inputs = (Tensor**) malloc(num_inputs * sizeof(Tensor*));
        _prev->inputs[0] = input;
        _prev->inputs[1] = weights;
        if (bias != nullptr) _prev->inputs[2] = bias;
        _prev->num_inputs = num_inputs;

        _prev->ctx = nullptr;
        _prev->num_ctx = 0;

        _prev->backward = linear_functional_backward;
        output->prev = _prev;
    }
    
    return output;
}
