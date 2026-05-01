#include <cstdlib>

#include "../../include/nn.h"
#include "../../include/tensor.h"
#include "../../include/functional.h"

Linear::Linear(int in_features, int out_features, int has_bias, int requires_grad) {
    this->in_features = in_features;
    this->out_features = out_features;
    this->has_bias = has_bias;
    this->requires_grad = requires_grad;
    this->bias = nullptr;

    int wshape[] = {in_features, out_features};
    this->weights = tensor_create(2, wshape, this->requires_grad, 1);
    
    normal_xavier_init(this->weights, in_features, out_features);
}

Tensor* Linear::forward(Tensor* input) {
    return linear_functional_forward(
        input, this->weights, (this->requires_grad || input->requires_grad)
    );
}