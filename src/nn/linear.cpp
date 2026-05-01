#include <cstdlib>
#include <cuda_runtime.h>

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

    if (this->has_bias) {
        int bshape[] = {out_features};
        this->bias = tensor_create(1, bshape, this->requires_grad, 1);
        cudaMemset(this->bias->data, 0, this->bias->size * sizeof(float));
    }
}

Tensor* Linear::forward(Tensor* input) {
    return linear_functional_forward(
        input, this->weights, this->bias, (this->requires_grad || input->requires_grad)
    );
}
