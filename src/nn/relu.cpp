#include <cstdlib>

#include "../../include/nn.h"
#include "../../include/tensor.h"
#include "../../include/functional.h"

ReLU::ReLU(int requires_grad) {
    this->requires_grad = requires_grad;
}

Tensor* ReLU::forward(Tensor* input) {
    return relu_functional_forward(
        input, (this->requires_grad || input->requires_grad)
    );
}
