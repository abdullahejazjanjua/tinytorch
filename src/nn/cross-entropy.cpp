#include <cstdlib>

#include "../../include/nn.h"
#include "../../include/tensor.h"
#include "../../include/functional.h"

CrossEntropy::CrossEntropy(int requires_grad) {
    this->requires_grad = requires_grad;
}

Tensor* CrossEntropy::forward(Tensor* logits, Tensor *labels) {
    // (N, num_classes) -> (N)
    int ndim = 1;
    int* out_shape = (int*)malloc(ndim * sizeof(int));
    out_shape[0] = 1;
    Tensor *loss = cross_entropy_functional_forward(logits, labels, ndim, out_shape, this->requires_grad);

    return loss;
}