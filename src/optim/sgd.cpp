#include <cuda_runtime.h>

#include "../../include/optim.h"
#include "../../include/tensor.h"

SGD::SGD(std::vector<Tensor*> params, float lr) : params(params), lr(lr) {}

void SGD::step() {
    for (Tensor *p : params) {
        if (p && p->grad && p->requires_grad) {
            sgd_step_pass(p, lr);
        }
    }
}

void SGD::zero_grad() {
    for (Tensor *p : params) {
        if (p && p->grad) {
            cudaMemset(p->grad->data, 0, p->grad->size * sizeof(float));
        }
    }
}
