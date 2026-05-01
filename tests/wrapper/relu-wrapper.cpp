#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#include "../../include/tensor.h"
#include "../../include/autograd.h"
#include "../../include/nn.h"

void test_relu_autograd() {
    // 1. Setup Input: 1x1x2x2 with mixed positive/negative values
    int ndim = 4;
    int shape[] = {1, 1, 2, 2};

    Tensor* x = tensor_create(ndim, shape, 1, 1);

    // Mixed values: [-1.0, 2.0, 3.0, -4.0]
    float h_data[] = {-1.0f, 2.0f, 3.0f, -4.0f};
    cudaMemcpy(x->data, h_data, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Forward Pass
    ReLU relu_layer;
    Tensor* y = relu_layer.forward(x);

    // 3. Verify Forward Output: negatives clamped to 0, positives passed through
    float h_out[4];
    cudaMemcpy(h_out, y->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Forward Output: [" << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << ", " << h_out[3] << "]"
              << " (Expected: [0, 2, 3, 0])" << std::endl;

    assert(h_out[0] == 0.0f);
    assert(h_out[1] == 2.0f);
    assert(h_out[2] == 3.0f);
    assert(h_out[3] == 0.0f);

    // 4. Verify Graph Construction
    assert(y->prev != nullptr);
    assert(y->prev->inputs[0] == x);
    assert(y->prev->num_inputs == 1);
    std::cout << "Graph Node successfully wired." << std::endl;

    // 5. Backward Pass: dout = [10, 10, 10, 10]
    float h_dout[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    cudaMemcpy(y->grad->data, h_dout, 4 * sizeof(float), cudaMemcpyHostToDevice);

    y->prev->backward(y->prev, y->grad);

    // 6. Verify dx: passes dout where input > 0, zero elsewhere
    // x = [-1, 2, 3, -4] -> dx = [0, 10, 10, 0]
    float h_dx[4];
    cudaMemcpy(h_dx, x->grad->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Backward Output dX: [" << h_dx[0] << ", " << h_dx[1] << ", " << h_dx[2] << ", " << h_dx[3] << "]"
              << " (Expected: [0, 10, 10, 0])" << std::endl;

    assert(h_dx[0] == 0.0f);
    assert(h_dx[1] == 10.0f);
    assert(h_dx[2] == 10.0f);
    assert(h_dx[3] == 0.0f);

    std::cout << "SUCCESS: ReLU forward and backward passes verified." << std::endl;
}

int main() {
    test_relu_autograd();
    return 0;
}
