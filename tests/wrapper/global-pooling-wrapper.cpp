#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <assert.h>

#include "../../include/tensor.h"
#include "../../include/autograd.h"
#include "../../include/nn.h"

void test_global_pooling_autograd() {
    // 1. Setup Input: 1x1x2x2 (N, C, H, W)
    int ndim = 4;
    int shape[] = {1, 1, 2, 2};
    
    // Using your tensor_create: (ndim, shape, requires_grad, on_gpu)
    Tensor* x = tensor_create(ndim, shape, 1, 1); 

    // Fill data: [1.0, 2.0, 3.0, 4.0]
    float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    cudaMemcpy(x->data, h_data, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Forward Pass
    GlobalPooling pool_layer;
    Tensor* y = pool_layer.forward(x);

    // Check value: avg(1,2,3,4) = 2.5
    float h_out;
    cudaMemcpy(&h_out, y->data, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Forward Output: " << h_out << " (Expected: 2.5)" << std::endl;

    // 3. Verify Graph Construction
    assert(y->prev != nullptr);
    assert(y->prev->inputs[0] == x);
    std::cout << "Graph Node successfully wired." << std::endl;

    // 4. Backward Pass Simulation
    // In a real training loop, dout comes from the loss function.
    // Here we manually set the output gradient (y->grad) to 1.0.
    float h_dout = 1.0f;
    cudaMemcpy(y->grad->data, &h_dout, y->grad->size * sizeof(float), cudaMemcpyHostToDevice);

    // Call the functional backward linked in the Node
    y->prev->backward(y->prev, y->grad);

    // 5. Verification
    // Gradient for avg pooling is dout / (H * W)
    // 1.0 / 4 = 0.25
    float h_x_grad[4];
    cudaMemcpy(h_x_grad, x->grad->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for(int i = 0; i < 4; i++) {
        if (h_x_grad[i] != 0.25f) passed = false;
    }

    if (passed) {
        std::cout << "SUCCESS: Gradients correctly propagated!" << std::endl;
    } else {
        std::cout << "FAILURE: Gradients are incorrect." << std::endl;
    }
}

int main() {
    test_global_pooling_autograd();
    return 0;
}