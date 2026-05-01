#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

void test_linear_bias_autograd() {
    int batch_size = 2;
    int in_features = 3;
    int out_features = 4;

    // 1. Setup Input X (Shape: [batch_size, in_features] = [2, 3]) all 1.0f
    int x_shape[] = {batch_size, in_features};
    Tensor* x = tensor_create(2, x_shape, 1, 1);
    float h_x[6];
    for (int i = 0; i < 6; i++) h_x[i] = 1.0f;
    cudaMemcpy(x->data, h_x, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Setup Linear layer with bias enabled
    Linear linear_layer(in_features, out_features, /*has_bias=*/1, 1);
    assert(linear_layer.bias != nullptr);
    assert(linear_layer.bias->shape[0] == out_features);

    // Set Weights W (Shape: [3, 4]) to all 1.0f
    float h_w[12];
    for (int i = 0; i < 12; i++) h_w[i] = 1.0f;
    cudaMemcpy(linear_layer.weights->data, h_w, 12 * sizeof(float), cudaMemcpyHostToDevice);

    // Set Bias b (Shape: [4]) to [1, 2, 3, 4]
    float h_b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cudaMemcpy(linear_layer.bias->data, h_b, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Forward Pass: Y = X @ W + b
    Tensor* y = linear_layer.forward(x);

    // Expected Y: each row is [1*3 + 1, 1*3 + 2, 1*3 + 3, 1*3 + 4] = [4, 5, 6, 7]
    float h_y[8];
    cudaMemcpy(h_y, y->data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Forward Output Y row 0: [" << h_y[0] << ", " << h_y[1] << ", " << h_y[2] << ", " << h_y[3] << "]"
              << " (Expected: [4, 5, 6, 7])" << std::endl;
    for (int row = 0; row < batch_size; row++) {
        for (int col = 0; col < out_features; col++) {
            float expected = 3.0f + h_b[col];
            assert(fabs(h_y[row * out_features + col] - expected) < 1e-5);
        }
    }

    // 4. Verify graph construction (3 inputs: input, weights, bias)
    assert(y->prev != nullptr);
    assert(y->prev->num_inputs == 3);
    assert(y->prev->inputs[0] == x);
    assert(y->prev->inputs[1] == linear_layer.weights);
    assert(y->prev->inputs[2] == linear_layer.bias);
    std::cout << "Graph Node has bias wired as 3rd input." << std::endl;

    // 5. Backward Pass: dY = all 1.0f
    float h_dy[8];
    for (int i = 0; i < 8; i++) h_dy[i] = 1.0f;
    cudaMemcpy(y->grad->data, h_dy, 8 * sizeof(float), cudaMemcpyHostToDevice);

    y->prev->backward(y->prev, y->grad);

    // 6a. dW = X^T @ dY -> [3, 2] @ [2, 4] -> each entry sums two 1.0s = 2.0
    float h_dw[12];
    cudaMemcpy(h_dw, linear_layer.weights->grad->data, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "dW[0]: " << h_dw[0] << " (Expected: 2.0)" << std::endl;
    for (int i = 0; i < 12; i++) assert(fabs(h_dw[i] - 2.0f) < 1e-5);

    // 6b. dX = dY @ W^T -> [2, 4] @ [4, 3] -> each entry sums four 1.0s = 4.0
    float h_dx[6];
    cudaMemcpy(h_dx, x->grad->data, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "dX[0]: " << h_dx[0] << " (Expected: 4.0)" << std::endl;
    for (int i = 0; i < 6; i++) assert(fabs(h_dx[i] - 4.0f) < 1e-5);

    // 6c. dbias = sum(dY, axis=0) -> sum across batch = [2, 2, 2, 2]
    float h_db[4];
    cudaMemcpy(h_db, linear_layer.bias->grad->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "dbias: [" << h_db[0] << ", " << h_db[1] << ", " << h_db[2] << ", " << h_db[3] << "]"
              << " (Expected: [2, 2, 2, 2])" << std::endl;
    for (int i = 0; i < 4; i++) assert(fabs(h_db[i] - 2.0f) < 1e-5);

    std::cout << "SUCCESS: Linear-with-bias forward and backward passes verified." << std::endl;
}

int main() {
    test_linear_bias_autograd();
    return 0;
}
