#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

void test_linear_autograd() {
    int batch_size = 2;
    int in_features = 3;
    int out_features = 4;

    // 1. Setup Input X (Shape: [batch_size, in_features] = [2, 3])
    int x_shape[] = {batch_size, in_features};
    Tensor* x = tensor_create(2, x_shape, 1, 1);
    float h_x[6];
    for(int i = 0; i < 6; i++) h_x[i] = 1.0f;
    cudaMemcpy(x->data, h_x, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Setup Linear Layer
    Linear linear_layer(in_features, out_features, 1);

    // Set Weights W (Shape: [in_features, out_features] = [3, 4])
    float h_w[12];
    for(int i = 0; i < 12; i++) h_w[i] = 1.0f;
    cudaMemcpy(linear_layer.weights->data, h_w, 12 * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Forward Pass: Y = X * W
    Tensor* y = linear_layer.forward(x);

    std::cout << "Forward Output Y Shape: [" << y->shape[0] << ", " << y->shape[1] << "]" << std::endl;

    // Verify Forward Output (Expected: 1.0 * 1.0 * 3 = 3.0f for all elements)
    float h_y[8];
    cudaMemcpy(h_y, y->data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    assert(y->shape[0] == 2 && y->shape[1] == 4);
    assert(h_y[0] == 3.0f); 

    // 4. Backward Pass Setup: dY = 1.0f
    float h_dy[8];
    for(int i = 0; i < 8; i++) h_dy[i] = 1.0f;
    cudaMemcpy(y->grad->data, h_dy, 8 * sizeof(float), cudaMemcpyHostToDevice);

    // Trigger backward pass through autograd node
    if (y->prev && y->prev->backward) {
        y->prev->backward(y->prev, y->grad);
    }

    std::cout << "Backward Output dW Shape: [" << linear_layer.weights->grad->shape[0] << ", " << linear_layer.weights->grad->shape[1] << "]" << std::endl;
    std::cout << "Backward Output dX Shape: [" << x->grad->shape[0] << ", " << x->grad->shape[1] << "]" << std::endl;

    // dW = X^T * dY -> [3, 2] * [2, 4] -> Each element sums 2 ones = 2.0f
    float h_dw[12];
    cudaMemcpy(h_dw, linear_layer.weights->grad->data, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "dW[0] Expected: 2.0f, Got: " << h_dw[0] << std::endl;
    assert(h_dw[0] == 2);

    // dX = dY * W^T -> [2, 4] * [4, 3] -> Each element sums 4 ones = 4.0f
    float h_dx[6];
    cudaMemcpy(h_dx, x->grad->data, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "dX[0] Expected: 4.0f, Got: " << h_dx[0] << std::endl;
    assert(h_dx[0] == 4);

    std::cout << "SUCCESS: Linear layer forward and backward passes verified." << std::endl;
}

int main() {
    test_linear_autograd();
    return 0;
}