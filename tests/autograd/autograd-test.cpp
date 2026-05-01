#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

void test_linear_ce_backward() {
    int batch_size = 2;
    int in_features = 4;
    int num_classes = 3;

    // 1. Setup Input X (Shape: [2, 4])
    int x_shape[] = {batch_size, in_features};
    Tensor* x = tensor_create(2, x_shape, 1, 1);
    float h_x[8] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    cudaMemcpy(x->data, h_x, 8 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Setup Labels (Shape: [2])
    int labels_shape[] = {batch_size};
    Tensor* labels = tensor_create(1, labels_shape, 0, 1); 
    float h_labels[2] = {1.0f, 2.0f}; 
    cudaMemcpy(labels->data, h_labels, 2 * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Initialize Layers
    Linear linear_layer(in_features, num_classes, 0, 1);
    CrossEntropy ce_layer(1);

    // Explicitly initialize weights to 1.0f for deterministic gradient verification
    float h_w[12];
    for (int i = 0; i < 12; i++) h_w[i] = 1.0f;
    cudaMemcpy(linear_layer.weights->data, h_w, 12 * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Forward Pass
    Tensor* logits = linear_layer.forward(x);
    Tensor* loss = ce_layer.forward(logits, labels);

    // 5. Backward Pass
    // float h_dout[1] = {1.0f};
    // cudaMemcpy(loss->grad->data, h_dout, 1 * sizeof(float), cudaMemcpyHostToDevice);

    backward(loss); // Invoking the global backward function

    // 6. Verify Gradients for Weights
    // Logits Softmax is [1/3, 1/3, 1/3] for both samples
    // dLogits (scaled by 1/2):
    //   Sample 0 (Target 1): [1/6, -2/6, 1/6] = [0.1666, -0.3333, 0.1666]
    //   Sample 1 (Target 2): [1/6, 1/6, -2/6] = [0.1666, 0.1666, -0.3333]
    // dW = X^T * dLogits
    // X^T rows are all [1, 2]
    // dW row = 1 * [1/6, -2/6, 1/6] + 2 * [1/6, 1/6, -2/6]
    //        = [3/6, 0/6, -3/6] = [0.5, 0.0, -0.5]
    
    float h_dw[12];
    cudaMemcpy(h_dw, linear_layer.weights->grad->data, 12 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < in_features; i++) {
        assert(fabs(h_dw[i * num_classes + 0] - 0.5f) < 1e-4);
        assert(fabs(h_dw[i * num_classes + 1] - 0.0f) < 1e-4);
        assert(fabs(h_dw[i * num_classes + 2] - (-0.5f)) < 1e-4);
    }

    std::cout << "SUCCESS: Linear + CrossEntropy backward pass verified." << std::endl;
}

int main() {
    test_linear_ce_backward();
    return 0;
}