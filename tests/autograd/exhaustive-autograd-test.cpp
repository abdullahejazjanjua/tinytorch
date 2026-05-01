#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

void test_exhaustive_network() {
    int batch_size = 1;

    // 1. Setup Input X (Shape: [1, 1, 3, 3])
    int x_shape[] = {batch_size, 1, 3, 3};
    Tensor* x = tensor_create(4, x_shape, 1, 1);
    float h_x[9] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    cudaMemcpy(x->data, h_x, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Setup Labels (Shape: [1])
    int labels_shape[] = {batch_size};
    Tensor* labels = tensor_create(1, labels_shape, 0, 1);
    float h_labels[1] = {0.0f}; 
    cudaMemcpy(labels->data, h_labels, 1 * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Initialize Layers
    Conv2D conv_layer(1, 2, 3, 0, 1); 
    GlobalPooling pool_layer(1);
    Linear linear_layer(2, 2, 0, 1);     
    CrossEntropy ce_layer(1);

    // Initialize Conv2D weights to 1.0f (Shape: [2, 1, 3, 3] -> 18 elements)
    float h_w_conv[18];
    for (int i = 0; i < 18; i++) h_w_conv[i] = 1.0f;
    cudaMemcpy(conv_layer.weights->data, h_w_conv, 18 * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize Linear weights (Shape: [2, 2] -> 4 elements)
    float h_w_lin[4] = {0.5f, -0.5f, -0.5f, 0.5f};
    cudaMemcpy(linear_layer.weights->data, h_w_lin, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Forward Pass
    Tensor* out_conv = conv_layer.forward(x);         
    Tensor* out_pool = pool_layer.forward(out_conv);  
    Tensor* logits = linear_layer.forward(out_pool);  
    Tensor* loss = ce_layer.forward(logits, labels);  

    // Verify Forward Pass
    float h_loss[1];
    cudaMemcpy(h_loss, loss->data, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    float expected_loss = -logf(0.5f); 
    assert(fabs(h_loss[0] - expected_loss) < 1e-4);

    // 5. Backward Pass
    backward(loss);

    // 6. Verify Gradients for Linear Weights
    // Linear Weight Shape is [in_features, out_features] = [2, 2]
    // dLogits = [-0.5, 0.5]
    // dW_lin = out_pool^T * dLogits = [9.0; 9.0] * [-0.5, 0.5] = [-4.5, 4.5; -4.5, 4.5]
    float h_dw_lin[4];
    cudaMemcpy(h_dw_lin, linear_layer.weights->grad->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    assert(fabs(h_dw_lin[0] - (-4.5f)) < 1e-4);
    assert(fabs(h_dw_lin[1] - (4.5f)) < 1e-4);
    assert(fabs(h_dw_lin[2] - (-4.5f)) < 1e-4);
    assert(fabs(h_dw_lin[3] - (4.5f)) < 1e-4);

    // 7. Verify Gradients for Conv2D Weights
    // d_out_pool = dLogits * W_lin^T = [-0.5, 0.5] * [0.5, -0.5; -0.5, 0.5] = [-0.5, 0.5]
    // d_out_conv = d_out_pool / (1 * 1) = [-0.5, 0.5]
    float h_dw_conv[18];
    cudaMemcpy(h_dw_conv, conv_layer.weights->grad->data, 18 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; i++) {
        assert(fabs(h_dw_conv[i] - (-0.5f)) < 1e-4);     
        assert(fabs(h_dw_conv[i + 9] - (0.5f)) < 1e-4);  
    }

    std::cout << "SUCCESS: Exhaustive Conv2D (3x3) + GP + Linear + CE backward pass verified." << std::endl;
}

int main() {
    test_exhaustive_network();
    return 0;
}