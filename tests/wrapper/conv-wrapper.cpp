#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

void test_conv2d_autograd_same() {
    int in_shape[] = {1, 1, 5, 5};
    Tensor* x = tensor_create(4, in_shape, 1, 1); 

    float h_input[25]; 
    for(int i=0; i<25; i++) h_input[i] = 1.0f;
    cudaMemcpy(x->data, h_input, 25 * sizeof(float), cudaMemcpyHostToDevice);

    int padding = 1; 
    Conv2D conv_layer(1, 1, 3, padding, 1);

    float h_weight[9];
    for(int i=0; i<9; i++) h_weight[i] = 1.0f;
    cudaMemcpy(conv_layer.weights->data, h_weight, 9 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor* y = conv_layer.forward(x);
    
    assert(y->shape[2] == 5 && y->shape[3] == 5);

    float h_dout[25];
    for(int i=0; i<25; i++) h_dout[i] = 1.0f;
    cudaMemcpy(y->grad->data, h_dout, 25 * sizeof(float), cudaMemcpyHostToDevice);

    if (y->prev && y->prev->backward) {
        y->prev->backward(y->prev, y->grad);
    } else {
        std::cerr << "Error: No autograd node found!" << std::endl;
        return;
    }

    float h_w_grad[9];
    cudaMemcpy(h_w_grad, conv_layer.weights->grad->data, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    assert(h_w_grad[0] == 16.0f); // Top-left weight hits 4x4 valid input region
    assert(h_w_grad[1] == 20.0f); // Top-middle weight hits 4x5 valid input region
    assert(h_w_grad[2] == 16.0f); // Top-right weight hits 4x4 valid input region
    assert(h_w_grad[3] == 20.0f); // Middle-left weight hits 5x4 valid input region
    assert(h_w_grad[4] == 25.0f); // Center weight hits 5x5 valid input region
    assert(h_w_grad[5] == 20.0f); // Middle-right weight hits 5x4 valid input region
    assert(h_w_grad[6] == 16.0f); // Bottom-left weight hits 4x4 valid input region
    assert(h_w_grad[7] == 20.0f); // Bottom-middle weight hits 4x5 valid input region
    assert(h_w_grad[8] == 16.0f); // Bottom-right weight hits 4x4 valid input region

    std::cout << "SUCCESS: SAME padding gradients match exact spatial expectations." << std::endl;
}

int main() {
    test_conv2d_autograd_same();
    return 0;
}