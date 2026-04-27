#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

void test_cross_entropy_autograd() {
    int batch_size = 2;
    int num_classes = 3;

    int logits_shape[] = {batch_size, num_classes};
    Tensor* logits = tensor_create(2, logits_shape, 1, 1);
    
    float h_logits[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpy(logits->data, h_logits, 6 * sizeof(float), cudaMemcpyHostToDevice);

    int labels_shape[] = {batch_size};
    Tensor* labels = tensor_create(1, labels_shape, 0, 1); 
    
    float h_labels[2] = {2.0f, 0.0f}; 
    cudaMemcpy(labels->data, h_labels, 2 * sizeof(float), cudaMemcpyHostToDevice);

    CrossEntropy ce_layer(1);
    Tensor* loss = ce_layer.forward(logits, labels);

    assert(loss->shape[0] == 1);

    float h_loss[1];
    cudaMemcpy(h_loss, loss->data, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float expected_loss = -logf(1.0f / 3.0f);
    std::cout << "Forward Pass: Expected Loss: " << expected_loss << ", Got: " << h_loss[0] << " [Verified]" << std::endl;
    assert(fabs(h_loss[0] - expected_loss) < 1e-4);
    
    if (loss->prev && loss->prev->backward) {
        loss->prev->backward(loss->prev, nullptr);
    }

    float h_logits_grad[6];
    cudaMemcpy(h_logits_grad, logits->grad->data, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    // With mean reduction: (Softmax - OneHot) / BatchSize
    // Softmax is [1/3, 1/3, 1/3], BatchSize is 2
    // Target index grad: (1/3 - 1) / 2 = -0.3333f
    float expected_grad_target = -0.333333f; 
    
    std::cout << "Backward Pass (Sample 0, Class 2): Expected Grad: " << expected_grad_target << ", Got: " << h_logits_grad[2] << " [Verified]" << std::endl;
    std::cout << "Backward Pass (Sample 1, Class 0): Expected Grad: " << expected_grad_target << ", Got: " << h_logits_grad[3] << " [Verified]" << std::endl;

    assert(fabs(h_logits_grad[2] - expected_grad_target) < 1e-3);
    assert(fabs(h_logits_grad[3] - expected_grad_target) < 1e-3);

    std::cout << "SUCCESS: CrossEntropy Autograd Verification Complete." << std::endl;
}

int main() {
    test_cross_entropy_autograd();
    return 0;
}