#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../../include/tensor.h"
#include "../../include/optim.h"

void test_sgd_step_and_zero_grad() {
    // 1. Setup a parameter tensor on GPU with requires_grad=1
    int shape[] = {4};
    Tensor* param = tensor_create(1, shape, 1, 1);

    // Set initial param values: [1.0, 2.0, 3.0, 4.0]
    float h_param[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cudaMemcpy(param->data, h_param, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Set gradient values: [0.5, 1.0, 1.5, 2.0]
    float h_grad[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    cudaMemcpy(param->grad->data, h_grad, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Construct SGD with lr = 0.1
    float lr = 0.1f;
    SGD optimizer(std::vector<Tensor*>{param}, lr);

    // 3. step()
    optimizer.step();

    // Expected: param[i] -= lr * grad[i]
    // [1.0 - 0.05, 2.0 - 0.1, 3.0 - 0.15, 4.0 - 0.2] = [0.95, 1.9, 2.85, 3.8]
    float h_param_after[4];
    cudaMemcpy(h_param_after, param->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "After step(): [" << h_param_after[0] << ", " << h_param_after[1]
              << ", " << h_param_after[2] << ", " << h_param_after[3] << "]"
              << " (Expected: [0.95, 1.9, 2.85, 3.8])" << std::endl;

    assert(fabs(h_param_after[0] - 0.95f) < 1e-5);
    assert(fabs(h_param_after[1] - 1.9f)  < 1e-5);
    assert(fabs(h_param_after[2] - 2.85f) < 1e-5);
    assert(fabs(h_param_after[3] - 3.8f)  < 1e-5);

    // 4. zero_grad()
    optimizer.zero_grad();

    float h_grad_after[4];
    cudaMemcpy(h_grad_after, param->grad->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "After zero_grad(): [" << h_grad_after[0] << ", " << h_grad_after[1]
              << ", " << h_grad_after[2] << ", " << h_grad_after[3] << "]"
              << " (Expected: all 0)" << std::endl;

    for (int i = 0; i < 4; i++) {
        assert(h_grad_after[i] == 0.0f);
    }

    std::cout << "SUCCESS: SGD step() and zero_grad() verified." << std::endl;

    tensor_free(param);
}

void test_sgd_multi_param() {
    // Same idea but with two params of different shapes to make sure the optimizer
    // iterates over the std::vector correctly.
    int s1[] = {2};
    int s2[] = {3};
    Tensor* p1 = tensor_create(1, s1, 1, 1);
    Tensor* p2 = tensor_create(1, s2, 1, 1);

    float h_p1[2] = {10.0f, 20.0f};
    float h_p2[3] = {100.0f, 200.0f, 300.0f};
    cudaMemcpy(p1->data, h_p1, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p2->data, h_p2, 3 * sizeof(float), cudaMemcpyHostToDevice);

    float h_g1[2] = {1.0f, 2.0f};
    float h_g2[3] = {10.0f, 20.0f, 30.0f};
    cudaMemcpy(p1->grad->data, h_g1, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p2->grad->data, h_g2, 3 * sizeof(float), cudaMemcpyHostToDevice);

    SGD optimizer(std::vector<Tensor*>{p1, p2}, 1.0f);
    optimizer.step();

    float h_p1_after[2], h_p2_after[3];
    cudaMemcpy(h_p1_after, p1->data, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p2_after, p2->data, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: p1 = [9, 18], p2 = [90, 180, 270]
    assert(fabs(h_p1_after[0] - 9.0f)   < 1e-5);
    assert(fabs(h_p1_after[1] - 18.0f)  < 1e-5);
    assert(fabs(h_p2_after[0] - 90.0f)  < 1e-4);
    assert(fabs(h_p2_after[1] - 180.0f) < 1e-4);
    assert(fabs(h_p2_after[2] - 270.0f) < 1e-4);

    std::cout << "SUCCESS: SGD multi-parameter step verified." << std::endl;

    tensor_free(p1);
    tensor_free(p2);
}

int main() {
    test_sgd_step_and_zero_grad();
    test_sgd_multi_param();
    return 0;
}
