#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"
#include "../../include/optim.h"

static void assert_near(float a, float b, float tol, const char *msg) {
    if (fabsf(a - b) > tol) {
        std::cerr << msg << ": expected ~" << b << ", got " << a << std::endl;
        assert(false);
    }
}

/**
 * Same network as full-pipeline-test.cpp, then after backward(loss):
 *   SGD.step() on { conv weights, linear weights, bias } and verify
 *       param_new = param_old - lr * grad
 *   SGD.zero_grad() and verify grad buffers are zero.
 */
void test_full_pipeline_backward_then_sgd() {
    constexpr float lr = 0.01f;
    constexpr float tol = 5e-4f;

    int batch_size = 1;

    int x_shape[] = {batch_size, 1, 3, 3};
    Tensor* x = tensor_create(4, x_shape, 1, 1);
    float h_x[9] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    cudaMemcpy(x->data, h_x, 9 * sizeof(float), cudaMemcpyHostToDevice);

    int labels_shape[] = {batch_size};
    Tensor* labels = tensor_create(1, labels_shape, 0, 1);
    float h_labels[1] = {0.0f};
    cudaMemcpy(labels->data, h_labels, 1 * sizeof(float), cudaMemcpyHostToDevice);

    Conv2D conv_layer(1, 2, 3, 0, 1);
    ReLU relu_layer(1);
    GlobalPooling pool_layer(1);
    Linear linear_layer(2, 2, 1, 1);
    CrossEntropy ce_layer(1);

    float h_w_conv[18];
    for (int i = 0; i < 18; i++) h_w_conv[i] = 1.0f;
    cudaMemcpy(conv_layer.weights->data, h_w_conv, 18 * sizeof(float), cudaMemcpyHostToDevice);

    float h_w_lin[4] = {0.5f, -0.5f, -0.5f, 0.5f};
    cudaMemcpy(linear_layer.weights->data, h_w_lin, 4 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor* out_conv = conv_layer.forward(x);
    Tensor* out_relu = relu_layer.forward(out_conv);
    Tensor* out_pool = pool_layer.forward(out_relu);
    Tensor* logits = linear_layer.forward(out_pool);
    Tensor* loss = ce_layer.forward(logits, labels);

    float h_loss[1];
    cudaMemcpy(h_loss, loss->data, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    assert_near(h_loss[0], -logf(0.5f), 1e-4f, "loss");

    backward(loss);

    /* Snapshot weights + grads before SGD */
    int nc = conv_layer.weights->size;
    int nw = linear_layer.weights->size;
    int nb = linear_layer.bias->size;

    std::vector<float> conv_w_old(nc), conv_g(nc);
    std::vector<float> lin_w_old(nw), lin_g(nw);
    std::vector<float> bias_old(nb), bias_g(nb);

    cudaMemcpy(conv_w_old.data(), conv_layer.weights->data, nc * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(conv_g.data(), conv_layer.weights->grad->data, nc * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(lin_w_old.data(), linear_layer.weights->data, nw * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(lin_g.data(), linear_layer.weights->grad->data, nw * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_old.data(), linear_layer.bias->data, nb * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_g.data(), linear_layer.bias->grad->data, nb * sizeof(float), cudaMemcpyDeviceToHost);

    SGD optimizer(std::vector<Tensor*>{
        conv_layer.weights,
        linear_layer.weights,
        linear_layer.bias
    }, lr);
    optimizer.step();

    auto check_updates = [&](Tensor *t, const std::vector<float> &old_w, const std::vector<float> &g,
                             const char *name) {
        std::vector<float> new_w(t->size);
        cudaMemcpy(new_w.data(), t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < t->size; i++) {
            float expected = old_w[i] - lr * g[i];
            assert_near(new_w[i], expected, tol, name);
        }
    };

    check_updates(conv_layer.weights, conv_w_old, conv_g, "conv_w");
    check_updates(linear_layer.weights, lin_w_old, lin_g, "lin_w");
    check_updates(linear_layer.bias, bias_old, bias_g, "bias");

    optimizer.zero_grad();

    std::vector<float> gz(std::max({nc, nw, nb}), 0.f);
    cudaMemcpy(gz.data(), conv_layer.weights->grad->data, nc * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nc; i++) assert_near(gz[i], 0.f, 1e-6f, "conv_grad after zero_grad");
    cudaMemcpy(gz.data(), linear_layer.weights->grad->data, nw * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nw; i++) assert_near(gz[i], 0.f, 1e-6f, "lin_grad after zero_grad");
    cudaMemcpy(gz.data(), linear_layer.bias->grad->data, nb * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nb; i++) assert_near(gz[i], 0.f, 1e-6f, "bias_grad after zero_grad");

    std::cout << "SUCCESS: Full pipeline backward + one SGD step + zero_grad verified." << std::endl;
}

int main() {
    test_full_pipeline_backward_then_sgd();
    return 0;
}
