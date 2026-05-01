#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#include "../../include/tensor.h"
#include "../../include/nn.h"
#include "../../include/autograd.h"

/**
 * One end-to-end graph touching every trainable path in common use:
 *   Conv2D -> ReLU -> GlobalAvgPool -> Linear (with bias) -> CrossEntropy (fused softmax+CE)
 *
 * Then backward(loss) walks the full autograd graph once.
 *
 * Weights / data match the exhaustive-autograd fixture, with ReLU inserted (identity on
 * positive conv activations) and Linear uses bias=0 so forward logits and grads for
 * conv/linear weights match that test; we additionally assert bias->grad.
 */
void test_full_pipeline_forward_backward() {
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
    Linear linear_layer(2, 2, /*has_bias=*/1, 1);
    CrossEntropy ce_layer(1);

    assert(linear_layer.bias != nullptr);

    float h_w_conv[18];
    for (int i = 0; i < 18; i++) h_w_conv[i] = 1.0f;
    cudaMemcpy(conv_layer.weights->data, h_w_conv, 18 * sizeof(float), cudaMemcpyHostToDevice);

    float h_w_lin[4] = {0.5f, -0.5f, -0.5f, 0.5f};
    cudaMemcpy(linear_layer.weights->data, h_w_lin, 4 * sizeof(float), cudaMemcpyHostToDevice);

    /* Bias stays zero-init from ctor; logits match no-bias case. */

    Tensor* out_conv = conv_layer.forward(x);
    Tensor* out_relu = relu_layer.forward(out_conv);
    Tensor* out_pool = pool_layer.forward(out_relu);
    Tensor* logits = linear_layer.forward(out_pool);
    Tensor* loss = ce_layer.forward(logits, labels);

    float h_loss[1];
    cudaMemcpy(h_loss, loss->data, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    float expected_loss = -logf(0.5f);
    assert(fabs(h_loss[0] - expected_loss) < 1e-4f);

    backward(loss);

    float h_dw_lin[4];
    cudaMemcpy(h_dw_lin, linear_layer.weights->grad->data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    assert(fabs(h_dw_lin[0] - (-4.5f)) < 1e-4f);
    assert(fabs(h_dw_lin[1] - (4.5f)) < 1e-4f);
    assert(fabs(h_dw_lin[2] - (-4.5f)) < 1e-4f);
    assert(fabs(h_dw_lin[3] - (4.5f)) < 1e-4f);

    /* bias grad = sum_batch d logits; batch M=1, dlogits = [-0.5, 0.5] */
    float h_db[2];
    cudaMemcpy(h_db, linear_layer.bias->grad->data, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    assert(fabs(h_db[0] - (-0.5f)) < 1e-4f);
    assert(fabs(h_db[1] - (0.5f)) < 1e-4f);

    float h_dw_conv[18];
    cudaMemcpy(h_dw_conv, conv_layer.weights->grad->data, 18 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; i++) {
        assert(fabs(h_dw_conv[i] - (-0.5f)) < 1e-4f);
        assert(fabs(h_dw_conv[i + 9] - (0.5f)) < 1e-4f);
    }

    std::cout << "SUCCESS: Full pipeline Conv2D -> ReLU -> GlobalPool -> Linear(bias) -> CE "
              << "(forward + one backward(loss)) verified." << std::endl;
}

int main() {
    test_full_pipeline_forward_backward();
    return 0;
}
