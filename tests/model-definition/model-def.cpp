#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "../../include/tensor.h"

void save_tensor(const char* filename, Tensor* t) {
    if (!t || !t->data) return;
    float* host_data = (float*)malloc(t->size * sizeof(float));
    cudaMemcpy(host_data, t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
    std::string full_path = std::string("/content/") + filename;
    std::ofstream outfile(full_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<char*>(host_data), t->size * sizeof(float));
        outfile.close();
    }
    free(host_data);
}

void fill_random(Tensor *t) {
    float* h = (float*)malloc(t->size * sizeof(float));
    for (int i = 0; i < t->size; i++) h[i] = (float)rand() / (float)RAND_MAX;
    cudaMemcpy(t->data, h, t->size * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

int main() {
    srand(42); 
    int in_s[] = {1, 3, 32, 32}, filt_s[] = {8, 3, 3, 3}, out_s[] = {1, 8, 30, 30}, pool_s[] = {1, 8}, label_s[] = {1}, loss_s[] = {1};

    Tensor *input = tensor_create(4, in_s, 1);
    Tensor *weights = tensor_create(4, filt_s, 1);
    Tensor *conv_out = tensor_create(4, out_s, 1);
    Tensor *pooled = tensor_create(2, pool_s, 1);
    Tensor *labels = tensor_create(1, label_s, 0);
    Tensor *loss = tensor_create(1, loss_s, 0);

    tensor_to_gpu(input);
    tensor_to_gpu(weights);
    tensor_to_gpu(conv_out);
    tensor_to_gpu(pooled);
    tensor_to_gpu(labels);
    tensor_to_gpu(loss);

    fill_random(input);
    float label_val = 3.0f;
    float* h_labels = (float*)malloc(sizeof(float));
    h_labels[0] = label_val;
    cudaMemcpy(labels->data, h_labels, sizeof(float), cudaMemcpyHostToDevice);
    free(h_labels);

    normal_xavier_init(weights, 3*3*3, 8*3*3);

    // Forward Pass
    conv2d_forward_pass(input, weights, 0, conv_out);
    global_pooling_forward_pass(conv_out, pooled);
    softmax_ce_forward(pooled, labels, loss);
    cudaDeviceSynchronize();

    // Backward Pass
    softmax_ce_backward(pooled, labels, pooled->grad);            // Grads w.r.t Logits
    global_pooling_backward_pass(pooled->grad, conv_out->grad);   // Grads w.r.t Conv Output
    conv2d_backward_pass_weight(input, conv_out->grad, 0, weights->grad);
    conv2d_backward_pass_input(weights, conv_out->grad, 0, input->grad);
    cudaDeviceSynchronize();

    // Exhaustive Export
    save_tensor("input.bin", input);
    save_tensor("weights.bin", weights);
    save_tensor("labels.bin", labels);
    save_tensor("fwd_logits.bin", pooled);
    
    // Intermediate Gradients
    save_tensor("grad_logits.bin", pooled->grad);       // Output of Softmax-CE backprop
    save_tensor("grad_conv_out.bin", conv_out->grad);   // Output of Pooling backprop
    save_tensor("grad_weights.bin", weights->grad);     // Output of Conv weight backprop
    save_tensor("grad_input.bin", input->grad);         // Output of Conv input backprop

    tensor_free(input); tensor_free(weights); tensor_free(conv_out);
    tensor_free(pooled); tensor_free(labels); tensor_free(loss);
    return 0;
}