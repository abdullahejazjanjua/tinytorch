#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>

#include "../../include/common.cuh"
#include "../../mnist-dataloader/mnist.h"
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

int argmax(float *f, int size) {
    float max = -FLT_MAX;
    int max_idx = 0;

    for (int i = 0; i < size; i++) {
        if (f[i] > max) {
            max = f[i];
            max_idx = i;
        }
    }

    return max_idx;
}

int main() {
    srand(42); 
    int in_s[] = {4, 1, 28, 28}; 
    int filt_s[] = {8, 1, 3, 3}; 
    int out_s[] = {4, 8, 26, 26}; 
    int pool_s[] = {4, 8}; 
    int fc_w_s[] = {8, 10}; 
    int logits_s[] = {4, 10}; 
    int label_s[] = {4}; 
    int loss_s[] = {1};

    MNISTData* train_dataset = load_dataset_in_ram("../../data/train-images.idx3-ubyte", "../../data/train-labels.idx1-ubyte", 60000);
    int* train_indices = create_indices(60000);

    Tensor *input = tensor_create(4, in_s, 1, 1);
    Tensor *weights = tensor_create(4, filt_s, 1, 1);
    Tensor *conv_out = tensor_create(4, out_s, 1, 1);
    Tensor *pooled = tensor_create(2, pool_s, 1, 1);
    Tensor *fc_weights = tensor_create(2, fc_w_s, 1, 1);
    Tensor *logits = tensor_create(2, logits_s, 1, 1);
    Tensor *labels = tensor_create(1, label_s, 0, 1);
    Tensor *loss = tensor_create(1, loss_s, 0, 1);

    load_batch_to_tensor(train_dataset, 4, 8, train_indices, input, labels);

    tensor_to_gpu(input);
    tensor_to_gpu(weights);
    tensor_to_gpu(conv_out);
    tensor_to_gpu(pooled);
    tensor_to_gpu(fc_weights);
    tensor_to_gpu(logits);
    tensor_to_gpu(labels);
    tensor_to_gpu(loss);

    normal_xavier_init(weights, 9, 72);
    normal_xavier_init(fc_weights, 8, 10);

    std :: cout << "Starting forward pass...\n";
    conv2d_forward_pass(input, weights, 0, conv_out);
    global_pooling_forward_pass(conv_out, pooled);
    matmul_forward_pass(pooled, fc_weights, nullptr, logits);
    softmax_ce_forward(logits, labels, loss);
    cudaDeviceSynchronize();

    float h_loss;
    CUDA_CHECK( cudaMemcpy(&h_loss, loss->data, sizeof(float), cudaMemcpyDeviceToHost) );
    std::cout << "Loss: " << h_loss << std::endl;

    float *h_logits = nullptr;
    h_logits = (float*) malloc(logits->size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, logits->data, logits->size * sizeof(float), cudaMemcpyDeviceToHost));

    float *h_labels = nullptr;
    h_labels = (float*) malloc(labels->size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_labels, labels->data, labels->size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < logits->shape[0]; i++) {
        std:: cout << "Batch: " << i << "\n";
        std :: cout << "    ";
        for (int j = 0; j < logits->shape[1]; j++) {
            std :: cout << h_logits[i * logits->shape[1] + j];
        }
        std :: cout << "\n";
        int pred_label = argmax(h_logits + (i * logits->shape[1]), logits->shape[1]);
        std :: cout << "    Prediciton: " << pred_label << " Ground Truth: " << h_labels[i] << "\n";
    }
    std :: cout << "\n";

    // Zero Gradients
    cudaMemset(logits->grad->data, 0, logits->size * sizeof(float));
    cudaMemset(fc_weights->grad->data, 0, fc_weights->size * sizeof(float));
    cudaMemset(pooled->grad->data, 0, pooled->size * sizeof(float));
    cudaMemset(conv_out->grad->data, 0, conv_out->size * sizeof(float));
    cudaMemset(weights->grad->data, 0, weights->size * sizeof(float));
    cudaMemset(input->grad->data, 0, input->size * sizeof(float));

    std :: cout << "Starting backward pass...\n";
    softmax_ce_backward(logits, labels, logits->grad);            
    matmul_backward_pass_A(pooled, fc_weights, logits->grad, pooled->grad);
    matmul_backward_pass_B(pooled, fc_weights, logits->grad, pooled->grad, fc_weights->grad);
    global_pooling_backward_pass(pooled->grad, conv_out->grad);   
    conv2d_backward_pass_weight(input, conv_out->grad, 0, weights->grad);
    conv2d_backward_pass_input(weights, conv_out->grad, 0, input->grad);
    cudaDeviceSynchronize();

    // Export Tensors
    save_tensor("input.bin", input);
    save_tensor("weights.bin", weights);
    save_tensor("fc_weights.bin", fc_weights);
    save_tensor("labels.bin", labels);
    save_tensor("fwd_logits.bin", logits);
    
    // Export Gradients
    save_tensor("grad_logits.bin", logits->grad);       
    save_tensor("grad_fc_weights.bin", fc_weights->grad);
    save_tensor("grad_pooled.bin", pooled->grad);
    save_tensor("grad_conv_out.bin", conv_out->grad);   
    save_tensor("grad_weights.bin", weights->grad);     
    save_tensor("grad_input.bin", input->grad);         

    tensor_free(input); 
    tensor_free(weights); 
    tensor_free(conv_out);
    tensor_free(pooled); 
    tensor_free(fc_weights);
    tensor_free(logits);
    tensor_free(labels); 
    tensor_free(loss);

    free(train_dataset->images);
    free(train_dataset->labels);
    free(train_dataset);
    free(train_indices);

    return 0;
}