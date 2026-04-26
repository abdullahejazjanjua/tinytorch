#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
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

int main() {
    srand(42); 
    int in_s[] = {4, 1, 28, 28}; 
    int filt_s[] = {8, 1, 3, 3}; 
    int out_s[] = {4, 8, 26, 26}; 
    int pool_s[] = {4, 8}; 
    int label_s[] = {4}; 
    int loss_s[] = {1};

    MNISTData* train_dataset = load_dataset_in_ram("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000);
    int* train_indices = create_indices(60000);

    Tensor *input = tensor_create(4, in_s, 1);
    Tensor *weights = tensor_create(4, filt_s, 1);
    Tensor *conv_out = tensor_create(4, out_s, 1);
    Tensor *pooled = tensor_create(2, pool_s, 1);
    Tensor *labels = tensor_create(1, label_s, 0);
    Tensor *loss = tensor_create(1, loss_s, 0);

    // Load actual data into host tensors
    load_batch_to_tensor(train_dataset, 4, 8, train_indices, input, labels);

    // Move to device
    tensor_to_gpu(input);
    tensor_to_gpu(weights);
    tensor_to_gpu(conv_out);
    tensor_to_gpu(pooled);
    tensor_to_gpu(labels);
    tensor_to_gpu(loss);

    // Initialize weights
    normal_xavier_init(weights, 9, 72);

    // Forward Pass
    conv2d_forward_pass(input, weights, 0, conv_out);
    global_pooling_forward_pass(conv_out, pooled);
    softmax_ce_forward(pooled, labels, loss);
    cudaDeviceSynchronize();

    // Backward Pass
    softmax_ce_backward(pooled, labels, pooled->grad);            
    global_pooling_backward_pass(pooled->grad, conv_out->grad);   
    conv2d_backward_pass_weight(input, conv_out->grad, 0, weights->grad);
    conv2d_backward_pass_input(weights, conv_out->grad, 0, input->grad);
    cudaDeviceSynchronize();

    // Exhaustive Export
    save_tensor("input.bin", input);
    save_tensor("weights.bin", weights);
    save_tensor("labels.bin", labels);
    save_tensor("fwd_logits.bin", pooled);
    
    // Intermediate Gradients
    save_tensor("grad_logits.bin", pooled->grad);       
    save_tensor("grad_conv_out.bin", conv_out->grad);   
    save_tensor("grad_weights.bin", weights->grad);     
    save_tensor("grad_input.bin", input->grad);         

    tensor_free(input); 
    tensor_free(weights); 
    tensor_free(conv_out);
    tensor_free(pooled); 
    tensor_free(labels); 
    tensor_free(loss);

    free(train_dataset->images);
    free(train_dataset->labels);
    free(train_dataset);
    free(train_indices);

    return 0;
}