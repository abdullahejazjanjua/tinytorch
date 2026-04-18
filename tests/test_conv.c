#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tensor.h"

void load_bin(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        exit(1);
    }
    fread(data, sizeof(float), size, f);
    fclose(f);
}

void verify_results(const float *custom_grad, const float *ref_grad, int size) {
    float max_diff = 0.0f;
    float tolerance = 1e-4f;
    int mismatches = 0;

    for (int i = 0; i < size; i++) {
        float diff = fabs(custom_grad[i] - ref_grad[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > tolerance) {
            if (mismatches < 5) {
                printf("Mismatch at index %d | Custom: %f | PyTorch: %f | Diff: %f\n", 
                       i, custom_grad[i], ref_grad[i], diff);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        printf("Verification FAILED. Total mismatches: %d. Max diff: %e\n", mismatches, max_diff);
    } else {
        printf("Verification PASSED. Max diff: %e\n", max_diff);
    }
}

void test_conv2d() {
    int meta[7]; 
    FILE *f = fopen("tests/data/meta.bin", "rb");
    if (!f) { 
        fprintf("[%s:%s]: meta file not found\n", __FILE__, __LINE__);
        return 1;
    }
    fread(meta, sizeof(int), 7, f);
    fclose(f);

    int batch_size   = meta[0];
    int in_channels  = meta[1];
    int input_height = meta[2];
    int input_width  = meta[3];
    int out_channels = meta[4];
    int kernel_size  = meta[5];
    int padding      = meta[6];

    int output_height = padding ? input_height : (input_height - kernel_size + 1);
    int output_width  = padding ? input_width  : (input_width - kernel_size + 1);

    int in_shape[]     = {batch_size, in_channels, input_height, input_width};
    int filter_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    int out_shape[]    = {batch_size, out_channels, output_height, output_width};

    Tensor *input   = tensor_create(4, in_shape);
    Tensor *filters = tensor_create(4, filter_shape);
    Tensor *output  = tensor_create(4, out_shape);
    Tensor *dout    = tensor_create(4, out_shape); // dout has same shape as output

    load_bin("data/input.bin", input->data, input->size);
    load_bin("data/filters.bin", filters->data, filters->size);
    
    float *ref_out = (float*)malloc(output->size * sizeof(float));
    load_bin("data/output_ref.bin", ref_out, output->size);

    printf("Testing forward pass...\n");
    conv2d_forward_pass(input, filters, padding, output);
    verify_results(output->data, ref_out, output->size);

    load_bin("data/dout.bin", dout->data, dout->size);
    float *ref_grad_w = (float*)malloc(filters->size * sizeof(float));
    load_bin("data/grad_w_ref.bin", ref_grad_w, filters->size);

    printf("Testing backward pass for weight...\n");
    conv2d_backward_pass_w(input, dout, padding, filters);
    verify_results(filters->grad->data, ref_grad_w, filters->size);

    free(ref_out);
    free(ref_grad_w);
    tensor_free(input);
    tensor_free(filters);
    tensor_free(output);
    tensor_free(dout);
}