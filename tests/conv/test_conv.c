#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/tensor.h"

void load_bin(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("File error"); exit(1); }
    fread(data, sizeof(float), size, f);
    fclose(f);
}

void verify(const char* label, const float *custom, const float *ref, int size) {
    float max_diff = 0.0f;
    float tol = 1e-4f; 
    int errs = 0;
    for (int i = 0; i < size; i++) {
        float d = fabs(custom[i] - ref[i]);
        if (d > max_diff) max_diff = d;
        if (d > tol) {
            if (errs < 3) printf("[%s] Error at %d: Custom %f, Ref %f\n", label, i, custom[i], ref[i]);
            errs++;
        }
    }
    if (errs > 0) printf("[%s] FAILED. Mismatches: %d, Max Diff: %e\n", label, errs, max_diff);
    else printf("[%s] PASSED. Max Diff: %e\n", label, max_diff);
}

int main(int argc, char **argv) {
    if (argc < 8) {
        printf("Usage: %s <N> <Ci> <Hi> <Wi> <Co> <K> <P:0|1>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]), Ci = atoi(argv[2]), Hi = atoi(argv[3]), Wi = atoi(argv[4]);
    int Co = atoi(argv[5]), K = atoi(argv[6]), P = atoi(argv[7]);

    int Ho = (P == 1) ? Hi : (Hi - K + 1);
    int Wo = (P == 1) ? Wi : (Wi - K + 1);

    Tensor *input = tensor_create(4, (int[]){N, Ci, Hi, Wi});
    Tensor *filt  = tensor_create(4, (int[]){Co, Ci, K, K});
    Tensor *dout  = tensor_create(4, (int[]){N, Co, Ho, Wo});
    Tensor *out   = tensor_create(4, (int[]){N, Co, Ho, Wo});
    filt->grad    = tensor_create(4, (int[]){Co, Ci, K, K});
    Tensor *grad_x = tensor_create(4, (int[]){N, Ci, Hi, Wi});

    load_bin("data/input.bin", input->data, input->size);
    load_bin("data/filters.bin", filt->data, filt->size);
    load_bin("data/dout.bin", dout->data, dout->size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms_f, ms_w, ms_i;

    // --- FORWARD ---
    conv2d_forward_pass(input, filt, P, out);
    cudaEventRecord(start, 0); // Added ', 0' for C compatibility
    for(int i=0; i<10; i++) conv2d_forward_pass(input, filt, P, out);
    cudaEventRecord(stop, 0);  // Added ', 0'
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_f, start, stop);

    // --- BWD WEIGHT ---
    conv2d_backward_pass_weight(input, dout, P, filt->grad);
    cudaEventRecord(start, 0); // Added ', 0'
    for(int i=0; i<10; i++) conv2d_backward_pass_weight(input, dout, P, filt->grad);
    cudaEventRecord(stop, 0);  // Added ', 0'
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_w, start, stop);

    // --- BWD INPUT ---
    conv2d_backward_pass_input(filt, dout, P, grad_x);
    cudaEventRecord(start, 0); // Added ', 0'
    for(int i=0; i<10; i++) conv2d_backward_pass_input(filt, dout, P, grad_x);
    cudaEventRecord(stop, 0);  // Added ', 0'
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_i, start, stop);

    printf("Metrics: FWD: %.3fms | BWD_W: %.3fms | BWD_I: %.3fms\n", ms_f/10, ms_w/10, ms_i/10);

    float *r_f = (float*)malloc(out->size * sizeof(float));
    float *r_w = (float*)malloc(filt->grad->size * sizeof(float));
    float *r_x = (float*)malloc(grad_x->size * sizeof(float));

    load_bin("data/out_ref.bin", r_f, out->size);
    load_bin("data/gw_ref.bin", r_w, filt->grad->size);
    load_bin("data/gx_ref.bin", r_x, grad_x->size);

    verify("Forward", out->data, r_f, out->size);
    verify("Grad_W ", filt->grad->data, r_w, filt->grad->size);
    verify("Grad_X ", grad_x->data, r_x, grad_x->size);

    return 0;
}