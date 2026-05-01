#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../../include/tensor.h"

void load_bin(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("File error"); exit(1); }
    (void)fread(data, sizeof(float), size, f);
    fclose(f);
}

void verify(const char* label, const float *custom, const float *ref, int size) {
    float max_diff = 0.0f;
    float atol = 1e-3f;
    float rtol = 1e-3f;
    int errs = 0;

    for (int i = 0; i < size; i++) {
        float d = fabs(custom[i] - ref[i]);
        if (d > max_diff) max_diff = d;

        if (d > (atol + rtol * fabs(ref[i]))) {
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

    /* Named shape arrays — C++ forbids `(int[]) { … }` as a pointer source (temporary). */
    int shape_input[]    = {N, Ci, Hi, Wi};
    int shape_filt[]     = {Co, Ci, K, K};
    int shape_output[]   = {N, Co, Ho, Wo};

    // CPU-side tensors that get loaded from disk, then moved to GPU
    Tensor *input = tensor_create(4, shape_input,  0, 0);
    Tensor *filt  = tensor_create(4, shape_filt,   0, 0);
    Tensor *dout  = tensor_create(4, shape_output, 0, 0);

    // Output tensors created directly on GPU
    Tensor *out    = tensor_create(4, shape_output, 0, 1);
    Tensor *grad_w = tensor_create(4, shape_filt,   0, 1);
    Tensor *grad_x = tensor_create(4, shape_input,  0, 1);

    load_bin("data/input.bin",   input->data, input->size);
    load_bin("data/filters.bin", filt->data,  filt->size);
    load_bin("data/dout.bin",    dout->data,  dout->size);

    tensor_to_gpu(input);
    tensor_to_gpu(filt);
    tensor_to_gpu(dout);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms_f, ms_w, ms_i;

    // --- FORWARD ---
    conv2d_forward_pass(input, filt, P, out);
    cudaEventRecord(start, 0);
    for(int i=0; i<10; i++) conv2d_forward_pass(input, filt, P, out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_f, start, stop);

    // --- BWD WEIGHT ---
    conv2d_backward_pass_weight(input, dout, P, grad_w);
    cudaEventRecord(start, 0);
    for(int i=0; i<10; i++) conv2d_backward_pass_weight(input, dout, P, grad_w);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_w, start, stop);

    // --- BWD INPUT ---
    conv2d_backward_pass_input(filt, dout, P, grad_x);
    cudaEventRecord(start, 0);
    for(int i=0; i<10; i++) conv2d_backward_pass_input(filt, dout, P, grad_x);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_i, start, stop);

    printf("Metrics: FWD: %.3fms | BWD_W: %.3fms | BWD_I: %.3fms\n", ms_f/10, ms_w/10, ms_i/10);

    // copy GPU outputs back to host for verification
    float *h_out = (float*)malloc(out->size    * sizeof(float));
    float *h_gw  = (float*)malloc(grad_w->size * sizeof(float));
    float *h_gx  = (float*)malloc(grad_x->size * sizeof(float));

    cudaMemcpy(h_out, out->data,    out->size    * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gw,  grad_w->data, grad_w->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gx,  grad_x->data, grad_x->size * sizeof(float), cudaMemcpyDeviceToHost);

    float *r_f = (float*)malloc(out->size    * sizeof(float));
    float *r_w = (float*)malloc(grad_w->size * sizeof(float));
    float *r_x = (float*)malloc(grad_x->size * sizeof(float));

    load_bin("data/out_ref.bin", r_f, out->size);
    load_bin("data/gw_ref.bin",  r_w, grad_w->size);
    load_bin("data/gx_ref.bin",  r_x, grad_x->size);

    verify("Forward", h_out, r_f, out->size);
    verify("Grad_W ", h_gw,  r_w, grad_w->size);
    verify("Grad_X ", h_gx,  r_x, grad_x->size);

    free(h_out); free(h_gw); free(h_gx);
    free(r_f);   free(r_w);  free(r_x);

    tensor_free(input);  tensor_free(filt);   tensor_free(dout);
    tensor_free(out);    tensor_free(grad_w); tensor_free(grad_x);

    return 0;
}
