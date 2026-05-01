#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../../include/tensor.h"

void load_bin(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { 
        perror("File error"); 
        exit(1); 
    }
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
            if (errs < 3)
                printf("[%s] Error at %d: Custom %f, Ref %f\n",
                       label, i, custom[i], ref[i]);
            errs++;
        }
    }

    if (errs > 0)
        printf("[%s] FAILED. Mismatches: %d, Max Diff: %e\n", label, errs, max_diff);
    else
        printf("[%s] PASSED. Max Diff: %e\n", label, max_diff);
}

int main(int argc, char **argv) {

    if (argc < 4) {
        printf("Usage: %s <M> <K> <N>\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    /* Named shape arrays — C++ forbids `(int[]) { … }` as a pointer source (temporary). */
    int shape_A_MK[] = {M, K};
    int shape_B_KN[] = {K, N};
    int shape_C_MN[] = {M, N};

    // CPU-side tensors that get loaded from disk, then moved to GPU
    Tensor *A  = tensor_create(2, shape_A_MK, 0, 0);
    Tensor *B  = tensor_create(2, shape_B_KN, 0, 0);
    Tensor *dC = tensor_create(2, shape_C_MN, 0, 0);

    // Output tensors created directly on GPU
    Tensor *C  = tensor_create(2, shape_C_MN, 0, 1);
    Tensor *dA = tensor_create(2, shape_A_MK, 0, 1);
    Tensor *dB = tensor_create(2, shape_B_KN, 0, 1);

    char pathA[64], pathB[64], pathC[64], pathdC[64], pathdA[64], pathdB[64];

    sprintf(pathA,  "data/A_%d_%d_%d.bin", M, K, N);
    sprintf(pathB,  "data/B_%d_%d_%d.bin", M, K, N);
    sprintf(pathC,  "data/C_ref_%d_%d_%d.bin", M, K, N);
    sprintf(pathdC, "data/dC_%d_%d_%d.bin", M, K, N);
    sprintf(pathdA, "data/dA_ref_%d_%d_%d.bin", M, K, N);
    sprintf(pathdB, "data/dB_ref_%d_%d_%d.bin", M, K, N);

    load_bin(pathA,  A->data,  A->size);
    load_bin(pathB,  B->data,  B->size);
    load_bin(pathdC, dC->data, dC->size);

    tensor_to_gpu(A);
    tensor_to_gpu(B);
    tensor_to_gpu(dC);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms_f, ms_a, ms_b;

    // forward
    matmul_forward_pass(A, B, NULL, C);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 10; i++)
        matmul_forward_pass(A, B, NULL, C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_f, start, stop);

    // backward A
    matmul_backward_pass_A(A, B, dC, dA);
    cudaEventRecord(start, 0);
    for (int i = 0; i < 10; i++)
        matmul_backward_pass_A(A, B, dC, dA);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_a, start, stop);

    // backward B
    matmul_backward_pass_B(A, B, dC, dB);
    cudaEventRecord(start, 0);
    for (int i = 0; i < 10; i++)
        matmul_backward_pass_B(A, B, dC, dB);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_b, start, stop);

    printf("Metrics: FWD: %.3fms | BWD_A: %.3fms | BWD_B: %.3fms\n", ms_f/10, ms_a/10, ms_b/10);

    // copy GPU outputs back to host for verification
    float *h_C  = (float*)malloc(C->size  * sizeof(float));
    float *h_dA = (float*)malloc(dA->size * sizeof(float));
    float *h_dB = (float*)malloc(dB->size * sizeof(float));

    cudaMemcpy(h_C,  C->data,  C->size  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dA, dA->data, dA->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB, dB->data, dB->size * sizeof(float), cudaMemcpyDeviceToHost);

    // load references
    float *r_C  = (float*)malloc(C->size  * sizeof(float));
    float *r_dA = (float*)malloc(dA->size * sizeof(float));
    float *r_dB = (float*)malloc(dB->size * sizeof(float));

    load_bin(pathC,  r_C,  C->size);
    load_bin(pathdA, r_dA, dA->size);
    load_bin(pathdB, r_dB, dB->size);

    // verification
    verify("Forward ", h_C,  r_C,  C->size);
    verify("Grad_A  ", h_dA, r_dA, dA->size);
    verify("Grad_B  ", h_dB, r_dB, dB->size);

    free(h_C);  free(h_dA); free(h_dB);
    free(r_C);  free(r_dA); free(r_dB);

    tensor_free(A);  tensor_free(B);  tensor_free(C);
    tensor_free(dC); tensor_free(dA); tensor_free(dB);

    return 0;
}
