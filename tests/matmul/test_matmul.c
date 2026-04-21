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
    fread(data, sizeof(float), size, f);
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

    Tensor *A = tensor_create(2, (int[]){M, K});
    Tensor *B = tensor_create(2, (int[]){K, N});
    Tensor *C = tensor_create(2, (int[]){M, N});

    Tensor *dC = tensor_create(2, (int[]){M, N});
    Tensor *dA = tensor_create(2, (int[]){M, K});
    Tensor *dB = tensor_create(2, (int[]){K, N});

    // ---------------- FILE LOADING (FIXED) ----------------

    char pathA[64], pathB[64], pathC[64], pathdC[64], pathdA[64], pathdB[64];

    sprintf(pathA,  "data/A_%d_%d_%d.bin", M, K, N);
    sprintf(pathB,  "data/B_%d_%d_%d.bin", M, K, N);
    sprintf(pathC,  "data/C_ref_%d_%d_%d.bin", M, K, N);
    sprintf(pathdC, "data/dC_%d_%d_%d.bin", M, K, N);
    sprintf(pathdA, "data/dA_ref_%d_%d_%d.bin", M, K, N);
    sprintf(pathdB, "data/dB_ref_%d_%d_%d.bin", M, K, N);

    load_bin(pathA, A->data, A->size);
    load_bin(pathB, B->data, B->size);
    load_bin(pathC, C->data, C->size);
    load_bin(pathdC, dC->data, dC->size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms_f, ms_b;

    // ---------------- FORWARD ----------------
    matmul_forward_pass(A, B, C);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 10; i++)
        matmul_forward_pass(A, B, C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_f, start, stop);

    // ---------------- BACKWARD ----------------
    matmul_backward_pass(A, B, dC, dA, dB);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 10; i++)
        matmul_backward_pass(A, B, dC, dA, dB);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_b, start, stop);

    printf("Metrics: FWD: %.3fms | BWD: %.3fms\n", ms_f/10, ms_b/10);

    // ---------------- LOAD REFERENCES ----------------
    float *r_C  = (float*)malloc(C->size * sizeof(float));
    float *r_dA = (float*)malloc(dA->size * sizeof(float));
    float *r_dB = (float*)malloc(dB->size * sizeof(float));

    load_bin(pathC,  r_C,  C->size);
    load_bin(pathdA, r_dA, dA->size);
    load_bin(pathdB, r_dB, dB->size);

    // ---------------- VERIFY ----------------
    verify("Forward ", C->data, r_C, C->size);
    verify("Grad_A  ", dA->data, r_dA, dA->size);
    verify("Grad_B  ", dB->data, r_dB, dB->size);

    free(r_C);
    free(r_dA);
    free(r_dB);

    return 0;
}
