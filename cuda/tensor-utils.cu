#include <curand.h>
#include <curand_kernel.h>

#include "../include/tensor.h"
#include "../include/common.cuh"


void normal_xavier_init(Tensor* t, int nin, int nout) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    float stddev = sqrt(2.0f / (float)(nin + nout));

    // Curand Normal requires an EVEN number of elements https://stackoverflow.com/questions/21561988/curand-generator-fails-on-odd-number-of-elements
    if (t->size % 2 == 0) {
        curandGenerateNormal(gen, t->data, t->size, 0.0f, stddev);
    } else {
        curandGenerateNormal(gen, t->data, t->size - 1, 0.0f, stddev);
        
        float* d_last;
        cudaMalloc(&d_last, sizeof(float) * 2); 
        curandGenerateNormal(gen, d_last, 2, 0.0f, stddev);
        cudaMemcpy(t->data + (t->size - 1), d_last, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_last);
    }

    curandDestroyGenerator(gen);
}