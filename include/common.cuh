#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(code)                                                       \
  do {                                                                         \
    cudaError_t err = (code);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "[%s:%d] GPU Error: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err = (call);                                               \
    if (err != CURAND_STATUS_SUCCESS)                                          \
    {                                                                          \
        fprintf(stderr, "[%s:%d] CURAND Error Code: %d\n", __FILE__, __LINE__, \
              (int)err);                                                       \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// Use inline to allow definition in the header file across multiple units
__device__ __host__ inline int cdiv(int size, int block_size) {
    return (size + block_size - 1) / block_size;
}

#endif