#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(code)                                                       \
  do {                                                                         \
    if ((code) != cudaSuccess) {                                               \
      std::cerr << "GPU ERROR in " << __FILE__ << ":" << __LINE__              \
                << " := " << cudaGetErrorString(code) << "\n";                 \
      exit(code);                                                              \
    }                                                                          \
  } while (0)


int cdiv(int size, int block_size) {
    return (size + block_size - 1) / block_size;
}