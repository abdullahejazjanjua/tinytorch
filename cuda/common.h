#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(code)                                                       \
  do {                                                                         \
    if ((code) != cudaSuccess) {                                               \
      std::cerr << "GPU ERROR in " << __FILE__ << ":" << __LINE__              \
                << " := " << cudaGetErrorString(code) << "\n";                 \
      exit(code);                                                              \
    }                                                                          \
  } while (0)