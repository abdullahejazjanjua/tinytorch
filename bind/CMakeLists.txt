cmake_minimum_required(VERSION 3.18)
project(DL_Backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86) 
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Dependencies
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 2. Source Grouping
set(BACKEND_SRCS
    ../include/tensor.cpp
    ../mnist-dataloader/mnist.cpp
    ../src/autograd/backward.cpp
    ../src/optim/sgd.cpp
)
file(GLOB FUNCTIONAL_SRCS "../src/functional/*.cpp")
file(GLOB NN_SRCS "../src/nn/*.cpp")
file(GLOB CUDA_SRCS "../cuda/*.cu")

# 3. Core Shared Library
add_library(dl_core SHARED 
    ${BACKEND_SRCS} 
    ${FUNCTIONAL_SRCS} 
    ${NN_SRCS} 
    ${CUDA_SRCS}
)

target_include_directories(dl_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../mnist-dataloader
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(dl_core PUBLIC 
    CUDA::cudart
    CUDA::curand
)

# 4. Define Three Separate Python Modules
# Module 1: base (Tensor struct and helper functions)
pybind11_add_module(base helper-bind.cpp)
target_link_libraries(base PRIVATE dl_core)

# Module 2: nn (Layers)
pybind11_add_module(nn layers-bind.cpp)
target_link_libraries(nn PRIVATE dl_core)

# Module 3: optim (Optimizers)
pybind11_add_module(optim optim-bind.cpp)
target_link_libraries(optim PRIVATE dl_core)

pybind11_add_module(mnist_io data-bind.cpp)
target_link_libraries(mnist_io PRIVATE dl_core)

set_target_properties(base nn optim mnist_io PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)