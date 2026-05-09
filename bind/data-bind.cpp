#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../mnist-dataloader/mnist.h"
#include "../include/tensor.h"

namespace py = pybind11;

// in many places, we have used logic of putting data inside a vector, this is done because pybind11/stl.h
// can perform type conversions, which in this case turns a vector into a python list

PYBIND11_MODULE(mnist_io, m) {
    py::class_<MNISTData>(m, "MNISTData");
    
    // [](...) is a lambda function definition
    m.def("create_indices", [](int num_images) {
        int* raw_ptr = create_indices(num_images);
        std::vector<int> indices(raw_ptr, raw_ptr + num_images);
        free(raw_ptr); 
        return indices;
    });

    m.def("free_mnist_data", &free_mnist_data);

    m.def("load_dataset_in_ram", &load_dataset_in_ram, py::return_value_policy::reference);

    m.def("load_batch_to_tensor", [](MNISTData *dataset, int batch_start, int batch_end, std::vector<int>& indices, Tensor *img_batch, Tensor *labels_batch) {
        load_batch_to_tensor(dataset, batch_start, batch_end, indices.data(), img_batch, labels_batch);
    });
}