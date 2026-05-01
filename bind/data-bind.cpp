#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../mnist-dataloader/mnist.h"
#include "../include/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(mnist_io, m) {
    py::class_<MNISTData>(m, "MNISTData");

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