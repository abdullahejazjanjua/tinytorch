#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "../include/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(base, m) {

    py::class_<Tensor>(m, "Tensor")
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("on_gpu", &Tensor::on_gpu)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_readwrite("grad", &Tensor::grad, py::return_value_policy::reference)

        .def_property_readonly("data", [](Tensor &t) {
            return py::array_t<float>(
                {t.size},           // Shape
                {sizeof(float)},    // Stride
                t.data,             // Pointer
                py::cast(&t)        // Lifetime parent
            );
        })
        .def_property_readonly("shape", [](Tensor &t) {
            std::vector<int> s;
            for (int i = 0; i < t.ndim; i++) s.push_back(t.shape[i]);
            return s;
        });

    m.def("tensor_create", [](std::vector<int> shape, int requires_grad, int on_gpu) {
        return tensor_create(static_cast<int>(shape.size()), shape.data(), requires_grad, on_gpu);
    }, py::return_value_policy::reference);

    m.def("tensor_free", &tensor_free);
    m.def("tensor_to_gpu", &tensor_to_gpu);
    m.def("tensor_to_cpu", &tensor_to_cpu);
}