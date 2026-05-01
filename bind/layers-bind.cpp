#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "../include/nn.h"
#include "../include/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(nn, m) {

    py::class_<GlobalPooling>(m, "GlobalPooling")
        .def(py::init<int>(), py::arg("requires_grad") = 1)
        .def("forward", &GlobalPooling::forward, py::return_value_policy::reference);

    py::class_<ReLU>(m, "ReLU")
        .def(py::init<int>(), py::arg("requires_grad") = 1)
        .def("forward", &ReLU::forward, py::return_value_policy::reference);

    py::class_<CrossEntropy>(m, "CrossEntropy")
        .def(py::init<int>(), py::arg("requires_grad") = 1)
        .def("forward", &CrossEntropy::forward, py::return_value_policy::reference);

    py::class_<Conv2D>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>(),
             py::arg("in_channels"), py::arg("out_channels"), 
             py::arg("kernel_size"), py::arg("padding"), py::arg("requires_grad"))
        .def_readwrite("padding", &Conv2D::padding)
        .def_readwrite("in_channels", &Conv2D::in_channels)
        .def_readwrite("out_channels", &Conv2D::out_channels)
        .def_readwrite("kernel_size", &Conv2D::kernel_size)
        .def_readwrite("requires_grad", &Conv2D::requires_grad)
        .def_readwrite("weights", &Conv2D::weights, py::return_value_policy::reference)
        .def("forward", &Conv2D::forward, py::return_value_policy::reference);

    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, int, int>(),
             py::arg("in_features"), py::arg("out_features"), 
             py::arg("has_bias"), py::arg("requires_grad"))
        .def_readwrite("in_features", &Linear::in_features)
        .def_readwrite("out_features", &Linear::out_features)
        .def_readwrite("has_bias", &Linear::has_bias)
        .def_readwrite("requires_grad", &Linear::requires_grad)
        .def_readwrite("weights", &Linear::weights, py::return_value_policy::reference)
        .def_readwrite("bias", &Linear::bias, py::return_value_policy::reference)
        .def("forward", &Linear::forward, py::return_value_policy::reference);
}