#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/optim.h"
#include "../include/tensor.h"
#include "../include/backward.h"

namespace py = pybind11;

PYBIND11_MODULE(optim, m) {

    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<Tensor*>, float>(), 
             py::arg("params"), 
             py::arg("lr"))
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    m.def("backward", &backward);
}