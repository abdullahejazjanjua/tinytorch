#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "../include/tensor.h"

namespace py = pybind11;

// for pointers, we have to define a return policy
// for others, the pybind11 simply sends a copy of the value, so there is no memory management involved
// py::return_value_policy::reference means python can read, modify with this memory but can't manage its lifecycle

PYBIND11_MODULE(base, m) {

    py::class_<Tensor>(m, "Tensor")
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("on_gpu", &Tensor::on_gpu)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_readwrite("grad", &Tensor::grad, py::return_value_policy::reference)

        // bind the data as a numpy array (allows compability with all libraries that support numpy arrays)
        .def_property_readonly("data", [](Tensor &t) {
            return py::array_t<float>(
                {t.size},
                {sizeof(float)},
                t.data,             // The numpy array holds the location to data instead of data itself
                py::cast(&t)        // Ensure that as long as numpy array exists, the tensor t also exists
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