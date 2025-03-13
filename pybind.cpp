#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SwiftTD.h"

namespace py = pybind11;

PYBIND11_MODULE(swift_td, m) {
    m.doc() = "Python bindings for SwiftTD reinforcement learning algorithm"; // Module docstring

    // Bind the Math class
    py::class_<Math>(m, "Math")
        .def_static("dot_product", &Math::DotProduct, 
            "Compute dot product of two vectors",
            py::arg("a"), py::arg("b"));

    // Bind the base SwiftTD class
    py::class_<SwiftTD>(m, "SwiftTD")
        .def("get_weights", &SwiftTD::GetWeights, 
            "Get the current weight vector")
        .def("set_gamma", &SwiftTD::SetGamma, 
            "Set the discount factor gamma",
            py::arg("gamma"));

    // Bind SwiftTDDense class
    py::class_<SwiftTDDense, SwiftTD>(m, "SwiftTDDense")
        .def(py::init<int, float, float, float, float, float, float, float>(),
            "Initialize SwiftTDDense algorithm",
            py::arg("num_features"),
            py::arg("lambda"),
            py::arg("initial_alpha"),
            py::arg("gamma"),
            py::arg("eps"),
            py::arg("max_step_size"),
            py::arg("step_size_decay"),
            py::arg("meta_step_size"))
        .def("step", &SwiftTDDense::Step,
            "Perform one step of learning",
            py::arg("features"),
            py::arg("reward"));

    // Bind SwiftTDSparse class
    py::class_<SwiftTDSparse, SwiftTD>(m, "SwiftTDSparse")
        .def(py::init<int, float, float, float, float, float, float, float>(),
            "Initialize SwiftTDSparse algorithm",
            py::arg("num_features"),
            py::arg("lambda"),
            py::arg("initial_alpha"),
            py::arg("gamma"),
            py::arg("eps"),
            py::arg("max_step_size"),
            py::arg("step_size_decay"),
            py::arg("meta_step_size"))
        .def("step", &SwiftTDSparse::Step,
            "Perform one step of learning with sparse features",
            py::arg("features_indices"),
            py::arg("reward"));
}
