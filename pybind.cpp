#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SwiftTD.h"

namespace py = pybind11;

PYBIND11_MODULE(swift_td, m) {
    m.doc() = "Python bindings for SwiftTD reinforcement learning algorithm"; // Module docstring

    // Bind SwiftTDDense class
    py::class_<SwiftTDDense>(m, "SwiftTDDense")
        .def(py::init<int, float, float, float, float, float, float, float>(),
            "Initialize SwiftTDDense algorithm",
            py::arg("num_features"),
            py::arg("lambda_"),
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
    py::class_<SwiftTDSparseAndBinaryFeatures>(m, "SwiftTDSparse")
        .def(py::init<int, float, float, float, float, float, float, float>(),
            "Initialize SwiftTDSparse algorithm",
            py::arg("num_features"),
            py::arg("lambda_"),
            py::arg("initial_alpha"),
            py::arg("gamma"),
            py::arg("eps"),
            py::arg("max_step_size"),
            py::arg("step_size_decay"),
            py::arg("meta_step_size"))
        .def("step", &SwiftTDSparseAndBinaryFeatures::Step,
            "Perform one step of learning with sparse features",
            py::arg("features_indices"),
            py::arg("reward"));

    // Bind SwiftTDSparseAndNonBinaryFeatures class
    py::class_<SwiftTDSparseAndNonBinaryFeatures>(m, "SwiftTDSparseNonBinary")
        .def(py::init<int, float, float, float, float, float, float, float>(),
            "Initialize SwiftTDSparseAndNonBinaryFeatures algorithm",
            py::arg("num_features"),
            py::arg("lambda_"),
            py::arg("initial_alpha"),
            py::arg("gamma"),
            py::arg("eps"),
            py::arg("max_step_size"),
            py::arg("step_size_decay"),
            py::arg("meta_step_size"))
        .def("step", &SwiftTDSparseAndNonBinaryFeatures::Step,
            "Perform one step of learning with sparse non-binary features",
            py::arg("feature_indices_values"),
            py::arg("reward"));
}
