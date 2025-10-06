#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SwiftTD.h"

namespace py = pybind11;

PYBIND11_MODULE(swift_td, m)
{
     m.doc() = "Python bindings for SwiftTD reinforcement learning algorithm"; // Module docstring
     py::class_<SwiftTDNonSparse>(m, "SwiftTDNonSparse")
          .def(py::init<int, float, float, float, float, float, float, float>(),
               "Initialize the SwiftTDNonSparse algorithm",
               py::arg("num_of_features"),
               py::arg("lambda"),
               py::arg("alpha"),
               py::arg("gamma"),
               py::arg("epsilon"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("meta_step_size"))
          .def("step", &SwiftTDNonSparse::Step,
               "Perform one step of learning",
               py::arg("features"),
               py::arg("reward"));

     // Bind SwiftTDSparse class
     py::class_<SwiftTDBinaryFeatures>(m, "SwiftTDBinaryFeatures")
          .def(py::init<int, float, float, float, float, float, float, float>(),
               "Initialize the SwiftTDBinaryFeatures algorithm",
               py::arg("num_of_features"),
               py::arg("lambda"),
               py::arg("alpha"),
               py::arg("gamma"),
               py::arg("epsilon"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("meta_step_size"))
          .def("step", &SwiftTDBinaryFeatures::Step,
               "Perform one step of learning with sparse features",
               py::arg("features_indices"),
               py::arg("reward"));

     // Bind SwiftTDSparseAndNonBinaryFeatures class
     py::class_<SwiftTD>(m, "SwiftTD")
          .def(py::init<int, float, float, float, float, float, float, float>(),
               "Initialize the SwiftTD algorithm",
               py::arg("num_of_features"),
               py::arg("lambda"),
               py::arg("alpha"),
               py::arg("gamma"),
               py::arg("epsilon"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("meta_step_size"))
          .def("step", &SwiftTD::Step,
               "Perform one step of learning with sparse non-binary features",
               py::arg("feature_indices_values"),
               py::arg("reward"));
}
