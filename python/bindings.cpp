#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "lif.h"
#include "linear.h"


namespace py = pybind11;

// void test_overflow_in_snn();

PYBIND11_MODULE(mytensor, m) {
    m.doc() = "Custom tensor library for SNN project";
    
    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        
        // Properties
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("ndim", &Tensor::ndim)
        
        // Methods
        .def("ones", &Tensor::ones)
        .def("zeros", &Tensor::zeros)
        .def("randn", &Tensor::randn)
        .def("matmul", &Tensor::matmul)
        .def("relu", &Tensor::relu)
        .def("print", &Tensor::print)
        
        // Operator overloads for Python
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("sum", &Tensor::sum)
        .def("__mul__", [](const Tensor& t, const Tensor& other) { return t * other; })
        .def("__mul__", [](const Tensor& t, float scalar) { return t * scalar; })
        .def("__rmul__", [](const Tensor& t, float scalar) { return t * scalar; })
        .def("__truediv__", [](const Tensor& t, float scalar) { return t / scalar; })
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return a / b; })
        .def("__rtruediv__", [](const Tensor& t, float scalar) { return t / scalar; })
        
        // array-style access
        // .def("__getitem__", [](Tensor& t, std::pair<size_t, size_t> idx) {
        //     std::vector<size_t> indices = {idx.first, idx.second};
        //     return t(indices);
        // })

        .def("__getitem__", [](Tensor& t, std::vector<size_t> indices) {
            return t(indices);
        })

        .def("__setitem__", [](Tensor& t, std::vector<size_t> indices, float value) {
            t(indices) = value;
        })

        .def("transpose", &Tensor::transpose);

        // .def("__setitem__", [](Tensor& t, std::pair<size_t, size_t> idx, float value) {
        //     std::vector<size_t> indices = {idx.first, idx.second};
        //     t(indices) = value;
        // });


    // LinearLayer class
    py::class_<LinearLayer>(m, "LinearLayer")
        .def(py::init<size_t, size_t, bool>(), 
            py::arg("in_features"), 
            py::arg("out_features"),
            py::arg("use_bias") = true)
        .def("forward", &LinearLayer::forward)
        .def("in_features", &LinearLayer::in_features)
        .def("out_features", &LinearLayer::out_features)
        .def("print_info", &LinearLayer::print_info);

    // LIFNeuron class 
    py::class_<LIFNeuron>(m, "LIFNeuron")
        .def(py::init<float, float, float, float, float, float, int, float>(),
            py::arg("tau_mem") = 20.0f,
            py::arg("tau_syn") = 5.0f,           
            py::arg("V_thresh") = -55.0f,
            py::arg("V_rest") = -70.0f,
            py::arg("V_reset") = -70.0f,
            py::arg("R") = 1.0f,
            py::arg("refractory_period") = 2,
            py::arg("dt") = 1.0f)                
        .def("update", &LIFNeuron::update)
        .def("reset", &LIFNeuron::reset)
        .def("get_membrane_potential", &LIFNeuron::get_membrane_potential)
        .def("get_synaptic_current", &LIFNeuron::get_synaptic_current)  
        .def("set_dt", &LIFNeuron::set_dt);                             

    // LIFLayer class - UPDATE THIS SECTION
    py::class_<LIFLayer>(m, "LIFLayer")
        .def(py::init<int, float, float, float>(),
            py::arg("size"),
            py::arg("tau_mem") = 20.0f,
            py::arg("tau_syn") = 5.0f,           
            py::arg("dt") = 1.0f)                
        .def("forward", &LIFLayer::forward)
        .def("reset", &LIFLayer::reset)
        .def("size", &LIFLayer::size)
        .def("set_dt", &LIFLayer::set_dt)                           
        .def("get_membrane_potentials", &LIFLayer::get_membrane_potentials)  
        .def("get_synaptic_currents", &LIFLayer::get_synaptic_currents);     
    
    // Helper functions
    m.def("randn", [](const std::vector<size_t>& shape, float mean, float stddev) {
        Tensor t(shape);
        t.randn(mean, stddev);
        return t;
    }, py::arg("shape"), py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f);
    
    m.def("ones", [](const std::vector<size_t>& shape) {
        Tensor t(shape);
        t.ones();
        return t;
    });
    
    m.def("zeros", [](const std::vector<size_t>& shape) {
        Tensor t(shape);
        t.zeros();
        return t;
    });

    // m.def("test_overflow_in_snn", &test_overflow_in_snn, "Test function that forces buffer overflow");
}
