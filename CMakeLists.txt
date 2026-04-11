cmake_minimum_required(VERSION 3.14)
project(SNN VERSION 0.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for your CPU
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the Python module
pybind11_add_module(mytensor
    src/tensor.cpp
    src/lif.cpp
    src/linear.cpp
    python/bindings.cpp
)

target_include_directories(mytensor PRIVATE include)

# Link any libraries (none yet!)
target_link_libraries(mytensor PRIVATE pybind11::headers)

# Install target
install(TARGETS mytensor DESTINATION ${CMAKE_INSTALL_PREFIX})