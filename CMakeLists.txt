cmake_minimum_required(VERSION 3.14)
project(SNN VERSION 0.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for your CPU
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

option(USE_NATIVE_OPT "Build with -march=native for maximum speed on this machine" OFF)

if(USE_NATIVE_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64-v3")
endif()

# ======= stack protection =======

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-D_FORTIFY_SOURCE=2)  # Bounded buffer checks

    # ASLR / PIE (Memory Randomization)
    add_compile_options(-fPIE)
     # Platform-specific linker flags
    if(WIN32 OR MINGW OR MSYS)
        # Windows / MinGW
        add_link_options(-Wl,--dynamicbase)  # ASLR
        add_link_options(-Wl,--nxcompat)     # DEP
        message(STATUS "Security: Windows ASLR + DEP enabled")
    elseif(APPLE)
        # macOS
        add_link_options(-Wl,-bind_at_load)
        message(STATUS "Security: macOS hardening enabled")
    else()
        # Linux / Unix
        add_link_options(-Wl,-z,relro,-z,now)  # Full RELRO
        add_link_options(-Wl,-z,noexecstack)   # Non-executable stack
        message(STATUS "Security: Linux RELRO + NOEXECSTACK enabled")
    endif()
endif()

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
target_link_libraries(mytensor PRIVATE pybind11::headers ssp)
# Install target
install(TARGETS mytensor DESTINATION ${CMAKE_INSTALL_PREFIX})
