#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

class Tensor {
private:
    std::vector<float> data_;           // actual numbers
    std::vector<size_t> shape_;         // dimensions
    std::vector<size_t> strides_;       // how to jump through memory
    size_t size_;                       // total elements
    
    void compute_strides();
    
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
    
    // Basic properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    
    // Element access
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;
    
    // Fill with values
    void fill(float value);
    void ones();
    void zeros();
    
    // Random initialization
    void randn(float mean = 0.0f, float stddev = 1.0f);
    
    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor sum(int axis = -1) const;  
    Tensor operator*(float scalar) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(float scalar) const;           
    Tensor operator/(const Tensor& other) const;  
    
    // Matrix multiplication (will implement fully later)
    Tensor matmul(const Tensor& other) const;

    Tensor transpose() const;
    
    // Activation functions
    Tensor relu() const;
    
    // Utility
    void print() const;
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
};

#endif // TENSOR_H