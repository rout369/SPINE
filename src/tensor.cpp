#include "tensor.h"
#include <random>
#include <cstring>

// Compute strides for row-major layout
void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

// Default constructor
Tensor::Tensor() : size_(0) {}

// Constructor with shape
Tensor::Tensor(const std::vector<size_t>& shape) 
    : shape_(shape), size_(1) {
    for (auto dim : shape) {
        size_ *= dim;
    }
    data_.resize(size_, 0.0f);
    compute_strides();
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
    : shape_(shape), data_(data) {
    // Calculate expected size from shape
    size_t expected_size = 1;
    for (auto dim : shape) {
        expected_size *= dim;
    }
    
    // Verify data size matches shape
    if (expected_size != data.size()) {
        std::cerr << "Error: Shape requires " << expected_size 
                  << " elements but got " << data.size() << std::endl;
        assert(false && "Data size doesn't match shape");
    }
    
    size_ = data.size();
    compute_strides();
}

// Element access
float& Tensor::operator()(const std::vector<size_t>& indices) {
    assert(indices.size() == shape_.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[i] < shape_[i]);
        offset += indices[i] * strides_[i];
    }
    return data_[offset];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    assert(indices.size() == shape_.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[i] < shape_[i]);
        offset += indices[i] * strides_[i];
    }
    return data_[offset];
}

// Fill all elements
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::ones() {
    fill(1.0f);
}

void Tensor::zeros() {
    fill(0.0f);
}

// Random initialization (Gaussian)
void Tensor::randn(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    for (size_t i = 0; i < size_; i++) {
        data_[i] = dist(gen);
    }
}

// Addition
// Replace your existing operator+ with this broadcasting version
Tensor Tensor::operator+(const Tensor& other) const {
    // Case 1: Same shape - direct addition
    if (shape_ == other.shape_) {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; i++) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    // Case 2: Broadcasting - 2D matrix + 1D vector (bias addition)
    else if (ndim() == 2 && other.ndim() == 1 && shape_[1] == other.shape_[0]) {
        Tensor result(shape_);
        for (size_t i = 0; i < shape_[0]; i++) {
            for (size_t j = 0; j < shape_[1]; j++) {
                result({i, j}) = (*this)({i, j}) + other.data_[j];
            }
        }
        return result;
    }
    
    // Case 3: 1D vector + 2D matrix (reverse broadcasting)
    else if (ndim() == 1 && other.ndim() == 2 && shape_[0] == other.shape_[1]) {
        Tensor result(other.shape_);
        for (size_t i = 0; i < other.shape_[0]; i++) {
            for (size_t j = 0; j < other.shape_[1]; j++) {
                result({i, j}) = data_[j] + other({i, j});
            }
        }
        return result;
    }
    
    else {
        std::cerr << "Shape mismatch: ";
        for (auto s : shape_) std::cerr << s << " ";
        std::cerr << "vs ";
        for (auto s : other.shape_) std::cerr << s << " ";
        std::cerr << std::endl;
        assert(false && "Shapes do not match and broadcasting not supported");
        return Tensor();
    }
}

// Subtraction operator
Tensor Tensor::operator-(const Tensor& other) const {
    assert(shape_ == other.shape_);
    Tensor result(shape_);
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// Sum reduction (for bias gradients)
Tensor Tensor::sum(int axis) const {
    if (axis == -1) {
        // Sum all elements
        float total = 0.0f;
        for (size_t i = 0; i < size_; i++) {
            total += data_[i];
        }
        Tensor result({1});
        result.data_[0] = total;
        return result;
    }
    else if (axis == 0 && ndim() == 2) {
        // Sum across rows (batch dimension)
        // Input: [batch, features] -> Output: [features]
        Tensor result({shape_[1]});
        result.zeros();
        for (size_t j = 0; j < shape_[1]; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < shape_[0]; i++) {
                sum += (*this)({i, j});
            }
            result.data_[j] = sum;
        }
        return result;
    }
    else {
        assert(false && "Sum with this axis not implemented");
        return Tensor();
    }
}

// Scalar multiplication
Tensor Tensor::operator*(const Tensor& other) const {
    assert(shape_ == other.shape_);  // Must have same shape
    Tensor result(shape_);
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// Scalar division: tensor / scalar
Tensor Tensor::operator/(float scalar) const {
    Tensor result(shape_);
    float inv_scalar = 1.0f / scalar;  // Precompute reciprocal for speed
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = data_[i] * inv_scalar;
    }
    return result;
}

// Element-wise division: tensor / tensor
Tensor Tensor::operator/(const Tensor& other) const {
    assert(shape_ == other.shape_);  // Must have same shape
    Tensor result(shape_);
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

// Matrix multiplication 
Tensor Tensor::matmul(const Tensor& other) const {
    // Check shapes: (a,b) @ (b,c) -> (a,c)
    assert(ndim() == 2 && other.ndim() == 2);
    assert(shape_[1] == other.shape_[0]);
    
    size_t a = shape_[0];
    size_t b = shape_[1];
    size_t c = other.shape_[1];
    
    Tensor result({a, c});
    result.zeros();

    // get row pointer for direct memory acess (faster than opertaor())

    const float* A = data_.data();
    const float* B = other.data_.data();
    float* C = result.data_.data();

    // Block size - tune this for your CPU (try 32, 64, or 128)
    const size_t BLOCK = 64;
    // { old neive approach 

    // for (size_t i = 0; i < a; i++) {
    //     for (size_t j = 0; j < c; j++) {
    //         float sum = 0.0f;
    //         for (size_t k = 0; k < b; k++) {
    //             sum += (*this)({i, k}) * other({k, j});
    //         }
    //         result({i, j}) = sum;
    //     }
    // }

// }    


    // Blocked matrix multiplication - processes matrix in small chunks
    for(size_t i = 0 ;i < a ; i += BLOCK){
        for(size_t j = 0; j < c; j += BLOCK){
            for(size_t k = 0; k < b; k += BLOCK){
                // Compute block process BLOCK X BLOCK chunks
                size_t i_max = std::min(i + BLOCK, a);
                size_t j_max = std::min(j + BLOCK, c);
                size_t k_max = std::min(k + BLOCK, b);

                for(size_t ii = i ; ii < i_max; ii++){
                    for(size_t kk = k ; kk < k_max ; kk++){
                        float aik = A[ii * b + kk];
                        for(size_t jj = j ; jj < j_max ; jj++){
                            C[ii * c + jj] += aik * B[kk * c + jj];
                        }
                    }
                }
            }
        }
    }

    return result;
}

// ReLU activation
Tensor Tensor::relu() const {
    Tensor result(shape_);
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}


Tensor Tensor::transpose() const {
    assert(ndim() == 2);
    
    // New shape: swap dimensions
    std::vector<size_t> new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape);
    
    // Copy data with transpose
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[1]; j++) {
            result({j, i}) = (*this)({i, j});
        }
    }
    
    return result;
}


// Print for debugging
void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); i++) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]\nData:\n";
    
    if (ndim() == 2) {
        for (size_t i = 0; i < shape_[0]; i++) {
            std::cout << "[";
            for (size_t j = 0; j < shape_[1]; j++) {
                std::cout << (*this)({i, j});
                if (j < shape_[1] - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    } else {
        // For 1D or other dimensions, just print flat
        std::cout << "[";
        for (size_t i = 0; i < size_ && i < 20; i++) {
            std::cout << data_[i];
            if (i < size_ - 1 && i < 19) std::cout << ", ";
        }
        if (size_ > 20) std::cout << ", ...";
        std::cout << "]\n";
    }
}


// Test for the buffer overflow vulnerability in SNN code - this is a separate function that will be called from Python to trigger the overflow and test the stack protector

// void test_overflow_in_snn() {
//     char buffer[10];
//     // Force the compiler to NOT optimize this away
//     volatile char* p = buffer;
    
//     // Overflow with a specific pattern
//     for (int i = 0; i < 100; i++) {
//         p[i] = 'A' + (i % 26);
//     }
    
//     // Use the buffer so compiler can't optimize it out
//     volatile char dummy = p[0];
//     (void)dummy;
// }
