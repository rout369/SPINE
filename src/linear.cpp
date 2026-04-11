#include "linear.h"
#include <cmath>
#include <iostream>

LinearLayer::LinearLayer(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
    
    // Initialize weights with Xavier/Glorot initialization
    // This helps with gradient flow in deep networks
    weights_ = Tensor({out_features, in_features});
    float scale = std::sqrt(2.0f / (in_features + out_features));
    weights_.randn(0.0f, scale);
    
    // Initialize bias to zeros
    if (use_bias) {
        bias_ = Tensor({out_features});
        bias_.zeros();
    }
}

Tensor LinearLayer::forward(const Tensor& input) {
    // Check input dimensions
    assert(input.ndim() == 2);
    assert(input.shape()[1] == in_features_);
    
    // Transpose weights to [in_features, out_features] for matmul
    Tensor weights_T = weights_.transpose();
    
    // Matrix multiply: [batch, in] @ [in, out] = [batch, out]
    Tensor output = input.matmul(weights_T);
    
    // Add bias if needed
    if (use_bias_) {
        size_t batch_size = input.shape()[0];
        Tensor output_with_bias(output.shape());
        
        const float* out_data = output.data();
        const float* bias_data = bias_.data();
        float* result_data = output_with_bias.data();
        
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features_; j++) {
                result_data[i * out_features_ + j] = 
                    out_data[i * out_features_ + j] + bias_data[j];
            }
        }
        return output_with_bias;
    }
    
    return output;
}

void LinearLayer::print_info() const {
    std::cout << "LinearLayer: " << in_features_ << " -> " << out_features_ << std::endl;
    std::cout << "Weights shape: [" << out_features_ << ", " << in_features_ << "]" << std::endl;
    if (use_bias_) {
        std::cout << "Bias shape: [" << out_features_ << "]" << std::endl;
    }
}