#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class LinearLayer {
private:
    Tensor weights_;      // [out_features, in_features]
    Tensor bias_;         // [out_features]
    bool use_bias_;
    size_t in_features_;
    size_t out_features_;
    
public:
    // Constructor
    LinearLayer(size_t in_features, size_t out_features, bool use_bias = true);
    
    // Forward pass: output = input @ weights^T + bias
    // Input shape: [batch_size, in_features]
    // Output shape: [batch_size, out_features]
    Tensor forward(const Tensor& input);
    
    // Getters
    const Tensor& weights() const { return weights_; }
    Tensor& weights() { return weights_; }
    const Tensor& bias() const { return bias_; }
    Tensor& bias() { return bias_; }
    
    // Info
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    void print_info() const;
};

#endif // LINEAR_H