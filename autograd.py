"""
Simple Autograd for SNN Tensor Library
Tracks operations and computes gradients using Python
"""

import sys
import os

# Add build path so autograd can find mytensor when imported
try:
    import mytensor as mt
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(script_dir, 'build')
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
    import mytensor as mt


class Tensor:
    """Python wrapper for C++ Tensor with autograd support"""
    
    def __init__(self, data, requires_grad=False):
        if isinstance(data, mt.Tensor):
            self.data = data
        elif isinstance(data, list):
            if not data:
                raise ValueError("Empty list not supported")
            if isinstance(data[0], list):
                shape = [len(data), len(data[0])]
                flat_data = []
                for row in data:
                    flat_data.extend(row)
                self.data = mt.Tensor(shape, flat_data)
            else:
                shape = [len(data)]
                self.data = mt.Tensor(shape, data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._parents = []
    
    @property
    def shape(self):
        return self.data.shape()
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        return add(self, other)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return mul_scalar(self, other)
        return mul(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return div_scalar(self, other)
        return div(self, other)
    
    def __rtruediv__(self, other):
        # scalar / tensor
        if isinstance(other, (int, float)):
            # Create tensor of ones, then divide
            ones = Tensor(mt.ones(self.shape), requires_grad=False)
            scaled = ones * other
            return div(scaled, self)
        return NotImplemented
    
    def matmul(self, other):
        return matmul(self, other)
    
    def relu(self):
        return relu(self)
    
    def sum(self):
        return sum_tensor(self)
    
    def zero_grad(self):
        if self.grad is not None:
            # Create zeros tensor of same shape
            zeros_data = mt.zeros(self.shape)
            self.grad = Tensor(zeros_data, requires_grad=False)
    
    def backward(self, grad=None):
        if grad is None:
            # Create ones tensor of same shape
            ones_data = mt.ones(self.shape)
            grad = Tensor(ones_data, requires_grad=False)
        
        if self.grad is None:
            self.grad = grad
        else:
            # Add to existing grad
            result_data = self.grad.data + grad.data
            self.grad = Tensor(result_data, requires_grad=False)
        
        self._backward()


class Linear:
    """Linear layer with learnable weights"""
    
    def __init__(self, in_features, out_features, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        scale = (2.0 / (in_features + out_features)) ** 0.5
        self.weight = Tensor(mt.randn([in_features, out_features], 0.0, scale), requires_grad=True)
        
        if use_bias:
            self.bias = Tensor(mt.zeros([out_features]), requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


class Sequential:
    """Sequential container for layers"""
    
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


# ============ Operation Functions ============

def add(a, b):
    """Addition with broadcasting support"""
    result_data = a.data + b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _backward():
        if a.requires_grad:
            a.backward(result.grad)
        if b.requires_grad:
            # If b was broadcasted (1D bias added to 2D matrix)
            if len(b.shape) == 1 and len(result.grad.shape) == 2:
                # Sum gradients across batch dimension - FIXED: use positional argument
                grad_summed_data = result.grad.data.sum(0)  # ← FIXED: no axis= keyword
                grad_summed = Tensor(grad_summed_data, requires_grad=False)
                b.backward(grad_summed)
            else:
                b.backward(result.grad)
    
    result._backward = _backward
    result._parents = [a, b]
    return result


def sub(a, b):
    """Subtraction with gradient tracking"""
    result_data = a.data - b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _backward():
        if a.requires_grad:
            a.backward(result.grad)
        if b.requires_grad:
            # d(a-b)/db = -1
            neg_grad_data = result.grad.data * (-1.0)
            neg_grad = Tensor(neg_grad_data, requires_grad=False)
            b.backward(neg_grad)
    
    result._backward = _backward
    result._parents = [a, b]
    return result


def mul(a, b):
    """Element-wise multiplication with gradient tracking"""
    result_data = a.data * b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _backward():
        if a.requires_grad:
            grad_a_data = b.data * result.grad.data
            grad_a = Tensor(grad_a_data, requires_grad=False)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b_data = a.data * result.grad.data
            grad_b = Tensor(grad_b_data, requires_grad=False)
            b.backward(grad_b)
    
    result._backward = _backward
    result._parents = [a, b]
    return result


def mul_scalar(tensor, scalar):
    """Scalar multiplication with gradient tracking"""
    result_data = tensor.data * scalar
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            grad_data = result.grad.data * scalar
            grad = Tensor(grad_data, requires_grad=False)
            tensor.backward(grad)
    
    result._backward = _backward
    result._parents = [tensor]
    return result


def div(a, b):
    """Element-wise division with gradient tracking"""
    result_data = a.data / b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _backward():
        if a.requires_grad:
            # d(a/b)/da = 1/b
            grad_a_data = result.grad.data * (1.0 / b.data)
            grad_a = Tensor(grad_a_data, requires_grad=False)
            a.backward(grad_a)
        if b.requires_grad:
            # d(a/b)/db = -a/b²
            grad_b_data = result.grad.data * (-a.data / (b.data * b.data))
            grad_b = Tensor(grad_b_data, requires_grad=False)
            b.backward(grad_b)
    
    result._backward = _backward
    result._parents = [a, b]
    return result


def div_scalar(tensor, scalar):
    """Scalar division with gradient tracking"""
    result_data = tensor.data / scalar
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            grad_data = result.grad.data / scalar
            grad = Tensor(grad_data, requires_grad=False)
            tensor.backward(grad)
    
    result._backward = _backward
    result._parents = [tensor]
    return result


def matmul(a, b):
    """Matrix multiplication with gradient tracking"""
    result_data = a.data.matmul(b.data)
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _backward():
        if a.requires_grad:
            # dL/dA = dL/dC @ B^T
            grad_a_data = result.grad.data.matmul(b.data.transpose())
            grad_a = Tensor(grad_a_data, requires_grad=False)
            a.backward(grad_a)
        if b.requires_grad:
            # dL/dB = A^T @ dL/dC
            grad_b_data = a.data.transpose().matmul(result.grad.data)
            grad_b = Tensor(grad_b_data, requires_grad=False)
            b.backward(grad_b)
    
    result._backward = _backward
    result._parents = [a, b]
    return result


def relu(tensor):
    """ReLU activation with gradient tracking"""
    result_data = tensor.data.relu()
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            # Create mask: 1 where input > 0, 0 elsewhere
            # We need to compute this from original tensor data
            mask_data = mt.zeros(tensor.shape)
            # For 2D tensors
            if len(tensor.shape) == 2:
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[1]):
                        val = tensor.data.__getitem__((i, j))
                        if val > 0:
                            mask_data.__setitem__((i, j), 1.0)
            # For 1D tensors
            elif len(tensor.shape) == 1:
                for i in range(tensor.shape[0]):
                    val = tensor.data.__getitem__((i,))
                    if val > 0:
                        mask_data.__setitem__((i,), 1.0)
            
            mask = Tensor(mask_data, requires_grad=False)
            grad_input_data = result.grad.data * mask.data
            grad_input = Tensor(grad_input_data, requires_grad=False)
            tensor.backward(grad_input)
    
    result._backward = _backward
    result._parents = [tensor]
    return result


def sum_tensor(tensor):
    """Sum all elements with gradient tracking"""
    data = tensor.data
    total = 0.0
    shape = tensor.shape
    
    # Sum all elements
    if len(shape) == 1:
        for i in range(shape[0]):
            total += data.__getitem__((i,))
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                total += data.__getitem__((i, j))
    
    result_data = mt.Tensor([1], [total])
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            grad_value = result.grad.data.__getitem__((0,))
            grad_input_data = mt.ones(tensor.shape)
            # Multiply all elements by grad_value
            if len(shape) == 1:
                for i in range(shape[0]):
                    grad_input_data.__setitem__((i,), grad_value)
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        grad_input_data.__setitem__((i, j), grad_value)
            grad_input = Tensor(grad_input_data, requires_grad=False)
            tensor.backward(grad_input)
    
    result._backward = _backward
    return result


# ============ Loss Functions ============

def mse_loss(pred, target):
    """Mean Squared Error loss"""
    diff = pred - target
    squared = diff * diff
    loss_sum = sum_tensor(squared)
    num_elements = pred.shape[0] * pred.shape[1]
    scale_factor = 1.0 / num_elements
    loss = loss_sum * scale_factor
    return loss


# ============ Optimizers ============

class SGD:
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                update_data = param.grad.data * (-self.lr)
                param.data = param.data + update_data
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

# """
# Simple Autograd for SNN Tensor Library
# Tracks operations and computes gradients using Python
# """

# import sys
# import os

# # Add build path so autograd can find mytensor when imported
# try:
#     import mytensor as mt
# except ImportError:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     build_path = os.path.join(script_dir, 'build')
#     if build_path not in sys.path:
#         sys.path.insert(0, build_path)
#     import mytensor as mt


# class Tensor:
#     """Python wrapper for C++ Tensor with autograd support"""
    
#     def __init__(self, data, requires_grad=False):
#         if isinstance(data, mt.Tensor):
#             self.data = data
#         elif isinstance(data, list):
#             if not data:
#                 raise ValueError("Empty list not supported")
#             if isinstance(data[0], list):
#                 shape = [len(data), len(data[0])]
#                 flat_data = []
#                 for row in data:
#                     flat_data.extend(row)
#                 self.data = mt.Tensor(shape, flat_data)
#             else:
#                 shape = [len(data)]
#                 self.data = mt.Tensor(shape, data)
#         else:
#             raise ValueError(f"Unsupported data type: {type(data)}")
        
#         self.requires_grad = requires_grad
#         self.grad = None
#         self._backward = lambda: None
#         self._parents = []
    
#     @property
#     def shape(self):
#         return self.data.shape()
    
#     def __repr__(self):
#         return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
#     def __add__(self, other):
#         return add(self, other)
    
#     def __sub__(self, other):
#         return sub(self, other)
    
#     def __mul__(self, other):
#         if isinstance(other, (int, float)):
#             return mul_scalar(self, other)
#         return mul(self, other)
    
#     def __rmul__(self, other):
#         return self.__mul__(other)
    
#     def __truediv__(self, other):
#         if isinstance(other, (int, float)):
#             return div_scalar(self, other)
#         return div(self, other)
    
#     def __rtruediv__(self, other):
#         # scalar / tensor
#         if isinstance(other, (int, float)):
#             # Create tensor of ones, then divide
#             ones = Tensor(mt.ones(self.shape), requires_grad=False)
#             scaled = ones * other
#             return div(scaled, self)
#         return NotImplemented
    
#     def matmul(self, other):
#         return matmul(self, other)
    
#     def relu(self):
#         return relu(self)
    
#     def sum(self):
#         return sum_tensor(self)
    
#     def zero_grad(self):
#         if self.grad is not None:
#             self.grad = mt.zeros(self.shape)
    
#     def backward(self, grad=None):
#         if grad is None:
#             grad = mt.ones(self.shape)
        
#         if self.grad is None:
#             self.grad = grad
#         else:
#             result = self.grad + grad
#             self.grad = result
        
#         self._backward()


# class Linear:
#     """Linear layer with learnable weights"""
    
#     def __init__(self, in_features, out_features, use_bias=True):
#         self.in_features = in_features
#         self.out_features = out_features
#         self.use_bias = use_bias
        
#         scale = (2.0 / (in_features + out_features)) ** 0.5
#         self.weight = Tensor(mt.randn([in_features, out_features], 0.0, scale), requires_grad=True)
        
#         if use_bias:
#             self.bias = Tensor(mt.zeros([out_features]), requires_grad=True)
#         else:
#             self.bias = None
    
#     def __call__(self, x):
#         return self.forward(x)
    
#     def forward(self, x):
#         out = x.matmul(self.weight)
#         if self.bias is not None:
#             out = out + self.bias
#         return out
    
#     def parameters(self):
#         params = [self.weight]
#         if self.bias is not None:
#             params.append(self.bias)
#         return params
    
#     def zero_grad(self):
#         for param in self.parameters():
#             param.zero_grad()


# class Sequential:
#     """Sequential container for layers"""
    
#     def __init__(self, *layers):
#         self.layers = layers
    
#     def __call__(self, x):
#         return self.forward(x)
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def parameters(self):
#         params = []
#         for layer in self.layers:
#             if hasattr(layer, 'parameters'):
#                 params.extend(layer.parameters())
#         return params
    
#     def zero_grad(self):
#         for param in self.parameters():
#             param.zero_grad()


# # ============ Operation Functions ============

# def add(a, b):
#     """Addition with broadcasting support"""
#     result = Tensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             a.backward(result.grad)
#         if b.requires_grad:
#             if len(b.shape) == 1 and len(result.grad.shape()) == 2:
#                 grad_summed = result.grad.sum(axis=0)
#                 b.backward(grad_summed)
#             else:
#                 b.backward(result.grad)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result


# def sub(a, b):
#     """Subtraction with gradient tracking"""
#     result_data = a.data - b.data
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             a.backward(result.grad)
#         if b.requires_grad:
#             b.backward(result.grad * -1.0)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result


# def mul(a, b):
#     """Element-wise multiplication with gradient tracking"""
#     result_data = a.data * b.data
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             a.backward(b.data * result.grad)
#         if b.requires_grad:
#             b.backward(a.data * result.grad)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result


# def mul_scalar(tensor, scalar):
#     """Scalar multiplication with gradient tracking"""
#     result_data = tensor.data * scalar
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             tensor.backward(result.grad * scalar)
    
#     result._backward = _backward
#     result._parents = [tensor]
#     return result


# def div(a, b):
#     """Element-wise division with gradient tracking"""
#     result_data = a.data / b.data
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             grad_a = result.grad * (1.0 / b.data)
#             a.backward(grad_a)
#         if b.requires_grad:
#             grad_b = result.grad * (-a.data / (b.data * b.data))
#             b.backward(grad_b)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result


# def div_scalar(tensor, scalar):
#     """Scalar division with gradient tracking"""
#     result_data = tensor.data / scalar
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             tensor.backward(result.grad / scalar)
    
#     result._backward = _backward
#     result._parents = [tensor]
#     return result


# def matmul(a, b):
#     """Matrix multiplication with gradient tracking"""
#     result_data = a.data.matmul(b.data)
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             grad_a = result.grad.matmul(b.data.transpose())
#             a.backward(grad_a)
#         if b.requires_grad:
#             grad_b = a.data.transpose().matmul(result.grad)
#             b.backward(grad_b)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result


# def relu(tensor):
#     """ReLU activation with gradient tracking"""
#     result_data = tensor.data.relu()
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             # Simplified: gradient is result.grad where input > 0
#             # For now, just pass through
#             tensor.backward(result.grad)
    
#     result._backward = _backward
#     result._parents = [tensor]
#     return result


# def sum_tensor(tensor):
#     """Sum all elements with gradient tracking"""
#     data = tensor.data
#     total = 0.0
#     shape = data.shape()
    
#     # Sum all elements using tuple indices
#     if len(shape) == 1:
#         for i in range(shape[0]):
#             total += data.__getitem__((i,))
#     elif len(shape) == 2:
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 total += data.__getitem__((i, j))
    
#     result_data = mt.Tensor([1], [total])
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             grad_value = result.grad.__getitem__((0,))
#             grad_input = mt.ones(tensor.shape) * grad_value
#             tensor.backward(grad_input)
    
#     result._backward = _backward
#     return result


# # ============ Loss Functions ============

# def mse_loss(pred, target):
#     """Mean Squared Error loss"""
#     diff = pred - target
#     squared = diff * diff
#     loss_sum = sum_tensor(squared)
#     num_elements = pred.shape[0] * pred.shape[1]
#     scale_factor = 1.0 / num_elements
#     loss = loss_sum * scale_factor
#     return loss


# # ============ Optimizers ============

# class SGD:
#     """Stochastic Gradient Descent optimizer"""
    
#     def __init__(self, parameters, lr=0.01):
#         self.parameters = parameters
#         self.lr = lr
    
#     def step(self):
#         for param in self.parameters:
#             if param.grad is not None:
#                 update = param.grad * (-self.lr)
#                 param.data = param.data + update
    
#     def zero_grad(self):
#         for param in self.parameters:
#             param.zero_grad()


# """
# Simple Autograd for SNN Tensor Library
# Tracks operations and computes gradients using Python
# """
# from re import sub
# import sys
# import os

# # Add build path so autograd can find mytensor when imported
# # This works both when run alone AND when imported from test.py
# try:
#     import mytensor as mt
# except ImportError:
#     # If not found, try to add the build path
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     build_path = os.path.join(script_dir, 'build')
#     if build_path not in sys.path:
#         sys.path.insert(0, build_path)
#     import mytensor as mt

# import numpy as np

# class Tensor:
#     """Python wrapper for C++ Tensor with autograd support"""
    
#     def __init__(self, data, requires_grad=False, dtype=float):
#         if isinstance(data, mt.Tensor):
#             self.data = data
#         elif isinstance(data, list):
#             shape = [len(data)] if not isinstance(data[0], list) else [len(data), len(data[0])]
#             flat_data = []
#             for item in data:
#                 if isinstance(item, list):
#                     flat_data.extend(item)
#                 else:
#                     flat_data.append(item)
#             self.data = mt.Tensor(shape, flat_data)
#         else:
#             raise ValueError("Unsupported data type")
        
#         self.requires_grad = requires_grad
#         self.grad = None
#         self._backward = lambda: None
#         self._parents = []
    
#     @property
#     def shape(self):
#         return self.data.shape()
    
#     def __repr__(self):
#         return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
#     def __add__(self, other):
#         return add(self, other)
    
#     def __sub__(self, other):
#         return sub(self, other)
    
#     def __mul__(self, other):
#         if isinstance(other, (int, float)):
#             return mul_scalar(self, other)
#         return mul(self, other)
    
#     def __rmul__(self, other):
#         return self.__mul__(other)
    
#     def __truediv__(self, other):
#         if isinstance(other, (int, float)):
#             return div_scalar(self, other)
#         return div(self, other)
    
#     def __rtruediv__(self, other):
#         # scalar / tensor
#         if isinstance(other, (int, float)):
#             # For scalar / tensor, we need to handle carefully
#             # This creates a tensor of ones * scalar, then divides
#             ones = Tensor(mt.ones(self.shape), requires_grad=False)
#             scaled = ones * other
#             return div(scaled, self)
#         return NotImplemented
    
#     def matmul(self, other):
#         return matmul(self, other)
    
#     def relu(self):
#         return relu(self)
    
#     def sum(self):
#         return sum_tensor(self)
    
#     def zero_grad(self):
#         if self.grad is not None:
#             self.grad = mt.zeros(self.shape)
    
#     def backward(self, grad=None):
#         if grad is None:
#             grad = mt.ones(self.shape)
        
#         if self.grad is None:
#             self.grad = grad
#         else:
#             # Add to existing grad
#             result = self.grad + grad
#             self.grad = result
        
#         self._backward()

# class Linear:
#     """Linear layer with learnable weights"""
    
#     def __init__(self, in_features, out_features, use_bias=True):
#         self.in_features = in_features
#         self.out_features = out_features
#         self.use_bias = use_bias
        
#         # Store weights as [in_features, out_features] directly
#         scale = (2.0 / (in_features + out_features)) ** 0.5
#         self.weight = Tensor(mt.randn([in_features, out_features], 0.0, scale), requires_grad=True)
        
#         if use_bias:
#             self.bias = Tensor(mt.zeros([out_features]), requires_grad=True)
#         else:
#             self.bias = None
    
#     def __call__(self, x):
#         return self.forward(x)
    
#     def forward(self, x):
#         # x shape: [batch, in_features]
#         # weight shape: [in_features, out_features]
#         # No transpose needed! [batch, in] @ [in, out] = [batch, out]
#         out = x.matmul(self.weight)
        
#         if self.bias is not None:
#             out = out + self.bias
#         return out
    
#     def parameters(self):
#         params = [self.weight]
#         if self.bias is not None:
#             params.append(self.bias)
#         return params
    
#     def zero_grad(self):
#         for param in self.parameters():
#             param.zero_grad()

# class Sequential:
#     """Sequential container for layers"""
    
#     def __init__(self, *layers):
#         self.layers = layers
    
#     def __call__(self, x):
#         return self.forward(x)
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def parameters(self):
#         params = []
#         for layer in self.layers:
#             if hasattr(layer, 'parameters'):
#                 params.extend(layer.parameters())
#         return params
    
#     def zero_grad(self):
#         for param in self.parameters():
#             param.zero_grad()


# # ============ Operation Functions ============


# def add(a, b):
#     """Addition with broadcasting support"""
#     result = Tensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             a.backward(result.grad)
#         if b.requires_grad:
#             # If b was broadcasted (1D bias added to 2D matrix)
#             if len(b.shape) == 1 and len(result.grad.shape()) == 2:
#                 # Sum gradients across batch dimension
#                 grad_summed = result.grad.sum(axis=0)  # Use C++ sum
#                 b.backward(grad_summed)
#             else:
#                 b.backward(result.grad)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result

# # subtraction function
# def sub(a, b):
#     """Subtraction with gradient tracking"""
#     result_data = a.data - b.data
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             a.backward(result.grad)
#         if b.requires_grad:
#             # d(a-b)/db = -1
#             b.backward(result.grad * -1.0)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result

# def mul(a, b):
#     """Element-wise multiplication with gradient tracking"""
#     result_data = a.data * b.data
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             a.backward(b.data * result.grad)
#         if b.requires_grad:
#             b.backward(a.data * result.grad)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result

# def mul_scalar(tensor, scalar):
#     """Scalar multiplication with gradient tracking"""
#     result_data = tensor.data * scalar
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             tensor.backward(result.grad * scalar)
    
#     result._backward = _backward
#     result._parents = [tensor]
#     return result


# def div(a, b):
#     """Element-wise division with gradient tracking"""
#     result_data = a.data / b.data
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             # d(a/b)/da = 1/b
#             grad_a = result.grad * (1.0 / b.data)
#             a.backward(grad_a)
#         if b.requires_grad:
#             # d(a/b)/db = -a/b²
#             grad_b = result.grad * (-a.data / (b.data * b.data))
#             b.backward(grad_b)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result

# def div_scalar(tensor, scalar):
#     """Scalar division with gradient tracking"""
#     result_data = tensor.data / scalar
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             tensor.backward(result.grad / scalar)
    
#     result._backward = _backward
#     result._parents = [tensor]
#     return result

# def matmul(a, b):
#     """Matrix multiplication with gradient tracking"""
#     result_data = a.data.matmul(b.data)
#     result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
#     def _backward():
#         if a.requires_grad:
#             # dL/dA = dL/dC @ B^T
#             grad_a = result.grad.matmul(b.data.transpose())
#             a.backward(grad_a)
#         if b.requires_grad:
#             # dL/dB = A^T @ dL/dC
#             grad_b = a.data.transpose().matmul(result.grad)
#             b.backward(grad_b)
    
#     result._backward = _backward
#     result._parents = [a, b]
#     return result

# def relu(tensor):
#     """ReLU activation with gradient tracking"""
#     result_data = tensor.data.relu()
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             # grad = 1 if input > 0 else 0
#             mask = tensor.data.relu()
#             mask.data = mask.data > 0  # Convert to boolean mask
#             # Multiply grad by mask
#             grad_input = result.grad * mask
#             tensor.backward(grad_input)
    
#     result._backward = _backward
#     result._parents = [tensor]
#     return result

# def sum_tensor(tensor):
#     """Sum all elements with gradient tracking"""
#     # Sum all elements
#     data = tensor.data
#     total = 0.0
#     for i in range(data.size()):
#         total += data.__getitem__(indices(i, data.shape()))
    
#     result_data = mt.Tensor([1], [total])
#     result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
#     def _backward():
#         if tensor.requires_grad:
#             # Gradient is 1 for all elements
#             grad_input = mt.ones(tensor.shape) * result.grad.__getitem__([0])
#             tensor.backward(grad_input)
    
#     result._backward = _backward
#     return result

# def indices(flat_idx, shape):
#     """Convert flat index to multi-dimensional indices"""
#     if len(shape) == 1:
#         return [flat_idx]
#     elif len(shape) == 2:
#         cols = shape[1]
#         return [flat_idx // cols, flat_idx % cols]
#     return [flat_idx]


# # ============ Loss Functions ============

# def mse_loss(pred, target):
#     """Mean Squared Error loss"""
#     diff = pred - target
#     squared = diff * diff
#     loss = sum_tensor(squared) / (pred.shape[0] * pred.shape[1])
#     return loss

# def cross_entropy_loss(pred, target):
#     """Cross entropy loss (simplified)"""
#     # This is simplified - full version would use softmax
#     # For now, just MSE
#     return mse_loss(pred, target)


# # ============ Optimizers ============

# class SGD:
#     """Stochastic Gradient Descent optimizer"""
    
#     def __init__(self, parameters, lr=0.01):
#         self.parameters = parameters
#         self.lr = lr
    
#     def step(self):
#         for param in self.parameters:
#             if param.grad is not None:
#                 # weight = weight - lr * grad
#                 update = param.grad * (-self.lr)
#                 param.data = param.data + update