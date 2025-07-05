# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triton fallback implementations for Apple Silicon (MPS) where Triton is not available
"""

import torch
import warnings
from typing import Optional, Callable, Any

# Global flag to check if we should use fallbacks
USE_TRITON_FALLBACKS = False

def enable_triton_fallbacks():
    """Enable Triton fallback implementations for Apple Silicon"""
    global USE_TRITON_FALLBACKS
    USE_TRITON_FALLBACKS = True

def triton_fallback_warning(func_name: str):
    """Issue warning about using fallback implementation"""
    warnings.warn(
        f"Using PyTorch fallback for {func_name} on Apple Silicon. "
        f"Performance may be slower than optimized Triton kernels.",
        UserWarning,
        stacklevel=3
    )

# Mock Triton decorators and functions for Apple Silicon
class MockTritonJIT:
    """Mock triton.jit decorator"""
    def __init__(self, *args, **kwargs):
        # Store jit parameters for potential future use
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, func):
        # Return the original function, but mark it as a fallback
        func._is_triton_fallback = True
        return func

class MockTritonAutoTune:
    """Mock triton.autotune decorator"""
    def __init__(self, configs, key=None, prune_configs_by=None, *args, **kwargs):
        self.configs = configs
        self.key = key
        self.prune_configs_by = prune_configs_by
    
    def __call__(self, func):
        func._is_triton_fallback = True
        return func

class MockTritonHeuristics:
    """Mock triton.heuristics decorator"""
    def __init__(self, values, *args, **kwargs):
        self.values = values
    
    def __call__(self, func):
        func._is_triton_fallback = True
        return func

class MockTritonLanguage:
    """Mock triton.language module"""
    
    # Mock data types
    float16 = torch.float16
    float32 = torch.float32
    bfloat16 = torch.bfloat16
    int32 = torch.int32
    int64 = torch.int64
    
    @staticmethod
    def program_id(axis):
        """Mock program_id - not used in fallback implementations"""
        return 0
    
    @staticmethod
    def num_programs(axis):
        """Mock num_programs - not used in fallback implementations"""
        return 1
    
    @staticmethod
    def load(ptr, mask=None, other=None):
        """Mock load - not used in fallback implementations"""
        return ptr
    
    @staticmethod
    def store(ptr, value, mask=None):
        """Mock store - not used in fallback implementations"""
        pass
    
    @staticmethod
    def arange(start, end):
        """Mock arange"""
        return torch.arange(start, end)
    
    @staticmethod
    def broadcast_to(x, shape):
        """Mock broadcast_to"""
        return x.expand(shape)
    
    @staticmethod
    def sum(x, axis=None):
        """Mock sum"""
        return torch.sum(x, dim=axis)
    
    @staticmethod
    def exp(x):
        """Mock exp"""
        return torch.exp(x)
    
    @staticmethod
    def log(x):
        """Mock log"""
        return torch.log(x)
    
    @staticmethod
    def sqrt(x):
        """Mock sqrt"""
        return torch.sqrt(x)
    
    @staticmethod
    def rsqrt(x):
        """Mock rsqrt"""
        return torch.rsqrt(x)
    
    @staticmethod
    def where(condition, x, y):
        """Mock where"""
        return torch.where(condition, x, y)

class MockTriton:
    """Mock triton module for Apple Silicon compatibility"""
    
    __version__ = "0.0.0-mps-fallback"
    
    # Mock decorators
    jit = MockTritonJIT
    autotune = MockTritonAutoTune
    heuristics = MockTritonHeuristics
    
    # Mock language module
    language = MockTritonLanguage()
    
    @staticmethod
    def next_power_of_2(n):
        """Fallback implementation of next_power_of_2"""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()
    
    @staticmethod
    def cdiv(a, b):
        """Ceiling division"""
        return (a + b - 1) // b

# Fallback implementations for common Triton kernel patterns

def rms_layernorm_fallback(
    X: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PyTorch fallback for RMS LayerNorm"""
    if out is None:
        out = torch.empty_like(X)
    
    # RMS normalization: x / sqrt(mean(x^2) + eps) * weight
    mean_sq = torch.mean(X * X, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    normalized = X * inv_rms
    out.copy_(normalized * W)
    return out

def layernorm_fallback(
    X: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PyTorch fallback for LayerNorm"""
    if out is None:
        out = torch.empty_like(X)
    
    # Standard layer norm
    mean = torch.mean(X, dim=-1, keepdim=True)
    var = torch.var(X, dim=-1, keepdim=True, unbiased=False)
    normalized = (X - mean) / torch.sqrt(var + eps)
    
    if b is not None:
        out.copy_(normalized * W + b)
    else:
        out.copy_(normalized * W)
    return out

def swiglu_fallback(
    X: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PyTorch fallback for SwiGLU activation"""
    if out is None:
        out = torch.empty(X.shape[0], X.shape[1], W2.shape[0], dtype=X.dtype, device=X.device)
    
    # SwiGLU: silu(x @ W1) * (x @ W3) @ W2
    gate = torch.nn.functional.silu(torch.matmul(X, W1.t()))
    up = torch.matmul(X, W3.t())
    intermediate = gate * up
    result = torch.matmul(intermediate, W2.t())
    out.copy_(result)
    return out

def cross_entropy_fallback(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> torch.Tensor:
    """PyTorch fallback for cross entropy loss"""
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction=reduction
    )

# Registry of fallback functions
FALLBACK_REGISTRY = {
    'rms_layernorm': rms_layernorm_fallback,
    'layernorm': layernorm_fallback, 
    'swiglu': swiglu_fallback,
    'cross_entropy': cross_entropy_fallback,
}

def get_fallback_function(name: str) -> Optional[Callable]:
    """Get a fallback function by name"""
    return FALLBACK_REGISTRY.get(name)

def register_fallback(name: str, func: Callable):
    """Register a new fallback function"""
    FALLBACK_REGISTRY[name] = func