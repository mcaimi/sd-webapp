#!/usr/bin/env python

"""
    Utility functions

    - Spherical Tensor Interpolation
    - Linear Tensor Interpolation
    - Merge Tool
"""

try:
    import torch
    import torch.nn.functional as F
    from libs.globals.vars import MergeMethod, RANDOM_BIT_LENGTH
    from libs.shared.exceptions import MergeError
    import random
except Exception as e:
    print(f"funcs.py: Raised Exception: {e}")

# get random seed
def get_random_seed(seed_len: int = RANDOM_BIT_LENGTH) -> int:
    return random.getrandbits(seed_len)

# spherical interpolation function
def slerp(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors"""
    # Flatten tensors for calculation
    # n-dim to 1-dim size, save original shape
    original_shape = t0.shape
    original_target_shape = t1.shape

    assert original_shape == original_target_shape

    t0_flat = t0.flatten()
    t1_flat = t1.flatten()

    # Normalize vectors with respect to vector norm
    # calculate L(p) norm. default P=2, so this calculates L2 Norm on tensors
    t0_norm = F.normalize(t0_flat, dim=0)
    t1_norm = F.normalize(t1_flat, dim=0)

    # Calculate angle between vectors
    dot_product = torch.dot(t0_norm, t1_norm)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)

    # Handle parallel vectors
    if omega.abs() < 1e-6:
        return t0 * (1 - alpha) + t1 * alpha

    # Spherical interpolation
    sin_omega = torch.sin(omega)
    result = (torch.sin((1 - alpha) * omega) / sin_omega) * t0_flat + (torch.sin(alpha * omega) / sin_omega) * t1_flat

    # return interpolated tensor with original shape
    return result.reshape(original_shape)

# linear tensor interpolation
def linear(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """ linear interpolation between two tensors """

    # sanity check
    assert t0.shape == t1.shape

    # perform linear interpolation
    result: torch.Tensor = (t0 * alpha) + (t1 * (1 - alpha))

    # return result
    return result

def merge_tensors(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    method: MergeMethod,
    alpha: float,
    **kwargs,
) -> torch.Tensor:
    """Merge two tensors using specified method"""
    # Ensure consistent dtypes
    original_dtype = tensor_a.dtype
    tensor_a = tensor_a.to(torch.float32)
    tensor_b = tensor_b.to(torch.float32)

    if method == MergeMethod.LINEAR:
        result = linear(tensor_a, tensor_b, alpha)
    elif method == MergeMethod.SLERP:
        result = slerp(tensor_a, tensor_b, alpha)
    elif method == MergeMethod.ADDITIVE:
        # Add the difference from base model
        base_tensor = kwargs.get("base_tensor", tensor_a)
        diff = tensor_b - base_tensor
        result = tensor_a + diff * alpha
    elif method == MergeMethod.SUBTRACT:
        # Subtract features from base model
        base_tensor = kwargs.get("base_tensor", tensor_a)
        diff = tensor_b - base_tensor
        result = tensor_a - diff * alpha
    else:
        raise MergeError(f"Unsupported merge method: {method}")

    return result.to(original_dtype)