"""
Device abstraction layer for AirLLM to support both CUDA and ROCm.
This module provides utilities for detecting and working with different GPU backends.
"""

import torch
import warnings
from typing import Optional, Union


def is_rocm_available() -> bool:
    """
    Check if ROCm is available on the system.
    
    Returns:
        bool: True if ROCm is available, False otherwise
    """
    try:
        # Check if torch was built with ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return torch.cuda.is_available()
        return False
    except Exception:
        return False


def is_cuda_available() -> bool:
    """
    Check if CUDA (NVIDIA GPU) is available on the system.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        if torch.cuda.is_available():
            # Check if it's ROCm or CUDA
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return False  # It's ROCm, not CUDA
            return True
        return False
    except Exception:
        return False


def get_device_type() -> str:
    """
    Get the type of GPU device available.
    
    Returns:
        str: 'rocm', 'cuda', or 'cpu'
    """
    if is_rocm_available():
        return 'rocm'
    elif is_cuda_available():
        return 'cuda'
    else:
        return 'cpu'


def get_device_name(device_id: int = 0) -> str:
    """
    Get the name of the GPU device.
    
    Args:
        device_id: Device ID to query (default: 0)
        
    Returns:
        str: Name of the device
    """
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device_id)
        return "CPU"
    except Exception:
        return "Unknown"


def normalize_device_string(device: Union[str, torch.device]) -> str:
    """
    Normalize device string to handle both CUDA and ROCm.
    
    Args:
        device: Device string or torch.device object
        
    Returns:
        str: Normalized device string (e.g., 'cuda:0')
    """
    if isinstance(device, torch.device):
        device = str(device)
    
    # For ROCm, we use 'cuda' prefix since PyTorch ROCm uses cuda interface
    if device.startswith('hip'):
        device = device.replace('hip', 'cuda')
    
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get a torch.device object, handling ROCm/CUDA differences.
    
    Args:
        device: Device specification (e.g., 'cuda:0', 'cuda', 'cpu')
                If None, will auto-detect available device
        
    Returns:
        torch.device: A torch device object
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    
    device_str = normalize_device_string(device)
    return torch.device(device_str)


def empty_cache():
    """
    Empty the GPU cache. Works for both CUDA and ROCm.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: Optional[Union[str, int]] = None):
    """
    Synchronize the GPU. Works for both CUDA and ROCm.
    
    Args:
        device: Device to synchronize (optional)
    """
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()


def get_device_stream(device: Union[str, torch.device]):
    """
    Create a CUDA/ROCm stream for the specified device.
    
    Args:
        device: Device string or torch.device
        
    Returns:
        Stream object compatible with both CUDA and ROCm
    """
    device_obj = get_device(device)
    if device_obj.type == 'cuda':
        return torch.cuda.Stream()
    return None


def to_device(tensor: torch.Tensor, device: Union[str, torch.device], 
              dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Move tensor to device, handling ROCm/CUDA compatibility.
    
    Args:
        tensor: Input tensor
        device: Target device
        dtype: Optional target dtype
        
    Returns:
        torch.Tensor: Tensor on the target device
    """
    device_obj = get_device(device)
    
    if dtype is not None:
        return tensor.to(device=device_obj, dtype=dtype)
    return tensor.to(device=device_obj)


def print_device_info():
    """
    Print information about available GPU devices.
    """
    device_type = get_device_type()
    print(f"Device type: {device_type}")
    
    if device_type == 'rocm':
        print("ROCm detected")
        if hasattr(torch.version, 'hip'):
            print(f"ROCm version: {torch.version.hip}")
    elif device_type == 'cuda':
        print("CUDA detected")
        print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")


# Compatibility aliases for ease of migration
def cuda_is_available() -> bool:
    """
    Compatibility function that returns True for both CUDA and ROCm.
    This allows existing code using torch.cuda.is_available() to work with ROCm.
    
    Returns:
        bool: True if either CUDA or ROCm is available
    """
    return torch.cuda.is_available()
