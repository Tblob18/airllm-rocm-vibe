"""
Unit tests for device abstraction layer.
"""

import unittest
import torch
from ..airllm.device_utils import (
    is_rocm_available,
    is_cuda_available,
    get_device_type,
    get_device_name,
    normalize_device_string,
    get_device,
    empty_cache,
    to_device,
    cuda_is_available
)


class TestDeviceUtils(unittest.TestCase):
    """Test cases for device utilities"""
    
    def test_get_device_type(self):
        """Test device type detection"""
        device_type = get_device_type()
        self.assertIn(device_type, ['cpu', 'cuda', 'rocm'])
    
    def test_cuda_is_available(self):
        """Test CUDA/ROCm availability check"""
        # Should return True if either CUDA or ROCm is available
        result = cuda_is_available()
        self.assertIsInstance(result, bool)
    
    def test_normalize_device_string(self):
        """Test device string normalization"""
        # Test CUDA device string
        self.assertEqual(normalize_device_string('cuda:0'), 'cuda:0')
        self.assertEqual(normalize_device_string('cuda'), 'cuda')
        
        # Test HIP/ROCm device string normalization
        self.assertEqual(normalize_device_string('hip:0'), 'cuda:0')
        
        # Test torch.device object
        device_obj = torch.device('cuda:0')
        self.assertEqual(normalize_device_string(device_obj), 'cuda:0')
    
    def test_get_device(self):
        """Test device object creation"""
        # Test default device
        device = get_device()
        self.assertIsInstance(device, torch.device)
        
        # Test explicit CPU device
        device_cpu = get_device('cpu')
        self.assertEqual(device_cpu.type, 'cpu')
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            device_cuda = get_device('cuda:0')
            self.assertEqual(device_cuda.type, 'cuda')
    
    def test_to_device(self):
        """Test tensor device transfer"""
        tensor = torch.randn(3, 3)
        
        # Test CPU transfer
        tensor_cpu = to_device(tensor, 'cpu')
        self.assertEqual(tensor_cpu.device.type, 'cpu')
        
        # Test GPU transfer if available
        if torch.cuda.is_available():
            tensor_gpu = to_device(tensor, 'cuda:0')
            self.assertEqual(tensor_gpu.device.type, 'cuda')
            
            # Test with dtype
            tensor_gpu_fp16 = to_device(tensor, 'cuda:0', dtype=torch.float16)
            self.assertEqual(tensor_gpu_fp16.dtype, torch.float16)
    
    def test_empty_cache(self):
        """Test cache emptying"""
        # Should not raise any errors
        try:
            empty_cache()
        except Exception as e:
            self.fail(f"empty_cache raised an exception: {e}")
    
    def test_get_device_name(self):
        """Test device name retrieval"""
        name = get_device_name(0)
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)


class TestROCmDetection(unittest.TestCase):
    """Test cases specifically for ROCm detection"""
    
    def test_rocm_cuda_mutual_exclusivity(self):
        """Test that ROCm and CUDA detection are mutually exclusive"""
        rocm = is_rocm_available()
        cuda = is_cuda_available()
        
        # Both should not be True at the same time
        if torch.cuda.is_available():
            # Either ROCm or CUDA should be available, but not both
            self.assertTrue(rocm ^ cuda or (not rocm and not cuda))
    
    def test_device_type_consistency(self):
        """Test device type is consistent with availability checks"""
        device_type = get_device_type()
        
        if device_type == 'rocm':
            self.assertTrue(is_rocm_available())
            self.assertFalse(is_cuda_available())
        elif device_type == 'cuda':
            self.assertTrue(is_cuda_available())
            self.assertFalse(is_rocm_available())
        elif device_type == 'cpu':
            self.assertFalse(is_rocm_available())
            self.assertFalse(is_cuda_available())


if __name__ == '__main__':
    unittest.main()
