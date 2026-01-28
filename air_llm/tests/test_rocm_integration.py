"""
Integration tests for ROCm compatibility.

These tests verify that the device abstraction layer integrates
properly with the AirLLM model classes.
"""

import unittest
import torch
from unittest.mock import patch, MagicMock

# Import from airllm
from ..airllm.device_utils import (
    get_device,
    get_device_type,
    get_device_stream
)


class TestROCmIntegration(unittest.TestCase):
    """Integration tests for ROCm support."""
    
    def test_device_initialization(self):
        """Test that device objects can be created."""
        # Test CPU device
        device_cpu = get_device('cpu')
        self.assertEqual(device_cpu.type, 'cpu')
        
        # Test default device (should not crash)
        device_default = get_device()
        self.assertIsInstance(device_default, torch.device)
    
    def test_stream_creation(self):
        """Test that streams can be created appropriately."""
        # CPU should return None
        stream_cpu = get_device_stream('cpu')
        self.assertIsNone(stream_cpu)
        
        # CUDA/ROCm should return stream if available
        if torch.cuda.is_available():
            device = get_device('cuda:0')
            stream = get_device_stream(device)
            # Should either be a stream or None (if not supported)
            self.assertTrue(stream is None or hasattr(stream, 'wait_stream'))
    
    def test_device_type_detection(self):
        """Test device type is consistently detected."""
        device_type = get_device_type()
        self.assertIn(device_type, ['cpu', 'cuda', 'rocm'])
        
        # Verify consistency with torch.cuda.is_available()
        if device_type in ['cuda', 'rocm']:
            self.assertTrue(torch.cuda.is_available())
        else:
            self.assertFalse(torch.cuda.is_available())
    
    def test_tensor_device_movement(self):
        """Test tensor can be moved between devices."""
        from ..airllm.device_utils import to_device
        
        # Create a test tensor
        tensor = torch.randn(10, 10)
        
        # Move to CPU
        tensor_cpu = to_device(tensor, 'cpu')
        self.assertEqual(tensor_cpu.device.type, 'cpu')
        
        # If GPU available, test GPU transfer
        if torch.cuda.is_available():
            tensor_gpu = to_device(tensor, 'cuda:0')
            self.assertEqual(tensor_gpu.device.type, 'cuda')
            
            # Test with dtype conversion
            tensor_fp16 = to_device(tensor, 'cuda:0', dtype=torch.float16)
            self.assertEqual(tensor_fp16.dtype, torch.float16)
            self.assertEqual(tensor_fp16.device.type, 'cuda')


class TestROCmModelCompatibility(unittest.TestCase):
    """Test compatibility with model initialization."""
    
    def test_imports_work(self):
        """Test that all necessary modules can be imported."""
        try:
            from ..airllm import device_utils
            from ..airllm.airllm_base import AirLLMBaseModel
            from ..airllm.auto_model import AutoModel
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_device_utils_available_in_package(self):
        """Test that device_utils is accessible from the package."""
        try:
            from ..airllm import device_utils
            
            # Check key functions are available
            self.assertTrue(hasattr(device_utils, 'get_device'))
            self.assertTrue(hasattr(device_utils, 'get_device_type'))
            self.assertTrue(hasattr(device_utils, 'print_device_info'))
            self.assertTrue(hasattr(device_utils, 'is_rocm_available'))
            self.assertTrue(hasattr(device_utils, 'is_cuda_available'))
        except ImportError as e:
            self.fail(f"device_utils not accessible: {e}")


class TestROCmBackwardCompatibility(unittest.TestCase):
    """Test that existing code remains compatible."""
    
    def test_cuda_string_still_works(self):
        """Test that 'cuda:0' string still works with ROCm."""
        device = get_device('cuda:0')
        # Should create device successfully regardless of backend
        self.assertIsInstance(device, torch.device)
    
    def test_torch_cuda_api_compatibility(self):
        """Test that torch.cuda API still works."""
        # These should not raise errors
        try:
            available = torch.cuda.is_available()
            self.assertIsInstance(available, bool)
            
            if available:
                count = torch.cuda.device_count()
                self.assertIsInstance(count, int)
                self.assertGreater(count, 0)
        except Exception as e:
            self.fail(f"torch.cuda API compatibility broken: {e}")


if __name__ == '__main__':
    unittest.main()
