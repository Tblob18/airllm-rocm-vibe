# AirLLM ROCm Compatibility Guide

## Introduction

AirLLM now supports AMD GPUs through ROCm (Radeon Open Compute), enabling efficient large language model inference on AMD hardware. This guide provides comprehensive information on setting up and using AirLLM with AMD GPUs.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Benchmarking](#benchmarking)
- [Known Issues](#known-issues)
- [Community and Support](#community-and-support)

## System Requirements

### Hardware Requirements

- **Supported AMD GPUs**: 
  - AMD Radeon RX 6000 series (RDNA 2)
  - AMD Radeon RX 7000 series (RDNA 3)
  - AMD Instinct MI series (CDNA 2, CDNA 3)
  - Other GCN 4.0+ and later architectures

- **Minimum VRAM**: 4GB (for smaller models with quantization)
- **Recommended VRAM**: 8GB or more
- **System RAM**: 16GB minimum, 32GB+ recommended

### Software Requirements

- **Operating System**: 
  - Ubuntu 20.04/22.04 LTS (recommended)
  - RHEL 8.x/9.x
  - Other Linux distributions with ROCm support

- **ROCm Version**: 5.4.0 or later (5.7+ recommended)
- **Python**: 3.8+
- **PyTorch**: ROCm-enabled build

## Installation

### Step 1: Install ROCm

Follow AMD's official ROCm installation guide for your operating system:

**Ubuntu 22.04:**

```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt install ./amdgpu-install_5.7.50700-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to render and video groups
sudo usermod -a -G render,video $LOGNAME

# Reboot system
sudo reboot
```

**Verify ROCm installation:**

```bash
rocm-smi
```

You should see your AMD GPU(s) listed.

### Step 2: Install PyTorch with ROCm Support

```bash
# For ROCm 5.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verify PyTorch ROCm installation
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'ROCm version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')"
```

### Step 3: Install AirLLM

```bash
# Install AirLLM
pip install airllm

# Or install from source for latest features
git clone https://github.com/yourusername/airllm-rocm-vibe.git
cd airllm-rocm-vibe
pip install -e ./air_llm
```

### Step 4: Install ROCm-Specific Requirements

```bash
# Install additional dependencies
pip install -r requirements-rocm.txt
```

**Note on bitsandbytes**: The standard bitsandbytes library may not work with ROCm. For compression/quantization support, you may need to:
- Use a ROCm-compatible fork (e.g., `bitsandbytes-rocm`)
- Build bitsandbytes from source with ROCm support
- Or use AirLLM without compression for now

## Quick Start

### Basic Inference Example

```python
from airllm import AutoModel

# Initialize model with ROCm
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# Prepare input
input_text = ['What is the capital of United States?']
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=128, 
    padding=False
)

# Run inference on AMD GPU
generation_output = model.generate(
    input_tokens['input_ids'].cuda(),  # .cuda() works with ROCm
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

# Decode output
output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

### Check Device Information

```python
from airllm.device_utils import print_device_info, get_device_type

# Print detailed device information
print_device_info()

# Check device type
device_type = get_device_type()
print(f"Running on: {device_type}")  # Should print 'rocm'
```

## Configuration

### Device Selection

AirLLM automatically detects ROCm GPUs. You can specify a device just like with CUDA:

```python
# Use default GPU (ROCm:0)
model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    device="cuda:0"  # 'cuda' interface works with ROCm
)

# Use specific GPU
model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    device="cuda:1"  # Second ROCm GPU
)

# Use CPU
model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    device="cpu"
)
```

### Memory Management

AirLLM's memory-efficient architecture is particularly beneficial for AMD GPUs:

```python
# For 4GB VRAM (requires model compression)
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    compression='4bit',  # Note: May require ROCm-compatible bitsandbytes
    dtype=torch.float16
)

# For 8GB VRAM
model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    dtype=torch.float16
)

# For 16GB+ VRAM
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    dtype=torch.float16,
    max_seq_len=2048
)
```

### Layer Sharding

Control where split model layers are saved:

```python
model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    layer_shards_saving_path="/path/to/custom/location",
    delete_original=True  # Save disk space
)
```

## Performance Considerations

### Optimization Tips

1. **Use float16**: ROCm performs well with half-precision
   ```python
   model = AutoModel.from_pretrained(model_id, dtype=torch.float16)
   ```

2. **Batch Processing**: Process multiple inputs when possible
   ```python
   input_texts = [
       'What is AI?',
       'Explain quantum computing.',
       'What is machine learning?'
   ]
   # Process together for better throughput
   ```

3. **Prefetching**: Enable prefetching to overlap computation and data loading
   ```python
   model = AutoModel.from_pretrained(
       model_id,
       prefetching=True  # Default is True
   )
   ```

4. **Profiling**: Monitor performance to identify bottlenecks
   ```python
   model = AutoModel.from_pretrained(
       model_id,
       profiling_mode=True  # Enable timing information
   )
   ```

### Expected Performance

Performance varies based on GPU model, VRAM, and model size:

| GPU Model | VRAM | Model Size | Tokens/sec (approx) |
|-----------|------|------------|---------------------|
| RX 6600 XT | 8GB | Llama-2-7B | 15-20 |
| RX 6800 XT | 16GB | Llama-2-13B | 12-18 |
| RX 7900 XTX | 24GB | Llama-2-70B | 8-12 |
| MI250 | 128GB | Llama-2-70B | 25-35 |

*Note: These are approximate values. Actual performance depends on configuration.*

## Troubleshooting

### Common Issues and Solutions

#### 1. ROCm Not Detected

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# Check ROCm installation
rocm-smi

# Verify GPU is visible
ls /dev/kfd /dev/dri

# Check user permissions
groups  # Should include 'render' and 'video'

# If not, add user to groups
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

#### 2. PyTorch Not Built with ROCm

**Symptom**: PyTorch installed but doesn't detect GPU

**Solution**:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install ROCm-specific build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

#### 3. Out of Memory Errors

**Solutions**:
- Use smaller batch sizes
- Enable compression (if ROCm-compatible bitsandbytes available)
- Reduce `max_seq_len`
- Use float16 instead of float32

```python
# Lower memory configuration
model = AutoModel.from_pretrained(
    model_id,
    dtype=torch.float16,
    max_seq_len=512,  # Reduce from default
)
```

#### 4. Slow Inference Speed

**Solutions**:
- Ensure ROCm drivers are up to date
- Check system isn't thermal throttling: `rocm-smi`
- Enable prefetching: `prefetching=True`
- Use profiling mode to identify bottlenecks

```python
model = AutoModel.from_pretrained(
    model_id,
    profiling_mode=True,
    prefetching=True
)
```

#### 5. Compression/Quantization Not Working

**Symptom**: Errors when using `compression='4bit'` or `compression='8bit'`

**Solution**: Standard bitsandbytes doesn't support ROCm. Either:
- Skip compression for now
- Use a ROCm-compatible fork:
  ```bash
  pip uninstall bitsandbytes
  pip install git+https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6.git
  ```

#### 6. HIP/CUDA Compatibility Issues

**Symptom**: Errors mentioning "HIP" or CUDA compatibility

**Solution**: AirLLM handles this automatically through device abstraction. If issues persist:
```python
from airllm.device_utils import print_device_info
print_device_info()  # Check device detection
```

### Getting Help

If you encounter issues not covered here:

1. Check existing GitHub issues
2. Review ROCm documentation: https://rocm.docs.amd.com/
3. Join the AirLLM community (Discord/Forums)
4. Create a detailed bug report with:
   - System info (`rocm-smi`, `python --version`)
   - PyTorch version and ROCm detection output
   - Full error traceback
   - Minimal reproduction code

## Benchmarking

### Running Benchmarks

Create a benchmark script:

```python
import time
import torch
from airllm import AutoModel

def benchmark_inference(model_id, num_runs=10):
    """Benchmark inference speed on ROCm."""
    model = AutoModel.from_pretrained(model_id)
    
    input_text = ["What is the meaning of life?"]
    input_tokens = model.tokenizer(
        input_text,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )
    
    # Warmup
    _ = model.generate(
        input_tokens['input_ids'].cuda(),
        max_new_tokens=20,
        use_cache=True
    )
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.generate(
            input_tokens['input_ids'].cuda(),
            max_new_tokens=20,
            use_cache=True
        )
        torch.cuda.synchronize()  # Wait for GPU to finish
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = 20 / avg_time
    
    print(f"Average time: {avg_time:.3f}s")
    print(f"Tokens/second: {tokens_per_sec:.2f}")
    
    return avg_time, tokens_per_sec

# Run benchmark
benchmark_inference("garage-bAInd/Platypus2-7B")
```

### Comparing CUDA vs ROCm

If you have access to both NVIDIA and AMD hardware:

```python
from airllm.device_utils import get_device_type, print_device_info

print_device_info()
device_type = get_device_type()
print(f"Running on: {device_type}")

# Run same benchmark on both systems and compare
```

## Known Issues

### Current Limitations

1. **Compression Support**: Standard bitsandbytes doesn't support ROCm
   - **Workaround**: Use ROCm-compatible forks or skip compression
   
2. **Flash Attention**: May not be available on all ROCm configurations
   - **Workaround**: AirLLM falls back to standard attention automatically

3. **Some Models**: Certain model architectures may have limited testing on ROCm
   - **Status**: Core models (Llama, Mistral, QWen) are well-tested

### Reporting Issues

When reporting ROCm-specific issues:

1. Include ROCm version: `rocm-smi --showversion`
2. Include PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Include GPU model: `rocm-smi`
4. Include device detection output:
   ```python
   from airllm.device_utils import print_device_info
   print_device_info()
   ```

## Community and Support

### Resources

- **AirLLM Documentation**: [Main README](../README.md)
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/
- **AMD GPU Community**: https://community.amd.com/

### Contributing

We welcome contributions to improve ROCm support:

1. Testing on different AMD GPU models
2. Performance optimizations
3. Documentation improvements
4. Bug reports and fixes

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Acknowledgements

ROCm support in AirLLM builds on:
- AMD's ROCm platform
- PyTorch's ROCm backend
- Community testing and feedback

## Next Steps

1. Try the [Quick Start example](#quick-start)
2. Experiment with different models and configurations
3. Run benchmarks on your hardware
4. Share your results with the community
5. Report any issues or contribute improvements

---

**Note**: ROCm support is actively maintained. Check back for updates and improvements!
