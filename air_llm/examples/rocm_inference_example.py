#!/usr/bin/env python3
"""
AirLLM ROCm Inference Example

This script demonstrates how to use AirLLM with AMD GPUs via ROCm.
It includes device detection, model loading, and inference.

Usage:
    python rocm_inference_example.py [--model MODEL_ID] [--prompt PROMPT]

Examples:
    # Use default model and prompt
    python rocm_inference_example.py
    
    # Use custom model
    python rocm_inference_example.py --model "meta-llama/Llama-2-7b-hf"
    
    # Use custom prompt
    python rocm_inference_example.py --prompt "Explain quantum computing"
"""

import argparse
import sys
import time
import torch

try:
    from airllm import AutoModel
    from airllm.device_utils import (
        get_device_type,
        get_device_name,
        print_device_info,
        is_rocm_available,
        is_cuda_available
    )
except ImportError as e:
    print(f"Error: Failed to import AirLLM. Make sure it's installed.")
    print(f"Install with: pip install airllm")
    print(f"Error details: {e}")
    sys.exit(1)


def check_rocm_setup():
    """Check if ROCm is properly set up."""
    print("=" * 60)
    print("ROCm Setup Check")
    print("=" * 60)
    
    # Check device type
    device_type = get_device_type()
    print(f"Detected device type: {device_type}")
    
    if device_type == 'rocm':
        print("✓ ROCm detected successfully!")
        if hasattr(torch.version, 'hip'):
            print(f"✓ ROCm version: {torch.version.hip}")
    elif device_type == 'cuda':
        print("⚠ CUDA detected (NVIDIA GPU)")
        print("  This example is designed for ROCm (AMD GPU)")
        print("  But it will still work with CUDA GPUs")
    else:
        print("⚠ No GPU detected, will use CPU")
        print("  For ROCm support, ensure:")
        print("  1. ROCm drivers are installed")
        print("  2. PyTorch is built with ROCm support")
        print("  3. User is in 'render' and 'video' groups")
    
    # Print detailed device info
    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)
    print_device_info()
    print("=" * 60 + "\n")
    
    return device_type


def run_inference(model_id, prompt, device="cuda:0", max_new_tokens=50):
    """
    Run inference with AirLLM on ROCm.
    
    Args:
        model_id: HuggingFace model ID or local path
        prompt: Input prompt for generation
        device: Device to use (default: "cuda:0")
        max_new_tokens: Maximum number of tokens to generate
    """
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")
    
    start_time = time.time()
    
    try:
        # Initialize model
        model = AutoModel.from_pretrained(
            model_id,
            device=device,
            dtype=torch.float16,  # Use half precision for efficiency
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Prepare input
    print(f"\nInput prompt: {prompt}")
    print("-" * 60)
    
    input_tokens = model.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )
    
    # Run inference
    print("Generating response...")
    generation_start = time.time()
    
    try:
        generation_output = model.generate(
            input_tokens['input_ids'].cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True
        )
        
        generation_time = time.time() - generation_start
        
        # Decode output
        output = model.tokenizer.decode(generation_output.sequences[0])
        
        print(f"\nGenerated output:")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
        # Performance metrics
        tokens_generated = len(generation_output.sequences[0]) - len(input_tokens['input_ids'][0])
        tokens_per_sec = tokens_generated / generation_time
        
        print(f"\nPerformance Metrics:")
        print(f"  Generation time: {generation_time:.2f} seconds")
        print(f"  Tokens generated: {tokens_generated}")
        print(f"  Tokens per second: {tokens_per_sec:.2f}")
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="AirLLM ROCm Inference Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rocm_inference_example.py
  python rocm_inference_example.py --model "meta-llama/Llama-2-7b-hf"
  python rocm_inference_example.py --prompt "What is the capital of France?"
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='garage-bAInd/Platypus2-7B',
        help='HuggingFace model ID or local path (default: garage-bAInd/Platypus2-7B)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='What is the capital of United States?',
        help='Input prompt for generation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=50,
        help='Maximum number of tokens to generate (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Check ROCm setup
    device_type = check_rocm_setup()
    
    # Confirm to proceed if not ROCm
    if device_type != 'rocm' and device_type != 'cuda':
        response = input("\nNo GPU detected. Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            sys.exit(0)
    
    # Run inference
    print("\nStarting inference...")
    print("=" * 60 + "\n")
    
    run_inference(
        model_id=args.model,
        prompt=args.prompt,
        device=args.device,
        max_new_tokens=args.max_tokens
    )
    
    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
