#!/usr/bin/env python3
"""
AirLLM ROCm Performance Benchmarking Script

This script benchmarks inference performance on AMD GPUs with ROCm.
It measures throughput, latency, and memory usage for different configurations.

Usage:
    python rocm_benchmark.py [--model MODEL_ID] [--runs N] [--tokens N]

Examples:
    # Quick benchmark with defaults
    python rocm_benchmark.py
    
    # Benchmark specific model
    python rocm_benchmark.py --model "meta-llama/Llama-2-7b-hf"
    
    # More runs for accuracy
    python rocm_benchmark.py --runs 20 --tokens 100
"""

import argparse
import sys
import time
import json
from typing import Dict, List
import torch

try:
    from airllm import AutoModel
    from airllm.device_utils import (
        get_device_type,
        get_device_name,
        print_device_info,
        is_rocm_available,
        synchronize
    )
except ImportError as e:
    print(f"Error: Failed to import AirLLM. Make sure it's installed.")
    print(f"Install with: pip install airllm")
    sys.exit(1)


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        self.results = []
        self.system_info = {}
    
    def add_result(self, config: Dict, metrics: Dict):
        """Add a benchmark result."""
        self.results.append({
            'config': config,
            'metrics': metrics
        })
    
    def set_system_info(self, info: Dict):
        """Set system information."""
        self.system_info = info
    
    def get_summary(self) -> Dict:
        """Get summary of all results."""
        if not self.results:
            return {}
        
        avg_tokens_per_sec = sum(r['metrics']['tokens_per_sec'] for r in self.results) / len(self.results)
        avg_latency = sum(r['metrics']['avg_latency'] for r in self.results) / len(self.results)
        
        return {
            'num_runs': len(self.results),
            'avg_tokens_per_sec': avg_tokens_per_sec,
            'avg_latency': avg_latency,
            'min_tokens_per_sec': min(r['metrics']['tokens_per_sec'] for r in self.results),
            'max_tokens_per_sec': max(r['metrics']['tokens_per_sec'] for r in self.results),
        }
    
    def print_summary(self):
        """Print a formatted summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        print("\nSystem Information:")
        for key, value in self.system_info.items():
            print(f"  {key}: {value}")
        
        if not self.results:
            print("\nNo benchmark results available.")
            return
        
        summary = self.get_summary()
        print(f"\nResults from {summary['num_runs']} runs:")
        print(f"  Average tokens/sec: {summary['avg_tokens_per_sec']:.2f}")
        print(f"  Average latency: {summary['avg_latency']:.4f}s")
        print(f"  Min tokens/sec: {summary['min_tokens_per_sec']:.2f}")
        print(f"  Max tokens/sec: {summary['max_tokens_per_sec']:.2f}")
        
        print("\nPer-run details:")
        for i, result in enumerate(self.results, 1):
            metrics = result['metrics']
            print(f"  Run {i}: {metrics['tokens_per_sec']:.2f} tok/s, "
                  f"{metrics['avg_latency']:.4f}s latency, "
                  f"{metrics['total_tokens']} tokens")
        
        print("=" * 70)
    
    def save_to_file(self, filename: str):
        """Save results to JSON file."""
        data = {
            'system_info': self.system_info,
            'summary': self.get_summary(),
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def get_system_info() -> Dict:
    """Gather system information."""
    info = {
        'device_type': get_device_type(),
        'pytorch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = get_device_name(0)
        
        props = torch.cuda.get_device_properties(0)
        info['gpu_memory_gb'] = props.total_memory / (1024**3)
        
        if hasattr(torch.version, 'hip') and torch.version.hip:
            info['rocm_version'] = torch.version.hip
        elif hasattr(torch.version, 'cuda'):
            info['cuda_version'] = torch.version.cuda
    
    return info


def benchmark_model(model_id: str, num_runs: int = 10, max_new_tokens: int = 50,
                    device: str = "cuda:0", warmup_runs: int = 2) -> BenchmarkResults:
    """
    Benchmark a model's inference performance.
    
    Args:
        model_id: HuggingFace model ID or local path
        num_runs: Number of benchmark runs
        max_new_tokens: Number of tokens to generate per run
        device: Device to use
        warmup_runs: Number of warmup runs (not counted)
    
    Returns:
        BenchmarkResults object
    """
    results = BenchmarkResults()
    
    # Gather system info
    system_info = get_system_info()
    results.set_system_info(system_info)
    
    print(f"\nBenchmark Configuration:")
    print(f"  Model: {model_id}")
    print(f"  Device: {device}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Benchmark runs: {num_runs}")
    print(f"  Tokens per run: {max_new_tokens}")
    
    # Load model
    print(f"\nLoading model...")
    load_start = time.time()
    
    try:
        model = AutoModel.from_pretrained(
            model_id,
            device=device,
            dtype=torch.float16,
            profiling_mode=False
        )
        load_time = time.time() - load_start
        print(f"✓ Model loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return results
    
    # Prepare input
    test_prompt = "What is artificial intelligence and how does it work?"
    input_tokens = model.tokenizer(
        [test_prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )
    
    input_ids = input_tokens['input_ids'].cuda()
    
    config = {
        'model_id': model_id,
        'device': device,
        'max_new_tokens': max_new_tokens,
        'dtype': 'float16'
    }
    
    # Warmup runs
    print(f"\nRunning {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        try:
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
            if torch.cuda.is_available():
                synchronize()
        except Exception as e:
            print(f"Error in warmup run {i+1}: {e}")
            return results
    
    print("✓ Warmup complete")
    
    # Benchmark runs
    print(f"\nRunning {num_runs} benchmark iterations...")
    
    for run_num in range(num_runs):
        try:
            start_time = time.time()
            
            generation_output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True
            )
            
            if torch.cuda.is_available():
                synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            elapsed_time = end_time - start_time
            tokens_generated = len(generation_output.sequences[0]) - len(input_ids[0])
            tokens_per_sec = tokens_generated / elapsed_time
            
            metrics = {
                'run_num': run_num + 1,
                'total_time': elapsed_time,
                'avg_latency': elapsed_time / tokens_generated,
                'tokens_per_sec': tokens_per_sec,
                'total_tokens': tokens_generated
            }
            
            results.add_result(config, metrics)
            
            print(f"  Run {run_num + 1}/{num_runs}: "
                  f"{tokens_per_sec:.2f} tok/s, "
                  f"{elapsed_time:.3f}s total")
            
        except Exception as e:
            print(f"✗ Error in run {run_num + 1}: {e}")
            continue
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="AirLLM ROCm Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='garage-bAInd/Platypus2-7B',
        help='HuggingFace model ID or local path'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=10,
        help='Number of benchmark runs (default: 10)'
    )
    
    parser.add_argument(
        '--tokens',
        type=int,
        default=50,
        help='Number of tokens to generate per run (default: 50)'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Number of warmup runs (default: 2)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print("=" * 70)
    print("AirLLM ROCm Performance Benchmark")
    print("=" * 70)
    print_device_info()
    print("=" * 70)
    
    # Check for GPU
    device_type = get_device_type()
    if device_type == 'cpu':
        print("\n⚠ Warning: No GPU detected. Benchmarking on CPU.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            sys.exit(0)
    elif device_type == 'rocm':
        print("\n✓ ROCm GPU detected. Proceeding with benchmark.")
    else:
        print(f"\n✓ {device_type.upper()} GPU detected. Proceeding with benchmark.")
    
    # Run benchmark
    results = benchmark_model(
        model_id=args.model,
        num_runs=args.runs,
        max_new_tokens=args.tokens,
        device=args.device,
        warmup_runs=args.warmup
    )
    
    # Print results
    results.print_summary()
    
    # Save to file if requested
    if args.output:
        results.save_to_file(args.output)
    
    print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    main()
