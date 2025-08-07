#!/usr/bin/env python3
"""
Performance Comparison: Original vs Optimized Pipeline
Compare processing time and memory usage for large datasets
"""

import sys
import time
import psutil
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_pipeline import DataPipeline
from optimize_pipeline import PipelineOptimizer

def monitor_performance(func, *args, **kwargs):
    """Monitor performance of a function."""
    start_time = time.time()
    start_memory = psutil.virtual_memory().percent
    
    # Run function
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().percent
    
    return {
        'execution_time': end_time - start_time,
        'memory_usage_start': start_memory,
        'memory_usage_end': end_memory,
        'memory_increase': end_memory - start_memory,
        'result': result
    }

def run_original_pipeline(sample_size):
    """Run the original pipeline."""
    print("🔄 Running original pipeline...")
    data_pipeline = DataPipeline()
    return data_pipeline.run_complete_pipeline(sample_size=sample_size)

def run_optimized_pipeline(sample_size):
    """Run the optimized pipeline."""
    print("🚀 Running optimized pipeline...")
    optimizer = PipelineOptimizer()
    return optimizer.run_optimized_pipeline(sample_size=sample_size)

def compare_performance(sample_sizes=[100, 500, 1000]):
    """Compare performance across different sample sizes."""
    print("📊 Performance Comparison: Original vs Optimized Pipeline")
    print("="*70)
    
    results = {}
    
    for sample_size in sample_sizes:
        print(f"\n🔍 Testing with {sample_size} patients...")
        
        # Test original pipeline
        try:
            original_perf = monitor_performance(run_original_pipeline, sample_size)
            print(f"✅ Original pipeline completed")
        except Exception as e:
            print(f"❌ Original pipeline failed: {e}")
            original_perf = None
        
        # Test optimized pipeline
        try:
            optimized_perf = monitor_performance(run_optimized_pipeline, sample_size)
            print(f"✅ Optimized pipeline completed")
        except Exception as e:
            print(f"❌ Optimized pipeline failed: {e}")
            optimized_perf = None
        
        # Store results
        results[sample_size] = {
            'original': original_perf,
            'optimized': optimized_perf
        }
        
        # Print comparison
        if original_perf and optimized_perf:
            time_improvement = ((original_perf['execution_time'] - optimized_perf['execution_time']) / 
                              original_perf['execution_time']) * 100
            memory_improvement = original_perf['memory_increase'] - optimized_perf['memory_increase']
            
            print(f"📈 Performance Comparison for {sample_size} patients:")
            print(f"  ⏱️  Time: {original_perf['execution_time']:.2f}s → {optimized_perf['execution_time']:.2f}s ({time_improvement:+.1f}%)")
            print(f"  💾 Memory: {original_perf['memory_increase']:+.1f}% → {optimized_perf['memory_increase']:+.1f}% ({memory_improvement:+.1f}%)")
        
        # Cleanup
        import gc
        gc.collect()
    
    return results

def print_summary(results):
    """Print performance summary."""
    print("\n" + "="*70)
    print("📊 PERFORMANCE SUMMARY")
    print("="*70)
    
    for sample_size, perf_data in results.items():
        if perf_data['original'] and perf_data['optimized']:
            orig = perf_data['original']
            opt = perf_data['optimized']
            
            time_improvement = ((orig['execution_time'] - opt['execution_time']) / orig['execution_time']) * 100
            memory_improvement = orig['memory_increase'] - opt['memory_increase']
            
            print(f"\n📊 {sample_size} Patients:")
            print(f"  ⏱️  Execution Time:")
            print(f"     Original: {orig['execution_time']:.2f}s")
            print(f"     Optimized: {opt['execution_time']:.2f}s")
            print(f"     Improvement: {time_improvement:+.1f}%")
            print(f"  💾 Memory Usage:")
            print(f"     Original: {orig['memory_increase']:+.1f}%")
            print(f"     Optimized: {opt['memory_increase']:+.1f}%")
            print(f"     Improvement: {memory_improvement:+.1f}%")

def main():
    """Main comparison function."""
    print("🚀 COVID-19 TFT Pipeline Performance Comparison")
    print("="*70)
    
    # Test with different sample sizes
    sample_sizes = [100, 500, 1000]  # Start small for testing
    
    # Run comparison
    results = compare_performance(sample_sizes)
    
    # Print summary
    print_summary(results)
    
    # Recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("="*50)
    print("✅ Use optimized pipeline for datasets > 500 patients")
    print("✅ Enable parallel processing for feature engineering")
    print("✅ Use chunked processing to manage memory")
    print("✅ Increase batch sizes for better GPU utilization")
    print("✅ Enable caching for repeated operations")
    print("✅ Monitor memory usage and cleanup regularly")
    
    return results

if __name__ == "__main__":
    results = main() 