#!/usr/bin/env python3
"""
Optimized COVID-19 TFT Training - Simple Version
Based on the working train_with_progress_fixed.py but with optimizations for 12,000+ patients
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import gc
import psutil
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.tft_model import COVID19TFTModel
from training.trainer import COVID19Trainer
from evaluation.metrics import COVID19Metrics
from data.data_pipeline import DataPipeline
from data.tft_formatter import TFTDataset

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optimized_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def monitor_memory():
    """Monitor current memory usage."""
    return psutil.virtual_memory().percent

def check_memory_warning(threshold: float = 90.0):
    """Check if memory usage is above threshold."""
    memory_usage = monitor_memory()
    if memory_usage > threshold:
        print(f"⚠️ High memory usage: {memory_usage:.1f}%")
        return True
    return False

def run_data_pipeline_optimized(sample_size: int = 12000):
    """
    Run data pipeline with optimizations for large datasets.
    
    Args:
        sample_size (int): Number of patients to process
        
    Returns:
        Dict: Processed datasets or None if failed
    """
    print(f"🔄 Starting optimized data pipeline for {sample_size} patients...")
    
    start_time = time.time()
    start_memory = monitor_memory()
    
    try:
        # Use the existing DataPipeline but with optimizations
        data_pipeline = DataPipeline()
        
        # For large datasets, we'll process in chunks
        if sample_size > 1000:
            print("📊 Large dataset detected - using chunked processing...")
            
            # Process in chunks of 1000 patients
            chunk_size = 1000
            all_train_patients = []
            all_val_patients = []
            all_test_patients = []
            
            for i in range(0, sample_size, chunk_size):
                current_chunk_size = min(chunk_size, sample_size - i)
                print(f"🔄 Processing chunk {i//chunk_size + 1}/{(sample_size + chunk_size - 1)//chunk_size} ({current_chunk_size} patients)")
                
                # Run pipeline for this chunk
                chunk_results = data_pipeline.run_complete_pipeline(sample_size=current_chunk_size, start_index=i)
                # Extract underlying patient records from TFTDataset for each split
                def _extract_patients(split_dict):
                    if not split_dict:
                        return []
                    ds = split_dict.get('dataset', None)
                    if ds is None:
                        return []
                    return [ds[j] for j in range(len(ds))]

                chunk_train_patients = _extract_patients(chunk_results['datasets'].get('train', {}))
                chunk_val_patients = _extract_patients(chunk_results['datasets'].get('validation', {}))
                chunk_test_patients = _extract_patients(chunk_results['datasets'].get('test', {}))
                
                # Combine results
                all_train_patients.extend(chunk_train_patients)
                all_val_patients.extend(chunk_val_patients)
                all_test_patients.extend(chunk_test_patients)
                
                # Memory management
                gc.collect()
                
                if check_memory_warning():
                    print("⚠️ High memory usage during data processing")
                
                # Progress update
                processed = min(i + chunk_size, sample_size)
                print(f"📊 Progress: {processed}/{sample_size} patients processed")
            
            # Re-wrap combined patients into TFTDataset instances to match downstream expectations
            combined = {}
            for split_name, patients in [('train', all_train_patients), ('val', all_val_patients), ('test', all_test_patients)]:
                ds = TFTDataset(patients)
                combined[split_name] = {
                    'dataset': ds,
                    'patient_count': len(patients),
                    'feature_dimensions': ds.get_feature_dimensions()
                }
            processed_datasets = combined
        else:
            # For smaller datasets, use the original method
            print("📊 Using standard processing for smaller dataset...")
            pipeline_results = data_pipeline.run_complete_pipeline(sample_size=sample_size)
            processed_datasets = {
                'train': pipeline_results['datasets'].get('train', {}),
                'val': pipeline_results['datasets'].get('validation', {}),
                'test': pipeline_results['datasets'].get('test', {})
            }
        
        pipeline_time = time.time() - start_time
        end_memory = monitor_memory()
        
        print(f"✅ Data pipeline completed in {pipeline_time:.2f}s")
        print(f"💾 Memory usage: {end_memory - start_memory:.1f}%")
        # Correctly report patient counts per split
        def _count(split):
            try:
                return split.get('patient_count', len(split.get('dataset', [])))
            except Exception:
                return 0
        print(f"📊 Train samples: {_count(processed_datasets['train'])}")
        print(f"📊 Val samples: {_count(processed_datasets['val'])}")
        print(f"📊 Test samples: {_count(processed_datasets['test'])}")
        
        return processed_datasets
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_model_optimized(processed_datasets):
    """
    Train model with optimizations for large datasets.
    
    Args:
        processed_datasets (Dict): Processed datasets
        
    Returns:
        Dict: Training results or None if failed
    """
    print("🔄 Starting optimized model training...")
    
    start_time = time.time()
    
    try:
        # Create trainer and setup model properly
        trainer = COVID19Trainer()
        trainer.setup_model()
        
        # Load configuration
        with open("configs/training_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create data loaders
        print("📦 Creating data loaders...")
        data_loaders = {}
        for split_name, dataset_info in processed_datasets.items():
            loader = trainer.create_data_loader(dataset_info['dataset'], batch_size=16)
            data_loaders[split_name] = loader
            print(f"  - {split_name}: {len(loader)} batches")
        
        # Train model with optimized settings
        training_results = trainer.train(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val']
        )
        
        # Extract losses from training history
        train_losses = training_results['training_history']['train_loss']
        val_losses = training_results['training_history']['val_loss']
        
        train_time = time.time() - start_time
        
        print(f"✅ Training completed in {train_time:.2f}s")
        print(f"📊 Final train loss: {train_losses[-1]:.4f}")
        print(f"📊 Final val loss: {val_losses[-1]:.4f}")
        
        # Find best validation loss
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss) + 1
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_epoch': len(train_losses),
            'trainer': trainer,
            'data_loaders': data_loaders
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model_optimized(training_results):
    """
    Evaluate model with optimizations.
    
    Args:
        training_results (Dict): Training results
        
    Returns:
        Dict: Evaluation results or None if failed
    """
    print("🔄 Starting optimized model evaluation...")
    
    try:
        trainer = training_results['trainer']
        test_loader = training_results['data_loaders']['test']
        
        # Evaluate on test set
        test_loss, test_metrics = trainer.validate_epoch(test_loader)
        
        print(f"✅ Evaluation completed")
        print(f"📊 Test loss: {test_loss:.4f}")
        
        # Log metrics
        for metric_name, metric_value in test_metrics.items():
            print(f"📊 {metric_name}: {metric_value:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_metrics': test_metrics
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main optimized training function."""
    print("🚀 Optimized COVID-19 TFT Training for 12,000+ Patients")
    print("="*60)
    
    # Setup logging
    setup_logging()
    
    # Configuration
    sample_size = 500  # Reduce to 500 patients with stratified split
    
    print(f"🎯 Target: {sample_size} patients")
    print(f"💾 Initial memory usage: {monitor_memory():.1f}%")
    
    # Step 1: Optimized Data Pipeline
    print("\n" + "="*50)
    print("STEP 1: OPTIMIZED DATA PIPELINE")
    print("="*50)
    
    processed_datasets = run_data_pipeline_optimized(sample_size=sample_size)
    if processed_datasets is None:
        print("❌ Data pipeline failed. Exiting.")
        return
    
    # Step 2: Optimized Model Training
    print("\n" + "="*50)
    print("STEP 2: OPTIMIZED MODEL TRAINING")
    print("="*50)
    
    training_results = train_model_optimized(processed_datasets)
    if training_results is None:
        print("❌ Training failed. Exiting.")
        return
    
    # Step 3: Optimized Model Evaluation
    print("\n" + "="*50)
    print("STEP 3: OPTIMIZED MODEL EVALUATION")
    print("="*50)
    
    evaluation_results = evaluate_model_optimized(training_results)
    if evaluation_results is None:
        print("❌ Evaluation failed. Exiting.")
        return
    
    # Final Summary
    print("\n" + "="*60)
    print("🎉 OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    total_time = time.time() - main.start_time if hasattr(main, 'start_time') else 0
    final_memory = monitor_memory()
    
    print(f"⏱️  Total pipeline time: {total_time:.2f} seconds")
    print(f"💾 Final memory usage: {final_memory:.1f}%")
    print(f"👥 Patients processed: {sample_size}")
    # Correctly report patient counts per split
    def _count(split):
        try:
            return split.get('patient_count', len(split.get('dataset', [])))
        except Exception:
            return 0
    print(f"📊 Train samples: {_count(processed_datasets['train'])}")
    print(f"📊 Val samples: {_count(processed_datasets['val'])}")
    print(f"📊 Test samples: {_count(processed_datasets['test'])}")
    print(f"🏆 Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"📈 Best epoch: {training_results['best_epoch']}")
    
    if 'test_loss' in evaluation_results:
        print(f"📊 Test loss: {evaluation_results['test_loss']:.4f}")
    
    print("="*60)
    
    return {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'pipeline_time': total_time,
        'memory_usage': final_memory
    }

if __name__ == "__main__":
    # Set start time for main function
    main.start_time = time.time()
    results = main() 