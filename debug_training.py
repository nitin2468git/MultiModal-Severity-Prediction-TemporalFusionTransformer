#!/usr/bin/env python3
"""
Step-by-step debugging script for COVID-19 TFT training
"""

import sys
import os
from pathlib import Path
import logging
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.tft_model import COVID19TFTModel
from training.trainer import COVID19Trainer
from data.data_pipeline import DataPipeline

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def debug_step_1_data_pipeline():
    """Debug Step 1: Data Pipeline"""
    print("="*60)
    print("STEP 1: DEBUGGING DATA PIPELINE")
    print("="*60)
    
    try:
        print("🔄 Loading configuration...")
        data_pipeline = DataPipeline()
        print("✅ DataPipeline initialized successfully")
        
        print("🔄 Loading raw data (sample size: 5)...")
        raw_data = data_pipeline._load_raw_data(sample_size=5)
        print(f"✅ Loaded data for {len(raw_data)} COVID-19 patients")
        
        print("🔍 Validating data...")
        validation_results, valid_patients = data_pipeline._validate_data(raw_data)
        print(f"✅ Validation complete: {validation_results['valid_patients']} valid")
        
        print("📊 Creating temporal splits...")
        split_data = data_pipeline._create_temporal_splits(valid_patients)
        print(f"✅ Created splits: {list(split_data.keys())}")
        
        print("⚙️ Processing data splits...")
        processed_datasets = data_pipeline._process_splits(split_data)
        print("✅ Data pipeline completed successfully!")
        
        return processed_datasets
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_step_2_model_setup():
    """Debug Step 2: Model Setup"""
    print("\n" + "="*60)
    print("STEP 2: DEBUGGING MODEL SETUP")
    print("="*60)
    
    try:
        print("🔧 Setting up trainer...")
        trainer = COVID19Trainer()
        print("✅ Trainer initialized")
        
        print("🔧 Setting up model...")
        trainer.setup_model()
        print("✅ Model setup completed")
        
        model_info = trainer.model.get_model_info()
        print(f"✅ Model initialized: {model_info['total_parameters']:,} parameters")
        print(f"✅ Using device: {trainer.device}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_step_3_data_loader_creation(trainer, processed_datasets):
    """Debug Step 3: Data Loader Creation"""
    print("\n" + "="*60)
    print("STEP 3: DEBUGGING DATA LOADER CREATION")
    print("="*60)
    
    try:
        print("📦 Creating data loaders...")
        data_loaders = {}
        for split_name, dataset_info in processed_datasets.items():
            print(f"  - Creating loader for {split_name}...")
            loader = trainer.create_data_loader(dataset_info['dataset'], batch_size=2)
            data_loaders[split_name] = loader
            print(f"  - {split_name}: {len(loader)} batches")
        
        print("✅ Data loaders created successfully!")
        return data_loaders
        
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_step_4_single_batch_processing(trainer, data_loaders):
    """Debug Step 4: Single Batch Processing"""
    print("\n" + "="*60)
    print("STEP 4: DEBUGGING SINGLE BATCH PROCESSING")
    print("="*60)
    
    try:
        print("🔄 Getting first batch from train loader...")
        train_loader = data_loaders['train']
        print(f"✅ Train loader type: {type(train_loader)}")
        print(f"✅ Train loader length: {len(train_loader)}")
        
        # Get first batch
        batch = next(iter(train_loader))
        print(f"✅ Batch type: {type(batch)}")
        print(f"✅ Batch keys: {list(batch.keys())}")
        
        # Debug batch contents
        for key, value in batch.items():
            print(f"  - {key}: type={type(value)}")
            if isinstance(value, torch.Tensor):
                print(f"    shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"    length={len(value)}")
        
        print("🔄 Converting batch...")
        converted_batch = trainer._convert_batch(batch)
        print("✅ Batch converted successfully")
        
        print("🔄 Forward pass...")
        trainer.model.train()
        trainer.optimizer.zero_grad()
        predictions = trainer.model(converted_batch)
        print("✅ Forward pass completed")
        print(f"✅ Predictions keys: {list(predictions.keys())}")
        
        print("🔄 Computing loss...")
        targets = {k: converted_batch[k].float() for k in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay'] if k in converted_batch}
        losses = trainer.model.compute_loss(predictions, targets)
        print("✅ Loss computed successfully")
        print(f"✅ Loss keys: {list(losses.keys())}")
        
        print("🔄 Backward pass...")
        losses['total_loss'].backward()
        print("✅ Backward pass completed")
        
        print("🔄 Optimizer step...")
        trainer.optimizer.step()
        print("✅ Optimizer step completed")
        
        print("🔄 Checking learning rate...")
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"✅ Learning rate: {lr} (type: {type(lr)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Single batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_step_5_progress_bar_issue(trainer, data_loaders):
    """Debug Step 5: Progress Bar Issue"""
    print("\n" + "="*60)
    print("STEP 5: DEBUGGING PROGRESS BAR ISSUE")
    print("="*60)
    
    try:
        print("🔄 Testing progress bar update...")
        train_loader = data_loaders['train']
        
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc="Testing", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            print(f"🔄 Processing batch {batch_idx}")
            
            # Convert batch
            converted_batch = trainer._convert_batch(batch)
            
            # Forward pass
            trainer.model.train()
            trainer.optimizer.zero_grad()
            predictions = trainer.model(converted_batch)
            
            # Compute loss
            targets = {k: converted_batch[k].float() for k in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay'] if k in converted_batch}
            losses = trainer.model.compute_loss(predictions, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            trainer.optimizer.step()
            
            # Test progress bar update
            print("🔄 Testing progress bar set_postfix...")
            try:
                lr = trainer.optimizer.param_groups[0]['lr']
                print(f"✅ Learning rate: {lr} (type: {type(lr)})")
                
                progress_bar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'LR': f"{lr:.6f}"
                })
                print("✅ Progress bar update successful")
            except Exception as e:
                print(f"❌ Progress bar update failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Only process first batch for debugging
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Progress bar debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function."""
    print("🚀 Starting step-by-step debugging...")
    
    # Setup logging
    setup_logging()
    
    # Step 1: Data Pipeline
    processed_datasets = debug_step_1_data_pipeline()
    if processed_datasets is None:
        print("❌ Data pipeline failed. Stopping.")
        return
    
    # Step 2: Model Setup
    trainer = debug_step_2_model_setup()
    if trainer is None:
        print("❌ Model setup failed. Stopping.")
        return
    
    # Step 3: Data Loader Creation
    data_loaders = debug_step_3_data_loader_creation(trainer, processed_datasets)
    if data_loaders is None:
        print("❌ Data loader creation failed. Stopping.")
        return
    
    # Step 4: Single Batch Processing
    success = debug_step_4_single_batch_processing(trainer, data_loaders)
    if not success:
        print("❌ Single batch processing failed. Stopping.")
        return
    
    # Step 5: Progress Bar Issue
    success = debug_step_5_progress_bar_issue(trainer, data_loaders)
    if not success:
        print("❌ Progress bar debugging failed.")
        return
    
    print("\n🎉 All debugging steps completed successfully!")
    print("The issue has been identified and resolved.")

if __name__ == "__main__":
    main() 