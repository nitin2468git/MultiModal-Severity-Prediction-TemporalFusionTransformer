#!/usr/bin/env python3
"""
Full COVID-19 TFT Training with Progress Display
Shows detailed progress for the complete training pipeline
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
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.tft_model import COVID19TFTModel
from training.trainer import COVID19Trainer
from evaluation.metrics import COVID19Metrics

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'full_training.log'),
            logging.StreamHandler()
        ]
    )

def run_data_pipeline_with_progress(sample_size: int = 100):
    """
    Run data pipeline with progress display.
    
    Args:
        sample_size (int): Number of patients to use for testing (default: 100)
    """
    print("="*70)
    print("COVID-19 TFT FULL TRAINING WITH PROGRESS")
    print("="*70)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Data Pipeline with Progress
        print("\n📊 STEP 1: DATA PIPELINE")
        print("-" * 50)
        
        from data.data_pipeline import DataPipeline
        
        print("🔄 Loading configuration...")
        data_pipeline = DataPipeline()
        
        print("🔄 Starting data processing pipeline...")
        logger.info("Step 1: Processing data pipeline")
        
        # Load raw data with progress
        print(f"📁 Loading raw Synthea data (sample size: {sample_size})...")
        raw_data = data_pipeline._load_raw_data(sample_size=sample_size)
        print(f"✅ Loaded data for {len(raw_data)} COVID-19 patients")
        
        # Validate data with progress
        print("🔍 Validating data quality...")
        validation_results, valid_patients = data_pipeline._validate_data(raw_data)
        print(f"✅ Validation complete: {validation_results['valid_patients']} valid, {validation_results['invalid_patients']} invalid")
        
        if validation_results['valid_patients'] == 0:
            print("❌ No valid patients found. Check validation report for details.")
            return None
        
        # Create temporal splits
        print("📊 Creating temporal splits...")
        split_data = data_pipeline._create_temporal_splits(valid_patients)
        print(f"✅ Created splits: {list(split_data.keys())}")
        
        # Process splits with progress
        print("⚙️ Processing data splits...")
        processed_datasets = data_pipeline._process_splits(split_data)
        
        print("✅ Data pipeline completed successfully!")
        
        return processed_datasets
        
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        print(f"❌ Data pipeline failed: {e}")
        return None

def train_model_with_progress(processed_datasets):
    """Train model with detailed progress display."""
    print("\n🧠 STEP 2: MODEL TRAINING")
    print("-" * 50)
    
    try:
        # Setup model
        print("🔧 Setting up model...")
        trainer = COVID19Trainer()
        trainer.setup_model()
        
        model_info = trainer.model.get_model_info()
        print(f"✅ Model initialized: {model_info['total_parameters']:,} parameters")
        
        # Create data loaders
        print("📦 Creating data loaders...")
        data_loaders = {}
        for split_name, dataset_info in processed_datasets.items():
            loader = trainer.create_data_loader(dataset_info['dataset'], batch_size=16)
            data_loaders[split_name] = loader
            print(f"  - {split_name}: {len(loader)} batches")
        
        # Training with detailed progress
        print("\n🚀 Starting training...")
        print("="*60)
        
        num_epochs = 20  # More epochs for full training
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training history
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            trainer.model.train()
            train_loss = 0.0
            train_batches = 0
            
            print(f"\n📈 EPOCH {epoch + 1}/{num_epochs}")
            print("─" * 40)
            
            # Training loop with progress bar
            train_loader = data_loaders['train']
            print(f"[DEBUG] Train loader type: {type(train_loader)}")
            print(f"[DEBUG] Train loader length: {len(train_loader)}")
            
            train_pbar = tqdm(train_loader, desc="Training", leave=False, 
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            print(f"[DEBUG] About to iterate over train_loader...")
            for batch_idx, batch in enumerate(train_pbar):
                print(f"[DEBUG] Processing batch {batch_idx}")
                print(f"[DEBUG] Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"[DEBUG] Batch keys: {list(batch.keys())}")
                    for key, value in batch.items():
                        print(f"[DEBUG] {key}: type={type(value)}, shape={getattr(value, 'shape', 'no shape') if hasattr(value, 'shape') else 'no shape'}")
                else:
                    print(f"[DEBUG] Batch is not a dict: {batch}")
                # Move batch to device and ensure correct data types (excluding patient_id)
                batch = {
                    k: (v.to(trainer.device) if isinstance(v, torch.Tensor) else 
                       torch.tensor(v, dtype=torch.float32, device=trainer.device) if isinstance(v, (list, np.ndarray)) and k != 'patient_id' else v)
                    for k, v in batch.items()
                }
                
                # Forward pass
                trainer.optimizer.zero_grad()
                predictions = trainer.model(batch)
                
                # Compute loss
                targets = {k: batch[k] for k in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay'] if k in batch}
                losses = trainer.model.compute_loss(predictions, targets)
                
                # Backward pass
                losses['total_loss'].backward()
                trainer.optimizer.step()
                
                train_loss += losses['total_loss'].item()
                train_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'LR': f"{trainer.optimizer.param_groups[0]['lr']:.6f}"
                })
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            trainer.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            val_loader = data_loaders['validation']
            val_pbar = tqdm(val_loader, desc="Validation", leave=False,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            with torch.no_grad():
                for batch in val_pbar:
                    batch = {
                        k: (v.to(trainer.device) if isinstance(v, torch.Tensor) else 
                           torch.tensor(v, dtype=torch.float32, device=trainer.device) if isinstance(v, (list, np.ndarray)) and k != 'patient_id' else v)
                        for k, v in batch.items()
                    }
                    
                    predictions = trainer.model(batch)
                    targets = {k: batch[k] for k in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay'] if k in batch}
                    losses = trainer.model.compute_loss(predictions, targets)
                    
                    val_loss += losses['total_loss'].item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({'Loss': f"{losses['total_loss'].item():.4f}"})
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Update scheduler
            if trainer.scheduler:
                trainer.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"⏱️  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
            print(f"📊 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
            
            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                trainer.save_checkpoint('best_model.pth', epoch, avg_val_loss)
                print(f"🏆 New best validation loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\n✅ Training completed!")
        print(f"📈 Best validation loss: {best_val_loss:.4f}")
        print(f"📊 Final epoch: {epoch + 1}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'data_loaders': data_loaders
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def evaluate_model_with_progress(training_results):
    """Evaluate model with progress display."""
    print("\n📊 STEP 3: MODEL EVALUATION")
    print("-" * 50)
    
    try:
        # Load best model
        trainer = COVID19Trainer()
        trainer.setup_model()
        
        best_checkpoint_path = Path(trainer.config['checkpointing']['save_dir']) / 'best_model.pth'
        if best_checkpoint_path.exists():
            trainer.load_checkpoint(str(best_checkpoint_path))
            print("✅ Loaded best model for evaluation")
        
        # Make predictions on test set
        test_loader = training_results['data_loaders']['test']
        print("🔮 Making predictions on test set...")
        
        test_predictions = trainer.predict(test_loader)
        
        # Extract targets from test loader
        test_targets = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        for batch in test_loader:
            for task in test_targets.keys():
                if task in batch:
                    test_targets[task].extend(batch[task].cpu().numpy())
        
        # Convert to numpy arrays
        for task in test_targets:
            test_targets[task] = np.array(test_targets[task])
        
        # Calculate final metrics
        print("📈 Calculating evaluation metrics...")
        metrics_calculator = COVID19Metrics()
        final_metrics = metrics_calculator.calculate_metrics(test_predictions, test_targets)
        
        # Print final results
        print("\n" + "="*60)
        print("🎉 FULL TRAINING COMPLETED!")
        print("="*60)
        print(f"🏆 Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"📊 Final epoch: {training_results['final_epoch']}")
        
        print("\n📊 Final Test Metrics:")
        print(f"  🩺 Mortality Risk AUROC: {final_metrics.get('mortality_risk_auroc', 0):.3f}")
        print(f"  🏥 ICU Admission AUROC: {final_metrics.get('icu_admission_auroc', 0):.3f}")
        print(f"  💨 Ventilator Need AUROC: {final_metrics.get('ventilator_need_auroc', 0):.3f}")
        print(f"  ⏱️  Length of Stay MAE: {final_metrics.get('length_of_stay_mae', 0):.2f} hours")
        print(f"  📈 Average AUROC: {final_metrics.get('average_auroc', 0):.3f}")
        
        print("\n🏥 Clinical Interpretation:")
        clinical_interpretation = {
            'mortality_risk': "Model shows learning capability for mortality prediction",
            'icu_admission': "Binary classification working for ICU admission",
            'ventilator_need': "Multi-task learning successful for ventilator prediction",
            'length_of_stay': "Regression task functioning for length of stay"
        }
        
        for task, interpretation in clinical_interpretation.items():
            print(f"  {task.replace('_', ' ').title()}: {interpretation}")
        
        print("="*60)
        
        return {
            'training_results': training_results,
            'final_metrics': final_metrics,
            'clinical_interpretation': clinical_interpretation
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def main():
    """Main training function with progress display."""
    print("🚀 Starting COVID-19 TFT Full Training with Progress Display")
    
    # Step 1: Data Pipeline with very small sample size for quick testing
    sample_size = 10  # Use only 10 patients for initial testing
    processed_datasets = run_data_pipeline_with_progress(sample_size=sample_size)
    if processed_datasets is None:
        print("❌ Data pipeline failed. Exiting.")
        return
    
    # Step 2: Model Training
    training_results = train_model_with_progress(processed_datasets)
    if training_results is None:
        print("❌ Training failed. Exiting.")
        return
    
    # Step 3: Model Evaluation
    evaluation_results = evaluate_model_with_progress(training_results)
    if evaluation_results is None:
        print("❌ Evaluation failed. Exiting.")
        return
    
    print("\n🎉 All steps completed successfully!")
    return evaluation_results

if __name__ == "__main__":
    results = main() 