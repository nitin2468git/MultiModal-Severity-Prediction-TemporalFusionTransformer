#!/usr/bin/env python3
"""
Sample Training Script for COVID-19 TFT
Tests the training pipeline on a small subset of data (100-200 patients)
to ensure everything works before running on the full dataset.
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
import random

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_pipeline import DataPipeline
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
            logging.FileHandler(log_dir / 'sample_training.log'),
            logging.StreamHandler()
        ]
    )

def create_sample_data_pipeline(sample_size=200):
    """Create a data pipeline with only a sample of patients."""
    print("="*70)
    print("COVID-19 TFT SAMPLE TRAINING (Testing Pipeline)")
    print(f"Using {sample_size} patients for quick testing")
    print("="*70)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Data Pipeline with Sample
        print("\n📊 STEP 1: SAMPLE DATA PIPELINE")
        print("-" * 50)
        
        print("🔄 Loading configuration...")
        data_pipeline = DataPipeline()
        
        print("🔄 Loading raw data...")
        raw_data = data_pipeline._load_raw_data()
        print(f"✅ Loaded data for {len(raw_data)} COVID-19 patients")
        
        # Take a sample of patients
        print(f"📊 Taking sample of {sample_size} patients...")
        if len(raw_data) > sample_size:
            # Use random sampling for reproducibility
            random.seed(42)  # For reproducible results
            sample_patients = random.sample(list(raw_data.keys()), sample_size)
            raw_data_sample = {patient_id: raw_data[patient_id] for patient_id in sample_patients}
        else:
            raw_data_sample = raw_data
            print(f"⚠️ Dataset has only {len(raw_data)} patients, using all")
        
        print(f"✅ Sample created: {len(raw_data_sample)} patients")
        
        # Validate sample data
        print("🔍 Validating sample data quality...")
        validation_results, valid_patients = data_pipeline._validate_data(raw_data_sample)
        print(f"✅ Validation complete: {validation_results['valid_patients']} valid, {validation_results['invalid_patients']} invalid")
        
        if validation_results['valid_patients'] == 0:
            print("❌ No valid patients in sample. Check validation report for details.")
            return None
        
        # Create temporal splits for sample
        print("📊 Creating temporal splits for sample...")
        split_data = data_pipeline._create_temporal_splits(valid_patients)
        print(f"✅ Created splits: {list(split_data.keys())}")
        
        # Process splits with progress
        print("⚙️ Processing sample data splits...")
        processed_datasets = data_pipeline._process_splits(split_data)
        
        print("✅ Sample data pipeline completed successfully!")
        
        # Print sample statistics
        for split_name, dataset in processed_datasets.items():
            print(f"  {split_name}: {len(dataset)} samples")
        
        return processed_datasets
        
    except Exception as e:
        logger.error(f"Sample data pipeline failed: {e}")
        print(f"❌ Sample data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_sample_model_with_progress(processed_datasets):
    """Train model on sample data with detailed progress display."""
    print("\n🧠 STEP 2: SAMPLE MODEL TRAINING")
    print("-" * 50)
    
    try:
        # Setup trainer with reduced epochs for quick testing
        print("🔄 Setting up trainer...")
        trainer = COVID19Trainer()
        
        # Modify config for sample training (fewer epochs, smaller batch size)
        trainer.config['training']['epochs'] = 5  # Reduced from default
        trainer.config['training']['batch_size'] = 16  # Smaller batch size
        trainer.config['training']['patience'] = 3  # Early stopping patience
        
        print("🔄 Setting up model...")
        trainer.setup_model()
        
        # Get model info
        model_info = trainer.model.get_model_info()
        print(f"✅ Model initialized:")
        print(f"  - Total parameters: {model_info['total_parameters']:,}")
        print(f"  - Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Create data loaders for sample
        print("🔄 Creating sample data loaders...")
        train_loader = trainer.create_data_loader(
            processed_datasets['train'], 
            batch_size=16  # Smaller batch size for sample
        )
        
        val_loader = trainer.create_data_loader(
            processed_datasets['validation'], 
            batch_size=16
        )
        
        test_loader = trainer.create_data_loader(
            processed_datasets['test'], 
            batch_size=16
        )
        
        print(f"✅ Sample data loaders created:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Training with progress
        print("\n🚀 Starting sample training...")
        print(f"Training for {trainer.config['training']['epochs']} epochs")
        
        training_results = trainer.train(train_loader, val_loader, test_loader)
        
        print("✅ Sample training completed successfully!")
        return training_results
        
    except Exception as e:
        print(f"❌ Sample training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_sample_model_with_progress(training_results, processed_datasets):
    """Evaluate sample model with progress display."""
    print("\n📈 STEP 3: SAMPLE MODEL EVALUATION")
    print("-" * 50)
    
    try:
        # Load best model
        trainer = COVID19Trainer()
        best_checkpoint_path = Path(trainer.config['checkpointing']['save_dir']) / 'best_model.pth'
        
        if best_checkpoint_path.exists():
            trainer.load_checkpoint(str(best_checkpoint_path))
            print("✅ Loaded best model for evaluation")
        else:
            print("⚠️ No best model checkpoint found")
            return None
        
        # Make predictions on test set
        print("🔄 Making predictions on test set...")
        test_loader = trainer.create_data_loader(
            processed_datasets['test'], 
            batch_size=16
        )
        
        test_predictions = trainer.predict(test_loader)
        
        # Extract targets from test loader
        print("🔄 Extracting test targets...")
        test_targets = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        for batch in test_loader:
            for task in test_targets.keys():
                if task in batch:
                    test_targets[task].extend(batch[task].cpu().numpy())
        
        # Convert to numpy arrays
        for task in test_targets:
            test_targets[task] = np.array(test_targets[task])
        
        # Calculate final metrics
        print("🔄 Calculating sample metrics...")
        metrics_calculator = COVID19Metrics()
        final_metrics = metrics_calculator.calculate_metrics(test_predictions, test_targets)
        
        # Generate evaluation report
        evaluation_report = metrics_calculator.generate_evaluation_report(
            test_predictions, test_targets
        )
        
        print("✅ Sample evaluation completed!")
        
        # Print sample results
        print("\n📊 SAMPLE TRAINING RESULTS:")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"Final epoch: {training_results['final_epoch'] + 1}")
        print(f"Test loss: {training_results['test_results']['test_loss']:.4f}")
        
        print("\nSample Test Metrics:")
        print(f"  Mortality Risk AUROC: {final_metrics.get('mortality_risk_auroc', 0):.3f}")
        print(f"  ICU Admission AUROC: {final_metrics.get('icu_admission_auroc', 0):.3f}")
        print(f"  Ventilator Need AUROC: {final_metrics.get('ventilator_need_auroc', 0):.3f}")
        print(f"  Length of Stay MAE: {final_metrics.get('length_of_stay_mae', 0):.2f} hours")
        print(f"  Average AUROC: {final_metrics.get('average_auroc', 0):.3f}")
        
        return {
            'training_results': training_results,
            'final_metrics': final_metrics,
            'evaluation_report': evaluation_report
        }
        
    except Exception as e:
        print(f"❌ Sample evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for sample training."""
    print("🎯 COVID-19 TFT SAMPLE TRAINING")
    print("Testing pipeline on small dataset before full training")
    print("="*70)
    
    # Step 1: Create sample data pipeline
    processed_datasets = create_sample_data_pipeline(sample_size=200)
    if processed_datasets is None:
        print("❌ Sample data pipeline failed. Exiting.")
        return
    
    # Step 2: Train model on sample
    training_results = train_sample_model_with_progress(processed_datasets)
    if training_results is None:
        print("❌ Sample training failed. Exiting.")
        return
    
    # Step 3: Evaluate sample model
    evaluation_results = evaluate_sample_model_with_progress(training_results, processed_datasets)
    if evaluation_results is None:
        print("❌ Sample evaluation failed. Exiting.")
        return
    
    # Step 4: Save sample results
    print("\n💾 STEP 4: SAVING SAMPLE RESULTS")
    print("-" * 50)
    
    results = {
        'training_results': training_results,
        'final_metrics': evaluation_results['final_metrics'],
        'evaluation_report': evaluation_results['evaluation_report'],
        'sample_size': 200,
        'timestamp': datetime.now().isoformat(),
        'note': 'Sample training for pipeline testing'
    }
    
    results_path = Path("experiments/results/sample_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Sample results saved to: {results_path}")
    
    print("\n" + "="*70)
    print("🎉 SAMPLE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("✅ Pipeline is working correctly")
    print("✅ Ready for full training on complete dataset")
    print("✅ Sample results saved for reference")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main() 