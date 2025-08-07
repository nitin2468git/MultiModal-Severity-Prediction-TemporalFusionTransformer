#!/usr/bin/env python3
"""
Main training script for COVID-19 TFT Severity Prediction
Part of the COVID-19 TFT Severity Prediction project
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

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import DataPipeline
from training.trainer import COVID19Trainer
from evaluation.metrics import COVID19Metrics

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main training function."""
    print("="*60)
    print("COVID-19 TFT SEVERITY PREDICTION TRAINING")
    print("="*60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Data Processing
        logger.info("Step 1: Processing data pipeline")
        data_pipeline = DataPipeline()
        pipeline_results = data_pipeline.run_complete_pipeline()
        
        logger.info(f"Data processing completed:")
        logger.info(f"  - Valid patients: {pipeline_results['processing_stats']['valid_patients']}")
        logger.info(f"  - Train split: {pipeline_results['processing_stats']['splits_created']['train']}")
        logger.info(f"  - Validation split: {pipeline_results['processing_stats']['splits_created']['validation']}")
        logger.info(f"  - Test split: {pipeline_results['processing_stats']['splits_created']['test']}")
        
        # Step 2: Setup Training
        logger.info("Step 2: Setting up training pipeline")
        trainer = COVID19Trainer()
        trainer.setup_model()
        
        # Get model info
        model_info = trainer.model.get_model_info()
        logger.info(f"Model initialized:")
        logger.info(f"  - Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"  - Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Step 3: Create Data Loaders
        logger.info("Step 3: Creating data loaders")
        processed_datasets = pipeline_results['datasets']
        
        train_loader = trainer.get_data_loaders(
            {'train': processed_datasets['train']}, 
            batch_size=32
        )['train']
        
        val_loader = trainer.get_data_loaders(
            {'validation': processed_datasets['validation']}, 
            batch_size=32
        )['validation']
        
        test_loader = trainer.get_data_loaders(
            {'test': processed_datasets['test']}, 
            batch_size=32
        )['test']
        
        logger.info(f"Data loaders created:")
        logger.info(f"  - Train batches: {len(train_loader)}")
        logger.info(f"  - Validation batches: {len(val_loader)}")
        logger.info(f"  - Test batches: {len(test_loader)}")
        
        # Step 4: Training
        logger.info("Step 4: Starting training process")
        training_results = trainer.train(train_loader, val_loader, test_loader)
        
        # Step 5: Final Evaluation
        logger.info("Step 5: Final evaluation")
        
        # Load best model
        best_checkpoint_path = Path(trainer.config['checkpointing']['save_dir']) / 'best_model.pth'
        if best_checkpoint_path.exists():
            trainer.load_checkpoint(str(best_checkpoint_path))
            logger.info("Loaded best model for final evaluation")
        
        # Make predictions on test set
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
        metrics_calculator = COVID19Metrics()
        final_metrics = metrics_calculator.calculate_metrics(test_predictions, test_targets)
        
        # Generate evaluation report
        evaluation_report = metrics_calculator.generate_evaluation_report(
            test_predictions, test_targets
        )
        
        # Step 6: Save Results
        logger.info("Step 6: Saving results")
        
        # Save final results
        results = {
            'training_results': training_results,
            'final_metrics': final_metrics,
            'evaluation_report': evaluation_report,
            'model_info': model_info,
            'pipeline_results': pipeline_results,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = Path("experiments/results/final_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"Final epoch: {training_results['final_epoch'] + 1}")
        print(f"Test loss: {training_results['test_results']['test_loss']:.4f}")
        
        print("\nFinal Test Metrics:")
        print(f"  Mortality Risk AUROC: {final_metrics.get('mortality_risk_auroc', 0):.3f}")
        print(f"  ICU Admission AUROC: {final_metrics.get('icu_admission_auroc', 0):.3f}")
        print(f"  Ventilator Need AUROC: {final_metrics.get('ventilator_need_auroc', 0):.3f}")
        print(f"  Length of Stay MAE: {final_metrics.get('length_of_stay_mae', 0):.2f} hours")
        print(f"  Average AUROC: {final_metrics.get('average_auroc', 0):.3f}")
        
        print("\nClinical Interpretation:")
        clinical_interpretation = evaluation_report['clinical_interpretation']
        for task, interpretation in clinical_interpretation.items():
            print(f"  {task.replace('_', ' ').title()}: {interpretation}")
        
        print(f"\nResults saved to: {results_path}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    results = main() 