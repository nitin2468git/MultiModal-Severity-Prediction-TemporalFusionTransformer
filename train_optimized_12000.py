#!/usr/bin/env python3
"""
Optimized COVID-19 TFT Training for 12,000+ Patients
Comprehensive pipeline with memory management and performance optimizations
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
import multiprocessing as mp
import gc
import psutil
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.tft_model import COVID19TFTModel
from training.trainer import COVID19Trainer
from evaluation.metrics import COVID19Metrics
from data.synthea_loader import SyntheaLoader
from data.data_validator import DataValidator
from data.feature_engineer import FeatureEngineer
from data.timeline_builder_optimized import OptimizedTimelineBuilder
from data.tft_formatter import TFTFormatter

class OptimizedCOVID19Trainer:
    """Optimized trainer for large-scale COVID-19 TFT training."""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
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
        return logging.getLogger(__name__)
    
    def _monitor_memory(self) -> float:
        """Monitor current memory usage."""
        return psutil.virtual_memory().percent
    
    def _check_memory_warning(self, threshold: float = 90.0) -> bool:
        """Check if memory usage is above threshold."""
        memory_usage = self._monitor_memory()
        if memory_usage > threshold:
            self.logger.warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
            return True
        return False
    
    def load_data_optimized(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Load data with memory optimization."""
        self.logger.info("🔄 Loading patient data...")
        
        start_time = time.time()
        start_memory = self._monitor_memory()
        
        try:
            # Load data in chunks if large dataset
            loader = SyntheaLoader()
            
            if sample_size and sample_size > 5000:
                # For very large datasets, load in chunks
                chunk_size = 2000
                all_patient_data = {}
                
                for i in range(0, sample_size, chunk_size):
                    current_chunk_size = min(chunk_size, sample_size - i)
                    self.logger.info(f"🔄 Loading chunk {i//chunk_size + 1}/{(sample_size + chunk_size - 1)//chunk_size}")
                    
                    chunk_data = loader.load_patient_data(sample_size=current_chunk_size, start_index=i)
                    all_patient_data.update(chunk_data)
                    
                    # Memory management
                    gc.collect()
                    
                    if self._check_memory_warning():
                        self.logger.warning("⚠️ High memory usage during data loading")
            else:
                all_patient_data = loader.load_patient_data(sample_size=sample_size)
            
            load_time = time.time() - start_time
            end_memory = self._monitor_memory()
            
            self.logger.info(f"✅ Data loading complete: {len(all_patient_data)} patients in {load_time:.2f}s")
            self.logger.info(f"💾 Memory usage: {end_memory - start_memory:.1f}%")
            
            return all_patient_data
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {str(e)}")
            raise
    
    def validate_data_optimized(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Validate data with optimized processing."""
        self.logger.info("🔄 Validating patient data...")
        
        start_time = time.time()
        
        try:
            validator = DataValidator()
            validation_results, valid_patients = validator.validate_data_quality(all_patient_data)
            
            validation_time = time.time() - start_time
            
            self.logger.info(f"✅ Validation complete: {len(valid_patients)} valid patients in {validation_time:.2f}s")
            
            return valid_patients
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {str(e)}")
            raise
    
    def build_timelines_optimized(self, valid_patients: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """Build timelines with memory optimization."""
        self.logger.info("🔄 Building patient timelines...")
        
        start_time = time.time()
        start_memory = self._monitor_memory()
        
        try:
            # Use optimized timeline builder
            timeline_builder = OptimizedTimelineBuilder(
                chunk_size=100,  # Process 100 patients at a time
                max_workers=min(4, mp.cpu_count())  # Use up to 4 workers
            )
            
            # Choose processing method based on dataset size
            total_patients = len(valid_patients)
            use_parallel = total_patients > 1000
            
            timelines = timeline_builder.build_patient_timelines(
                valid_patients, 
                use_parallel=use_parallel
            )
            
            build_time = time.time() - start_time
            end_memory = self._monitor_memory()
            
            self.logger.info(f"✅ Timeline building complete: {len(timelines)} timelines in {build_time:.2f}s")
            self.logger.info(f"💾 Memory usage: {end_memory - start_memory:.1f}%")
            
            return timelines
            
        except Exception as e:
            self.logger.error(f"❌ Timeline building failed: {str(e)}")
            raise
    
    def extract_features_optimized(self, timelines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract features with memory management."""
        self.logger.info("🔄 Extracting features...")
        
        start_time = time.time()
        
        try:
            feature_engineer = FeatureEngineer()
            processed_features = []
            
            # Process in chunks to manage memory
            chunk_size = 50
            total_timelines = len(timelines)
            
            for i in range(0, total_timelines, chunk_size):
                chunk = timelines[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                total_chunks = (total_timelines + chunk_size - 1) // chunk_size
                
                self.logger.info(f"🔄 Processing feature chunk {chunk_num}/{total_chunks}")
                
                chunk_features = []
                for timeline in chunk:
                    try:
                        features = feature_engineer.extract_features(timeline)
                        if features is not None:
                            chunk_features.append(features)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Feature extraction failed for patient {timeline.get('patient_id', 'unknown')}: {str(e)}")
                        continue
                
                processed_features.extend(chunk_features)
                
                # Memory management
                gc.collect()
                
                if self._check_memory_warning():
                    self.logger.warning("⚠️ High memory usage during feature extraction")
            
            extract_time = time.time() - start_time
            
            self.logger.info(f"✅ Feature extraction complete: {len(processed_features)} features in {extract_time:.2f}s")
            
            return processed_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {str(e)}")
            raise
    
    def create_tft_datasets_optimized(self, processed_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create TFT datasets with optimization."""
        self.logger.info("🔄 Creating TFT datasets...")
        
        start_time = time.time()
        
        try:
            tft_formatter = TFTFormatter()
            
            # Split data for train/val/test
            total_samples = len(processed_features)
            train_size = int(0.7 * total_samples)
            val_size = int(0.15 * total_samples)
            
            train_data = processed_features[:train_size]
            val_data = processed_features[train_size:train_size + val_size]
            test_data = processed_features[train_size + val_size:]
            
            # Create datasets
            train_dataset = tft_formatter.create_tft_dataset(train_data)
            val_dataset = tft_formatter.create_tft_dataset(val_data)
            test_dataset = tft_formatter.create_tft_dataset(test_data)
            
            # Create data loaders with optimized settings
            train_loader = tft_formatter.create_data_loader(
                train_dataset, 
                batch_size=self.config['training']['batch_size'],
                num_workers=min(4, mp.cpu_count())
            )
            
            val_loader = tft_formatter.create_data_loader(
                val_dataset, 
                batch_size=self.config['training']['batch_size'],
                num_workers=min(4, mp.cpu_count())
            )
            
            test_loader = tft_formatter.create_data_loader(
                test_dataset, 
                batch_size=self.config['training']['batch_size'],
                num_workers=min(4, mp.cpu_count())
            )
            
            create_time = time.time() - start_time
            
            self.logger.info(f"✅ TFT datasets created in {create_time:.2f}s")
            self.logger.info(f"📊 Train samples: {len(train_dataset)}")
            self.logger.info(f"📊 Val samples: {len(val_dataset)}")
            self.logger.info(f"📊 Test samples: {len(test_dataset)}")
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset
            }
            
        except Exception as e:
            self.logger.error(f"❌ TFT dataset creation failed: {str(e)}")
            raise
    
    def train_model_optimized(self, data_loaders: Dict[str, Any]) -> COVID19Trainer:
        """Train model with optimization."""
        self.logger.info("🔄 Training model...")
        
        start_time = time.time()
        
        try:
            # Create model
            model = COVID19TFTModel()
            model.to(self.device)
            
            # Create trainer
            trainer = COVID19Trainer(model)
            
            # Train model
            train_losses, val_losses = trainer.train(
                train_loader=data_loaders['train_loader'],
                val_loader=data_loaders['val_loader'],
                epochs=self.config['training']['epochs'],
                learning_rate=self.config['training']['learning_rate']
            )
            
            train_time = time.time() - start_time
            
            self.logger.info(f"✅ Training complete in {train_time:.2f}s")
            self.logger.info(f"📊 Final train loss: {train_losses[-1]:.4f}")
            self.logger.info(f"📊 Final val loss: {val_losses[-1]:.4f}")
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise
    
    def evaluate_model_optimized(self, trainer: COVID19Trainer, test_loader: Any) -> Dict[str, Any]:
        """Evaluate model with optimization."""
        self.logger.info("🔄 Evaluating model...")
        
        try:
            # Evaluate on test set
            test_loss, test_metrics = trainer.evaluate(test_loader)
            
            self.logger.info(f"✅ Evaluation complete")
            self.logger.info(f"📊 Test loss: {test_loss:.4f}")
            
            # Log metrics
            for metric_name, metric_value in test_metrics.items():
                self.logger.info(f"📊 {metric_name}: {metric_value:.4f}")
            
            return {
                'test_loss': test_loss,
                'test_metrics': test_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {str(e)}")
            return {
                'test_loss': float('inf'),
                'test_metrics': {},
                'error': str(e)
            }
    
    def run_optimized_pipeline(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete optimized pipeline."""
        self.logger.info("🚀 Starting optimized COVID-19 TFT pipeline...")
        
        pipeline_start_time = time.time()
        pipeline_start_memory = self._monitor_memory()
        
        try:
            # Step 1: Load data
            all_patient_data = self.load_data_optimized(sample_size)
            
            # Step 2: Validate data
            valid_patients = self.validate_data_optimized(all_patient_data)
            
            # Step 3: Build timelines
            timelines = self.build_timelines_optimized(valid_patients)
            
            # Step 4: Extract features
            processed_features = self.extract_features_optimized(timelines)
            
            # Step 5: Create TFT datasets
            data_loaders = self.create_tft_datasets_optimized(processed_features)
            
            # Step 6: Train model
            trainer = self.train_model_optimized(data_loaders)
            
            # Step 7: Evaluate model
            evaluation_results = self.evaluate_model_optimized(trainer, data_loaders['test_loader'])
            
            # Calculate pipeline metrics
            pipeline_time = time.time() - pipeline_start_time
            pipeline_end_memory = self._monitor_memory()
            
            results = {
                'pipeline_time': pipeline_time,
                'memory_usage': pipeline_end_memory - pipeline_start_memory,
                'patients_processed': len(valid_patients),
                'timelines_created': len(timelines),
                'features_extracted': len(processed_features),
                'train_samples': len(data_loaders['train_dataset']),
                'val_samples': len(data_loaders['val_dataset']),
                'test_samples': len(data_loaders['test_dataset']),
                'evaluation_results': evaluation_results,
                'trainer': trainer
            }
            
            self.logger.info(f"✅ Optimized pipeline completed successfully!")
            self.logger.info(f"⏱️ Total pipeline time: {pipeline_time:.2f}s")
            self.logger.info(f"💾 Total memory usage: {pipeline_end_memory - pipeline_start_memory:.1f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            raise

def main():
    """Main function for optimized training."""
    print("🚀 Optimized COVID-19 TFT Training for 12,000+ Patients")
    print("="*60)
    
    # Initialize optimized trainer
    trainer = OptimizedCOVID19Trainer()
    
    # Run pipeline with 12,000 patients
    sample_size = 12000
    
    try:
        results = trainer.run_optimized_pipeline(sample_size=sample_size)
        
        # Print summary
        print("\n📊 Pipeline Results Summary:")
        print("="*40)
        print(f"⏱️ Total time: {results['pipeline_time']:.2f} seconds")
        print(f"💾 Memory usage: {results['memory_usage']:.1f}%")
        print(f"👥 Patients processed: {results['patients_processed']}")
        print(f"📈 Timelines created: {results['timelines_created']}")
        print(f"🔧 Features extracted: {results['features_extracted']}")
        print(f"📊 Train samples: {results['train_samples']}")
        print(f"📊 Val samples: {results['val_samples']}")
        print(f"📊 Test samples: {results['test_samples']}")
        
        if 'test_loss' in results['evaluation_results']:
            print(f"📊 Test loss: {results['evaluation_results']['test_loss']:.4f}")
        
        print("\n✅ Pipeline completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return None

if __name__ == "__main__":
    results = main() 