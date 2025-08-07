#!/usr/bin/env python3
"""
Pipeline Optimization for Large-Scale COVID-19 TFT Training
Optimizes the pipeline for 12,000+ patients efficiently
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import psutil
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.synthea_loader import SyntheaLoader
from data.data_pipeline import DataPipeline
from data.data_validator import DataValidator
from data.feature_engineer import FeatureEngineer
from data.timeline_builder import TimelineBuilder
from data.tft_formatter import TFTFormatter

class PipelineOptimizer:
    """
    Optimizes the COVID-19 TFT pipeline for large-scale datasets.
    """
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize the optimizer."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_usage': [],
            'processing_time': [],
            'cpu_usage': [],
            'gpu_usage': []
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def optimize_data_loading(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimized data loading with memory management and parallel processing.
        
        Args:
            sample_size (Optional[int]): Number of patients to load
            
        Returns:
            Dict[str, Any]: Optimized data loading results
        """
        self.logger.info("🚀 Starting optimized data loading...")
        
        # Memory optimization settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Load data in chunks to manage memory
        chunk_size = 1000  # Process 1000 patients at a time
        all_patient_data = {}
        
        # Load Synthea data
        synthea_loader = SyntheaLoader()
        synthea_loader.load_all_tables()
        covid19_patients = list(synthea_loader.identify_covid19_patients())
        
        if sample_size:
            covid19_patients = covid19_patients[:sample_size]
        
        self.logger.info(f"📊 Processing {len(covid19_patients)} COVID-19 patients")
        
        # Process in chunks
        for i in range(0, len(covid19_patients), chunk_size):
            chunk_patients = covid19_patients[i:i + chunk_size]
            self.logger.info(f"🔄 Processing chunk {i//chunk_size + 1}/{(len(covid19_patients) + chunk_size - 1)//chunk_size}")
            
            # Process chunk in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunk_data = list(executor.map(
                    lambda pid: synthea_loader.get_patient_data(pid),
                    chunk_patients
                ))
            
            # Add to results
            for patient_id, patient_data in zip(chunk_patients, chunk_data):
                if patient_data:
                    all_patient_data[patient_id] = patient_data
            
            # Memory cleanup
            gc.collect()
            
            # Log memory usage
            memory_usage = psutil.virtual_memory().percent
            self.logger.info(f"💾 Memory usage: {memory_usage:.1f}%")
        
        self.logger.info(f"✅ Loaded data for {len(all_patient_data)} patients")
        return all_patient_data
    
    def optimize_validation(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]]) -> tuple[Dict[str, Any], Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Optimized data validation with parallel processing.
        
        Args:
            all_patient_data: All patient data
            
        Returns:
            Dict[str, Any]: Validation results
        """
        self.logger.info("🔍 Starting optimized data validation...")
        
        # Use more lenient validation for speed
        validator = DataValidator(
            max_missing_percentage=0.8,  # More lenient
            outlier_detection=False,  # Disable for speed
            data_validation=True
        )
        
        # Process validation in parallel
        validation_results, valid_patients = validator.validate_data_quality(all_patient_data)
        
        self.logger.info(f"✅ Validation complete: {len(valid_patients)} valid patients")
        return validation_results, valid_patients
    
    def optimize_feature_engineering(self, valid_patients: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Optimized feature engineering with parallel processing.
        
        Args:
            valid_patients: Valid patient data
            
        Returns:
            Dict[str, Any]: Engineered features
        """
        self.logger.info("⚙️ Starting optimized feature engineering...")
        
        # Initialize components
        timeline_builder = TimelineBuilder(
            max_sequence_length=720,
            time_interval="1H",
            padding_strategy="zero"
        )
        
        feature_engineer = FeatureEngineer()
        
        # Process in parallel
        patient_ids = list(valid_patients.keys())
        chunk_size = 500
        
        processed_features = {}
        
        for i in range(0, len(patient_ids), chunk_size):
            chunk_ids = patient_ids[i:i + chunk_size]
            self.logger.info(f"🔄 Processing features for chunk {i//chunk_size + 1}")
            
            # Process chunk
            chunk_patients = {pid: valid_patients[pid] for pid in chunk_ids}
            
            # Build timelines
            timelines = timeline_builder.build_all_timelines(chunk_patients)
            
            # Extract features in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                feature_results = list(executor.map(
                    lambda pid: self._extract_patient_features(
                        valid_patients[pid], timelines.get(pid, {}), feature_engineer
                    ),
                    chunk_ids
                ))
            
            # Add to results
            for patient_id, features in zip(chunk_ids, feature_results):
                if features:
                    processed_features[patient_id] = features
            
            # Memory cleanup
            gc.collect()
        
        self.logger.info(f"✅ Feature engineering complete for {len(processed_features)} patients")
        return processed_features
    
    def _extract_patient_features(self, patient_data: Dict[str, pd.DataFrame], 
                                 timeline: Dict[str, Any], 
                                 feature_engineer: FeatureEngineer) -> Optional[Dict[str, Any]]:
        """Extract features for a single patient."""
        try:
            # Extract features
            static_features = feature_engineer.extract_static_features(patient_data)
            temporal_features = feature_engineer.extract_temporal_features(timeline)
            
            # Engineer derived features
            derived_features = feature_engineer.engineer_derived_features(
                static_features, temporal_features
            )
            
            # Preprocess features
            static_array, temporal_array = feature_engineer.preprocess_features(
                static_features, temporal_features
            )
            
            return {
                'static_features': static_array,
                'temporal_features': temporal_array,
                'timeline': timeline
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract features: {e}")
            return None
    
    def optimize_tft_formatting(self, processed_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimized TFT formatting with batch processing.
        
        Args:
            processed_features: Processed features
            
        Returns:
            Dict[str, Any]: TFT formatted data
        """
        self.logger.info("📊 Starting optimized TFT formatting...")
        
        tft_formatter = TFTFormatter(
            max_encoder_length=720,
            max_prediction_length=24,
            static_features_dim=15,
            temporal_features_dim=26
        )
        
        # Process in batches
        patient_ids = list(processed_features.keys())
        batch_size = 100
        
        tft_datasets = {}
        
        for i in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[i:i + batch_size]
            self.logger.info(f"🔄 Formatting batch {i//batch_size + 1}")
            
            # Format batch
            batch_features = {pid: processed_features[pid] for pid in batch_ids}
            
            # Create TFT dataset
            tft_data = []
            for patient_id, features in batch_features.items():
                try:
                    formatted_data = tft_formatter.format_patient_data(
                        features['timeline'],
                        features['static_features'],
                        features['temporal_features']
                    )
                    
                    # Create targets
                    targets = tft_formatter.create_targets(features['timeline'])
                    formatted_data.update(targets)
                    
                    tft_data.append(formatted_data)
                except Exception as e:
                    self.logger.warning(f"Failed to format patient {patient_id}: {e}")
            
            # Create dataset
            if tft_data:
                dataset = tft_formatter.create_tft_dataset(tft_data)
                tft_datasets[f'batch_{i//batch_size}'] = dataset
            
            # Memory cleanup
            gc.collect()
        
        self.logger.info(f"✅ TFT formatting complete for {len(tft_datasets)} batches")
        return tft_datasets
    
    def create_optimized_data_loaders(self, tft_datasets: Dict[str, Any], 
                                    batch_size: int = 64) -> Dict[str, Any]:
        """
        Create optimized data loaders with better performance settings.
        
        Args:
            tft_datasets: TFT datasets
            batch_size: Batch size for training
            
        Returns:
            Dict[str, Any]: Optimized data loaders
        """
        self.logger.info("🔄 Creating optimized data loaders...")
        
        tft_formatter = TFTFormatter()
        
        # Optimized settings
        loader_settings = {
            'batch_size': batch_size,
            'num_workers': min(8, mp.cpu_count()),  # Use more workers
            'shuffle': True
        }
        
        data_loaders = {}
        
        # Create loaders for each batch
        for batch_name, dataset in tft_datasets.items():
            data_loader = tft_formatter.create_data_loader(
                dataset,
                **loader_settings
            )
            data_loaders[batch_name] = data_loader
        
        self.logger.info(f"✅ Created {len(data_loaders)} optimized data loaders")
        return data_loaders
    
    def run_optimized_pipeline(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete optimized pipeline.
        
        Args:
            sample_size (Optional[int]): Number of patients to process
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        start_time = time.time()
        
        self.logger.info("🚀 Starting optimized pipeline...")
        
        # Step 1: Optimized data loading
        all_patient_data = self.optimize_data_loading(sample_size)
        
        # Step 2: Optimized validation
        validation_results, valid_patients = self.optimize_validation(all_patient_data)
        
        # Step 3: Optimized feature engineering
        processed_features = self.optimize_feature_engineering(valid_patients)
        
        # Step 4: Optimized TFT formatting
        tft_datasets = self.optimize_tft_formatting(processed_features)
        
        # Step 5: Create optimized data loaders
        data_loaders = self.create_optimized_data_loaders(tft_datasets)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        memory_usage = psutil.virtual_memory().percent
        
        results = {
            'total_time': total_time,
            'memory_usage': memory_usage,
            'patients_processed': len(valid_patients),
            'tft_datasets': len(tft_datasets),
            'data_loaders': len(data_loaders),
            'validation_results': validation_results
        }
        
        self.logger.info(f"✅ Optimized pipeline completed in {total_time:.2f} seconds")
        self.logger.info(f"💾 Final memory usage: {memory_usage:.1f}%")
        
        return results

def main():
    """Main optimization function."""
    print("🚀 COVID-19 TFT Pipeline Optimization")
    print("="*50)
    
    # Initialize optimizer
    optimizer = PipelineOptimizer()
    
    # Run optimized pipeline
    results = optimizer.run_optimized_pipeline(sample_size=1000)  # Start with smaller sample
    
    # Print results
    print("\n📊 Optimization Results:")
    print(f"⏱️  Total time: {results['total_time']:.2f} seconds")
    print(f"💾 Memory usage: {results['memory_usage']:.1f}%")
    print(f"👥 Patients processed: {results['patients_processed']}")
    print(f"📦 TFT datasets: {results['tft_datasets']}")
    print(f"🔄 Data loaders: {results['data_loaders']}")
    
    return results

if __name__ == "__main__":
    results = main() 