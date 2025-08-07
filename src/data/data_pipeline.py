#!/usr/bin/env python3
"""
DataPipeline: Complete data processing pipeline for TFT model
Part of the COVID-19 TFT Severity Prediction project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
import torch
from torch.utils.data import DataLoader
import yaml
import pickle
import json

from data.synthea_loader import SyntheaLoader
from data.timeline_builder import TimelineBuilder
from data.feature_engineer import FeatureEngineer
from data.tft_formatter import TFTFormatter, TFTDataset
from data.data_validator import DataValidator

class DataPipeline:
    """
    Complete data processing pipeline for COVID-19 TFT severity prediction.
    
    This class orchestrates the entire data processing pipeline from raw Synthea data
    to TFT-ready datasets, including loading, validation, feature engineering,
    timeline building, and formatting.
    
    Attributes:
        config (Dict): Pipeline configuration
        synthea_loader (SyntheaLoader): Data loader
        timeline_builder (TimelineBuilder): Timeline constructor
        feature_engineer (FeatureEngineer): Feature extractor and engineer
        tft_formatter (TFTFormatter): TFT data formatter
        data_validator (DataValidator): Data quality validator
        logger (logging.Logger): Pipeline logger
    """
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize DataPipeline.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Initialize components
        self.synthea_loader = SyntheaLoader(
            data_path=self.config['data']['raw_data_path']
        )
        
        self.timeline_builder = TimelineBuilder(
            max_sequence_length=self.config['processing']['timeline']['max_sequence_length'],
            time_interval=self.config['processing']['timeline']['time_interval'],
            padding_strategy=self.config['processing']['timeline']['padding_strategy']
        )
        
        self.feature_engineer = FeatureEngineer()
        
        self.tft_formatter = TFTFormatter(
            max_encoder_length=self.config['processing']['timeline']['max_sequence_length'],
            max_prediction_length=24,  # 24 hours ahead
            static_features_dim=15,
            temporal_features_dim=26  # Updated to match actual features
        )
        
        self.data_validator = DataValidator(
            max_missing_percentage=self.config['processing']['quality']['max_missing_percentage'],
            outlier_detection=self.config['processing']['quality']['outlier_detection'],
            data_validation=self.config['processing']['quality']['data_validation']
        )
        
        # Create output directories
        self._create_output_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for pipeline operations."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.config['data']['processed_data_path'],
            self.config['data']['cache_path'],
            "data/validation_reports",
            "data/processed_datasets"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_complete_pipeline(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline.
        
        Args:
            sample_size (Optional[int]): If provided, only process this many patients for testing
        
        Returns:
            Dict[str, Any]: Pipeline results with datasets and metadata
        """
        self.logger.info("Starting complete data processing pipeline")
        
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'datasets': {},
            'metadata': {},
            'validation_results': {},
            'processing_stats': {}
        }
        
        try:
            # Step 1: Load raw data
            self.logger.info("Step 1: Loading raw Synthea data")
            raw_data = self._load_raw_data(sample_size=sample_size)
            pipeline_results['processing_stats']['raw_data_loaded'] = len(raw_data)
            
            # Step 2: Validate data quality
            self.logger.info("Step 2: Validating data quality")
            validation_results, valid_patients = self._validate_data(raw_data)
            pipeline_results['validation_results'] = validation_results
            pipeline_results['processing_stats']['valid_patients'] = len(valid_patients)
            
            # Step 3: Create temporal splits
            self.logger.info("Step 3: Creating temporal splits")
            split_data = self._create_temporal_splits(valid_patients)
            pipeline_results['processing_stats']['splits_created'] = {
                'train': len(split_data['train']),
                'validation': len(split_data['validation']),
                'test': len(split_data['test'])
            }
            
            # Step 4: Process each split
            self.logger.info("Step 4: Processing data splits")
            processed_datasets = self._process_splits(split_data)
            pipeline_results['datasets'] = processed_datasets
            
            # Step 5: Save processed data
            self.logger.info("Step 5: Saving processed data")
            self._save_processed_data(processed_datasets, pipeline_results)
            
            # Step 6: Generate metadata
            self.logger.info("Step 6: Generating metadata")
            pipeline_results['metadata'] = self._generate_metadata(processed_datasets)
            
            self.logger.info("Data processing pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            raise
        
        return pipeline_results
    
    def _load_raw_data(self, sample_size: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load raw data from Synthea dataset.
        
        Args:
            sample_size (Optional[int]): If provided, only load this many patients for testing
        """
        # Load all tables
        self.synthea_loader.load_all_tables()
        
        # Identify COVID-19 patients
        covid19_patients = self.synthea_loader.identify_covid19_patients()
        
        # If sample_size is provided, take only that many patients
        if sample_size is not None:
            covid19_patients = list(covid19_patients)[:sample_size]
            self.logger.info(f"Using sample of {sample_size} patients for testing")
        
        # Get data for COVID-19 patients
        all_patient_data = {}
        for patient_id in covid19_patients:
            patient_data = self.synthea_loader.get_patient_data(patient_id)
            if patient_data:
                all_patient_data[patient_id] = patient_data
        
        self.logger.info(f"Loaded data for {len(all_patient_data)} COVID-19 patients")
        return all_patient_data
    
    def _validate_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, pd.DataFrame]]]:
        """Validate data quality and return valid patients."""
        validation_results, valid_patients = self.data_validator.validate_data_quality(raw_data)
        
        # Save validation report
        self.data_validator.save_validation_report(
            validation_results,
            f"data/validation_reports/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        return validation_results, valid_patients
    
    def _create_temporal_splits(self, valid_patients: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """Create temporal splits for train/validation/test."""
        return self.data_validator.create_temporal_split(
            valid_patients,
            test_size=0.1,
            validation_size=0.15
        )
    
    def _process_splits(self, split_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]]) -> Dict[str, Any]:
        """Process each data split through the complete pipeline."""
        processed_datasets = {}
        
        for split_name, patient_data in split_data.items():
            self.logger.info(f"Processing {split_name} split with {len(patient_data)} patients")
            
            # Build timelines
            timelines = self.timeline_builder.build_all_timelines(patient_data)
            
            # Process each patient
            processed_patients = []
            
            for patient_id, patient_data_dict in patient_data.items():
                if patient_id in timelines:
                    timeline = timelines[patient_id]
                    
                    # Extract features
                    static_features = self.feature_engineer.extract_static_features(patient_data_dict)
                    temporal_features = self.feature_engineer.extract_temporal_features(timeline)
                    
                    # Engineer derived features
                    derived_features = self.feature_engineer.engineer_derived_features(
                        static_features, temporal_features
                    )
                    
                    # Preprocess features
                    static_array, temporal_array = self.feature_engineer.preprocess_features(
                        static_features, temporal_features
                    )
                    
                    # Format for TFT
                    tft_data = self.tft_formatter.format_patient_data(
                        timeline, static_array, temporal_array
                    )
                    
                    # Create targets
                    targets = self.tft_formatter.create_targets(timeline)
                    tft_data.update(targets)
                    
                    processed_patients.append(tft_data)
            
            # Create dataset
            dataset = self.tft_formatter.create_tft_dataset(processed_patients)
            
            processed_datasets[split_name] = {
                'dataset': dataset,
                'patient_count': len(processed_patients),
                'feature_dimensions': dataset.get_feature_dimensions()
            }
            
            self.logger.info(f"Processed {split_name} split: {len(processed_patients)} patients")
        
        return processed_datasets
    
    def _save_processed_data(self, processed_datasets: Dict[str, Any], pipeline_results: Dict[str, Any]):
        """Save processed data and pipeline results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save pipeline results
        results_path = f"data/processed_datasets/pipeline_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Save datasets (save metadata only, not the actual tensors)
        datasets_metadata = {}
        for split_name, split_data in processed_datasets.items():
            datasets_metadata[split_name] = {
                'patient_count': split_data['patient_count'],
                'feature_dimensions': split_data['feature_dimensions']
            }
        
        metadata_path = f"data/processed_datasets/datasets_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(datasets_metadata, f, indent=2)
        
        self.logger.info(f"Saved pipeline results to {results_path}")
        self.logger.info(f"Saved datasets metadata to {metadata_path}")
    
    def _generate_metadata(self, processed_datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata about the processed datasets."""
        metadata = {
            'total_patients': 0,
            'feature_dimensions': {},
            'split_distribution': {},
            'data_quality_metrics': {},
            'processing_timestamp': datetime.now().isoformat()
        }
        
        total_patients = 0
        for split_name, split_data in processed_datasets.items():
            patient_count = split_data['patient_count']
            total_patients += patient_count
            
            metadata['split_distribution'][split_name] = {
                'patient_count': patient_count,
                'feature_dimensions': split_data['feature_dimensions']
            }
        
        metadata['total_patients'] = total_patients
        
        # Calculate feature dimensions (should be consistent across splits)
        if processed_datasets:
            first_split = list(processed_datasets.values())[0]
            metadata['feature_dimensions'] = first_split['feature_dimensions']
        
        return metadata
    
    def get_data_loaders(self, processed_datasets: Dict[str, Any], 
                         batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing.
        
        Args:
            processed_datasets (Dict[str, Any]): Processed datasets
            batch_size (int): Batch size for DataLoaders
            
        Returns:
            Dict[str, DataLoader]: DataLoaders for each split
        """
        data_loaders = {}
        
        for split_name, split_data in processed_datasets.items():
            dataset = split_data['dataset']
            
            # Use different shuffle settings for different splits
            shuffle = split_name == 'train'
            
            data_loader = self.tft_formatter.create_data_loader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4
            )
            
            data_loaders[split_name] = data_loader
        
        return data_loaders
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features used in the pipeline."""
        static_names, temporal_names = self.feature_engineer.get_feature_names()
        
        return {
            'static_features': {
                'names': static_names,
                'count': len(static_names),
                'dimensions': 15
            },
            'temporal_features': {
                'names': temporal_names,
                'count': len(temporal_names),
                'dimensions': 50
            },
            'targets': {
                'mortality_risk': 'binary_classification',
                'icu_admission': 'binary_classification',
                'ventilator_need': 'binary_classification',
                'length_of_stay': 'regression'
            }
        }
    
    def save_pipeline_state(self, output_path: str = "data/pipeline_state.pkl"):
        """Save pipeline state for reproducibility."""
        pipeline_state = {
            'config': self.config,
            'feature_engineer_scalers': self.feature_engineer.scalers,
            'feature_engineer_encoders': self.feature_engineer.encoders,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        self.logger.info(f"Pipeline state saved to {output_path}")
    
    def load_pipeline_state(self, state_path: str = "data/pipeline_state.pkl"):
        """Load pipeline state for reproducibility."""
        with open(state_path, 'rb') as f:
            pipeline_state = pickle.load(f)
        
        # Restore scalers and encoders
        self.feature_engineer.scalers = pipeline_state['feature_engineer_scalers']
        self.feature_engineer.encoders = pipeline_state['feature_engineer_encoders']
        
        self.logger.info(f"Pipeline state loaded from {state_path}")


def main():
    """Main function to run the data pipeline."""
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Total patients processed: {results['processing_stats']['valid_patients']}")
    print(f"Train split: {results['processing_stats']['splits_created']['train']} patients")
    print(f"Validation split: {results['processing_stats']['splits_created']['validation']} patients")
    print(f"Test split: {results['processing_stats']['splits_created']['test']} patients")
    print("="*50)
    
    # Save pipeline state
    pipeline.save_pipeline_state()
    
    return results


if __name__ == "__main__":
    main() 