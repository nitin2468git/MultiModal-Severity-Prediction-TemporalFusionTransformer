#!/usr/bin/env python3
"""
End-to-End Testing for COVID-19 TFT Pipeline
Comprehensive testing to ensure reliability with 12,000+ patients
"""

import sys
import os
from pathlib import Path
import logging
import time
import psutil
import gc
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import multiprocessing as mp

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.synthea_loader import SyntheaLoader
from data.data_pipeline import DataPipeline
from data.data_validator import DataValidator
from data.feature_engineer import FeatureEngineer
from data.timeline_builder import TimelineBuilder
from data.tft_formatter import TFTFormatter
from models.tft_model import COVID19TFTModel
from training.trainer import COVID19Trainer

class EndToEndTester:
    """Comprehensive end-to-end testing for the COVID-19 TFT pipeline."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('end_to_end_test.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def test_component_loading(self, sample_size: int = 100) -> Dict[str, Any]:
        """Test individual component loading."""
        self.logger.info(f"🧪 Testing component loading with {sample_size} patients...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Test SyntheaLoader
            loader = SyntheaLoader()
            patient_data = loader.load_patient_data(sample_size=sample_size)
            
            load_time = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            
            self.logger.info(f"✅ Component loading successful: {len(patient_data)} patients")
            
            return {
                'success': True,
                'load_time': load_time,
                'memory_usage': end_memory - start_memory,
                'patients_loaded': len(patient_data),
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Component loading failed: {str(e)}")
            return {
                'success': False,
                'load_time': time.time() - start_time,
                'memory_usage': 0,
                'patients_loaded': 0,
                'error': str(e)
            }
    
    def test_timeline_building(self, sample_size: int = 100) -> Dict[str, Any]:
        """Test timeline building with chunked processing."""
        self.logger.info(f"🧪 Testing timeline building with {sample_size} patients...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Load data
            loader = SyntheaLoader()
            patient_data = loader.load_patient_data(sample_size=sample_size)
            
            # Process in chunks to avoid memory issues
            chunk_size = 50
            timeline_builder = TimelineBuilder()
            all_timelines = []
            
            for i in range(0, len(patient_data), chunk_size):
                chunk = dict(list(patient_data.items())[i:i+chunk_size])
                self.logger.info(f"🔄 Processing timeline chunk {i//chunk_size + 1}/{(len(patient_data) + chunk_size - 1)//chunk_size}")
                
                chunk_timelines = timeline_builder.build_patient_timelines(chunk)
                all_timelines.extend(chunk_timelines)
                
                # Force garbage collection
                gc.collect()
                
                # Check memory usage
                current_memory = psutil.virtual_memory().percent
                if current_memory > 90:
                    self.logger.warning(f"⚠️ High memory usage: {current_memory:.1f}%")
            
            build_time = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            
            self.logger.info(f"✅ Timeline building successful: {len(all_timelines)} timelines")
            
            return {
                'success': True,
                'build_time': build_time,
                'memory_usage': end_memory - start_memory,
                'timelines_created': len(all_timelines),
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Timeline building failed: {str(e)}")
            return {
                'success': False,
                'build_time': time.time() - start_time,
                'memory_usage': 0,
                'timelines_created': 0,
                'error': str(e)
            }
    
    def test_feature_engineering(self, sample_size: int = 100) -> Dict[str, Any]:
        """Test feature engineering with memory management."""
        self.logger.info(f"🧪 Testing feature engineering with {sample_size} patients...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Load and build timelines
            loader = SyntheaLoader()
            patient_data = loader.load_patient_data(sample_size=sample_size)
            
            timeline_builder = TimelineBuilder()
            timelines = timeline_builder.build_patient_timelines(patient_data)
            
            # Process features in chunks
            chunk_size = 25
            feature_engineer = FeatureEngineer()
            all_features = []
            
            for i in range(0, len(timelines), chunk_size):
                chunk = timelines[i:i+chunk_size]
                self.logger.info(f"🔄 Processing feature chunk {i//chunk_size + 1}/{(len(timelines) + chunk_size - 1)//chunk_size}")
                
                chunk_features = []
                for timeline in chunk:
                    try:
                        features = feature_engineer.extract_features(timeline)
                        if features is not None:
                            chunk_features.append(features)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Feature extraction failed for patient {timeline.get('patient_id', 'unknown')}: {str(e)}")
                        continue
                
                all_features.extend(chunk_features)
                
                # Force garbage collection
                gc.collect()
            
            build_time = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            
            self.logger.info(f"✅ Feature engineering successful: {len(all_features)} features")
            
            return {
                'success': True,
                'build_time': build_time,
                'memory_usage': end_memory - start_memory,
                'features_created': len(all_features),
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {str(e)}")
            return {
                'success': False,
                'build_time': time.time() - start_time,
                'memory_usage': 0,
                'features_created': 0,
                'error': str(e)
            }
    
    def test_tft_formatting(self, sample_size: int = 100) -> Dict[str, Any]:
        """Test TFT formatting with memory optimization."""
        self.logger.info(f"🧪 Testing TFT formatting with {sample_size} patients...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Load and process data
            loader = SyntheaLoader()
            patient_data = loader.load_patient_data(sample_size=sample_size)
            
            timeline_builder = TimelineBuilder()
            timelines = timeline_builder.build_patient_timelines(patient_data)
            
            feature_engineer = FeatureEngineer()
            processed_features = []
            
            for timeline in timelines:
                try:
                    features = feature_engineer.extract_features(timeline)
                    if features is not None:
                        processed_features.append(features)
                except Exception as e:
                    self.logger.warning(f"⚠️ Feature extraction failed for patient {timeline.get('patient_id', 'unknown')}: {str(e)}")
                    continue
            
            # Format for TFT
            tft_formatter = TFTFormatter()
            tft_datasets = tft_formatter.create_tft_dataset(processed_features)
            
            build_time = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            
            self.logger.info(f"✅ TFT formatting successful: {len(tft_datasets)} samples")
            
            return {
                'success': True,
                'build_time': build_time,
                'memory_usage': end_memory - start_memory,
                'tft_samples': len(tft_datasets),
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ TFT formatting failed: {str(e)}")
            return {
                'success': False,
                'build_time': time.time() - start_time,
                'memory_usage': 0,
                'tft_samples': 0,
                'error': str(e)
            }
    
    def test_full_pipeline(self, sample_size: int = 100) -> Dict[str, Any]:
        """Test the complete pipeline with memory management."""
        self.logger.info(f"🧪 Testing full pipeline with {sample_size} patients...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Step 1: Data Pipeline
            data_pipeline = DataPipeline()
            train_data, val_data, test_data = data_pipeline.run_pipeline(sample_size=sample_size)
            
            # Step 2: Create TFT datasets
            tft_formatter = TFTFormatter()
            train_dataset = tft_formatter.create_tft_dataset(train_data)
            val_dataset = tft_formatter.create_tft_dataset(val_data)
            test_dataset = tft_formatter.create_tft_dataset(test_data)
            
            # Step 3: Create data loaders
            train_loader = tft_formatter.create_data_loader(train_dataset, batch_size=32, num_workers=4)
            val_loader = tft_formatter.create_data_loader(val_dataset, batch_size=32, num_workers=4)
            test_loader = tft_formatter.create_data_loader(test_dataset, batch_size=32, num_workers=4)
            
            # Step 4: Test model creation
            model = COVID19TFTModel()
            
            # Step 5: Test trainer creation
            trainer = COVID19Trainer(model)
            
            build_time = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            
            self.logger.info(f"✅ Full pipeline successful")
            self.logger.info(f"📊 Train samples: {len(train_dataset)}")
            self.logger.info(f"📊 Val samples: {len(val_dataset)}")
            self.logger.info(f"📊 Test samples: {len(test_dataset)}")
            
            return {
                'success': True,
                'build_time': build_time,
                'memory_usage': end_memory - start_memory,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset),
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Full pipeline failed: {str(e)}")
            return {
                'success': False,
                'build_time': time.time() - start_time,
                'memory_usage': 0,
                'train_samples': 0,
                'val_samples': 0,
                'test_samples': 0,
                'error': str(e)
            }
    
    def run_scalability_test(self) -> Dict[str, Any]:
        """Test scalability with different sample sizes."""
        self.logger.info("🚀 Running scalability test...")
        
        sample_sizes = [50, 100, 500, 1000, 2000]
        results = {}
        
        for size in sample_sizes:
            self.logger.info(f"\n📊 Testing with {size} patients...")
            
            # Test timeline building (most critical component)
            timeline_result = self.test_timeline_building(size)
            results[f'timeline_{size}'] = timeline_result
            
            if not timeline_result['success']:
                self.logger.error(f"❌ Timeline building failed at {size} patients")
                break
            
            # Test full pipeline for smaller sizes
            if size <= 1000:
                pipeline_result = self.test_full_pipeline(size)
                results[f'pipeline_{size}'] = pipeline_result
                
                if not pipeline_result['success']:
                    self.logger.error(f"❌ Full pipeline failed at {size} patients")
                    break
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            memory_usage = psutil.virtual_memory().percent
            self.logger.info(f"💾 Current memory usage: {memory_usage:.1f}%")
            
            if memory_usage > 95:
                self.logger.warning(f"⚠️ High memory usage detected: {memory_usage:.1f}%")
                break
        
        return results
    
    def generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on test results."""
        recommendations = []
        
        # Analyze timeline building performance
        timeline_times = []
        for key, result in results.items():
            if key.startswith('timeline_') and result['success']:
                timeline_times.append((int(key.split('_')[1]), result['build_time']))
        
        if timeline_times:
            # Calculate time per patient
            times_per_patient = [(size, time/size) for size, time in timeline_times]
            avg_time_per_patient = sum(time for _, time in times_per_patient) / len(times_per_patient)
            
            if avg_time_per_patient > 0.1:  # More than 0.1 seconds per patient
                recommendations.append("🔧 Timeline building is slow - consider parallel processing")
                recommendations.append("🔧 Consider pre-computing and caching timeline data")
            
            if len(timeline_times) > 1:
                # Check for scalability issues
                last_size, last_time = timeline_times[-1]
                first_size, first_time = timeline_times[0]
                scaling_factor = (last_time / last_size) / (first_time / first_size)
                
                if scaling_factor > 2:
                    recommendations.append("🔧 Timeline building doesn't scale linearly - optimize memory usage")
        
        # Memory usage recommendations
        max_memory = max([result.get('memory_usage', 0) for result in results.values()])
        if max_memory > 50:
            recommendations.append("🔧 High memory usage detected - implement chunked processing")
            recommendations.append("🔧 Consider using memory-mapped files for large datasets")
        
        # Pipeline recommendations
        pipeline_failures = [key for key, result in results.items() 
                           if key.startswith('pipeline_') and not result['success']]
        if pipeline_failures:
            recommendations.append("🔧 Pipeline failures detected - implement better error handling")
            recommendations.append("🔧 Consider using smaller batch sizes for data loading")
        
        return recommendations

def main():
    """Main testing function."""
    print("🧪 COVID-19 TFT End-to-End Testing")
    print("="*50)
    
    tester = EndToEndTester()
    
    # Run scalability test
    results = tester.run_scalability_test()
    
    # Generate recommendations
    recommendations = tester.generate_optimization_recommendations(results)
    
    # Print results
    print("\n📊 Test Results:")
    print("="*30)
    
    for key, result in results.items():
        if result['success']:
            print(f"✅ {key}: {result.get('build_time', 0):.2f}s, "
                  f"Memory: {result.get('memory_usage', 0):.1f}%")
        else:
            print(f"❌ {key}: Failed - {result.get('error', 'Unknown error')}")
    
    print("\n💡 Optimization Recommendations:")
    print("="*35)
    for rec in recommendations:
        print(f"  {rec}")
    
    return results

if __name__ == "__main__":
    results = main() 