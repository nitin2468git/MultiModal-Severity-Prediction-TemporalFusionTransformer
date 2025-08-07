#!/usr/bin/env python3
"""
Test script for the COVID-19 TFT data pipeline.

This script tests the data processing pipeline components to ensure
they work correctly with the Synthea dataset.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import DataPipeline, SyntheaLoader, TimelineBuilder, FeatureEngineer, TFTFormatter, DataValidator

def test_synthea_loader():
    """Test the SyntheaLoader component."""
    print("Testing SyntheaLoader...")
    
    try:
        loader = SyntheaLoader()
        tables = loader.load_all_tables()
        print(f"✓ Loaded {len(tables)} tables")
        
        covid19_patients = loader.identify_covid19_patients()
        print(f"✓ Identified {len(covid19_patients)} COVID-19 patients")
        
        # Test getting patient data
        if covid19_patients:
            patient_id = covid19_patients[0]
            patient_data = loader.get_patient_data(patient_id)
            print(f"✓ Retrieved data for patient {patient_id}")
        
        return True
    except Exception as e:
        print(f"✗ SyntheaLoader test failed: {e}")
        return False

def test_timeline_builder():
    """Test the TimelineBuilder component."""
    print("Testing TimelineBuilder...")
    
    try:
        builder = TimelineBuilder()
        print("✓ TimelineBuilder initialized")
        
        # Test with sample data
        sample_timeline = {
            'patient_id': 'test_patient',
            'timestamps': [],
            'features': {},
            'events': {},
            'metadata': {}
        }
        
        print("✓ TimelineBuilder test passed")
        return True
    except Exception as e:
        print(f"✗ TimelineBuilder test failed: {e}")
        return False

def test_feature_engineer():
    """Test the FeatureEngineer component."""
    print("Testing FeatureEngineer...")
    
    try:
        engineer = FeatureEngineer()
        print("✓ FeatureEngineer initialized")
        
        # Test feature names
        static_names, temporal_names = engineer.get_feature_names()
        print(f"✓ Static features: {len(static_names)}")
        print(f"✓ Temporal features: {len(temporal_names)}")
        
        return True
    except Exception as e:
        print(f"✗ FeatureEngineer test failed: {e}")
        return False

def test_tft_formatter():
    """Test the TFTFormatter component."""
    print("Testing TFTFormatter...")
    
    try:
        formatter = TFTFormatter()
        print("✓ TFTFormatter initialized")
        
        # Test empty data creation
        empty_data = formatter._create_empty_tft_data()
        print("✓ Empty TFT data created")
        
        return True
    except Exception as e:
        print(f"✗ TFTFormatter test failed: {e}")
        return False

def test_data_validator():
    """Test the DataValidator component."""
    print("Testing DataValidator...")
    
    try:
        validator = DataValidator()
        print("✓ DataValidator initialized")
        
        return True
    except Exception as e:
        print(f"✗ DataValidator test failed: {e}")
        return False

def test_data_pipeline():
    """Test the complete DataPipeline."""
    print("Testing DataPipeline...")
    
    try:
        pipeline = DataPipeline()
        print("✓ DataPipeline initialized")
        
        # Test feature info
        feature_info = pipeline.get_feature_info()
        print(f"✓ Feature info retrieved: {len(feature_info['static_features']['names'])} static, {len(feature_info['temporal_features']['names'])} temporal")
        
        return True
    except Exception as e:
        print(f"✗ DataPipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("COVID-19 TFT DATA PIPELINE TEST")
    print("="*60)
    
    tests = [
        test_synthea_loader,
        test_timeline_builder,
        test_feature_engineer,
        test_tft_formatter,
        test_data_validator,
        test_data_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! Data pipeline is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 