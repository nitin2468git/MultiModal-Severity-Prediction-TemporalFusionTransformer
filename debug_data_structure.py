#!/usr/bin/env python3
"""
Debug script to check data structure and validation issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from data.synthea_loader import SyntheaLoader

def debug_data_structure():
    print("🔍 Debugging data structure...")
    
    # Load data
    loader = SyntheaLoader()
    loader.load_all_tables()
    covid_patients = loader.identify_covid19_patients()
    
    print(f"📊 Found {len(covid_patients)} COVID-19 patients")
    
    # Check sample patient data
    sample_patient = covid_patients[0]
    print(f"🔍 Checking patient: {sample_patient}")
    
    patient_data = loader.get_patient_data(sample_patient)
    print(f"📋 Patient data keys: {list(patient_data.keys())}")
    
    # Check each table
    for table_name, table_data in patient_data.items():
        print(f"  📄 {table_name}: {table_data.shape if hasattr(table_data, 'shape') else 'No shape'}")
        if not table_data.empty:
            print(f"    Columns: {list(table_data.columns)}")
            print(f"    Sample data:")
            print(table_data.head(2))
            print()
    
    # Check if 'patients' table exists
    if 'patients' in patient_data:
        print("✅ 'patients' table found!")
        print(f"   Shape: {patient_data['patients'].shape}")
        print(f"   Columns: {list(patient_data['patients'].columns)}")
    else:
        print("❌ 'patients' table NOT found!")
        print("Available tables:", list(patient_data.keys()))

if __name__ == "__main__":
    debug_data_structure() 