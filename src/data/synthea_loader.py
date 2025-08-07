#!/usr/bin/env python3
"""
SyntheaLoader: Load and validate Synthea COVID-19 dataset
Part of the COVID-19 TFT Severity Prediction project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import warnings

class SyntheaLoader:
    """
    Load and validate Synthea COVID-19 dataset for TFT model training.
    
    This class handles loading all CSV files from the Synthea dataset,
    identifying COVID-19 patients, and preparing data for the TFT model.
    
    Attributes:
        data_path (Path): Path to the Synthea CSV dataset
        tables (Dict[str, pd.DataFrame]): Loaded CSV tables
        covid19_patients (List[str]): List of patient IDs with COVID-19
        logger (logging.Logger): Logger for data loading operations
    """
    
    def __init__(self, data_path: str = "10k_synthea_covid19_csv"):
        """
        Initialize SyntheaLoader.
        
        Args:
            data_path (str): Path to the Synthea CSV dataset directory
        """
        self.data_path = Path(data_path)
        self.tables: Dict[str, pd.DataFrame] = {}
        self.covid19_patients: List[str] = []
        self.logger = self._setup_logger()
        
        # COVID-19 related keywords for patient identification
        self.covid19_keywords = [
            'covid', 'coronavirus', 'sars-cov-2', 'covid-19', 'covid19',
            'pneumonia', 'respiratory', 'acute respiratory'
        ]
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data loading operations."""
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
    
    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the Synthea dataset.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of table names to DataFrames
        """
        self.logger.info(f"Loading tables from {self.data_path}")
        
        csv_files = list(self.data_path.glob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            table_name = csv_file.stem
            try:
                self.tables[table_name] = pd.read_csv(csv_file)
                self.logger.info(f"Loaded {table_name}: {self.tables[table_name].shape}")
            except Exception as e:
                self.logger.error(f"Failed to load {table_name}: {e}")
                
        return self.tables
    
    def identify_covid19_patients(self) -> List[str]:
        """
        Identify patients with COVID-19 from conditions and observations.
        
        Returns:
            List[str]: List of patient IDs with COVID-19
        """
        self.logger.info("Identifying COVID-19 patients...")
        
        covid19_patients = set()
        
        # Check conditions table
        if 'conditions' in self.tables:
            conditions_df = self.tables['conditions']
            covid19_conditions = conditions_df[
                conditions_df['DESCRIPTION'].str.lower().str.contains(
                    '|'.join(self.covid19_keywords), 
                    case=False, 
                    na=False
                )
            ]
            covid19_patients.update(covid19_conditions['PATIENT'].unique())
            self.logger.info(f"Found {len(covid19_conditions)} COVID-19 conditions")
        
        # Check observations table
        if 'observations' in self.tables:
            observations_df = self.tables['observations']
            covid19_observations = observations_df[
                observations_df['DESCRIPTION'].str.lower().str.contains(
                    '|'.join(self.covid19_keywords), 
                    case=False, 
                    na=False
                )
            ]
            covid19_patients.update(covid19_observations['PATIENT'].unique())
            self.logger.info(f"Found {len(covid19_observations)} COVID-19 observations")
        
        self.covid19_patients = list(covid19_patients)
        self.logger.info(f"Total COVID-19 patients identified: {len(self.covid19_patients)}")
        
        return self.covid19_patients
    
    def get_patient_data(self, patient_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific patient.
        
        Args:
            patient_id (str): Patient ID to retrieve data for
            
        Returns:
            Dict[str, pd.DataFrame]: Patient data organized by table
        """
        patient_data = {}
        
        for table_name, table_df in self.tables.items():
            if 'PATIENT' in table_df.columns:
                patient_data[table_name] = table_df[
                    table_df['PATIENT'] == patient_id
                ].copy()
        
        return patient_data
    
    def validate_data_quality(self) -> Dict[str, any]:
        """
        Validate data quality and completeness.
        
        Returns:
            Dict[str, any]: Data quality metrics
        """
        self.logger.info("Validating data quality...")
        
        quality_metrics = {
            'total_patients': len(self.tables.get('patients', pd.DataFrame()).get('Id', []).unique()),
            'covid19_patients': len(self.covid19_patients),
            'tables_loaded': len(self.tables),
            'missing_values': {},
            'data_types': {},
            'temporal_coverage': {}
        }
        
        # Check missing values
        for table_name, table_df in self.tables.items():
            quality_metrics['missing_values'][table_name] = table_df.isnull().sum().to_dict()
            
        # Check data types
        for table_name, table_df in self.tables.items():
            quality_metrics['data_types'][table_name] = table_df.dtypes.to_dict()
            
        # Check temporal coverage for encounters
        if 'encounters' in self.tables:
            encounters_df = self.tables['encounters']
            if 'START' in encounters_df.columns:
                start_dates = pd.to_datetime(encounters_df['START'], errors='coerce')
                end_dates = pd.to_datetime(encounters_df.get('END', encounters_df['START']), errors='coerce')
                
                quality_metrics['temporal_coverage'] = {
                    'start_date': start_dates.min(),
                    'end_date': end_dates.max(),
                    'date_range_days': (end_dates.max() - start_dates.min()).days
                }
        
        self.logger.info(f"Data quality validation complete: {quality_metrics['total_patients']} total patients, {quality_metrics['covid19_patients']} COVID-19 patients")
        
        return quality_metrics
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get comprehensive data summary for the dataset.
        
        Returns:
            Dict[str, any]: Data summary including patient demographics, conditions, etc.
        """
        summary = {
            'dataset_info': {
                'name': 'Synthea COVID-19 Dataset',
                'tables_loaded': len(self.tables),
                'total_patients': len(self.tables.get('patients', pd.DataFrame()).get('Id', []).unique()),
                'covid19_patients': len(self.covid19_patients)
            },
            'table_info': {},
            'covid19_analysis': {}
        }
        
        # Table information
        for table_name, table_df in self.tables.items():
            summary['table_info'][table_name] = {
                'rows': len(table_df),
                'columns': len(table_df.columns),
                'memory_usage': table_df.memory_usage(deep=True).sum()
            }
        
        # COVID-19 specific analysis
        if self.covid19_patients:
            covid19_patients_df = self.tables.get('patients', pd.DataFrame())
            if not covid19_patients_df.empty and 'Id' in covid19_patients_df.columns:
                covid19_patient_data = covid19_patients_df[
                    covid19_patients_df['Id'].isin(self.covid19_patients)
                ]
                
                summary['covid19_analysis'] = {
                    'total_covid19_patients': len(self.covid19_patients),
                    'demographics': {
                        'age_distribution': covid19_patient_data.get('BIRTHDATE', pd.Series()).describe().to_dict() if 'BIRTHDATE' in covid19_patient_data.columns else {},
                        'gender_distribution': covid19_patient_data.get('GENDER', pd.Series()).value_counts().to_dict() if 'GENDER' in covid19_patient_data.columns else {},
                        'race_distribution': covid19_patient_data.get('RACE', pd.Series()).value_counts().to_dict() if 'RACE' in covid19_patient_data.columns else {}
                    }
                }
        
        return summary 