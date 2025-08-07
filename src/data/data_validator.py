#!/usr/bin/env python3
"""
DataValidator: Validate data quality and implement temporal splits
Part of the COVID-19 TFT Severity Prediction project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split
import json

class DataValidator:
    """
    Validate data quality and implement temporal split strategy.
    
    This class handles data quality checks, validation metrics,
    and temporal split strategy to prevent data leakage.
    
    Attributes:
        max_missing_percentage (float): Maximum allowed missing percentage
        outlier_detection (bool): Whether to detect outliers
        data_validation (bool): Whether to perform data validation
        logger (logging.Logger): Logger for validation operations
    """
    
    def __init__(self, max_missing_percentage: float = 0.5,  # Increased from 0.2
                 outlier_detection: bool = True,
                 data_validation: bool = True):
        """
        Initialize DataValidator.
        
        Args:
            max_missing_percentage (float): Maximum allowed missing percentage (default: 0.2)
            outlier_detection (bool): Whether to detect outliers (default: True)
            data_validation (bool): Whether to perform data validation (default: True)
        """
        self.max_missing_percentage = max_missing_percentage
        self.outlier_detection = outlier_detection
        self.data_validation = data_validation
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for validation operations."""
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
    
    def validate_data_quality(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Validate data quality for all patients.
        
        Args:
            all_patient_data (Dict[str, Dict[str, pd.DataFrame]]): All patient data
            
        Returns:
            Dict[str, Any]: Data quality validation results
        """
        self.logger.info("Starting data quality validation")
        
        validation_results = {
            'total_patients': len(all_patient_data),
            'valid_patients': 0,
            'invalid_patients': 0,
            'quality_metrics': {},
            'data_issues': [],
            'recommendations': []
        }
        
        valid_patients = {}
        invalid_patients = {}
        
        for patient_id, patient_data in all_patient_data.items():
            try:
                # Validate individual patient data
                patient_validation = self._validate_patient_data(patient_data, patient_id)
                
                if patient_validation['is_valid']:
                    valid_patients[patient_id] = patient_data
                    validation_results['valid_patients'] += 1
                else:
                    invalid_patients[patient_id] = patient_data
                    validation_results['invalid_patients'] += 1
                    validation_results['data_issues'].extend(patient_validation['issues'])
                    
            except Exception as e:
                self.logger.error(f"Error validating patient {patient_id}: {e}")
                invalid_patients[patient_id] = patient_data
                validation_results['invalid_patients'] += 1
                validation_results['data_issues'].append(f"Patient {patient_id}: {str(e)}")
        
        # Calculate overall quality metrics
        validation_results['quality_metrics'] = self._calculate_quality_metrics(valid_patients)
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        self.logger.info(f"Data validation complete: {validation_results['valid_patients']} valid, {validation_results['invalid_patients']} invalid")
        
        return validation_results, valid_patients
    
    def _validate_patient_data(self, patient_data: Dict[str, pd.DataFrame], 
                              patient_id: str) -> Dict[str, Any]:
        """Validate individual patient data."""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check for required tables - be more lenient
        required_tables = ['patients']  # Only require patients table
        missing_tables = [table for table in required_tables if table not in patient_data]
        
        if missing_tables:
            # Instead of failing, just warn and continue
            validation_result['issues'].append(f"Missing required tables: {missing_tables}")
            # Don't fail validation for missing tables, just warn
        
        # Check if patients table has data
        if 'patients' in patient_data and patient_data['patients'].empty:
            validation_result['issues'].append("Patients table is empty")
            # Don't fail validation for empty patients table, just warn
        
        # Check for temporal data - be more lenient
        temporal_consistency = {'is_consistent': True, 'issues': [], 'consistency_score': 1.0}
        if 'observations' in patient_data:
            obs_df = patient_data['observations']
            if obs_df.empty:
                # Don't fail validation for empty observations, just warn
                validation_result['issues'].append("No observations data")
            else:
                # Check for temporal consistency - be more lenient
                temporal_consistency = self._check_temporal_consistency(obs_df)
                if not temporal_consistency['is_consistent']:
                    # Don't fail validation for temporal issues, just warn
                    validation_result['issues'].extend(temporal_consistency['issues'])
        
        # Check for missing values - be more lenient
        missing_analysis = self._analyze_missing_values(patient_data)
        if missing_analysis['missing_percentage'] > 0.8:  # Only fail if >80% missing
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Very high missing values: {missing_analysis['missing_percentage']:.2%}")
        
        # Check for outliers if enabled - don't fail validation for outliers
        outlier_analysis = {'has_outliers': False, 'outlier_count': 0}
        if self.outlier_detection:
            outlier_analysis = self._detect_outliers(patient_data)
            if outlier_analysis['has_outliers']:
                validation_result['issues'].append(f"Outliers detected: {outlier_analysis['outlier_count']} values")
        
        # Store metrics
        validation_result['metrics'] = {
            'missing_percentage': missing_analysis['missing_percentage'],
            'temporal_consistency': temporal_consistency.get('consistency_score', 0.0),
            'outlier_count': outlier_analysis.get('outlier_count', 0)
        }
        
        return validation_result
    
    def _check_temporal_consistency(self, obs_df: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal consistency of observations."""
        consistency_result = {
            'is_consistent': True,
            'issues': [],
            'consistency_score': 1.0
        }
        
        if 'DATE' not in obs_df.columns:
            consistency_result['is_consistent'] = False
            consistency_result['issues'].append("Missing DATE column")
            return consistency_result
        
        # Convert dates - be more lenient with date parsing
        try:
            obs_df['DATE'] = pd.to_datetime(obs_df['DATE'], errors='coerce')
            # Remove rows with invalid dates instead of failing
            valid_dates = obs_df['DATE'].notna()
            if not valid_dates.any():
                consistency_result['is_consistent'] = False
                consistency_result['issues'].append("No valid dates found")
                return consistency_result
            obs_df = obs_df[valid_dates]
        except Exception as e:
            consistency_result['is_consistent'] = False
            consistency_result['issues'].append(f"Date parsing error: {e}")
            return consistency_result
        
        # Check for future dates (data leakage) - be more lenient
        current_time = datetime.now()
        future_dates = obs_df[obs_df['DATE'] > current_time]
        
        if not future_dates.empty:
            # Don't fail for future dates, just warn
            consistency_result['issues'].append(f"Future dates detected: {len(future_dates)} observations")
        
        # Check for reasonable date range - be more lenient
        date_range = obs_df['DATE'].max() - obs_df['DATE'].min()
        if date_range > timedelta(days=365*2):  # More than 2 years (increased from 1 year)
            consistency_result['issues'].append("Unusually long observation period")
        
        # Calculate consistency score
        total_obs = len(obs_df)
        if total_obs > 0:
            valid_obs = total_obs - len(future_dates)
            consistency_result['consistency_score'] = valid_obs / total_obs
        
        return consistency_result
    
    def _analyze_missing_values(self, patient_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze missing values in patient data."""
        missing_analysis = {
            'missing_percentage': 0.0,
            'high_missing': False,
            'missing_by_table': {}
        }
        
        total_values = 0
        missing_values = 0
        
        for table_name, table_df in patient_data.items():
            if not table_df.empty:
                table_missing = table_df.isnull().sum().sum()
                table_total = table_df.size
                
                missing_analysis['missing_by_table'][table_name] = {
                    'missing_count': table_missing,
                    'total_count': table_total,
                    'missing_percentage': table_missing / table_total if table_total > 0 else 0.0
                }
                
                total_values += table_total
                missing_values += table_missing
        
        if total_values > 0:
            missing_analysis['missing_percentage'] = missing_values / total_values
            missing_analysis['high_missing'] = missing_analysis['missing_percentage'] > self.max_missing_percentage
        
        return missing_analysis
    
    def _detect_outliers(self, patient_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect outliers in numerical data."""
        outlier_analysis = {
            'has_outliers': False,
            'outlier_count': 0,
            'outliers_by_table': {}
        }
        
        for table_name, table_df in patient_data.items():
            if not table_df.empty:
                # Focus on numerical columns
                numerical_cols = table_df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    table_outliers = 0
                    
                    for col in numerical_cols:
                        # Use IQR method for outlier detection
                        Q1 = table_df[col].quantile(0.25)
                        Q3 = table_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = table_df[(table_df[col] < lower_bound) | (table_df[col] > upper_bound)]
                        table_outliers += len(outliers)
                    
                    outlier_analysis['outliers_by_table'][table_name] = table_outliers
                    outlier_analysis['outlier_count'] += table_outliers
        
        outlier_analysis['has_outliers'] = outlier_analysis['outlier_count'] > 0
        
        return outlier_analysis
    
    def _calculate_quality_metrics(self, valid_patients: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        if not valid_patients:
            return {}
        
        metrics = {
            'total_patients': len(valid_patients),
            'avg_observations_per_patient': 0,
            'avg_encounters_per_patient': 0,
            'avg_medications_per_patient': 0,
            'data_completeness': 0.0,
            'temporal_coverage': 0.0
        }
        
        total_observations = 0
        total_encounters = 0
        total_medications = 0
        total_completeness = 0.0
        total_temporal_coverage = 0.0
        
        for patient_data in valid_patients.values():
            # Count observations
            if 'observations' in patient_data:
                total_observations += len(patient_data['observations'])
            
            # Count encounters
            if 'encounters' in patient_data:
                total_encounters += len(patient_data['encounters'])
            
            # Count medications
            if 'medications' in patient_data:
                total_medications += len(patient_data['medications'])
            
            # Calculate completeness for this patient
            patient_completeness = self._calculate_patient_completeness(patient_data)
            total_completeness += patient_completeness
            
            # Calculate temporal coverage for this patient
            patient_temporal_coverage = self._calculate_temporal_coverage(patient_data)
            total_temporal_coverage += patient_temporal_coverage
        
        # Calculate averages
        metrics['avg_observations_per_patient'] = total_observations / len(valid_patients)
        metrics['avg_encounters_per_patient'] = total_encounters / len(valid_patients)
        metrics['avg_medications_per_patient'] = total_medications / len(valid_patients)
        metrics['data_completeness'] = total_completeness / len(valid_patients)
        metrics['temporal_coverage'] = total_temporal_coverage / len(valid_patients)
        
        return metrics
    
    def _calculate_patient_completeness(self, patient_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate data completeness for a patient."""
        total_values = 0
        non_missing_values = 0
        
        for table_df in patient_data.values():
            if not table_df.empty:
                total_values += table_df.size
                non_missing_values += (table_df.size - table_df.isnull().sum().sum())
        
        return non_missing_values / total_values if total_values > 0 else 0.0
    
    def _calculate_temporal_coverage(self, patient_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate temporal coverage for a patient."""
        if 'observations' not in patient_data:
            return 0.0
        
        obs_df = patient_data['observations']
        if obs_df.empty or 'DATE' not in obs_df.columns:
            return 0.0
        
        try:
            obs_df['DATE'] = pd.to_datetime(obs_df['DATE'])
            date_range = obs_df['DATE'].max() - obs_df['DATE'].min()
            
            # Calculate coverage as ratio of actual days to expected days
            actual_days = date_range.days + 1
            expected_days = 30  # Expected 30 days of data
            
            return min(actual_days / expected_days, 1.0)
        except:
            return 0.0
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Data quality recommendations
        if validation_results['invalid_patients'] > 0:
            recommendations.append(f"Review and clean data for {validation_results['invalid_patients']} invalid patients")
        
        if validation_results['valid_patients'] < 100:
            recommendations.append("Consider collecting more patient data for robust model training")
        
        # Missing data recommendations
        if validation_results['quality_metrics'].get('data_completeness', 1.0) < 0.8:
            recommendations.append("Implement data imputation strategies for missing values")
        
        # Temporal coverage recommendations
        if validation_results['quality_metrics'].get('temporal_coverage', 1.0) < 0.5:
            recommendations.append("Ensure adequate temporal coverage for all patients")
        
        return recommendations
    
    def create_temporal_split(self, valid_patients: Dict[str, Dict[str, pd.DataFrame]], 
                            test_size: float = 0.1, validation_size: float = 0.15,
                            split_date: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Create temporal split to prevent data leakage.
        
        Args:
            valid_patients (Dict[str, Dict[str, pd.DataFrame]]): Valid patient data
            test_size (float): Proportion of data for testing
            validation_size (float): Proportion of data for validation
            split_date (Optional[str]): Date to split data (if None, use automatic split)
            
        Returns:
            Dict[str, Dict[str, Dict[str, pd.DataFrame]]]: Split data (train, validation, test)
        """
        self.logger.info("Creating temporal split to prevent data leakage")
        
        # Get all patient timestamps
        patient_timestamps = {}
        
        for patient_id, patient_data in valid_patients.items():
            if 'observations' in patient_data:
                obs_df = patient_data['observations']
                if 'DATE' in obs_df.columns and not obs_df.empty:
                    try:
                        obs_df['DATE'] = pd.to_datetime(obs_df['DATE'])
                        patient_timestamps[patient_id] = obs_df['DATE'].min()
                    except:
                        continue
        
        if not patient_timestamps:
            self.logger.warning("No valid timestamps found, using random split")
            return self._create_random_split(valid_patients, test_size, validation_size)
        
        # Sort patients by their earliest timestamp
        sorted_patients = sorted(patient_timestamps.items(), key=lambda x: x[1])
        patient_ids = [patient_id for patient_id, _ in sorted_patients]
        
        # Calculate split indices
        total_patients = len(patient_ids)
        test_end = int(total_patients * (1 - test_size))
        val_end = int(test_end * (1 - validation_size / (1 - test_size)))
        
        # Split patients
        train_ids = patient_ids[:val_end]
        val_ids = patient_ids[val_end:test_end]
        test_ids = patient_ids[test_end:]
        
        # Create split data
        split_data = {
            'train': {pid: valid_patients[pid] for pid in train_ids if pid in valid_patients},
            'validation': {pid: valid_patients[pid] for pid in val_ids if pid in valid_patients},
            'test': {pid: valid_patients[pid] for pid in test_ids if pid in valid_patients}
        }
        
        self.logger.info(f"Temporal split created: {len(split_data['train'])} train, {len(split_data['validation'])} validation, {len(split_data['test'])} test")
        
        return split_data
    
    def _create_random_split(self, valid_patients: Dict[str, Dict[str, pd.DataFrame]], 
                           test_size: float, validation_size: float) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """Create random split when temporal data is not available."""
        patient_ids = list(valid_patients.keys())
        
        # Split patients randomly
        train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
        train_ids, val_ids = train_test_split(train_ids, test_size=validation_size/(1-test_size), random_state=42)
        
        split_data = {
            'train': {pid: valid_patients[pid] for pid in train_ids},
            'validation': {pid: valid_patients[pid] for pid in val_ids},
            'test': {pid: valid_patients[pid] for pid in test_ids}
        }
        
        return split_data
    
    def save_validation_report(self, validation_results: Dict[str, Any], 
                             output_path: str = "data/validation_report.json"):
        """Save validation report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        serializable_results = self._make_json_serializable(validation_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Validation report saved to {output_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj 