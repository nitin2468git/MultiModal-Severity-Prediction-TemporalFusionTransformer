#!/usr/bin/env python3
"""
FeatureEngineer: Extract and engineer features for TFT model
Part of the COVID-19 TFT Severity Prediction project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    """
    Extract and engineer features for TFT model training.
    
    This class handles feature extraction, engineering, and preprocessing
    for both static (15 dimensions) and temporal (26 dimensions) features.
    
    Attributes:
        static_features (List[str]): List of static feature names
        temporal_features (List[str]): List of temporal feature names
        scalers (Dict): Dictionary of fitted scalers
        encoders (Dict): Dictionary of fitted encoders
        logger (logging.Logger): Logger for feature engineering operations
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.static_features = []
        self.temporal_features = []
        self.scalers = {}
        self.encoders = {}
        self.logger = self._setup_logger()
        
        # Define feature mappings
        self._setup_feature_mappings()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for feature engineering operations."""
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
    
    def _setup_feature_mappings(self):
        """Setup feature mappings for extraction."""
        # Static features (15 dimensions)
        self.static_feature_mappings = {
            'demographics': {
                'age': ['BIRTHDATE'],
                'gender': ['GENDER'],
                'race': ['RACE'],
                'ethnicity': ['ETHNICITY']
            },
            'comorbidities': {
                'diabetes': ['diabetes', 'diabetic'],
                'hypertension': ['hypertension', 'high blood pressure'],
                'obesity': ['obesity', 'bmi 30+'],
                'copd': ['copd', 'chronic obstructive'],
                'heart_disease': ['heart disease', 'cardiovascular']
            },
            'socioeconomic': {
                'healthcare_coverage': ['insurance', 'coverage'],
                'income_level': ['income', 'poverty']
            }
        }
        
        # Temporal features (26 dimensions)
        self.temporal_feature_mappings = {
            'vital_signs': {
                'temperature': ['temperature', 'temp'],
                'heart_rate': ['heart rate', 'pulse'],
                'blood_pressure_systolic': ['systolic', 'bp systolic'],
                'blood_pressure_diastolic': ['diastolic', 'bp diastolic'],
                'oxygen_saturation': ['oxygen saturation', 'o2 sat'],
                'respiratory_rate': ['respiratory rate', 'resp rate']
            },
            'laboratory_values': {
                'crp': ['crp', 'c-reactive protein'],
                'd_dimer': ['d-dimer', 'd dimer'],
                'white_blood_cells': ['white blood cells', 'wbc', 'leukocytes'],
                'platelets': ['platelets', 'platelet count'],
                'lymphocytes': ['lymphocytes', 'lymphocyte count'],
                'neutrophils': ['neutrophils', 'neutrophil count'],
                'creatinine': ['creatinine'],
                'troponin': ['troponin']
            },
            'medications': {
                'remdesivir': ['remdesivir'],
                'dexamethasone': ['dexamethasone'],
                'tocilizumab': ['tocilizumab'],
                'supportive_care': ['acetaminophen', 'ibuprofen', 'paracetamol'],
                'anticoagulants': ['heparin', 'enoxaparin', 'warfarin']
            },
            'procedures': {
                'intubation': ['intubation', 'endotracheal'],
                'mechanical_ventilation': ['mechanical ventilation', 'ventilator'],
                'diagnostic_tests': ['covid test', 'pcr test', 'antigen test'],
                'imaging_studies': ['chest x-ray', 'ct scan', 'chest ct']
            },
            'devices': {
                'ventilator_settings': ['ventilator', 'respirator'],
                'monitoring_equipment': ['monitor', 'ecg', 'ekg'],
                'life_support': ['ecmo', 'extracorporeal']
            }
        }
    
    def extract_static_features(self, patient_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract static features for a patient.
        
        Args:
            patient_data (Dict[str, pd.DataFrame]): Patient's clinical data
            
        Returns:
            Dict[str, Any]: Static features dictionary
        """
        static_features = {}
        
        # Extract demographics
        if 'patients' in patient_data:
            patient_info = patient_data['patients'].iloc[0]
            static_features.update(self._extract_demographics(patient_info))
        
        # Extract comorbidities from conditions
        if 'conditions' in patient_data:
            conditions_df = patient_data['conditions']
            static_features.update(self._extract_comorbidities(conditions_df))
        
        # Extract socioeconomic factors
        static_features.update(self._extract_socioeconomic_factors(patient_data))
        
        return static_features
    
    def _extract_demographics(self, patient_info: pd.Series) -> Dict[str, Any]:
        """Extract demographic features."""
        demographics = {}
        
        # Age calculation
        if 'BIRTHDATE' in patient_info:
            try:
                birthdate = pd.to_datetime(patient_info['BIRTHDATE'])
                demographics['age'] = (datetime.now() - birthdate).days / 365.25
            except:
                demographics['age'] = 0.0
        else:
            demographics['age'] = 0.0
        
        # Gender encoding
        gender = patient_info.get('GENDER', 'Unknown')
        if 'gender' not in self.encoders:
            self.encoders['gender'] = LabelEncoder()
            self.encoders['gender'].fit(['Unknown', 'F', 'M'])
        demographics['gender'] = float(self.encoders['gender'].transform([gender])[0])
        
        # Race encoding
        race = patient_info.get('RACE', 'Unknown')
        if 'race' not in self.encoders:
            self.encoders['race'] = LabelEncoder()
            self.encoders['race'].fit(['Unknown', 'white', 'black', 'asian', 'native', 'other'])
        demographics['race'] = float(self.encoders['race'].transform([race])[0])
        
        # Ethnicity encoding
        ethnicity = patient_info.get('ETHNICITY', 'Unknown')
        if 'ethnicity' not in self.encoders:
            self.encoders['ethnicity'] = LabelEncoder()
            self.encoders['ethnicity'].fit(['Unknown', 'nonhispanic', 'hispanic'])
        demographics['ethnicity'] = float(self.encoders['ethnicity'].transform([ethnicity])[0])
        
        return demographics
    
    def _extract_comorbidities(self, conditions_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comorbidity indicators."""
        comorbidities = {}
        
        # Initialize all comorbidities to False
        for comorbidity in self.static_feature_mappings['comorbidities'].keys():
            comorbidities[comorbidity] = 0.0
        
        # Check for each comorbidity in conditions
        if not conditions_df.empty:
            for comorbidity, keywords in self.static_feature_mappings['comorbidities'].items():
                # Check if any keyword matches in condition descriptions
                for keyword in keywords:
                    if conditions_df['DESCRIPTION'].str.contains(keyword, case=False, na=False).any():
                        comorbidities[comorbidity] = 1.0
                        break
        
        return comorbidities
    
    def _extract_socioeconomic_factors(self, patient_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract socioeconomic factors."""
        socioeconomic = {}
        
        # Initialize factors to 0
        for factor in self.static_feature_mappings['socioeconomic'].keys():
            socioeconomic[factor] = 0.0
        
        # Extract from available data (placeholder for now)
        # In practice, this would use actual socioeconomic data
        socioeconomic['healthcare_coverage'] = 1.0  # Assume covered
        socioeconomic['income_level'] = 0.5  # Middle income
        
        return socioeconomic
    
    def extract_temporal_features(self, timeline: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract temporal features from patient timeline.
        
        Args:
            timeline (Dict[str, Any]): Patient timeline
            
        Returns:
            Dict[str, List[float]]: Temporal features dictionary
        """
        temporal_features = {}
        
        # Extract from timeline features
        if 'features' in timeline:
            features = timeline['features']
            
            # Vital signs (6 dimensions)
            if 'vital_signs' in features:
                temporal_features['vital_signs'] = features['vital_signs']
            
            # Laboratory values (8 dimensions)
            if 'laboratory_values' in features:
                temporal_features['laboratory_values'] = features['laboratory_values']
            
            # Medications (5 dimensions)
            if 'medications' in features:
                temporal_features['medications'] = features['medications']
            
            # Procedures (4 dimensions)
            if 'procedures' in features:
                temporal_features['procedures'] = features['procedures']
            
            # Devices (3 dimensions)
            if 'devices' in features:
                temporal_features['devices'] = features['devices']
        
        return temporal_features
    
    def engineer_derived_features(self, static_features: Dict[str, Any], 
                                temporal_features: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Engineer derived features from static and temporal data.
        
        Args:
            static_features (Dict[str, Any]): Static features
            temporal_features (Dict[str, List[float]]): Temporal features
            
        Returns:
            Dict[str, Any]: Derived features
        """
        derived_features = {}
        
        # Age-based risk factors
        if 'age' in static_features:
            age = static_features['age']
            derived_features['age_risk'] = 1.0 if age > 65 else 0.0
            derived_features['age_group'] = self._categorize_age(age)
        
        # Comorbidity count
        comorbidity_count = sum([
            static_features.get('diabetes', 0),
            static_features.get('hypertension', 0),
            static_features.get('obesity', 0),
            static_features.get('copd', 0),
            static_features.get('heart_disease', 0)
        ])
        derived_features['comorbidity_count'] = comorbidity_count
        derived_features['high_comorbidity'] = 1.0 if comorbidity_count >= 2 else 0.0
        
        # Temporal feature engineering
        if temporal_features:
            derived_features.update(self._engineer_temporal_derived_features(temporal_features))
        
        return derived_features
    
    def _categorize_age(self, age: float) -> int:
        """Categorize age into groups."""
        if age < 18:
            return 0  # Pediatric
        elif age < 50:
            return 1  # Young adult
        elif age < 65:
            return 2  # Middle-aged
        else:
            return 3  # Elderly
    
    def _engineer_temporal_derived_features(self, temporal_features: Dict[str, List[float]]) -> Dict[str, Any]:
        """Engineer derived features from temporal data."""
        derived_features = {}
        
        # Calculate trends and patterns
        for feature_type, values in temporal_features.items():
            if isinstance(values, list) and len(values) > 0:
                # Convert to numpy array for calculations
                values_array = np.array(values)
                
                # Basic statistics
                derived_features[f'{feature_type}_mean'] = np.mean(values_array)
                derived_features[f'{feature_type}_std'] = np.std(values_array)
                derived_features[f'{feature_type}_max'] = np.max(values_array)
                
                # Trend analysis
                if len(values_array) > 1:
                    # Calculate slope using linear regression
                    x = np.arange(len(values_array))
                    slope = np.polyfit(x, values_array, 1)[0]
                    derived_features[f'{feature_type}_trend'] = slope
                else:
                    derived_features[f'{feature_type}_trend'] = 0.0
        
        return derived_features
    
    def preprocess_features(self, static_features: Dict[str, Any], 
                          temporal_features: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features for model input.
        
        Args:
            static_features (Dict[str, Any]): Static features
            temporal_features (Dict[str, List[float]]): Temporal features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed static and temporal features
        """
        # Preprocess static features
        static_array = self._preprocess_static_features(static_features)
        
        # Preprocess temporal features
        temporal_array = self._preprocess_temporal_features(temporal_features)
        
        return static_array, temporal_array
    
    def _preprocess_static_features(self, static_features: Dict[str, Any]) -> np.ndarray:
        """Preprocess static features."""
        # Define expected static features
        expected_features = [
            'age', 'gender', 'race', 'ethnicity',
            'diabetes', 'hypertension', 'obesity', 'copd', 'heart_disease',
            'healthcare_coverage', 'income_level',
            'age_risk', 'age_group', 'comorbidity_count', 'high_comorbidity'
        ]
        
        # Initialize array with zeros
        static_array = np.zeros(len(expected_features))
        
        # Fill array with available features
        for i, feature in enumerate(expected_features):
            if feature in static_features:
                static_array[i] = static_features[feature]
        
        # Normalize numerical features (categorical features are already encoded)
        static_array = self._normalize_numerical_features(static_array, expected_features)
        
        return static_array
    
    def _preprocess_temporal_features(self, temporal_features: Dict[str, List[float]]) -> np.ndarray:
        """Preprocess temporal features."""
        # Define expected temporal features
        expected_features = {
            'vital_signs': ['temperature', 'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                          'oxygen_saturation', 'respiratory_rate'],
            'laboratory_values': ['crp', 'd_dimer', 'white_blood_cells', 'platelets', 'lymphocytes',
                                'neutrophils', 'creatinine', 'troponin'],
            'medications': ['remdesivir', 'dexamethasone', 'tocilizumab', 'supportive_care', 'anticoagulants'],
            'procedures': ['intubation', 'mechanical_ventilation', 'diagnostic_tests', 'imaging_studies'],
            'devices': ['ventilator_settings', 'monitoring_equipment', 'life_support']
        }
        
        # Initialize array with zeros
        temporal_arrays = []
        
        # Process each feature category
        for category, features in expected_features.items():
            for feature in features:
                if feature in temporal_features and isinstance(temporal_features[feature], list):
                    if temporal_features[feature]:
                        temporal_arrays.extend(temporal_features[feature])
                    else:
                        temporal_arrays.extend([0.0] * 721)  # Pad with zeros
                else:
                    temporal_arrays.append(0.0)  # Single value for missing feature
        
        # Convert to numpy array
        temporal_array = np.array(temporal_arrays)
        
        # Handle missing values
        temporal_array = self._handle_missing_values(temporal_array)
        
        # Normalize temporal features
        temporal_array = self._normalize_temporal_features(temporal_array)
        
        return temporal_array
    
    def _encode_categorical_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Encode categorical features."""
        encoded_features = features.copy()
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in ['gender', 'race', 'ethnicity']:
                # Create or get encoder for this feature
                if feature_name not in self.encoders:
                    self.encoders[feature_name] = LabelEncoder()
                    # Fit on all possible values
                    all_values = ['Unknown', 'F', 'M', 'white', 'black', 'asian', 'native', 'other', 
                                'nonhispanic', 'hispanic']
                    self.encoders[feature_name].fit(all_values)
                
                # Encode the value
                try:
                    encoded_features[i] = self.encoders[feature_name].transform([features[i]])[0]
                except:
                    encoded_features[i] = 0  # Default to first category
        
        return encoded_features
    
    def _normalize_numerical_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Normalize numerical features."""
        normalized_features = features.copy()
        
        # Identify numerical features
        numerical_features = ['age', 'diabetes', 'hypertension', 'obesity', 'copd', 'heart_disease',
                           'healthcare_coverage', 'income_level']
        
        numerical_indices = [i for i, name in enumerate(feature_names) if name in numerical_features]
        
        if numerical_indices:
            numerical_values = features[numerical_indices].astype(float)
            
            # Create or get scaler
            if 'static_scaler' not in self.scalers:
                self.scalers['static_scaler'] = StandardScaler()
                # Fit on a sample of data (in practice, this would be fit on training data)
                sample_data = np.random.randn(100, len(numerical_indices))
                self.scalers['static_scaler'].fit(sample_data)
            
            # Normalize
            normalized_numerical = self.scalers['static_scaler'].transform(numerical_values.reshape(1, -1))
            normalized_features[numerical_indices] = normalized_numerical.flatten()
        
        return normalized_features
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in temporal features."""
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        return features
    
    def _normalize_temporal_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize temporal features."""
        if len(features) == 0:
            return features
        
        # Create or get scaler
        if 'temporal_scaler' not in self.scalers:
            self.scalers['temporal_scaler'] = StandardScaler()
            # Fit on a sample of data
            sample_data = np.random.randn(100, features.shape[0])
            self.scalers['temporal_scaler'].fit(sample_data)
        
        # Normalize
        normalized_features = self.scalers['temporal_scaler'].transform(features.reshape(1, -1))
        return normalized_features.flatten()
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get feature names."""
        static_features = [
            'age', 'gender', 'race', 'ethnicity',
            'diabetes', 'hypertension', 'obesity', 'copd', 'heart_disease',
            'healthcare_coverage', 'income_level',
            'age_risk', 'age_group', 'comorbidity_count', 'high_comorbidity'
        ]
        
        temporal_features = []
        for category, features in self.temporal_feature_mappings.items():
            temporal_features.extend(list(features.keys()))
        
        return static_features, temporal_features