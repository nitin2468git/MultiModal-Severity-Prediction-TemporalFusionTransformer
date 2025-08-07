#!/usr/bin/env python3
"""
TimelineBuilder: Construct patient timelines for TFT model
Part of the COVID-19 TFT Severity Prediction project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings

class TimelineBuilder:
    """
    Build patient timelines with hourly intervals for TFT model training.
    
    This class constructs temporal sequences for each patient, organizing
    clinical events into hourly intervals for the Temporal Fusion Transformer.
    
    Attributes:
        max_sequence_length (int): Maximum timeline length in hours
        time_interval (str): Time interval for timeline construction
        padding_strategy (str): Strategy for handling missing time periods
        logger (logging.Logger): Logger for timeline operations
    """
    
    def __init__(self, max_sequence_length: int = 720, time_interval: str = "1H", 
                 padding_strategy: str = "zero"):
        """
        Initialize TimelineBuilder.
        
        Args:
            max_sequence_length (int): Maximum timeline length in hours (default: 720 = 30 days)
            time_interval (str): Time interval for timeline construction (default: "1H")
            padding_strategy (str): Strategy for handling missing periods (default: "zero")
        """
        self.max_sequence_length = max_sequence_length
        self.time_interval = time_interval
        self.padding_strategy = padding_strategy
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for timeline operations."""
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
    
    def build_patient_timeline(self, patient_data: Dict[str, pd.DataFrame], 
                              patient_id: str) -> Dict[str, Any]:
        """
        Build timeline for a single patient.
        
        Args:
            patient_data (Dict[str, pd.DataFrame]): Patient's clinical data
            patient_id (str): Patient identifier
            
        Returns:
            Dict[str, Any]: Patient timeline with temporal features
        """
        self.logger.info(f"Building timeline for patient {patient_id}")
        
        # Initialize timeline
        timeline = {
            'patient_id': patient_id,
            'timestamps': [],
            'features': {},
            'events': {},
            'metadata': {}
        }
        
        # Get all timestamps from patient data
        all_timestamps = self._extract_all_timestamps(patient_data)
        
        if not all_timestamps:
            self.logger.warning(f"No temporal data found for patient {patient_id}")
            return timeline
        
        # Create hourly timeline - ensure all timestamps are timezone-naive
        timeline_start = min(all_timestamps)
        timeline_end = min(timeline_start + timedelta(hours=self.max_sequence_length), 
                          max(all_timestamps))
        
        # Ensure both start and end are timezone-naive
        if hasattr(timeline_start, 'tz') and timeline_start.tz is not None:
            timeline_start = timeline_start.tz_localize(None)
        if hasattr(timeline_end, 'tz') and timeline_end.tz is not None:
            timeline_end = timeline_end.tz_localize(None)
        
        # Generate hourly intervals (720 points)
        time_range = pd.date_range(
            start=timeline_start,
            periods=720,  # Fixed number of periods to match model expectation
            freq='h'  # Use 'h' instead of 'H' to avoid deprecation warning
        )
        
        timeline['timestamps'] = time_range.tolist()
        timeline['features'] = self._extract_temporal_features(patient_data, time_range)
        timeline['events'] = self._extract_events(patient_data, time_range)
        timeline['metadata'] = self._extract_metadata(patient_data)
        
        self.logger.info(f"Built timeline for patient {patient_id}: {len(time_range)} timepoints")
        return timeline
    
    def _extract_all_timestamps(self, patient_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Extract all timestamps from patient data."""
        timestamps = []
        
        # Extract from observations
        if 'observations' in patient_data:
            obs_df = patient_data['observations']
            if 'DATE' in obs_df.columns:
                # Convert to timezone-naive timestamps
                try:
                    obs_timestamps = pd.to_datetime(obs_df['DATE'])
                    # Handle timezone-aware timestamps
                    if obs_timestamps.dt.tz is not None:
                        obs_timestamps = obs_timestamps.dt.tz_localize(None)
                    timestamps.extend(obs_timestamps.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to process observation timestamps: {e}")
        
        # Extract from encounters
        if 'encounters' in patient_data:
            enc_df = patient_data['encounters']
            if 'START' in enc_df.columns:
                # Convert to timezone-naive timestamps
                try:
                    start_timestamps = pd.to_datetime(enc_df['START'])
                    # Handle timezone-aware timestamps
                    if start_timestamps.dt.tz is not None:
                        start_timestamps = start_timestamps.dt.tz_localize(None)
                    timestamps.extend(start_timestamps.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to process encounter start timestamps: {e}")
            if 'STOP' in enc_df.columns:
                # Convert to timezone-naive timestamps
                try:
                    stop_timestamps = pd.to_datetime(enc_df['STOP'])
                    # Handle timezone-aware timestamps
                    if stop_timestamps.dt.tz is not None:
                        stop_timestamps = stop_timestamps.dt.tz_localize(None)
                    timestamps.extend(stop_timestamps.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to process encounter stop timestamps: {e}")
        
        # Extract from medications
        if 'medications' in patient_data:
            med_df = patient_data['medications']
            if 'START' in med_df.columns:
                # Convert to timezone-naive timestamps
                try:
                    med_start_timestamps = pd.to_datetime(med_df['START'])
                    # Handle timezone-aware timestamps
                    if med_start_timestamps.dt.tz is not None:
                        med_start_timestamps = med_start_timestamps.dt.tz_localize(None)
                    timestamps.extend(med_start_timestamps.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to process medication start timestamps: {e}")
            if 'STOP' in med_df.columns:
                # Convert to timezone-naive timestamps
                try:
                    med_stop_timestamps = pd.to_datetime(med_df['STOP'])
                    # Handle timezone-aware timestamps
                    if med_stop_timestamps.dt.tz is not None:
                        med_stop_timestamps = med_stop_timestamps.dt.tz_localize(None)
                    timestamps.extend(med_stop_timestamps.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to process medication stop timestamps: {e}")
        
        # Extract from procedures
        if 'procedures' in patient_data:
            proc_df = patient_data['procedures']
            if 'DATE' in proc_df.columns:
                # Convert to timezone-naive timestamps
                try:
                    proc_timestamps = pd.to_datetime(proc_df['DATE'])
                    # Handle timezone-aware timestamps
                    if proc_timestamps.dt.tz is not None:
                        proc_timestamps = proc_timestamps.dt.tz_localize(None)
                    timestamps.extend(proc_timestamps.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to process procedure timestamps: {e}")
        
        return timestamps
    
    def _extract_temporal_features(self, patient_data: Dict[str, pd.DataFrame], 
                                  time_range: pd.DatetimeIndex) -> Dict[str, List[float]]:
        """Extract temporal features for each timepoint."""
        features = {
            'vital_signs': [],
            'laboratory_values': [],
            'medications': [],
            'procedures': [],
            'devices': []
        }
        
        # Initialize feature arrays
        for feature_type in features:
            features[feature_type] = [0.0] * len(time_range)
        
        # Extract vital signs
        if 'observations' in patient_data:
            obs_df = patient_data['observations']
            features['vital_signs'] = self._extract_vital_signs(obs_df, time_range)
        
        # Extract laboratory values
        if 'observations' in patient_data:
            obs_df = patient_data['observations']
            features['laboratory_values'] = self._extract_laboratory_values(obs_df, time_range)
        
        # Extract medications
        if 'medications' in patient_data:
            med_df = patient_data['medications']
            features['medications'] = self._extract_medications(med_df, time_range)
        
        # Extract procedures
        if 'procedures' in patient_data:
            proc_df = patient_data['procedures']
            features['procedures'] = self._extract_procedures(proc_df, time_range)
        
        # Extract devices
        if 'devices' in patient_data:
            dev_df = patient_data['devices']
            features['devices'] = self._extract_devices(dev_df, time_range)
        
        return features
    
    def _extract_vital_signs(self, obs_df: pd.DataFrame, 
                            time_range: pd.DatetimeIndex) -> List[float]:
        """Extract vital signs for each timepoint."""
        vital_signs = [0.0] * len(time_range)
        
        # Vital signs keywords
        vital_keywords = {
            'temperature': ['temperature', 'temp'],
            'heart_rate': ['heart rate', 'pulse'],
            'blood_pressure_systolic': ['systolic', 'bp systolic'],
            'blood_pressure_diastolic': ['diastolic', 'bp diastolic'],
            'oxygen_saturation': ['oxygen saturation', 'o2 sat'],
            'respiratory_rate': ['respiratory rate', 'resp rate']
        }
        
        # Convert observation dates to timezone-naive
        obs_df = obs_df.copy()
        if 'DATE' in obs_df.columns:
            obs_df['DATE'] = pd.to_datetime(obs_df['DATE']).dt.tz_localize(None)
        
        for idx, timestamp in enumerate(time_range):
            # Find observations within 1 hour of this timestamp
            time_window = pd.Timedelta(hours=1)
            relevant_obs = obs_df[
                (obs_df['DATE'] >= timestamp - time_window) &
                (obs_df['DATE'] < timestamp + time_window)
            ]
            
            if not relevant_obs.empty:
                # Calculate average vital signs for this timepoint
                for vital_type, keywords in vital_keywords.items():
                    for keyword in keywords:
                        matching_obs = relevant_obs[
                            relevant_obs['DESCRIPTION'].str.contains(keyword, case=False, na=False)
                        ]
                        if not matching_obs.empty:
                            # Try to convert VALUE to numeric
                            try:
                                values = pd.to_numeric(matching_obs['VALUE'], errors='coerce')
                                if not values.isna().all():
                                    vital_signs[idx] = values.mean()
                                    break
                            except:
                                continue
        
        return vital_signs
    
    def _extract_laboratory_values(self, obs_df: pd.DataFrame, 
                                  time_range: pd.DatetimeIndex) -> List[float]:
        """Extract laboratory values for each timepoint."""
        lab_values = [0.0] * len(time_range)
        
        # Laboratory test keywords
        lab_keywords = {
            'glucose': ['glucose', 'blood glucose'],
            'creatinine': ['creatinine'],
            'sodium': ['sodium', 'na'],
            'potassium': ['potassium', 'k'],
            'hemoglobin': ['hemoglobin', 'hgb'],
            'white_blood_cells': ['white blood cell', 'wbc'],
            'platelets': ['platelet'],
            'c_reactive_protein': ['c-reactive protein', 'crp']
        }
        
        # Convert observation dates to timezone-naive
        obs_df = obs_df.copy()
        if 'DATE' in obs_df.columns:
            obs_df['DATE'] = pd.to_datetime(obs_df['DATE']).dt.tz_localize(None)
        
        for idx, timestamp in enumerate(time_range):
            # Find observations within 1 hour of this timestamp
            time_window = pd.Timedelta(hours=1)
            relevant_obs = obs_df[
                (obs_df['DATE'] >= timestamp - time_window) &
                (obs_df['DATE'] < timestamp + time_window)
            ]
            
            if not relevant_obs.empty:
                # Calculate average lab values for this timepoint
                for lab_type, keywords in lab_keywords.items():
                    for keyword in keywords:
                        matching_obs = relevant_obs[
                            relevant_obs['DESCRIPTION'].str.contains(keyword, case=False, na=False)
                        ]
                        if not matching_obs.empty:
                            try:
                                values = pd.to_numeric(matching_obs['VALUE'], errors='coerce')
                                if not values.isna().all():
                                    lab_values[idx] = values.mean()
                                    break
                            except:
                                continue
        
        return lab_values
    
    def _extract_medications(self, med_df: pd.DataFrame, 
                            time_range: pd.DatetimeIndex) -> List[float]:
        """Extract medication information for each timepoint."""
        medications = [0.0] * len(time_range)
        
        # COVID-19 related medications
        covid_medications = {
            'remdesivir': ['remdesivir'],
            'dexamethasone': ['dexamethasone'],
            'hydroxychloroquine': ['hydroxychloroquine'],
            'azithromycin': ['azithromycin'],
            'tocilizumab': ['tocilizumab'],
            'baricitinib': ['baricitinib']
        }
        
        # Convert medication dates to timezone-naive
        med_df = med_df.copy()
        if 'START' in med_df.columns:
            med_df['START'] = pd.to_datetime(med_df['START']).dt.tz_localize(None)
        
        for idx, timestamp in enumerate(time_range):
            # Find medications active at this timestamp
            active_meds = med_df[
                (med_df['START'] <= timestamp) &
                ((med_df['STOP'].isna()) | (pd.to_datetime(med_df['STOP']).dt.tz_localize(None) > timestamp))
            ]
            
            if not active_meds.empty:
                # Count COVID-19 related medications
                covid_med_count = 0
                for med_type, keywords in covid_medications.items():
                    for keyword in keywords:
                        matching_meds = active_meds[
                            active_meds['DESCRIPTION'].str.contains(keyword, case=False, na=False)
                        ]
                        if not matching_meds.empty:
                            covid_med_count += 1
                            break
                
                medications[idx] = covid_med_count
        
        return medications
    
    def _extract_procedures(self, proc_df: pd.DataFrame, 
                           time_range: pd.DatetimeIndex) -> List[float]:
        """Extract procedure information for each timepoint."""
        procedures = [0.0] * len(time_range)
        
        # COVID-19 related procedures
        covid_procedures = {
            'intubation': ['intubation', 'endotracheal'],
            'ventilation': ['ventilation', 'mechanical ventilation'],
            'ecmo': ['ecmo', 'extracorporeal'],
            'dialysis': ['dialysis', 'hemodialysis'],
            'tracheostomy': ['tracheostomy']
        }
        
        # Convert procedure dates to timezone-naive
        proc_df = proc_df.copy()
        if 'DATE' in proc_df.columns:
            proc_df['DATE'] = pd.to_datetime(proc_df['DATE']).dt.tz_localize(None)
        
        for idx, timestamp in enumerate(time_range):
            # Find procedures performed at this timestamp (within 1 hour)
            time_window = pd.Timedelta(hours=1)
            relevant_procs = proc_df[
                (proc_df['DATE'] >= timestamp - time_window) &
                (proc_df['DATE'] < timestamp + time_window)
            ]
            
            if not relevant_procs.empty:
                # Count COVID-19 related procedures
                covid_proc_count = 0
                for proc_type, keywords in covid_procedures.items():
                    for keyword in keywords:
                        matching_procs = relevant_procs[
                            relevant_procs['DESCRIPTION'].str.contains(keyword, case=False, na=False)
                        ]
                        if not matching_procs.empty:
                            covid_proc_count += 1
                            break
                
                procedures[idx] = covid_proc_count
        
        return procedures
    
    def _extract_devices(self, dev_df: pd.DataFrame, 
                        time_range: pd.DatetimeIndex) -> List[float]:
        """Extract device information for each timepoint."""
        devices = [0.0] * len(time_range)
        
        # COVID-19 related devices
        covid_devices = {
            'ventilator': ['ventilator', 'mechanical ventilator'],
            'ecmo': ['ecmo', 'extracorporeal'],
            'oxygen': ['oxygen', 'o2'],
            'monitor': ['monitor', 'cardiac monitor']
        }
        
        # Convert device dates to timezone-naive
        dev_df = dev_df.copy()
        if 'START' in dev_df.columns:
            dev_df['START'] = pd.to_datetime(dev_df['START']).dt.tz_localize(None)
        
        for idx, timestamp in enumerate(time_range):
            # Find devices active at this timestamp
            active_devices = dev_df[
                (dev_df['START'] <= timestamp) &
                ((dev_df['STOP'].isna()) | (pd.to_datetime(dev_df['STOP']).dt.tz_localize(None) > timestamp))
            ]
            
            if not active_devices.empty:
                # Count COVID-19 related devices
                covid_device_count = 0
                for device_type, keywords in covid_devices.items():
                    for keyword in keywords:
                        matching_devices = active_devices[
                            active_devices['DESCRIPTION'].str.contains(keyword, case=False, na=False)
                        ]
                        if not matching_devices.empty:
                            covid_device_count += 1
                            break
                
                devices[idx] = covid_device_count
        
        return devices
    
    def _extract_events(self, patient_data: Dict[str, pd.DataFrame], 
                       time_range: pd.DatetimeIndex) -> Dict[str, List[Any]]:
        """Extract clinical events for each timepoint."""
        events = {
            'conditions': [],
            'encounters': [],
            'medications': [],
            'procedures': []
        }
        
        for timestamp in time_range:
            time_window = pd.Timedelta(hours=1)
            
            # Extract conditions
            if 'conditions' in patient_data:
                cond_df = patient_data['conditions']
                # Convert timestamps to timezone-naive
                start_times = pd.to_datetime(cond_df['START'])
                if start_times.dt.tz is not None:
                    start_times = start_times.dt.tz_localize(None)
                relevant_conditions = cond_df[
                    (start_times >= timestamp - time_window) &
                    (start_times < timestamp + time_window)
                ]
                events['conditions'].append(relevant_conditions.to_dict('records'))
            else:
                events['conditions'].append([])
            
            # Extract encounters
            if 'encounters' in patient_data:
                enc_df = patient_data['encounters']
                # Convert timestamps to timezone-naive
                start_times = pd.to_datetime(enc_df['START'])
                if start_times.dt.tz is not None:
                    start_times = start_times.dt.tz_localize(None)
                relevant_encounters = enc_df[
                    (start_times >= timestamp - time_window) &
                    (start_times < timestamp + time_window)
                ]
                events['encounters'].append(relevant_encounters.to_dict('records'))
            else:
                events['encounters'].append([])
            
            # Extract medications
            if 'medications' in patient_data:
                med_df = patient_data['medications']
                # Convert timestamps to timezone-naive
                start_times = pd.to_datetime(med_df['START'])
                if start_times.dt.tz is not None:
                    start_times = start_times.dt.tz_localize(None)
                relevant_medications = med_df[
                    (start_times >= timestamp - time_window) &
                    (start_times < timestamp + time_window)
                ]
                events['medications'].append(relevant_medications.to_dict('records'))
            else:
                events['medications'].append([])
            
            # Extract procedures
            if 'procedures' in patient_data:
                proc_df = patient_data['procedures']
                # Convert timestamps to timezone-naive
                proc_times = pd.to_datetime(proc_df['DATE'])
                if proc_times.dt.tz is not None:
                    proc_times = proc_times.dt.tz_localize(None)
                relevant_procedures = proc_df[
                    (proc_times >= timestamp - time_window) &
                    (proc_times < timestamp + time_window)
                ]
                events['procedures'].append(relevant_procedures.to_dict('records'))
            else:
                events['procedures'].append([])
        
        return events
    
    def _extract_metadata(self, patient_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract patient metadata."""
        metadata = {
            'patient_id': None,
            'age': None,
            'gender': None,
            'race': None,
            'ethnicity': None,
            'timeline_start': None,
            'timeline_end': None,
            'total_timepoints': 0
        }
        
        # Extract patient demographics
        if 'patients' in patient_data:
            patient_info = patient_data['patients'].iloc[0]
            metadata['patient_id'] = patient_info.get('Id', None)
            metadata['age'] = patient_info.get('BIRTHDATE', None)
            metadata['gender'] = patient_info.get('GENDER', None)
            metadata['race'] = patient_info.get('RACE', None)
            metadata['ethnicity'] = patient_info.get('ETHNICITY', None)
        
        return metadata
    
    def build_all_timelines(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Any]]:
        """
        Build timelines for all patients.
        
        Args:
            all_patient_data (Dict[str, Dict[str, pd.DataFrame]]): Data for all patients
            
        Returns:
            Dict[str, Dict[str, Any]]: All patient timelines
        """
        self.logger.info(f"Building timelines for {len(all_patient_data)} patients")
        
        all_timelines = {}
        
        for patient_id, patient_data in all_patient_data.items():
            try:
                timeline = self.build_patient_timeline(patient_data, patient_id)
                all_timelines[patient_id] = timeline
            except Exception as e:
                self.logger.error(f"Failed to build timeline for patient {patient_id}: {e}")
                continue
        
        self.logger.info(f"Successfully built timelines for {len(all_timelines)} patients")
        return all_timelines 