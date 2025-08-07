#!/usr/bin/env python3
"""
TFTFormatter: Format data for Temporal Fusion Transformer model
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
from torch.utils.data import Dataset, DataLoader

class TFTFormatter:
    """
    Format data for Temporal Fusion Transformer model training.
    
    This class handles the conversion of patient timelines and features
    into the specific format required by the TFT model.
    
    Attributes:
        max_encoder_length (int): Maximum encoder sequence length
        max_prediction_length (int): Maximum prediction horizon
        static_features_dim (int): Dimension of static features
        temporal_features_dim (int): Dimension of temporal features
        logger (logging.Logger): Logger for formatting operations
    """
    
    def __init__(self, max_encoder_length: int = 720, max_prediction_length: int = 24,
                 static_features_dim: int = 15, temporal_features_dim: int = 26):
        """
        Initialize TFTFormatter.
        
        Args:
            max_encoder_length (int): Maximum encoder sequence length (default: 720 hours = 30 days)
            max_prediction_length (int): Maximum prediction horizon (default: 24 hours)
            static_features_dim (int): Dimension of static features (default: 15)
            temporal_features_dim (int): Dimension of temporal features (default: 26)
        """
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.static_features_dim = static_features_dim
        self.temporal_features_dim = temporal_features_dim
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for formatting operations."""
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
    
    def format_patient_data(self, timeline: Dict[str, Any], 
                           static_features: np.ndarray,
                           temporal_features: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Format patient data for TFT model input.
        
        Args:
            timeline (Dict[str, Any]): Patient timeline
            static_features (np.ndarray): Static features array
            temporal_features (np.ndarray): Temporal features array
            
        Returns:
            Dict[str, torch.Tensor]: Formatted data for TFT model
        """
        self.logger.info(f"Formatting data for patient {timeline.get('patient_id', 'unknown')}")
        
        # Extract timestamps
        timestamps = timeline.get('timestamps', [])
        
        if not timestamps:
            self.logger.warning("No timestamps found in timeline")
            return self._create_empty_tft_data()
        
        # Convert timestamps to relative time
        relative_times = self._calculate_relative_times(timestamps)
        
        # Pad or truncate temporal features
        padded_temporal = self._pad_temporal_features(temporal_features, len(timestamps))
        
        # Create TFT format data with proper data types and dimensions
        tft_data = {
            'static_features': torch.tensor(static_features, dtype=torch.float32),  # [static_dim]
            'temporal_features': torch.tensor(padded_temporal, dtype=torch.float32),  # [seq_len, temporal_dim]
            'time_index': torch.tensor(relative_times, dtype=torch.float32),  # [seq_len]
            'patient_id': timeline.get('patient_id', 'unknown'),  # Keep as string
            'sequence_length': len(timestamps)  # Keep as int, will be converted to tensor in collate_fn
        }
        
        # Add targets
        targets = self.create_targets(timeline)
        tft_data.update(targets)
        
        return tft_data
    
    def _calculate_relative_times(self, timestamps: List[datetime]) -> List[float]:
        """Calculate relative time indices from timestamps."""
        if not timestamps:
            return []
        
        # Convert to hours from the first timestamp
        start_time = timestamps[0]
        relative_times = []
        
        for timestamp in timestamps:
            time_diff = timestamp - start_time
            hours = time_diff.total_seconds() / 3600
            relative_times.append(hours)
        
        return relative_times
    
    def _pad_temporal_features(self, temporal_features: np.ndarray, 
                              sequence_length: int) -> np.ndarray:
        """
        Pad or truncate temporal features to match sequence length with consistent handling.
        
        Args:
            temporal_features (np.ndarray): Input temporal features
            sequence_length (int): Target sequence length
            
        Returns:
            np.ndarray: Padded/truncated features with shape (sequence_length, temporal_features_dim)
        """
        # Handle empty input
        if len(temporal_features) == 0:
            return np.zeros((sequence_length, self.temporal_features_dim), dtype=np.float32)
        
        # Convert to float32 for consistency
        temporal_features = temporal_features.astype(np.float32)
        
        # Reshape if needed
        if temporal_features.ndim == 1:
            temporal_features = temporal_features.reshape(1, -1)
        elif temporal_features.ndim > 2:
            raise ValueError(f"Temporal features must be 1D or 2D, got shape {temporal_features.shape}")
        
        current_seq_len, current_feat_dim = temporal_features.shape
        
        # Handle sequence length
        if current_seq_len > sequence_length:
            # Truncate from the end to keep most recent data
            temporal_features = temporal_features[-sequence_length:]
        elif current_seq_len < sequence_length:
            # Pad with zeros at the end
            padding = np.zeros((sequence_length - current_seq_len, current_feat_dim), dtype=np.float32)
            temporal_features = np.vstack([temporal_features, padding])
        
        # Handle feature dimension
        if current_feat_dim != self.temporal_features_dim:
            if current_feat_dim > self.temporal_features_dim:
                # Keep only the first temporal_features_dim features
                temporal_features = temporal_features[:, :self.temporal_features_dim]
            else:
                # Pad with zeros on the right
                padding = np.zeros((sequence_length, self.temporal_features_dim - current_feat_dim), dtype=np.float32)
                temporal_features = np.hstack([temporal_features, padding])
        
        # Final shape check
        assert temporal_features.shape == (sequence_length, self.temporal_features_dim), \
            f"Expected shape ({sequence_length}, {self.temporal_features_dim}), got {temporal_features.shape}"
        
        return temporal_features
    
    def _create_empty_tft_data(self) -> Dict[str, torch.Tensor]:
        """Create empty TFT data structure with consistent dimensions."""
        return {
            'static_features': torch.zeros((1, self.static_features_dim), dtype=torch.float32),  # [1, static_dim]
            'temporal_features': torch.zeros((self.max_encoder_length, self.temporal_features_dim), dtype=torch.float32),  # [seq_len, temporal_dim]
            'time_index': torch.zeros((1, self.max_encoder_length), dtype=torch.float32),  # [1, seq_len]
            'patient_id': 'unknown',
            'sequence_length': 0
        }
    
    def create_targets(self, timeline: Dict[str, Any], 
                      prediction_horizon: int = 24) -> Dict[str, torch.Tensor]:
        """
        Create target variables for multi-task prediction.
        
        Args:
            timeline (Dict[str, Any]): Patient timeline
            prediction_horizon (int): Prediction horizon in hours
            
        Returns:
            Dict[str, torch.Tensor]: Target variables
        """
        targets = {
            'mortality_risk': torch.zeros(1, dtype=torch.float32),
            'icu_admission': torch.zeros(1, dtype=torch.float32),
            'ventilator_need': torch.zeros(1, dtype=torch.float32),
            'length_of_stay': torch.zeros(1, dtype=torch.float32)
        }
        
        # Extract events from timeline
        events = timeline.get('events', {})
        
        # Determine mortality risk based on conditions and procedures
        mortality_indicators = self._extract_mortality_indicators(events)
        targets['mortality_risk'][0] = mortality_indicators
        
        # Determine ICU admission based on encounters
        icu_indicators = self._extract_icu_indicators(events)
        targets['icu_admission'][0] = icu_indicators
        
        # Determine ventilator need based on procedures and devices
        ventilator_indicators = self._extract_ventilator_indicators(events)
        targets['ventilator_need'][0] = ventilator_indicators
        
        # Calculate length of stay
        los = self._calculate_length_of_stay(timeline)
        targets['length_of_stay'][0] = los
        
        return targets
    
    def _extract_mortality_indicators(self, events: Dict[str, List[Any]]) -> float:
        """Extract mortality risk indicators from events."""
        mortality_score = 0.0
        
        # Check for severe conditions
        conditions = events.get('conditions', [])
        for condition_list in conditions:
            for condition in condition_list:
                description = condition.get('DESCRIPTION', '').lower()
                if any(keyword in description for keyword in ['severe', 'critical', 'acute respiratory', 'pneumonia']):
                    mortality_score += 0.3
        
        # Check for critical procedures
        procedures = events.get('procedures', [])
        for procedure_list in procedures:
            for procedure in procedure_list:
                description = procedure.get('DESCRIPTION', '').lower()
                if any(keyword in description for keyword in ['intubation', 'mechanical ventilation', 'ecmo']):
                    mortality_score += 0.4
        
        return min(mortality_score, 1.0)
    
    def _extract_icu_indicators(self, events: Dict[str, List[Any]]) -> float:
        """Extract ICU admission indicators from events."""
        icu_score = 0.0
        
        # Check for ICU encounters
        encounters = events.get('encounters', [])
        for encounter_list in encounters:
            for encounter in encounter_list:
                description = encounter.get('DESCRIPTION', '').lower()
                if any(keyword in description for keyword in ['icu', 'intensive care', 'critical care']):
                    icu_score = 1.0
                    break
        
        # Check for critical procedures that indicate ICU care
        procedures = events.get('procedures', [])
        for procedure_list in procedures:
            for procedure in procedure_list:
                description = procedure.get('DESCRIPTION', '').lower()
                if any(keyword in description for keyword in ['intubation', 'mechanical ventilation']):
                    icu_score = 1.0
                    break
        
        return icu_score
    
    def _extract_ventilator_indicators(self, events: Dict[str, List[Any]]) -> float:
        """Extract ventilator need indicators from events."""
        ventilator_score = 0.0
        
        # Check for ventilator procedures
        procedures = events.get('procedures', [])
        for procedure_list in procedures:
            for procedure in procedure_list:
                description = procedure.get('DESCRIPTION', '').lower()
                if any(keyword in description for keyword in ['mechanical ventilation', 'ventilator']):
                    ventilator_score = 1.0
                    break
        
        # Check for ventilator devices
        # Note: devices are typically in the features, not events
        # This is a simplified implementation
        
        return ventilator_score
    
    def _calculate_length_of_stay(self, timeline: Dict[str, Any]) -> float:
        """Calculate length of stay in hours."""
        timestamps = timeline.get('timestamps', [])
        
        if len(timestamps) < 2:
            return 0.0
        
        # Ensure timestamps are datetime objects
        try:
            start_time = timestamps[0] if isinstance(timestamps[0], datetime) else pd.to_datetime(timestamps[0])
            end_time = timestamps[-1] if isinstance(timestamps[-1], datetime) else pd.to_datetime(timestamps[-1])
            
            # Ensure timestamps are timezone-naive
            if start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)
            
            duration = end_time - start_time
            los_hours = float(duration.total_seconds() / 3600)
            
            return los_hours
        except Exception as e:
            self.logger.warning(f"Failed to calculate length of stay: {e}")
            return 0.0
    
    def create_tft_dataset(self, all_patient_data: List[Dict[str, Any]]) -> 'TFTDataset':
        """
        Create TFT dataset from all patient data.
        
        Args:
            all_patient_data (List[Dict[str, Any]]): List of formatted patient data
            
        Returns:
            TFTDataset: TFT-compatible dataset
        """
        return TFTDataset(all_patient_data)
    
    def create_data_loader(self, dataset: 'TFTDataset', 
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 4) -> DataLoader:
        """
        Create DataLoader for TFT dataset.
        
        Args:
            dataset (TFTDataset): TFT dataset
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            num_workers (int): Number of worker processes
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        print(f"[DEBUG] Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
        print(f"[DEBUG] Dataset size: {len(dataset)}")
        
        # Test the first item to see what we're working with
        if len(dataset) > 0:
            first_item = dataset[0]
            print(f"[DEBUG] First dataset item type: {type(first_item)}")
            print(f"[DEBUG] First dataset item keys: {list(first_item.keys())}")
            for key, value in first_item.items():
                print(f"[DEBUG] {key}: type={type(value)}, shape={getattr(value, 'shape', 'no shape') if hasattr(value, 'shape') else 'no shape'}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for TFT data.
        
        Args:
            batch (List[Dict[str, Any]]): Batch of patient data
            
        Returns:
            Dict[str, torch.Tensor]: Batched data
        """
        print("[DEBUG] Starting collate function")
        print(f"[DEBUG] Batch type: {type(batch)}")
        print(f"[DEBUG] Batch length: {len(batch)}")
        
        if len(batch) == 0:
            print("[DEBUG] Empty batch!")
            return {}
        
        print(f"[DEBUG] First item type: {type(batch[0])}")
        print(f"[DEBUG] First item keys: {list(batch[0].keys())}")
        
        # Extract components and log shapes
        print(f"[DEBUG] Batch size: {len(batch)}")
        print(f"[DEBUG] Batch keys: {list(batch[0].keys())}")
        
        try:
            static_features = [item['static_features'] for item in batch]
            print(f"[DEBUG] Static features extracted successfully")
        except Exception as e:
            print(f"[DEBUG] Error extracting static_features: {e}")
            raise
        
        try:
            temporal_features = [item['temporal_features'] for item in batch]
            print(f"[DEBUG] Temporal features extracted successfully")
        except Exception as e:
            print(f"[DEBUG] Error extracting temporal_features: {e}")
            raise
        
        try:
            time_indices = [item['time_index'] for item in batch]
            print(f"[DEBUG] Time indices extracted successfully")
        except Exception as e:
            print(f"[DEBUG] Error extracting time_indices: {e}")
            raise
        
        print(f"[DEBUG] Static features types: {[type(sf) for sf in static_features]}")
        print(f"[DEBUG] Static features shapes: {[sf.shape if hasattr(sf, 'shape') else 'no shape' for sf in static_features]}")
        print(f"[DEBUG] Temporal features types: {[type(tf) for tf in temporal_features]}")
        print(f"[DEBUG] Temporal features shapes: {[tf.shape if hasattr(tf, 'shape') else 'no shape' for tf in temporal_features]}")
        print(f"[DEBUG] Time indices types: {[type(ti) for ti in time_indices]}")
        print(f"[DEBUG] Time indices shapes: {[ti.shape if hasattr(ti, 'shape') else 'no shape' for ti in time_indices]}")
        
        # Convert and stack tensors with proper type checking and dimension handling
        def ensure_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype)
            return torch.tensor(x, dtype=dtype)
        
        # Stack tensors and handle non-tensor data with proper dimensions
        batched_data = {
            'static_features': torch.stack([ensure_tensor(item['static_features']) for item in batch]),  # [batch_size, static_dim]
            'temporal_features': torch.stack([ensure_tensor(item['temporal_features']) for item in batch]),  # [batch_size, seq_len, temporal_dim]
            'time_index': torch.stack([ensure_tensor(item['time_index']) for item in batch]),  # [batch_size, seq_len]
            'patient_ids': [str(item['patient_id']) for item in batch],  # Keep as strings
            'sequence_length': torch.tensor([int(item['sequence_length']) for item in batch], dtype=torch.long)  # [batch_size]
        }
        
        # Add targets if available with proper dimensions
        if 'mortality_risk' in batch[0]:
            for target_name in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']:
                target_values = [ensure_tensor(item[target_name]).squeeze() for item in batch]  # Ensure 1D tensor
                batched_data[target_name] = torch.stack(target_values)  # [batch_size]
        
        return batched_data


class TFTDataset(Dataset):
    """
    TFT-compatible dataset for COVID-19 severity prediction.
    
    This dataset provides the interface between the data pipeline
    and the TFT model training.
    """
    
    def __init__(self, patient_data: List[Dict[str, Any]]):
        """
        Initialize TFTDataset.
        
        Args:
            patient_data (List[Dict[str, Any]]): List of formatted patient data
        """
        self.patient_data = patient_data
        self.logger = logging.getLogger(__name__)
    
    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.patient_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single patient's data.
        
        Args:
            idx (int): Patient index
            
        Returns:
            Dict[str, Any]: Patient data
        """
        return self.patient_data[idx]
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """
        Get feature dimensions.
        
        Returns:
            Tuple[int, int]: Static and temporal feature dimensions
        """
        if len(self.patient_data) == 0:
            return 0, 0
        
        sample = self.patient_data[0]
        static_dim = sample['static_features'].shape[0]
        temporal_dim = sample['temporal_features'].shape[1]
        
        return static_dim, temporal_dim 