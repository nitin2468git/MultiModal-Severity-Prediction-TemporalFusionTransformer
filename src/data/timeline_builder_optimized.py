#!/usr/bin/env python3
"""
Optimized Timeline Builder for COVID-19 TFT Project
Memory-efficient timeline building for large datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

class OptimizedTimelineBuilder:
    """
    Memory-efficient timeline builder for large patient datasets.
    
    Features:
    - Chunked processing to manage memory usage
    - Parallel processing for timeline construction
    - Automatic garbage collection
    - Memory monitoring and warnings
    - Progress tracking
    """
    
    def __init__(self, chunk_size: int = 50, max_workers: int = 4):
        """
        Initialize optimized timeline builder.
        
        Args:
            chunk_size (int): Number of patients to process in each chunk
            max_workers (int): Maximum number of parallel workers
        """
        self.chunk_size = chunk_size
        self.max_workers = min(max_workers, mp.cpu_count())
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for timeline building operations."""
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
    
    def _monitor_memory(self) -> float:
        """Monitor current memory usage."""
        return psutil.virtual_memory().percent
    
    def _check_memory_warning(self, threshold: float = 90.0) -> bool:
        """Check if memory usage is above threshold."""
        memory_usage = self._monitor_memory()
        if memory_usage > threshold:
            self.logger.warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
            return True
        return False
    
    def _process_patient_chunk(self, patient_chunk: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Process a chunk of patients to build timelines.
        
        Args:
            patient_chunk (Dict[str, Dict[str, pd.DataFrame]]): Chunk of patient data
            
        Returns:
            List[Dict[str, Any]]: List of patient timelines
        """
        timelines = []
        
        for patient_id, patient_data in patient_chunk.items():
            try:
                timeline = self._build_single_patient_timeline(patient_id, patient_data)
                if timeline is not None:
                    timelines.append(timeline)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to build timeline for patient {patient_id}: {str(e)}")
                continue
        
        return timelines
    
    def _build_single_patient_timeline(self, patient_id: str, 
                                     patient_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Build timeline for a single patient.
        
        Args:
            patient_id (str): Patient identifier
            patient_data (Dict[str, pd.DataFrame]): Patient's data tables
            
        Returns:
            Optional[Dict[str, Any]]: Patient timeline or None if failed
        """
        try:
            # Extract patient information
            if 'patients' not in patient_data or patient_data['patients'].empty:
                return None
            
            patient_info = patient_data['patients'].iloc[0]
            
            # Extract timestamps from all tables
            all_timestamps = []
            
            # Add observations timestamps
            if 'observations' in patient_data and not patient_data['observations'].empty:
                obs_timestamps = patient_data['observations']['date'].tolist()
                all_timestamps.extend(obs_timestamps)
            
            # Add conditions timestamps
            if 'conditions' in patient_data and not patient_data['conditions'].empty:
                cond_timestamps = patient_data['conditions']['start'].tolist()
                all_timestamps.extend(cond_timestamps)
            
            # Add medications timestamps
            if 'medications' in patient_data and not patient_data['medications'].empty:
                med_timestamps = patient_data['medications']['start'].tolist()
                all_timestamps.extend(med_timestamps)
            
            # Add procedures timestamps
            if 'procedures' in patient_data and not patient_data['procedures'].empty:
                proc_timestamps = patient_data['procedures']['date'].tolist()
                all_timestamps.extend(proc_timestamps)
            
            # Add encounters timestamps
            if 'encounters' in patient_data and not patient_data['encounters'].empty:
                enc_timestamps = patient_data['encounters']['start'].tolist()
                all_timestamps.extend(enc_timestamps)
            
            # Convert timestamps to datetime objects
            timestamps = []
            for ts in all_timestamps:
                try:
                    if isinstance(ts, str):
                        timestamp = pd.to_datetime(ts)
                    else:
                        timestamp = ts
                    timestamps.append(timestamp)
                except Exception:
                    continue
            
            # Remove duplicates and sort
            timestamps = sorted(list(set(timestamps)))
            
            if not timestamps:
                return None
            
            # Create timeline structure
            timeline = {
                'patient_id': patient_id,
                'timestamps': timestamps,
                'patient_info': patient_info.to_dict(),
                'events': {
                    'observations': patient_data.get('observations', pd.DataFrame()).to_dict('records'),
                    'conditions': patient_data.get('conditions', pd.DataFrame()).to_dict('records'),
                    'medications': patient_data.get('medications', pd.DataFrame()).to_dict('records'),
                    'procedures': patient_data.get('procedures', pd.DataFrame()).to_dict('records'),
                    'encounters': patient_data.get('encounters', pd.DataFrame()).to_dict('records')
                }
            }
            
            return timeline
            
        except Exception as e:
            self.logger.error(f"❌ Error building timeline for patient {patient_id}: {str(e)}")
            return None
    
    def build_patient_timelines_chunked(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Build timelines for all patients using chunked processing.
        
        Args:
            all_patient_data (Dict[str, Dict[str, pd.DataFrame]]): All patient data
            
        Returns:
            List[Dict[str, Any]]: List of all patient timelines
        """
        self.logger.info(f"🚀 Starting optimized timeline building for {len(all_patient_data)} patients...")
        
        total_patients = len(all_patient_data)
        all_timelines = []
        
        # Process in chunks
        patient_items = list(all_patient_data.items())
        
        for i in range(0, total_patients, self.chunk_size):
            chunk_end = min(i + self.chunk_size, total_patients)
            chunk_items = patient_items[i:chunk_end]
            chunk_data = dict(chunk_items)
            
            chunk_num = i // self.chunk_size + 1
            total_chunks = (total_patients + self.chunk_size - 1) // self.chunk_size
            
            self.logger.info(f"🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_data)} patients)")
            
            # Process chunk
            chunk_timelines = self._process_patient_chunk(chunk_data)
            all_timelines.extend(chunk_timelines)
            
            # Memory management
            gc.collect()
            
            # Check memory usage
            if self._check_memory_warning():
                self.logger.warning("⚠️ High memory usage detected, continuing with caution...")
            
            # Progress update
            processed = min(i + self.chunk_size, total_patients)
            self.logger.info(f"📊 Progress: {processed}/{total_patients} patients processed")
        
        self.logger.info(f"✅ Timeline building complete: {len(all_timelines)} timelines created")
        return all_timelines
    
    def build_patient_timelines_parallel(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Build timelines using parallel processing.
        
        Args:
            all_patient_data (Dict[str, Dict[str, pd.DataFrame]]): All patient data
            
        Returns:
            List[Dict[str, Any]]: List of all patient timelines
        """
        self.logger.info(f"🚀 Starting parallel timeline building for {len(all_patient_data)} patients...")
        
        # Split data into chunks for parallel processing
        patient_items = list(all_patient_data.items())
        chunks = []
        
        for i in range(0, len(patient_items), self.chunk_size):
            chunk_items = patient_items[i:i + self.chunk_size]
            chunks.append(dict(chunk_items))
        
        all_timelines = []
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_patient_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    chunk_timelines = future.result()
                    all_timelines.extend(chunk_timelines)
                    
                    self.logger.info(f"✅ Chunk {chunk_num + 1}/{len(chunks)} completed: {len(chunk_timelines)} timelines")
                    
                    # Memory management
                    gc.collect()
                    
                    # Check memory usage
                    if self._check_memory_warning():
                        self.logger.warning("⚠️ High memory usage detected in parallel processing")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing chunk {chunk_num + 1}: {str(e)}")
        
        self.logger.info(f"✅ Parallel timeline building complete: {len(all_timelines)} timelines created")
        return all_timelines
    
    def build_patient_timelines(self, all_patient_data: Dict[str, Dict[str, pd.DataFrame]], 
                              use_parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Build timelines for all patients with automatic optimization.
        
        Args:
            all_patient_data (Dict[str, Dict[str, pd.DataFrame]]): All patient data
            use_parallel (bool): Whether to use parallel processing
            
        Returns:
            List[Dict[str, Any]]: List of all patient timelines
        """
        total_patients = len(all_patient_data)
        
        # Choose processing method based on dataset size and available resources
        if total_patients > 1000 and use_parallel and self.max_workers > 1:
            self.logger.info("🔄 Using parallel processing for large dataset")
            return self.build_patient_timelines_parallel(all_patient_data)
        else:
            self.logger.info("🔄 Using chunked processing")
            return self.build_patient_timelines_chunked(all_patient_data) 