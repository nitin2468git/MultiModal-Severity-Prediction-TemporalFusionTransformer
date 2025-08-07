#!/usr/bin/env python3
"""
Unit tests for TimelineBuilder datetime handling
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.timeline_builder import TimelineBuilder

class TestTimelineBuilder(unittest.TestCase):
    """Test cases for TimelineBuilder datetime handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.timeline_builder = TimelineBuilder(
            max_sequence_length=24,  # 24 hours
            time_interval='h'
        )
        
        # Create test data with timezone-aware timestamps
        self.test_data = {
            'observations': pd.DataFrame({
                'DATE': pd.to_datetime([
                    '2023-01-01 10:00:00+00:00',  # UTC timezone
                    '2023-01-01 12:00:00+00:00',
                    '2023-01-01 14:00:00+00:00'
                ]),
                'CODE': ['TEST1', 'TEST2', 'TEST3'],
                'DESCRIPTION': ['Test Obs 1', 'Test Obs 2', 'Test Obs 3'],
                'VALUE': [100, 120, 110]
            }),
            'encounters': pd.DataFrame({
                'START': pd.to_datetime([
                    '2023-01-01 09:00:00+00:00'  # UTC timezone
                ]),
                'STOP': pd.to_datetime([
                    '2023-01-01 17:00:00+00:00'
                ]),
                'CODE': ['ENCOUNTER1']
            })
        }
    
    def test_timestamp_extraction(self):
        """Test that timestamp extraction handles timezone-aware dates correctly"""
        timestamps = self.timeline_builder._extract_all_timestamps(self.test_data)
        
        # Verify we got the expected number of timestamps
        self.assertEqual(len(timestamps), 5)  # 3 observations + 1 encounter start + 1 encounter stop
        
        # Verify all timestamps are timezone-naive
        for ts in timestamps:
            self.assertIsNone(ts.tzinfo, f"Timestamp {ts} should be timezone-naive")
    
    def test_timeline_generation(self):
        """Test that timeline generation works with the extracted timestamps"""
        try:
            timeline = self.timeline_builder.build_patient_timeline(
                self.test_data,
                patient_id="TEST001"
            )
            
            # Verify timeline was created
            self.assertIsNotNone(timeline)
            self.assertEqual(timeline['patient_id'], "TEST001")
            
            # Get the timestamps
            timestamps = timeline['timestamps']
            
            # Verify we have timestamps
            self.assertTrue(len(timestamps) > 0)
            
            # Verify timestamps are hourly intervals
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                self.assertEqual(diff, timedelta(hours=1))
            
        except Exception as e:
            self.fail(f"Timeline generation failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()