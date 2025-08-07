#!/usr/bin/env python3
"""
Test script for the COVID-19 TFT model.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.tft_model import COVID19TFTModel
import yaml

def test_model():
    """Test the TFT model with sample data."""
    print("Testing COVID-19 TFT Model...")
    
    # Load model config
    with open("configs/model_config.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Initialize model
    model = COVID19TFTModel(model_config)
    print(f"✓ Model initialized with {model.get_model_info()['total_parameters']:,} parameters")
    
    # Create sample batch
    batch_size = 4
    seq_len = 100
    static_dim = 15
    temporal_dim = 50
    
    sample_batch = {
        'static_features': torch.randn(batch_size, static_dim),
        'temporal_features': torch.randn(batch_size, seq_len, temporal_dim),
        'time_index': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float(),
        'mortality_risk': torch.randint(0, 2, (batch_size,)).float(),
        'icu_admission': torch.randint(0, 2, (batch_size,)).float(),
        'ventilator_need': torch.randint(0, 2, (batch_size,)).float(),
        'length_of_stay': torch.randn(batch_size,)
    }
    
    print(f"✓ Sample batch created: {batch_size} patients, {seq_len} timepoints")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(sample_batch)
    
    print("✓ Forward pass successful")
    
    # Check predictions
    for task, pred in predictions.items():
        print(f"  {task}: {pred.shape} - range [{pred.min():.3f}, {pred.max():.3f}]")
    
    # Test loss computation
    losses = model.compute_loss(predictions, {
        'mortality_risk': sample_batch['mortality_risk'],
        'icu_admission': sample_batch['icu_admission'],
        'ventilator_need': sample_batch['ventilator_need'],
        'length_of_stay': sample_batch['length_of_stay']
    })
    
    print("✓ Loss computation successful")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    print("\n🎉 Model test completed successfully!")
    return True

if __name__ == "__main__":
    test_model() 