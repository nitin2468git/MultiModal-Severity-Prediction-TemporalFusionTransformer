#!/usr/bin/env python3
"""
Training pipeline for COVID-19 TFT Severity Prediction
Part of the COVID-19 TFT Severity Prediction project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.tft_model import COVID19TFTModel
from evaluation.metrics import COVID19Metrics

class COVID19Trainer:
    """
    Training pipeline for COVID-19 TFT model.
    
    This class handles the complete training process including
    multi-task learning, validation, checkpointing, and monitoring.
    """
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Initialize trainer.
        
        Args:
            config_path (str): Path to training configuration
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Metrics calculator
        self.metrics_calculator = COVID19Metrics()
        
        # Data type mapping
        self.dtype_mapping = {
            'static_features': torch.float32,
            'temporal_features': torch.float32,
            'time_index': torch.float32,
            'sequence_length': torch.long,
            'mortality_risk': torch.float32,
            'icu_admission': torch.float32,
            'ventilator_need': torch.float32,
            'length_of_stay': torch.float32
        }
        
        # Non-tensor fields
        self.non_tensor_fields = {'patient_ids'}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training operations."""
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
        
    def _convert_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert batch data to appropriate types and devices.
        
        Args:
            batch (Dict[str, Any]): Input batch
            
        Returns:
            Dict[str, Any]: Converted batch
        """
        converted_batch = {}
        
        # Define fields that should remain as strings
        string_fields = {'patient_ids', 'patient_id'}
        
        for key, value in batch.items():
            if key in string_fields:
                # Keep string fields as is
                converted_batch[key] = value
            elif isinstance(value, torch.Tensor):
                # Convert existing tensors to correct dtype and device
                dtype = self.dtype_mapping.get(key, torch.float32)
                converted_batch[key] = value.to(dtype=dtype, device=self.device)
            elif isinstance(value, (list, np.ndarray)):
                # Convert lists/arrays to tensors
                if key in string_fields:
                    converted_batch[key] = value
                else:
                    dtype = self.dtype_mapping.get(key, torch.float32)
                    converted_batch[key] = torch.tensor(value, dtype=dtype, device=self.device)
            else:
                # Keep other types as is
                converted_batch[key] = value
        
        return converted_batch
    
    def setup_model(self, model_config_path: str = "configs/model_config.yaml"):
        """Setup model with configuration."""
        # Load model config
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Initialize model
        self.model = COVID19TFTModel(model_config).to(self.device)
        
        # Setup optimizer
        training_config = self.config['training']
        if training_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 0.01)
            )
        elif training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate']
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate']
            )
        
        # Setup scheduler
        if training_config.get('scheduler', {}).get('enabled', False):
            scheduler_config = training_config['scheduler']
            if scheduler_config['type'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=training_config['epochs'],
                    eta_min=scheduler_config.get('min_lr', 1e-6)
                )
            elif scheduler_config['type'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def create_data_loader(self, dataset, batch_size: int = 32) -> DataLoader:
        """Create a DataLoader from a dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            collate_fn=getattr(dataset, '_collate_fn', None)
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            Tuple[float, Dict[str, float]]: Average loss and metrics
        """
        print(f"[DEBUG] Starting train_epoch with train_loader: {type(train_loader)}")
        print(f"[DEBUG] Train loader length: {len(train_loader)}")
        
        self.model.train()
        total_loss = 0.0
        all_predictions = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        all_targets = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        
        print(f"[DEBUG] About to iterate over train_loader...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            print(f"[DEBUG] Processing batch {batch_idx}")
            print(f"[DEBUG] Batch type: {type(batch)}")
            if isinstance(batch, dict):
                print(f"[DEBUG] Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    print(f"[DEBUG] {key}: type={type(value)}, shape={getattr(value, 'shape', 'no shape') if hasattr(value, 'shape') else 'no shape'}")
            else:
                print(f"[DEBUG] Batch is not a dict: {batch}")
            # Convert batch data types consistently
            batch = self._convert_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            targets = {k: batch[k].float() for k in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay'] if k in batch}
            losses = self.model.compute_loss(predictions, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping with adaptive threshold
            if self.config['training'].get('gradient_clipping', {}).get('enabled', True):
                clip_config = self.config['training'].get('gradient_clipping', {})
                clip_type = clip_config.get('type', 'norm')  # 'norm' or 'value'
                clip_threshold = clip_config.get('threshold', 1.0)
                
                # Calculate gradient norm for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=float('inf'),  # Don't clip, just calculate norm
                    norm_type=2
                )
                
                # Adjust threshold if using adaptive clipping
                if clip_config.get('adaptive', False):
                    history_len = clip_config.get('history_length', 100)
                    if not hasattr(self, 'grad_norm_history'):
                        self.grad_norm_history = []
                    self.grad_norm_history.append(grad_norm.item())
                    if len(self.grad_norm_history) > history_len:
                        self.grad_norm_history = self.grad_norm_history[-history_len:]
                    
                    # Use percentile-based threshold
                    if len(self.grad_norm_history) > 10:
                        percentile = clip_config.get('adaptive_percentile', 95)
                        clip_threshold = np.percentile(self.grad_norm_history, percentile)
                
                # Apply clipping
                if clip_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=clip_threshold,
                        norm_type=2
                    )
                else:  # clip_type == 'value'
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        clip_value=clip_threshold
                    )
                
                # Log if gradient norm is high
                if grad_norm > clip_threshold * 2:
                    self.logger.warning(f"High gradient norm: {grad_norm:.2f} (threshold: {clip_threshold:.2f})")
            
            self.optimizer.step()
            
            # Accumulate loss and predictions
            total_loss += losses['total_loss'].item()
            
            # Store predictions and targets for metrics
            for task in all_predictions.keys():
                if task in predictions:
                    all_predictions[task].extend(predictions[task].detach().cpu().numpy())
                if task in targets:
                    all_targets[task].extend(targets[task].detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple[float, Dict[str, float]]: Average loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        all_targets = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Convert batch data types consistently
                batch = self._convert_batch(batch)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                targets = {k: batch[k].float() for k in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay'] if k in batch}
                losses = self.model.compute_loss(predictions, targets)
                
                # Accumulate loss and predictions
                total_loss += losses['total_loss'].item()
                
                # Store predictions and targets for metrics
                for task in all_predictions.keys():
                    if task in predictions:
                        all_predictions[task].extend(predictions[task].cpu().numpy())
                    if task in targets:
                        all_targets[task].extend(targets[task].cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Complete training process.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (Optional[DataLoader]): Test data loader
            
        Returns:
            Dict[str, Any]: Training results and history
        """
        print(f"[DEBUG] Starting training process")
        print(f"[DEBUG] Train loader type: {type(train_loader)}")
        print(f"[DEBUG] Train loader length: {len(train_loader)}")
        print(f"[DEBUG] Val loader type: {type(val_loader)}")
        print(f"[DEBUG] Val loader length: {len(val_loader)}")
        
        self.logger.info("Starting training process")
        
        # Early stopping setup
        early_stopping_config = self.config['training']['early_stopping']
        patience = early_stopping_config['patience']
        min_delta = early_stopping_config.get('min_delta', 1e-4)
        min_epochs = early_stopping_config.get('min_epochs', 10)
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Calculate relative improvement
            rel_improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss > 0 else float('inf')
            
            # Checkpointing and early stopping logic
            if val_loss < best_val_loss and rel_improvement > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                self.logger.info(f"New best validation loss: {val_loss:.4f} (improvement: {rel_improvement:.2%})")
            else:
                patience_counter += 1
                if patience_counter == patience // 2:
                    self.logger.warning(f"No improvement for {patience_counter} epochs")
            
            # Save latest checkpoint
            self.save_checkpoint('latest_model.pth', epoch, val_loss)
            
            # Early stopping checks
            if early_stopping_config['enabled'] and epoch >= min_epochs:
                # Check for divergence
                if val_loss > 1.5 * min(self.training_history['val_loss']):
                    self.logger.warning("Training diverging, stopping early")
                    break
                    
                # Check for plateau
                if patience_counter >= patience:
                    recent_losses = self.training_history['val_loss'][-patience:]
                    loss_std = np.std(recent_losses)
                    if loss_std < min_delta:
                        self.logger.info(f"Loss plateaued with std {loss_std:.6f}, stopping early")
                    else:
                        self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Final evaluation on test set
        test_results = None
        if test_loader is not None:
            test_loss, test_metrics = self.validate_epoch(test_loader)
            test_results = {'test_loss': test_loss, 'test_metrics': test_metrics}
            self.logger.info(f"Final test loss: {test_loss:.4f}")
        
        # Save training history
        self.save_training_history()
        
        # Generate plots
        self.plot_training_history()
        
        return {
            'training_history': self.training_history,
            'test_results': test_results,
            'best_val_loss': best_val_loss,
            'final_epoch': self.current_epoch
        }
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['checkpointing']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def save_training_history(self):
        """Save training history to file."""
        history_path = Path("experiments/results/training_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, values in self.training_history.items():
            if isinstance(values, list):
                serializable_history[key] = []
                for v in values:
                    if isinstance(v, (np.floating, np.integer)):
                        serializable_history[key].append(float(v))
                    elif isinstance(v, dict):
                        # Handle nested dictionaries (like metrics)
                        serializable_dict = {}
                        for k, val in v.items():
                            if isinstance(val, (np.floating, np.integer)):
                                serializable_dict[k] = float(val)
                            else:
                                serializable_dict[k] = val
                        serializable_history[key].append(serializable_dict)
                    else:
                        serializable_history[key].append(v)
            else:
                serializable_history[key] = values
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Generate training history plots."""
        plots_dir = Path("experiments/results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Metrics plots
        if self.training_history['val_metrics']:
            metrics = self.training_history['val_metrics'][-1]  # Latest metrics
            
            plt.subplot(2, 2, 2)
            tasks = ['mortality_risk', 'icu_admission', 'ventilator_need']
            auroc_values = [metrics.get(f'{task}_auroc', 0) for task in tasks]
            plt.bar(tasks, auroc_values)
            plt.title('AUROC by Task')
            plt.ylabel('AUROC')
            plt.ylim(0, 1)
        
        plt.subplot(2, 2, 3)
        los_mae = [metrics.get('length_of_stay_mae', 0) for metrics in self.training_history['val_metrics']]
        plt.plot(los_mae, label='Length of Stay MAE')
        plt.title('Length of Stay MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        # Plot learning rate if available
        if hasattr(self.optimizer, 'param_groups'):
            lr = self.optimizer.param_groups[0]['lr']
            plt.text(0.5, 0.5, f'Final LR: {lr:.6f}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Final Learning Rate')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to {plots_dir}")
    
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Make predictions on a dataset.
        
        Args:
            data_loader (DataLoader): Data loader for prediction
            
        Returns:
            Dict[str, np.ndarray]: Predictions for each task
        """
        self.model.eval()
        predictions = {task: [] for task in ['mortality_risk', 'icu_admission', 'ventilator_need', 'length_of_stay']}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Making predictions"):
                # Convert batch data types consistently
                batch = self._convert_batch(batch)
                
                # Get predictions
                batch_predictions = self.model.predict(batch)
                
                # Store predictions
                for task, pred in batch_predictions.items():
                    predictions[task].extend(pred.cpu().numpy())
        
        # Convert to numpy arrays
        for task in predictions:
            predictions[task] = np.array(predictions[task])
        
        return predictions 