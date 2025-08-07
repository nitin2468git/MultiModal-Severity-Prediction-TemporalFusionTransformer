#!/usr/bin/env python3
"""
Evaluation metrics for COVID-19 TFT Severity Prediction
Part of the COVID-19 TFT Severity Prediction project
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class COVID19Metrics:
    """
    Evaluation metrics for COVID-19 severity prediction.
    
    This class provides comprehensive evaluation metrics for multi-task
    COVID-19 severity prediction including clinical validation metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for metrics operations."""
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
    
    def calculate_metrics(self, predictions: Dict[str, np.ndarray], 
                         targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for all tasks.
        
        Args:
            predictions (Dict[str, np.ndarray]): Model predictions
            targets (Dict[str, np.ndarray]): Ground truth targets
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {}
        
        # Binary classification tasks
        binary_tasks = ['mortality_risk', 'icu_admission', 'ventilator_need']
        for task in binary_tasks:
            if task in predictions and task in targets:
                task_metrics = self._calculate_binary_metrics(
                    predictions[task], targets[task], task
                )
                metrics.update(task_metrics)
        
        # Regression task
        if 'length_of_stay' in predictions and 'length_of_stay' in targets:
            los_metrics = self._calculate_regression_metrics(
                predictions['length_of_stay'], targets['length_of_stay']
            )
            metrics.update(los_metrics)
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics(predictions, targets)
        metrics.update(overall_metrics)
        
        return metrics
    
    def _calculate_binary_metrics(self, predictions: np.ndarray, 
                                targets: np.ndarray, task_name: str) -> Dict[str, float]:
        """
        Calculate metrics for binary classification tasks.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            task_name (str): Name of the task
            
        Returns:
            Dict[str, float]: Binary classification metrics
        """
        metrics = {}
        
        # Convert to numpy arrays and ensure they are 1D
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Convert predictions to binary if needed
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            # Already probabilities
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            # Convert to probabilities
            predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid
            binary_predictions = (predictions > 0.5).astype(int)
        
        # Basic metrics
        metrics[f'{task_name}_auroc'] = roc_auc_score(targets, predictions)
        metrics[f'{task_name}_auprc'] = average_precision_score(targets, predictions)
        metrics[f'{task_name}_precision'] = precision_score(targets, binary_predictions, zero_division=0)
        metrics[f'{task_name}_recall'] = recall_score(targets, binary_predictions, zero_division=0)
        metrics[f'{task_name}_f1'] = f1_score(targets, binary_predictions, zero_division=0)
        
        # Confusion matrix metrics
        cm = confusion_matrix(targets, binary_predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics[f'{task_name}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics[f'{task_name}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[f'{task_name}_positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics[f'{task_name}_negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        else:
            # Only one class present
            metrics[f'{task_name}_specificity'] = 1.0
            metrics[f'{task_name}_sensitivity'] = 0.0
            metrics[f'{task_name}_positive_predictive_value'] = 0.0
            metrics[f'{task_name}_negative_predictive_value'] = 1.0
        
        return metrics
    
    def _calculate_regression_metrics(self, predictions: np.ndarray, 
                                    targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for regression task (length of stay).
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            
        Returns:
            Dict[str, float]: Regression metrics
        """
        metrics = {}
        
        # Convert to numpy arrays and ensure they are 1D
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Basic regression metrics
        metrics['length_of_stay_mae'] = mean_absolute_error(targets, predictions)
        metrics['length_of_stay_rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['length_of_stay_mape'] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        metrics['length_of_stay_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Clinical relevance metrics
        metrics['length_of_stay_within_24h'] = np.mean(np.abs(targets - predictions) <= 24)
        metrics['length_of_stay_within_48h'] = np.mean(np.abs(targets - predictions) <= 48)
        
        return metrics
    
    def _calculate_overall_metrics(self, predictions: Dict[str, np.ndarray], 
                                 targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate overall performance metrics.
        
        Args:
            predictions (Dict[str, np.ndarray]): All predictions
            targets (Dict[str, np.ndarray]): All targets
            
        Returns:
            Dict[str, float]: Overall metrics
        """
        metrics = {}
        
        # Average AUROC across binary tasks
        binary_tasks = ['mortality_risk', 'icu_admission', 'ventilator_need']
        auroc_scores = []
        for task in binary_tasks:
            if task in predictions and task in targets:
                auroc = roc_auc_score(targets[task], predictions[task])
                auroc_scores.append(auroc)
        
        if auroc_scores:
            metrics['average_auroc'] = np.mean(auroc_scores)
            metrics['auroc_std'] = np.std(auroc_scores)
        
        # Clinical composite score
        clinical_scores = []
        if 'mortality_risk' in predictions and 'mortality_risk' in targets:
            clinical_scores.append(roc_auc_score(targets['mortality_risk'], predictions['mortality_risk']))
        if 'icu_admission' in predictions and 'icu_admission' in targets:
            clinical_scores.append(roc_auc_score(targets['icu_admission'], predictions['icu_admission']))
        
        if clinical_scores:
            metrics['clinical_composite_score'] = np.mean(clinical_scores)
        
        return metrics
    
    def generate_evaluation_report(self, predictions: Dict[str, np.ndarray], 
                                 targets: Dict[str, np.ndarray],
                                 output_dir: str = "experiments/results") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            predictions (Dict[str, np.ndarray]): Model predictions
            targets (Dict[str, np.ndarray]): Ground truth targets
            output_dir (str): Output directory for reports
            
        Returns:
            Dict[str, Any]: Complete evaluation report
        """
        # Calculate all metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Generate plots
        plots = self._generate_evaluation_plots(predictions, targets, output_dir)
        
        # Create comprehensive report
        report = {
            'metrics': metrics,
            'plots': plots,
            'summary': self._generate_summary(metrics),
            'clinical_interpretation': self._generate_clinical_interpretation(metrics)
        }
        
        # Save report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'evaluation_report.json', 'w') as f:
            import json
            # Convert numpy types to native Python types
            serializable_report = self._make_json_serializable(report)
            json.dump(serializable_report, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {output_path / 'evaluation_report.json'}")
        
        return report
    
    def _generate_evaluation_plots(self, predictions: Dict[str, np.ndarray], 
                                 targets: Dict[str, np.ndarray],
                                 output_dir: str) -> Dict[str, str]:
        """
        Generate evaluation plots.
        
        Args:
            predictions (Dict[str, np.ndarray]): Model predictions
            targets (Dict[str, np.ndarray]): Ground truth targets
            output_dir (str): Output directory
            
        Returns:
            Dict[str, str]: Dictionary of plot file paths
        """
        plots = {}
        output_path = Path(output_dir) / 'plots'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ROC curves for binary tasks
        binary_tasks = ['mortality_risk', 'icu_admission', 'ventilator_need']
        plt.figure(figsize=(15, 5))
        
        for i, task in enumerate(binary_tasks):
            if task in predictions and task in targets:
                plt.subplot(1, 3, i + 1)
                self._plot_roc_curve(targets[task], predictions[task], task)
        
        plt.tight_layout()
        roc_plot_path = output_path / 'roc_curves.png'
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['roc_curves'] = str(roc_plot_path)
        
        # Length of stay regression plot
        if 'length_of_stay' in predictions and 'length_of_stay' in targets:
            plt.figure(figsize=(10, 8))
            self._plot_regression_results(targets['length_of_stay'], predictions['length_of_stay'])
            los_plot_path = output_path / 'length_of_stay_regression.png'
            plt.savefig(los_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['length_of_stay_regression'] = str(los_plot_path)
        
        # Confusion matrices
        plt.figure(figsize=(15, 5))
        for i, task in enumerate(binary_tasks):
            if task in predictions and task in targets:
                plt.subplot(1, 3, i + 1)
                self._plot_confusion_matrix(targets[task], predictions[task], task)
        
        plt.tight_layout()
        cm_plot_path = output_path / 'confusion_matrices.png'
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['confusion_matrices'] = str(cm_plot_path)
        
        return plots
    
    def _plot_roc_curve(self, targets: np.ndarray, predictions: np.ndarray, task_name: str):
        """Plot ROC curve for binary classification task."""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(targets, predictions)
        auroc = roc_auc_score(targets, predictions)
        
        plt.plot(fpr, tpr, label=f'{task_name} (AUROC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {task_name.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_regression_results(self, targets: np.ndarray, predictions: np.ndarray):
        """Plot regression results for length of stay."""
        plt.subplot(2, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual Length of Stay (hours)')
        plt.ylabel('Predicted Length of Stay (hours)')
        plt.title('Length of Stay: Predicted vs Actual')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        residuals = predictions - targets
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Length of Stay (hours)')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        error_24h = np.abs(residuals) <= 24
        error_48h = np.abs(residuals) <= 48
        plt.bar(['Within 24h', 'Within 48h'], 
                [np.mean(error_24h), np.mean(error_48h)])
        plt.ylabel('Proportion')
        plt.title('Prediction Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray, task_name: str):
        """Plot confusion matrix for binary classification task."""
        # Convert predictions to binary
        binary_predictions = (predictions > 0.5).astype(int)
        
        cm = confusion_matrix(targets, binary_predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {task_name.replace("_", " ").title()}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    def _generate_summary(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of model performance."""
        summary = {
            'overall_performance': {
                'average_auroc': metrics.get('average_auroc', 0.0),
                'clinical_composite_score': metrics.get('clinical_composite_score', 0.0)
            },
            'binary_classification': {},
            'regression': {}
        }
        
        # Binary classification summary
        binary_tasks = ['mortality_risk', 'icu_admission', 'ventilator_need']
        for task in binary_tasks:
            if f'{task}_auroc' in metrics:
                summary['binary_classification'][task] = {
                    'auroc': metrics[f'{task}_auroc'],
                    'precision': metrics[f'{task}_precision'],
                    'recall': metrics[f'{task}_recall'],
                    'f1_score': metrics[f'{task}_f1']
                }
        
        # Regression summary
        if 'length_of_stay_mae' in metrics:
            summary['regression']['length_of_stay'] = {
                'mae': metrics['length_of_stay_mae'],
                'rmse': metrics['length_of_stay_rmse'],
                'r2': metrics['length_of_stay_r2'],
                'within_24h': metrics['length_of_stay_within_24h'],
                'within_48h': metrics['length_of_stay_within_48h']
            }
        
        return summary
    
    def _generate_clinical_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate clinical interpretation of results."""
        interpretation = {}
        
        # Mortality risk interpretation
        if 'mortality_risk_auroc' in metrics:
            auroc = metrics['mortality_risk_auroc']
            if auroc >= 0.9:
                interpretation['mortality_risk'] = "Excellent discrimination for mortality risk prediction"
            elif auroc >= 0.8:
                interpretation['mortality_risk'] = "Good discrimination for mortality risk prediction"
            elif auroc >= 0.7:
                interpretation['mortality_risk'] = "Fair discrimination for mortality risk prediction"
            else:
                interpretation['mortality_risk'] = "Poor discrimination for mortality risk prediction"
        
        # ICU admission interpretation
        if 'icu_admission_auroc' in metrics:
            auroc = metrics['icu_admission_auroc']
            if auroc >= 0.85:
                interpretation['icu_admission'] = "Excellent discrimination for ICU admission prediction"
            elif auroc >= 0.75:
                interpretation['icu_admission'] = "Good discrimination for ICU admission prediction"
            else:
                interpretation['icu_admission'] = "Fair discrimination for ICU admission prediction"
        
        # Length of stay interpretation
        if 'length_of_stay_mae' in metrics:
            mae = metrics['length_of_stay_mae']
            within_24h = metrics.get('length_of_stay_within_24h', 0)
            
            if mae <= 24 and within_24h >= 0.8:
                interpretation['length_of_stay'] = "Excellent length of stay prediction accuracy"
            elif mae <= 48 and within_24h >= 0.6:
                interpretation['length_of_stay'] = "Good length of stay prediction accuracy"
            else:
                interpretation['length_of_stay'] = "Fair length of stay prediction accuracy"
        
        return interpretation
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj 