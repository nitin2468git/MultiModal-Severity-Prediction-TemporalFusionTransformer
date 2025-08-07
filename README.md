# COVID-19 Severity Prediction using Temporal Fusion Transformer

A comprehensive machine learning project for predicting COVID-19 severity outcomes using multi-modal patient data and Temporal Fusion Transformer (TFT) architecture.

## 🎯 Project Overview

This project implements a **4-day sprint methodology** to rapidly develop a Temporal Fusion Transformer model for COVID-19 severity prediction. The model predicts multiple clinical outcomes simultaneously:

- **Mortality Risk** (Binary Classification)
- **ICU Admission** (Binary Classification) 
- **Ventilator Need** (Binary Classification)
- **Length of Stay** (Regression)

## 🏗️ Architecture

### Multi-Modal Data Integration
- **Static Features** (15 dimensions): Demographics, comorbidities, socioeconomic factors
- **Temporal Features** (50 dimensions): Vital signs, laboratory values, medications, procedures, devices

### TFT Model Specifications
- **Backbone**: `pytorch_forecasting.TemporalFusionTransformer`
- **Hidden Size**: 128 (configurable)
- **Attention Heads**: 4 (configurable)
- **Sequence Length**: Max 720 hours (30 days)
- **Multi-Task Heads**: 4 separate prediction heads with weighted loss

## 📁 Project Structure

```
covid19-tft-severity-prediction/
├── src/
│   ├── data/
│   │   ├── synthea_loader.py      # Data loading and validation
│   │   ├── timeline_builder.py    # Patient temporal sequences
│   │   ├── feature_engineer.py    # Feature engineering
│   │   └── data_validator.py      # Data quality checks
│   ├── models/
│   │   ├── tft_model.py          # TFT model implementation
│   │   ├── loss_functions.py     # Multi-task loss functions
│   │   └── baseline_models.py    # Baseline model comparisons
│   ├── training/
│   │   ├── trainer.py            # Training pipeline
│   │   ├── callbacks.py          # Custom callbacks
│   │   └── metrics.py            # Evaluation metrics
│   └── evaluation/
│       ├── evaluator.py          # Model evaluation
│       │   ├── attention_analysis.py  # Attention pattern analysis
│       └── clinical_validation.py     # Clinical validation
├── configs/
│   ├── model_config.yaml         # Model architecture config
│   ├── training_config.yaml      # Training parameters
│   └── data_config.yaml          # Data processing config
├── notebooks/
│   ├── data_exploration.ipynb   # Data analysis
│   ├── model_development.ipynb   # Model development
│   └── results_analysis.ipynb    # Results analysis
├── experiments/
│   ├── baseline_experiments/     # Baseline model results
│   ├── tft_experiments/         # TFT model results
│   ├── results/                 # Final results
│   └── checkpoints/             # Model checkpoints
├── data/
│   ├── processed/               # Processed data
│   └── cache/                   # Data cache
└── 10k_synthea_covid19_csv/    # Raw Synthea dataset
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/nitin2468git/MultiModal-Severity-Prediction-TemporalFusionTransformer.git
cd MultiModal-Severity-Prediction-TemporalFusionTransformer

# Activate virtual environment
source activate_env.sh

# Or manually:
python -m venv covid19_tft_env
source covid19_tft_env/bin/activate  # On Windows: covid19_tft_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Loading
```python
from src.data.synthea_loader import SyntheaLoader

# Load and validate data
loader = SyntheaLoader("10k_synthea_covid19_csv")
tables = loader.load_all_tables()
covid19_patients = loader.identify_covid19_patients()
quality_metrics = loader.validate_data_quality()
```

### 3. Model Training
```python
from src.training.trainer import COVID19Trainer
from configs.training_config import load_config

# Load configuration
config = load_config("configs/training_config.yaml")

# Initialize trainer
trainer = COVID19Trainer(config)
trainer.train()
```

## 📊 Expected Results

### Performance Targets
- **Mortality AUROC**: > 0.85
- **ICU AUROC**: > 0.80  
- **Ventilator AUROC**: > 0.75
- **LOS RMSE**: < 5.0 days
- **Calibration**: ECE < 0.05

### Clinical Impact
- **Risk Stratification**: Clear thresholds for clinical action
- **Resource Allocation**: Optimization framework for healthcare resources
- **Treatment Timing**: Causal analysis for intervention windows
- **Uncertainty Communication**: Calibrated confidence intervals

## 🔬 Research Contributions

### Novel Features
- **Multi-Modal Integration**: Static + temporal patient data
- **Multi-Task Learning**: Simultaneous prediction of 4 outcomes
- **Attention-Based Causal Discovery**: Granger causality with attention weights
- **Uncertainty Quantification**: Bayesian methods for calibrated predictions
- **Clinical Interpretability**: Attention patterns aligned with medical knowledge

### Technical Innovations
- **Temporal Fusion Transformer**: Adapted for healthcare time-series
- **Multi-Task Loss Function**: Weighted combination of clinical outcomes
- **Robust Data Processing**: Handle missing data and class imbalance
- **Scalable Architecture**: Support for 10,000+ patient datasets

## 📈 Development Timeline

### 4-Day Sprint Methodology

#### **Day 1: Data Pipeline Development**
- ✅ Environment setup and dependency installation
- ✅ Data exploration and quality assessment
- 🔄 SyntheaLoader implementation
- 🔄 TimelineBuilder implementation
- 🔄 FeatureEngineer development
- 🔄 TFTFormatter for PyTorch tensors

#### **Day 2: Model Implementation**
- 🔄 TFT model architecture implementation
- 🔄 Multi-task prediction heads
- 🔄 Loss function development
- 🔄 Training pipeline implementation
- 🔄 PyTorch Lightning integration

#### **Day 3: Training and Evaluation**
- 🔄 Hyperparameter tuning setup
- 🔄 Cross-validation implementation
- 🔄 Full model training (30-50 epochs)
- 🔄 Evaluation pipeline implementation
- 🔄 Attention analysis development

#### **Day 4: Results and Documentation**
- 🔄 Final model evaluation
- 🔄 Statistical analysis
- 🔄 Paper writing (ACM format)
- 🔄 Presentation creation
- 🔄 Repository organization

## 🛠️ Technical Stack

### Core Dependencies
- **PyTorch**: Deep learning framework
- **PyTorch Forecasting**: Temporal Fusion Transformer implementation
- **PyTorch Lightning**: Training framework
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning utilities

### Visualization & Analysis
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Lifelines**: Survival analysis
- **Statsmodels**: Statistical analysis

### Development Tools
- **YAML**: Configuration management
- **Jupyter**: Interactive development
- **Git**: Version control
- **Docker**: Containerization (optional)

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **GPU**: RTX 3070 or equivalent (8GB VRAM)
- **RAM**: 16GB+
- **Storage**: 10GB+ for dataset and models

### Performance Requirements
- **Training Time**: < 4 hours on RTX 3070
- **Memory Usage**: < 8GB GPU VRAM
- **Inference Time**: < 500ms per patient prediction
- **Scalability**: Support for 10,000+ patient dataset

## 🔍 Data Description

### Synthea COVID-19 Dataset
- **Source**: Synthea synthetic patient data
- **Size**: 10,000 synthetic patients
- **Tables**: 17 CSV files (patients, encounters, conditions, medications, etc.)
- **COVID-19 Patients**: ~9,106 identified patients
- **Temporal Coverage**: Variable length patient timelines

### Data Quality
- **Missing Data**: Handle up to 20% missing values
- **Class Imbalance**: Address through focal loss and sampling
- **Outliers**: Robust loss functions (Huber loss for regression)
- **Data Drift**: Monitor and detect distribution shifts

## 📚 Documentation

### Key Documents
- **Design Document**: `tft_design_doc.md` - Comprehensive project blueprint
- **Architecture**: `tft_architecture.md` - Technical architecture details
- **Paper Layout**: `tft_paper_layout.md` - Research paper structure

### Configuration Files
- **Model Config**: `configs/model_config.yaml` - TFT architecture parameters
- **Training Config**: `configs/training_config.yaml` - Training pipeline settings
- **Data Config**: `configs/data_config.yaml` - Data processing parameters

## 🤝 Contributing

### Development Guidelines
- Follow the 4-day sprint methodology
- Implement type hints and comprehensive documentation
- Ensure clinical relevance and interpretability
- Maintain code quality and testing standards
- Track experiments and results systematically

### Code Standards
- **Naming**: PascalCase for classes, snake_case for functions
- **Documentation**: Comprehensive docstrings with clinical context
- **Testing**: Unit tests for all components
- **Logging**: Structured logging with clinical relevance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contact

- **Repository**: https://github.com/nitin2468git/MultiModal-Severity-Prediction-TemporalFusionTransformer
- **Author**: Nitin Bhatnagar
- **Project**: COVID-19 TFT Severity Prediction

## 🙏 Acknowledgments

- **Synthea**: Synthetic patient data generation
- **PyTorch Forecasting**: Temporal Fusion Transformer implementation
- **PyTorch Lightning**: Training framework
- **Research Community**: COVID-19 clinical research contributions 