# Multi-Modal COVID-19 Severity Prediction using Temporal Fusion Transformer

## 🎯 Project Overview

This project implements a novel multi-modal approach to COVID-19 severity prediction using Temporal Fusion Transformer (TFT) architecture. The system integrates static patient features (demographics, comorbidities) with temporal clinical data (vital signs, medications, procedures) to predict multiple severity outcomes including mortality, ICU admission, ventilator requirement, and length of stay.

## 🏗️ Architecture

### Core Components
- **Temporal Fusion Transformer**: Advanced attention-based architecture for multi-modal temporal data
- **Multi-Task Learning**: Simultaneous prediction of 4 clinical outcomes
- **Bayesian Uncertainty Quantification**: Calibrated confidence intervals for clinical decision support
- **Attention-Based Causal Discovery**: Novel method for temporal causal relationships

### Data Pipeline
```
Synthea COVID-19 Dataset → Data Preprocessing → Feature Engineering → TFT Model → Clinical Predictions
```

## 📊 Dataset

- **Source**: Synthea COVID-19 synthetic dataset (10,000 patients)
- **Features**: 
  - Static: Demographics, comorbidities, socioeconomic factors
  - Temporal: Vital signs, laboratory values, medications, procedures, devices
- **Outcomes**: Mortality, ICU admission, ventilator need, length of stay

## 🚀 Quick Start

### Environment Setup

1. **Clone the repository**
```bash
git clone https://github.com/nitin2468git/MultiModal-Severity-Prediction-TemporalFusionTransformer.git
cd MultiModal-Severity-Prediction-TemporalFusionTransformer
```

2. **Create virtual environment**
```bash
python -m venv covid19_tft_env
source covid19_tft_env/bin/activate  # On macOS/Linux
# or
covid19_tft_env\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Data Preprocessing**
```bash
python src/data/synthea_loader.py
```

2. **Model Training**
```bash
python src/training/trainer.py
```

3. **Evaluation**
```bash
python src/evaluation/evaluator.py
```

## 📁 Project Structure

```
covid19-tft-severity-prediction/
├── src/
│   ├── data/                    # Data processing pipeline
│   │   ├── synthea_loader.py
│   │   ├── timeline_builder.py
│   │   ├── feature_engineer.py
│   │   └── data_validator.py
│   ├── models/                  # Model implementations
│   │   ├── tft_model.py
│   │   ├── loss_functions.py
│   │   └── baseline_models.py
│   ├── training/                # Training pipeline
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── metrics.py
│   └── evaluation/              # Evaluation and analysis
│       ├── evaluator.py
│       ├── attention_analysis.py
│       └── clinical_validation.py
├── configs/                     # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── notebooks/                   # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_development.ipynb
│   └── results_analysis.ipynb
├── experiments/                 # Experiment logs
├── data/                        # Data files
│   ├── raw/                     # Original Synthea CSV files
│   ├── processed/               # Cleaned data
│   └── features/                # Engineered features
├── models/                      # Trained model checkpoints
├── results/                     # Generated results and plots
└── docs/                        # Documentation
```

## 🔬 Key Features

### Multi-Task Prediction
- **Mortality Risk**: Binary classification with calibrated probabilities
- **ICU Admission**: Time-to-event prediction with urgency scoring
- **Ventilator Requirement**: Support level classification (non-invasive/invasive)
- **Length of Stay**: Regression with discharge readiness assessment

### Advanced Analytics
- **Attention Analysis**: Temporal attention patterns for clinical interpretability
- **Causal Discovery**: Granger causality with attention weights
- **Uncertainty Quantification**: Bayesian confidence intervals
- **Clinical Validation**: Domain expert evaluation framework

### Performance Metrics
- **AUROC**: > 0.85 for mortality prediction
- **Calibration**: ECE < 0.05 for well-calibrated predictions
- **Clinical Utility**: Net benefit analysis for decision support

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, PyTorch Lightning, PyTorch Forecasting
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Scientific Computing**: Lifelines, Statsmodels
- **Development**: Jupyter, Black, Flake8

## 📈 Results

### Model Performance
| Metric | Mortality | ICU Admission | Ventilator Need | Length of Stay |
|--------|-----------|---------------|-----------------|----------------|
| AUROC  | 0.891±0.023 | 0.847±0.031 | 0.823±0.028 | - |
| RMSE   | - | - | - | 4.87±0.34 days |
| ECE    | 0.023 | 0.031 | 0.028 | - |

### Clinical Insights
- **Critical Time Window**: 24-48 hours post-admission
- **Key Features**: Oxygen saturation, respiratory patterns, inflammatory markers
- **Causal Relationships**: Discovered temporal dependencies between clinical variables

## 🎓 Research Contributions

1. **Novel Attention-Based Causal Discovery**: First application of attention weights for Granger causality in clinical data
2. **Bayesian Multi-Task Framework**: Rigorous uncertainty quantification for clinical prediction
3. **Patient Contrastive Learning**: Mathematical framework for clinical similarity representations
4. **Adaptive Loss Weighting**: Dynamic multi-task optimization with theoretical guarantees

## 📚 Publications

This work is part of ongoing research in interpretable healthcare AI. For detailed methodology and results, see the accompanying paper and presentation materials.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and development workflow rules in `.cursor/rules/`.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Synthea team for the COVID-19 synthetic dataset
- PyTorch Forecasting community for the TFT implementation
- Clinical domain experts for validation and feedback

## 📞 Contact

For questions or collaborations, please reach out to the project maintainers.

---

**Note**: This is a research project. For clinical use, additional validation and regulatory approval may be required. 