#!/bin/bash

# COVID-19 TFT Project Environment Activation Script

echo "🚀 Activating COVID-19 TFT Project Environment..."

# Check if virtual environment exists
if [ ! -d "covid19_tft_env" ]; then
    echo "❌ Virtual environment not found. Creating new environment..."
    python -m venv covid19_tft_env
fi

# Activate virtual environment
source covid19_tft_env/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Installing/updating dependencies..."

# Install/update dependencies
pip install -r requirements.txt

echo "🎯 Environment ready for COVID-19 TFT development!"
echo ""
echo "📋 Available commands:"
echo "  - python src/data/synthea_loader.py    # Data preprocessing"
echo "  - python src/training/trainer.py       # Model training"
echo "  - python src/evaluation/evaluator.py   # Model evaluation"
echo "  - jupyter notebook                      # Start Jupyter"
echo ""
echo "📚 Project documentation: README.md"
echo "🔧 Development rules: .cursor/rules/" 