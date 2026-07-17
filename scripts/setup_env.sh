#!/bin/bash
# Setup environment for QuantuML

set -e

echo "=========================================="
echo "QuantuML Environment Setup"
echo "=========================================="

# Python version check
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Create directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/{models,logs,evaluations}
mkdir -p data/{raw,processed}
mkdir -p deployment/{victron,api}

echo ""
echo "=========================================="
echo "Setup complete! Activate with:"
echo "  source venv/bin/activate"
echo "=========================================="
