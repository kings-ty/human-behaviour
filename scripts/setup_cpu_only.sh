#!/bin/bash
# HRI30 Action Recognition - CPU-Only Setup
# For systems without CUDA/GPU support

echo "🖥️  HRI30 Action Recognition - CPU-Only Setup"
echo "=============================================="
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

echo "🐍 Using Python: $($PYTHON --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON -m venv venv_cpu
source venv_cpu/bin/activate

# Install CPU-only requirements
echo "⬇️  Installing CPU-only PyTorch and dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "
import torch
import torchvision
import cv2
print('✅ PyTorch:', torch.__version__)
print('✅ TorchVision:', torchvision.__version__)
print('✅ OpenCV:', cv2.__version__)
print('✅ Device:', 'CUDA available' if torch.cuda.is_available() else 'CPU-only')
"

echo ""
echo "🎯 CPU-Only Setup Complete!"
echo "============================"
echo ""
echo "🚀 To start training:"
echo "   source venv_cpu/bin/activate"
echo "   python run_training.py --device cpu --preset fast"
echo ""
echo "💡 Tips for CPU training:"
echo "   - Use smaller batch sizes (2-4)"
echo "   - Use smaller input resolution (224x224)"
echo "   - Use fewer epochs (30-50)"
echo "   - Consider using a subset of data for testing"
echo ""