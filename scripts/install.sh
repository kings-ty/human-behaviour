#!/bin/bash
# HRI30 Action Recognition - Linux/macOS Quick Setup

echo "🚀 HRI30 Action Recognition Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect Python
echo -e "${BLUE}🐍 Detecting Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
    echo -e "${GREEN}✅ Found: $(python3 --version)${NC}"
elif command -v python &> /dev/null; then
    PYTHON=python
    echo -e "${GREEN}✅ Found: $(python --version)${NC}"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
MIN_VERSION="3.8"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then 
    echo -e "${RED}❌ Python $PYTHON_VERSION found, but requires $MIN_VERSION or higher${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}📦 Creating virtual environment...${NC}"
$PYTHON -m venv venv

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}⬆️ Upgrading pip...${NC}"
pip install --upgrade pip

# Detect system and hardware
echo -e "${BLUE}🔍 Detecting system configuration...${NC}"

OS_TYPE="unknown"
REQUIREMENTS_FILE="requirements-cpu.txt"

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}🍎 Detected macOS${NC}"
    OS_TYPE="macos"
    REQUIREMENTS_FILE="requirements-macos.txt"
    
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo -e "${GREEN}🚀 Apple Silicon detected${NC}"
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${GREEN}🐧 Detected Linux${NC}"
    OS_TYPE="linux"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
            echo -e "${GREEN}🎮 Detected NVIDIA GPU: $GPU_INFO${NC}"
            REQUIREMENTS_FILE="requirements-cuda.txt"
        else
            echo -e "${YELLOW}⚠️  nvidia-smi found but no GPU detected${NC}"
        fi
    else
        echo -e "${YELLOW}💻 No NVIDIA GPU detected, using CPU version${NC}"
    fi
else
    echo -e "${YELLOW}❓ Unknown OS type: $OSTYPE${NC}"
    echo -e "${YELLOW}💻 Defaulting to CPU version${NC}"
fi

# Install requirements
echo -e "${BLUE}📦 Installing Python packages from $REQUIREMENTS_FILE...${NC}"
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Requirements installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install requirements${NC}"
        echo -e "${YELLOW}💡 Trying CPU fallback...${NC}"
        pip install -r "requirements-cpu.txt"
    fi
else
    echo -e "${YELLOW}⚠️  $REQUIREMENTS_FILE not found, using CPU requirements${NC}"
    pip install -r "requirements-cpu.txt"
fi

# Run setup script
echo -e "${BLUE}🔧 Setting up project structure...${NC}"
$PYTHON setup.py --action setup_all

# Check installation
echo -e "${BLUE}🧪 Testing installation...${NC}"
$PYTHON -c "
import torch
import cv2
import numpy as np
print('✅ Core packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
print('🎯 Installation test passed!')
"

# Create activation script
echo -e "${BLUE}📜 Creating activation script...${NC}"
cat > activate_hri30.sh << 'EOF'
#!/bin/bash
# Activate HRI30 environment
source venv/bin/activate
echo "🚀 HRI30 Environment Activated!"
echo "💻 Hardware: $(python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")"
echo "📁 Ready to train! Run: python run_training.py"
EOF
chmod +x activate_hri30.sh

echo -e "\n${GREEN}✅ Setup Complete!${NC}"
echo -e "=================================="
echo -e "${BLUE}🚀 To start using HRI30:${NC}"
echo -e "   ${YELLOW}source activate_hri30.sh${NC}  # Activate environment"
echo -e "   ${YELLOW}python run_training.py${NC}     # Start training"
echo -e ""
echo -e "${BLUE}📚 Quick commands:${NC}"
echo -e "   ${YELLOW}python run_training.py --preset fast${NC}      # Quick test"
echo -e "   ${YELLOW}python run_training.py --device cpu${NC}       # Force CPU"
echo -e "   ${YELLOW}python setup.py --action check_data${NC}       # Check dataset"
echo -e ""
echo -e "${BLUE}📖 For detailed help, see:${NC}"
echo -e "   ${YELLOW}QUICKSTART.md${NC} - Quick start guide"
echo -e "   ${YELLOW}README.md${NC}     - Full documentation"
echo -e ""
echo -e "${GREEN}🎯 Target: 85%+ accuracy on HRI30 dataset!${NC}"