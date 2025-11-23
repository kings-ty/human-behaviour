# 🚀 HRI30 Action Recognition

Deep learning framework for industrial human-robot interaction action recognition based on the HRI30 dataset.

## 📋 Overview

This project implements state-of-the-art action recognition models for the **HRI30: An Action Recognition Dataset for Industrial Human-Robot Interaction**. The framework is designed to work across different hardware configurations, from CPU-only systems to high-end GPUs.

### 🎯 Key Features

- **Universal Hardware Support**: CPU-only, NVIDIA GPUs, Apple Silicon
- **Multiple Model Architectures**: SlowOnly, TSN, ir-CSN implementations
- **Automatic Hardware Detection**: Optimal settings for your system
- **30 Action Classes**: Industrial human-robot interaction scenarios
- **Production Ready**: Easy deployment and team collaboration

### 📊 Dataset

- **Classes**: 30 industrial action categories
- **Total Videos**: 2,940 clips (98 per class)
- **Resolution**: 720×480 @ 30 FPS
- **Duration**: 1-2 seconds per clip
- **Categories**:
  - Human-Object Interaction (6 classes)
  - Body Motion (20 classes)
  - Human-Robot Collaboration (4 classes)

## 🚀 Quick Start

### Option 1: CPU-Only Setup (No GPU Required)

Perfect for development, testing, or systems without dedicated graphics cards.

```bash
# Linux/macOS
./setup_cpu_only.sh
source venv_cpu/bin/activate

# Windows
setup_cpu_only.bat
call venv_cpu\Scripts\activate

# Start training
python run_training.py --device cpu --preset fast
```

### Option 2: GPU Setup (NVIDIA CUDA)

For systems with NVIDIA graphics cards.

```bash
# Install CUDA dependencies
pip install -r requirements-cuda.txt

# Auto-detect and optimize for your GPU
python run_training.py

# Manual GPU specification
python run_training.py --device cuda --batch_size 16
```

### Option 3: Universal Setup

Works on any system - automatically detects best configuration.

```bash
# Install dependencies
pip install -r requirements.txt

# Setup project
python setup.py --action setup_all

# Start training (auto-detects hardware)
python run_training.py
```

## 💻 Hardware Configurations

The framework automatically optimizes settings based on detected hardware:

| Hardware | Batch Size | Resolution | Training Time | Expected Accuracy |
|----------|------------|------------|---------------|-------------------|
| **CPU Only** | 2 | 224×224 | ~45 min/epoch | 75-80% |
| **GTX 1060/MX450** | 4-8 | 224×224 | ~15 min/epoch | 78-82% |
| **RTX 2060/3060** | 16 | 256×256 | ~6 min/epoch | 82-85% |
| **RTX 3080+/Xavier** | 32 | 256×256 | ~3 min/epoch | 85-87% |

## 📁 Project Structure

```
human-bahviour/
├── src/                          # Core modules
│   ├── models/                   # Model architectures
│   ├── data/                     # Data loading & preprocessing
│   ├── training/                 # Training logic
│   └── utils/                    # Utilities
├── configs/                      # Configuration files
│   ├── cpu_optimized.py         # CPU-only config
│   ├── gpu_balanced.py          # GPU config
│   └── hardware_specific.py     # Auto-detection
├── scripts/                      # Setup & utility scripts
│   ├── setup_cpu_only.sh        # CPU setup (Linux/Mac)
│   └── setup_cpu_only.bat       # CPU setup (Windows)
├── data/                         # Data directory (gitignored)
│   ├── train_set/               # Training videos
│   └── annotations/             # Labels & metadata
├── experiments/                  # Training outputs
├── requirements.txt             # CPU dependencies (default)
├── requirements-cuda.txt        # GPU dependencies
└── README.md                    # This file
```

## 🔧 Advanced Usage

### Training Commands

```bash
# Quick test (5 epochs)
python run_training.py --preset fast

# Full training with monitoring
python run_training.py --epochs 100 --tensorboard

# Resume from checkpoint
python run_training.py --resume experiments/*/best_model.pth

# Custom configuration
python run_training.py --config configs/cpu_optimized.py

# Specific hardware targeting
python run_training.py --device mx450 --batch_size 4
```

### Evaluation & Testing

```bash
# Evaluate trained model
python evaluate.py --model_path experiments/*/best_model.pth

# Test on specific videos
python test_single.py --video_path data/test_video.mp4 --model_path model.pth

# Export results
python evaluate.py --export_results --format csv
```

### Monitoring Training

```bash
# TensorBoard (real-time monitoring)
tensorboard --logdir experiments/

# View training logs
tail -f experiments/*/training.log

# Check system resources
python monitor_resources.py
```

## 🛠️ Development Setup

### For Contributors

```bash
# Clone repository
git clone <repository_url>
cd human-bahviour

# Setup development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Adding Your Own Data

1. **Video Format**: `.mp4`, `.avi`, or `.mov`
2. **Naming Convention**: `v_{class_id}_g{group}_c{clip}.{ext}`
3. **Place in**: `data/train_set/` directory
4. **Annotations**: Update `data/annotations/labels.json`

```python
# Example annotation format
{
    "v_0_g1_c1.avi": {
        "class": "Pick_Up_Object",
        "class_id": 0,
        "duration": 1.5,
        "fps": 30
    }
}
```

## 🔍 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Use CPU mode
python run_training.py --device cpu

# Or reduce batch size
python run_training.py --batch_size 2
```

**"No videos found"**
```bash
# Check data directory
ls data/train_set/
# Ensure videos are in correct format and location
```

**"Package conflicts"**
```bash
# Clean install
pip install --force-reinstall -r requirements.txt
```

**Slow CPU training**
```bash
# Use optimized CPU config
python run_training.py --config configs/cpu_optimized.py --preset fast
```

### Platform-Specific Notes

**Windows:**
- Use Anaconda Prompt or PowerShell
- May need Visual Studio Build Tools for compilation

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- For Apple Silicon, ensure ARM64 Python

**Linux:**
- Install build essentials: `sudo apt install build-essential`
- For CUDA, ensure proper driver installation

## 📈 Model Performance

Based on HRI30 paper benchmarks:

| Model | Accuracy | Speed | Memory | Best For |
|-------|----------|-------|---------|----------|
| **SlowOnly** | 86.55% | Medium | High | Accuracy |
| **TSN** | 82.1% | Fast | Low | Speed |
| **ir-CSN** | 84.3% | Medium | Medium | Balanced |

## 🤝 Team Collaboration

### Sharing Project

1. **Share entire folder** (excluding data)
2. **Team members run**: Platform-specific setup script
3. **Each person gets**: Hardware-optimized configuration
4. **Results are**: Comparable across different systems

### CI/CD Integration

```yaml
# Example GitHub Actions
- name: Test CPU Training
  run: |
    ./setup_cpu_only.sh
    source venv_cpu/bin/activate
    python run_training.py --preset test --epochs 2
```

## 📝 License

[Specify your license here]

## 📞 Support

For issues, questions, or contributions:
- Open an issue in the repository
- Check the troubleshooting section above
- Review existing documentation

## 🙏 Acknowledgments

Based on the research paper: "HRI30: An Action Recognition Dataset for Industrial Human-Robot Interaction"