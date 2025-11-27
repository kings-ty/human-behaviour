#!/usr/bin/env python3
"""
CPU-only training script for HRI30 Action Recognition
Automatically bypasses user prompts and forces CPU training
"""

import os
import sys

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import after setting environment
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

from configs.config import get_config_for_device, ModelConfig, TrainingConfig
from src.data.data_loader import HRI30DataModule  
from src.training.trainer import HRI30Trainer

def main():
    """Main training function"""
    print("🎯 HRI30 Action Recognition Training (CPU Mode)")
    print("Paper: 'HRI30: An Action Recognition Dataset for Industrial Human-Robot Interaction'")
    print()
    
    # Get configuration optimized for CPU
    config = get_config_for_device("cpu_only")
    
    # Force CPU settings
    config['model'].device = "cpu"
    config['model'].batch_size = 1
    config['training'].use_amp = False
    
    print("================================================================================")
    print("🚀 HRI30 ACTION RECOGNITION TRAINING CONFIGURATION")
    print("================================================================================")
    print(f"📊 Dataset:")
    print(f"   Root: {config['data'].data_root}")
    print(f"   Split: 1")
    print(f"   Classes: {config['data'].num_classes}")
    print(f"   Input size: {config['data'].input_resolution}")
    print()
    print(f"🤖 Model:")
    print(f"   Architecture: {config['model'].model_name}")
    print(f"   Backbone: {config['model'].backbone}")
    print(f"   Pretrained: {config['model'].pretrained}")
    print()
    print(f"🏋️  Training:")
    print(f"   Epochs: {config['model'].epochs}")
    print(f"   Batch size: {config['model'].batch_size}")
    print(f"   Learning rate: {config['model'].learning_rate}")
    print(f"   Device: {config['model'].device}")
    print(f"   Mixed precision: {config['training'].use_amp}")
    print()
    print(f"💾 Experiment:")
    print(f"   Name: {config['training'].experiment_name}")
    print(f"   Save directory: {config['training'].save_dir}")
    print("================================================================================")
    print()
    
    try:
        # Setup data
        print("📚 Setting up data...")
        data_module = HRI30DataModule(
            data_config=config['data'],
            model_config=config['model'],
            batch_size=config['model'].batch_size,
            num_workers=config['model'].num_workers
        )
        
        # Create trainer
        print("🏗️  Creating trainer...")
        trainer = HRI30Trainer(
            model_config=config['model'],
            training_config=config['training'], 
            data_module=data_module,
            device="cpu"  # Force CPU
        )
        
        # Start training
        print("🚀 Starting training...")
        trainer.train()
        
        print("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)