#!/usr/bin/env python3
"""
Main training script for HRI30 Action Recognition
Easy-to-use script to start training with different configurations
"""

import os
import argparse
import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config_for_device, ModelConfig, TrainingConfig
from src.data.data_loader import HRI30DataModule
from src.training.trainer import HRI30Trainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='HRI30 Action Recognition Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic training on auto-detected hardware
  python run_training.py
  
  # Training with custom parameters
  python run_training.py --model slowonly --epochs 50 --batch_size 8
  
  # Training for MX450 (limited memory)
  python run_training.py --device mx450 --batch_size 4 --epochs 30
  
  # Training for Jetson Xavier AGX (more memory)
  python run_training.py --device xavier --batch_size 16 --epochs 100
  
  # Resume from checkpoint
  python run_training.py --resume experiments/hri30_action_recognition_*/best_checkpoint.pth
  
  # Training with different split
  python run_training.py --split 2
        '''
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='/home/ty/human-bahviour',
                       help='Root directory of HRI30 dataset')
    parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3],
                       help='Train/test split to use (1, 2, or 3)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='slowonly', 
                       choices=['slowonly', 'tsn', 'ircsn', 'tin'],
                       help='Model architecture to use')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='Backbone architecture')
    parser.add_argument('--pretrained', type=str, default='kinetics400',
                       choices=['kinetics400', 'imagenet', 'none'],
                       help='Pretrained weights to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    
    # Hardware arguments  
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mx450', 'xavier'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers (auto-detected if not specified)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='hri30_action_recognition',
                       help='Name for this experiment')
    parser.add_argument('--save_dir', type=str, default='/home/ty/human-bahviour/experiments',
                       help='Directory to save experiments')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10,
                       help='How often to log during training')
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='How often to evaluate during training')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='How often to save checkpoints')
    
    # Quick presets
    parser.add_argument('--preset', type=str, default=None,
                       choices=['fast', 'balanced', 'quality'],
                       help='Use preset configuration')
    
    return parser.parse_args()


def apply_preset(args, config):
    """Apply preset configurations for different use cases"""
    if args.preset == 'fast':
        # Fast training for testing/debugging
        config['model'].epochs = 20
        config['model'].batch_size = 4
        config['data'].input_resolution = (224, 224)
        config['training'].eval_interval = 2
        config['training'].save_interval = 5
        print("Applied 'fast' preset: Quick training for testing")
        
    elif args.preset == 'balanced':
        # Balanced setting for most hardware
        config['model'].epochs = 50
        config['model'].batch_size = 8
        config['data'].input_resolution = (256, 256)
        config['training'].eval_interval = 5
        config['training'].save_interval = 10
        print("Applied 'balanced' preset: Good performance/speed tradeoff")
        
    elif args.preset == 'quality':
        # High quality setting for best results
        config['model'].epochs = 100
        config['model'].batch_size = 16
        config['data'].input_resolution = (256, 256)
        config['training'].eval_interval = 5
        config['training'].save_interval = 10
        config['training'].use_amp = True
        print("Applied 'quality' preset: Best possible results")


def validate_setup(config):
    """Validate that everything is set up correctly"""
    data_config = config['data']
    
    # Check if data directory exists
    if not os.path.exists(data_config.data_root):
        print(f"❌ Error: Data directory {data_config.data_root} does not exist")
        print("Please make sure your HRI30 dataset is properly set up")
        return False
    
    # Check for video files
    train_dir = os.path.join(data_config.data_root, data_config.train_dir)
    if not os.path.exists(train_dir):
        print(f"⚠️  Warning: Train directory {train_dir} does not exist")
        print("Looking for video files in root directory...")
    
    # Check CUDA availability if using GPU
    if config['model'].device == 'cuda' and not torch.cuda.is_available():
        print("❌ Error: CUDA requested but not available")
        print("Switching to CPU mode...")
        config['model'].device = 'cpu'
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = config['model'].batch_size
        estimated_memory = batch_size * 3 * 8 * 256 * 256 * 4 / 1e9  # Rough estimate
        
        if estimated_memory > gpu_memory * 0.8:  # Use 80% of available memory
            new_batch_size = max(1, int(batch_size * gpu_memory * 0.8 / estimated_memory))
            print(f"⚠️  Warning: Batch size {batch_size} may be too large for {gpu_memory:.1f}GB GPU")
            print(f"Recommending batch size: {new_batch_size}")
            config['model'].batch_size = new_batch_size
    
    return True


def print_configuration(config, args):
    """Print final configuration summary"""
    print("\n" + "="*80)
    print("🚀 HRI30 ACTION RECOGNITION TRAINING CONFIGURATION")
    print("="*80)
    
    print(f"📊 Dataset:")
    print(f"   Root: {config['data'].data_root}")
    print(f"   Split: {args.split}")
    print(f"   Classes: {config['data'].num_classes}")
    print(f"   Input size: {config['data'].input_resolution}")
    
    print(f"\n🤖 Model:")
    print(f"   Architecture: {config['model'].model_name}")
    print(f"   Backbone: {config['model'].backbone}")
    print(f"   Pretrained: {config['model'].pretrained}")
    
    print(f"\n🏋️  Training:")
    print(f"   Epochs: {config['model'].epochs}")
    print(f"   Batch size: {config['model'].batch_size}")
    print(f"   Learning rate: {config['model'].learning_rate}")
    print(f"   Device: {config['model'].device}")
    print(f"   Mixed precision: {config['training'].use_amp}")
    
    print(f"\n💾 Experiment:")
    print(f"   Name: {args.experiment_name}")
    print(f"   Save directory: {args.save_dir}")
    
    if args.resume:
        print(f"   Resume from: {args.resume}")
    
    print("="*80)


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    print("🎯 HRI30 Action Recognition Training")
    print(f"Paper: 'HRI30: An Action Recognition Dataset for Industrial Human-Robot Interaction'")
    print(f"Target baseline: 86.55% Top-1 accuracy (SlowOnly + Kinetics400)\n")
    
    # Get base configuration
    device_type = args.device if args.device != 'auto' else 'auto'
    config = get_config_for_device(device_type)
    
    # Update configuration with command line arguments
    config['data'].data_root = args.data_root
    config['model'].model_name = args.model
    config['model'].backbone = args.backbone
    config['model'].pretrained = args.pretrained
    config['model'].epochs = args.epochs
    config['model'].learning_rate = args.learning_rate
    config['model'].weight_decay = args.weight_decay
    config['training'].experiment_name = args.experiment_name
    config['training'].save_dir = args.save_dir
    config['training'].log_interval = args.log_interval
    config['training'].eval_interval = args.eval_interval
    config['training'].save_interval = args.save_interval
    config['training'].resume_checkpoint = args.resume
    
    # Override batch size and num_workers if specified
    if args.batch_size is not None:
        config['model'].batch_size = args.batch_size
    if args.num_workers is not None:
        config['model'].num_workers = args.num_workers
    if args.mixed_precision:
        config['training'].use_amp = True
    
    # Apply preset if specified
    if args.preset:
        apply_preset(args, config)
    
    # Validate setup
    if not validate_setup(config):
        print("❌ Setup validation failed. Please fix the issues and try again.")
        return 1
    
    # Print final configuration
    print_configuration(config, args)
    
    # Ask for confirmation
    response = input("\n🤔 Continue with training? [Y/n]: ")
    if response.lower() in ['n', 'no']:
        print("Training cancelled by user.")
        return 0
    
    try:
        # Create data module
        print("\n📚 Setting up data...")
        data_module = HRI30DataModule(
            config['data'],
            config['model'],
            split_id=args.split
        )
        
        # Create trainer
        print("🏗️  Creating trainer...")
        trainer = HRI30Trainer(
            config['model'],
            config['training'],
            data_module,
            device=config['model'].device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("\n🚀 Starting training...")
        trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {trainer.experiment_dir}")
        print(f"🏆 Best validation accuracy: {trainer.best_val_accuracy:.2f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)