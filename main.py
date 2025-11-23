#!/usr/bin/env python3
"""
Main entry point for HRI30 Action Recognition
Simplified launcher for training and evaluation
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description='HRI30 Action Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Quick CPU training
  python main.py train --device cpu --preset fast
  
  # GPU training with auto-detection
  python main.py train --device auto
  
  # Evaluate model
  python main.py evaluate --model_path model.pth
  
  # Setup environment
  python main.py setup --platform cpu
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mx450', 'xavier'],
                             help='Target device')
    train_parser.add_argument('--preset', default='balanced', choices=['fast', 'balanced', 'accurate'],
                             help='Training preset')
    train_parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    train_parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model_path', required=True, help='Path to trained model')
    eval_parser.add_argument('--data_path', default='data/train_set', help='Path to test data')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup environment')
    setup_parser.add_argument('--platform', choices=['cpu', 'cuda', 'auto'], default='auto',
                             help='Platform to setup for')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from run_training import main as train_main
        # Convert args to format expected by run_training.py
        sys.argv = ['run_training.py']
        if args.device != 'auto':
            sys.argv.extend(['--device', args.device])
        if args.preset != 'balanced':
            sys.argv.extend(['--preset', args.preset])
        if args.epochs:
            sys.argv.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(['--batch_size', str(args.batch_size)])
        if args.config:
            sys.argv.extend(['--config', args.config])
        
        train_main()
        
    elif args.command == 'evaluate':
        from src.training.evaluate import main as eval_main
        sys.argv = ['evaluate.py', '--model_path', args.model_path, '--data_path', args.data_path]
        eval_main()
        
    elif args.command == 'setup':
        from src.utils.environment_setup import CrossPlatformSetup
        setup = CrossPlatformSetup()
        setup.run_full_setup()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()