"""
CPU-Optimized Configuration for HRI30 Action Recognition
For systems without CUDA/GPU support - maximum compatibility
"""

from config import get_config_for_device, DataConfig, ModelConfig, TrainingConfig

def get_cpu_optimized_config():
    """Get configuration optimized specifically for CPU training"""
    
    # Start with base configuration
    config = get_config_for_device("cpu_only")
    
    # CPU-specific optimizations
    config['model'].device = "cpu"
    config['model'].batch_size = 2  # Very small batch to avoid memory issues
    config['model'].num_workers = 1  # Reduced workers for CPU
    config['model'].pin_memory = False  # Disable for CPU
    
    # Reduce model complexity
    config['model'].epochs = 30  # Fewer epochs for faster training
    config['model'].learning_rate = 0.01  # Higher LR for faster convergence
    
    # Smaller input resolution for speed
    config['data'].input_resolution = (224, 224)  # Smaller than default
    config['data'].clip_duration = 1.5  # Shorter clips
    config['data'].sample_rate = 6  # Sample fewer frames
    
    # Training optimizations
    config['training'].use_amp = False  # No mixed precision on CPU
    config['training'].log_interval = 5  # More frequent logging
    config['training'].eval_interval = 10  # Less frequent evaluation
    config['training'].patience = 10  # Earlier stopping
    
    # Reduce model size for CPU
    config['model'].alpha = 8  # More temporal downsampling
    config['model'].num_segments = 4  # Fewer segments to sample
    
    return config

def get_cpu_fast_test_config():
    """Get configuration for quick testing on CPU (very fast but lower accuracy)"""
    
    config = get_cpu_optimized_config()
    
    # Even more aggressive settings for testing
    config['model'].batch_size = 1
    config['model'].epochs = 5  # Just a few epochs
    config['data'].input_resolution = (128, 128)  # Very small
    config['data'].clip_duration = 1.0  # Shortest clips
    config['data'].sample_rate = 8  # Very few frames
    config['model'].num_segments = 2  # Minimal segments
    config['training'].eval_interval = 2  # Evaluate often
    
    return config

def get_cpu_balanced_config():
    """Get balanced configuration for CPU (reasonable speed vs accuracy)"""
    
    config = get_cpu_optimized_config()
    
    # Balanced settings
    config['model'].batch_size = 4
    config['model'].epochs = 50
    config['data'].input_resolution = (224, 224)
    config['data'].clip_duration = 2.0
    config['data'].sample_rate = 4
    config['model'].num_segments = 6
    config['training'].patience = 15
    
    return config

# For direct import
CONFIG = get_cpu_optimized_config()
CONFIG_FAST = get_cpu_fast_test_config()
CONFIG_BALANCED = get_cpu_balanced_config()

if __name__ == "__main__":
    print("=== CPU-Optimized Configuration ===")
    config = get_cpu_optimized_config()
    
    print(f"Device: {config['model'].device}")
    print(f"Batch size: {config['model'].batch_size}")
    print(f"Input resolution: {config['data'].input_resolution}")
    print(f"Epochs: {config['model'].epochs}")
    print(f"Workers: {config['model'].num_workers}")
    print(f"Mixed precision: {config['training'].use_amp}")
    print()
    
    print("Available presets:")
    print("- CONFIG: Standard CPU optimization")
    print("- CONFIG_FAST: Fast testing (5 epochs)")
    print("- CONFIG_BALANCED: Balanced speed/accuracy")