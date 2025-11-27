"""
HRI30 Action Recognition Configuration
Based on the paper: "HRI30: An Action Recognition Dataset for Industrial Human-Robot Interaction"
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class DataConfig:
    """Data configuration based on HRI30 specifications"""
    # Dataset paths
    data_root: str = "/home/ty/human-bahviour"
    train_dir: str = "train_set"
    annotations_dir: str = "annotations"
    
    # Video specifications (from paper)
    frame_rate: int = 30
    original_resolution: Tuple[int, int] = (720, 480)
    input_resolution: Tuple[int, int] = (256, 256)  # Resized for training
    
    # Video processing
    sample_rate: int = 4  # Sample every 4th frame
    clip_duration: float = 2.0  # Most clips are 1-2 seconds
    
    # Data splits (from Table II in paper)
    train_split_ratios: List[float] = field(default_factory=lambda: [0.714, 0.786, 0.643])  # [2100/2940, 2310/2940, 1890/2940]
    test_split_ratios: List[float] = field(default_factory=lambda: [0.286, 0.214, 0.357])   # [840/2940, 630/2940, 1050/2940]
    
    # 30 Action classes from the paper
    action_classes: List[str] = field(default_factory=lambda: [
        "Deliver_Object",
        "Move_Backwards_While_Drilling", 
        "Move_Backwards_While_Polishing",
        "Move_Diagonally_Backward_Left_with_Drill",
        "Move_Diagonally_Backward_Left_with_Polisher",
        "Move_Diagonally_Backward_Right_with_Drill",
        "Move_Diagonally_Backward_Right_with_Polisher",
        "Move_Diagonally_Forward_Left_with_Drill",
        "Move_Diagonally_Forward_Left_with_Polisher",
        "Move_Diagonally_Forward_Right_with_Drill",
        "Move_Diagonally_Forward_Right_with_Polisher",
        "Move_Forward_While_Drilling",
        "Move_Forward_While_Polishing",
        "Move_Left_While_Drilling",
        "Move_Left_While_Polishing", 
        "Move_Right_While_Drilling",
        "Move_Right_While_Polishing",
        "No_Collaborative_with_Drill",
        "No_Collaborative_with_Polisher",
        "Pick_Up_Drill",
        "Pick_Up_Polisher",
        "Pick_Up_The_Object",
        "Put_Down_Drill",
        "Put_Down_Polisher",
        "Using_The_Drill",
        "Using_The_Polisher",
        "Walking",
        "Walking_with_Drill",
        "Walking_with_Object",
        "Walking_with_Polisher"
    ])
    
    # Action categories (from paper)
    action_categories: Dict[str, List[str]] = None
    
    def __post_init__(self):
        self.num_classes = len(self.action_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.action_classes)}
        
        # Define action categories
        self.action_categories = {
            "Human_Object_Interaction": [
                "Pick_Up_Drill", "Pick_Up_Polisher", "Pick_Up_The_Object",
                "Put_Down_Drill", "Put_Down_Polisher", "Deliver_Object"
            ],
            "Body_Motion_Only": [
                "Walking", "Walking_with_Drill", "Walking_with_Object", "Walking_with_Polisher",
                "Move_Backwards_While_Drilling", "Move_Backwards_While_Polishing",
                "Move_Forward_While_Drilling", "Move_Forward_While_Polishing",
                "Move_Left_While_Drilling", "Move_Left_While_Polishing",
                "Move_Right_While_Drilling", "Move_Right_While_Polishing",
                "Move_Diagonally_Backward_Left_with_Drill", "Move_Diagonally_Backward_Left_with_Polisher",
                "Move_Diagonally_Backward_Right_with_Drill", "Move_Diagonally_Backward_Right_with_Polisher",
                "Move_Diagonally_Forward_Left_with_Drill", "Move_Diagonally_Forward_Left_with_Polisher",
                "Move_Diagonally_Forward_Right_with_Drill", "Move_Diagonally_Forward_Right_with_Polisher"
            ],
            "Human_Robot_Collaboration": [
                "Using_The_Drill", "Using_The_Polisher",
                "No_Collaborative_with_Drill", "No_Collaborative_with_Polisher"
            ]
        }


@dataclass 
class ModelConfig:
    """Model configuration based on paper specifications"""
    # Architecture
    backbone: str = "resnet50"  # From Table III
    pretrained: str = "kinetics400"  # Best performing in paper
    
    # Training (from Section IV-A)
    batch_size: int = 8  # Reduced for MX450/Xavier limitations
    learning_rate: float = 0.001  # From paper
    optimizer: str = "sgd"  # From paper
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Training schedule
    epochs: int = 100
    warmup_epochs: int = 5
    lr_scheduler: str = "cosine"
    
    # Model specific (SlowOnly - best performing)
    model_name: str = "slowonly"
    alpha: int = 4  # Temporal downsampling
    beta: int = 1   # Channel ratio
    num_segments: int = 8  # Number of segments to sample
    
    # Hardware specific - CPU-first approach
    device: str = "cpu"  # Default to CPU for compatibility
    num_workers: int = 4
    pin_memory: bool = False  # Disable for CPU compatibility


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Experiment tracking
    experiment_name: str = "hri30_action_recognition"
    save_dir: str = "/home/ty/human-bahviour/experiments"
    log_interval: int = 10
    eval_interval: int = 5
    save_interval: int = 10
    
    # Augmentation (from paper - random flip mentioned)
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Evaluation metrics (from paper)
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "top5_accuracy", "f1_score", "confusion_matrix"])
    
    # Mixed precision training (for memory efficiency)
    use_amp: bool = True
    
    # Resume training
    resume_checkpoint: str = None


@dataclass
class HardwareConfig:
    """Hardware-specific configurations"""
    # For MX450
    mx450_config: Dict[str, Any] = None
    
    # For Jetson Xavier AGX  
    xavier_config: Dict[str, Any] = None
    
    def __post_init__(self):
        self.mx450_config = {
            "batch_size": 4,
            "input_size": (224, 224),
            "num_workers": 2,
            "memory_limit": "1.5GB",
            "fp16": True
        }
        
        self.xavier_config = {
            "batch_size": 16,
            "input_size": (256, 256), 
            "num_workers": 6,
            "memory_limit": "8GB",
            "fp16": True
        }


# Global configuration instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
HARDWARE_CONFIG = HardwareConfig()


def get_config_for_device(device_type: str = "auto") -> Dict[str, Any]:
    """Get configuration optimized for specific hardware"""
    if device_type == "auto":
        # Try to detect device safely (CPU-first approach)
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                try:
                    import subprocess
                    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", 
                                                     shell=True, text=True, stderr=subprocess.DEVNULL)
                    if "Xavier" in gpu_info:
                        device_type = "xavier"
                    elif "MX450" in gpu_info:
                        device_type = "mx450"
                    else:
                        device_type = "cpu_only"  # Default to CPU for compatibility
                except:
                    device_type = "cpu_only"
            else:
                device_type = "cpu_only"
        except ImportError:
            device_type = "cpu_only"
    
    base_config = {
        "data": DATA_CONFIG,
        "model": MODEL_CONFIG, 
        "training": TRAINING_CONFIG,
        "hardware": HARDWARE_CONFIG
    }
    
    if device_type == "xavier":
        # Optimize for Jetson Xavier AGX
        base_config["model"].batch_size = HARDWARE_CONFIG.xavier_config["batch_size"]
        base_config["model"].num_workers = HARDWARE_CONFIG.xavier_config["num_workers"]
        base_config["data"].input_resolution = HARDWARE_CONFIG.xavier_config["input_size"]
        base_config["model"].device = "cuda"
        base_config["model"].pin_memory = True
    elif device_type == "mx450":
        # Optimize for MX450 or limited GPU hardware
        base_config["model"].batch_size = HARDWARE_CONFIG.mx450_config["batch_size"]
        base_config["model"].num_workers = HARDWARE_CONFIG.mx450_config["num_workers"] 
        base_config["data"].input_resolution = HARDWARE_CONFIG.mx450_config["input_size"]
        base_config["model"].device = "cuda"
        base_config["model"].pin_memory = True
    else:
        # CPU-only configuration for maximum compatibility
        base_config["model"].batch_size = 2  # Very small batch for CPU
        base_config["model"].num_workers = 2  # Reduced workers for CPU
        base_config["data"].input_resolution = (224, 224)  # Smaller input
        base_config["model"].device = "cpu"
        base_config["model"].pin_memory = False
        base_config["training"].use_amp = False  # Disable mixed precision on CPU
    
    return base_config


if __name__ == "__main__":
    # Print configuration summary
    config = get_config_for_device()
    print("=== HRI30 Action Recognition Configuration ===")
    print(f"Number of classes: {config['data'].num_classes}")
    print(f"Action categories: {list(config['data'].action_categories.keys())}")
    print(f"Model: {config['model'].model_name} with {config['model'].backbone}")
    print(f"Batch size: {config['model'].batch_size}")
    print(f"Input resolution: {config['data'].input_resolution}")
    print(f"Device: {config['model'].device}")