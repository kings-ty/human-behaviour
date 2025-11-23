"""
Configuration files for different hardware setups
"""

from .config import get_config_for_device, ModelConfig, TrainingConfig, DataConfig, HardwareConfig

__all__ = ['get_config_for_device', 'ModelConfig', 'TrainingConfig', 'DataConfig', 'HardwareConfig']