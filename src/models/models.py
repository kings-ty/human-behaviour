"""
Action Recognition Models for HRI30
Implements SlowOnly, TSN, and other models mentioned in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any
import math
from configs.config import ModelConfig


class SlowOnlyModel(nn.Module):
    """
    SlowOnly model implementation based on the paper
    Best performing model from Table III: 86.55% Top-1 accuracy
    """
    
    def __init__(self, config: ModelConfig):
        super(SlowOnlyModel, self).__init__()
        self.config = config
        self.num_classes = 30  # HRI30 has 30 classes
        
        # Load lightweight backbone based on config
        if config.backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif config.backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)  # Default to lightest
        
        # Modify first conv layer for video input (3D convolution)
        self.backbone.conv1 = nn.Conv3d(
            3, 64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False
        )
        
        # Modify backbone for 3D processing
        self.backbone = self._convert_to_3d(self.backbone)
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add lightweight classification head  
        in_features = 512 if config.backbone == "resnet18" else 2048
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),  # Much smaller hidden layer
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        # Global average pooling for temporal dimension
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _convert_to_3d(self, model):
        """Convert 2D convolutions to 3D for video processing"""
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                # Convert 2D conv to 3D conv
                new_conv = nn.Conv3d(
                    module.in_channels,
                    module.out_channels,
                    kernel_size=(1, module.kernel_size[0], module.kernel_size[1]),
                    stride=(1, module.stride[0], module.stride[1]),
                    padding=(0, module.padding[0], module.padding[1]),
                    bias=(module.bias is not None)
                )
                # Copy weights
                with torch.no_grad():
                    new_conv.weight[:, :, 0, :, :] = module.weight
                    if module.bias is not None:
                        new_conv.bias = module.bias
                setattr(model, name, new_conv)
            
            elif isinstance(module, nn.BatchNorm2d):
                # Convert 2D batch norm to 3D
                new_bn = nn.BatchNorm3d(module.num_features)
                with torch.no_grad():
                    new_bn.weight = module.weight
                    new_bn.bias = module.bias
                    new_bn.running_mean = module.running_mean
                    new_bn.running_var = module.running_var
                setattr(model, name, new_bn)
                
            elif isinstance(module, nn.MaxPool2d):
                # Convert 2D max pool to 3D
                new_pool = nn.MaxPool3d(
                    kernel_size=(1, module.kernel_size, module.kernel_size),
                    stride=(1, module.stride, module.stride),
                    padding=(0, module.padding, module.padding)
                )
                setattr(model, name, new_pool)
                
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                # Convert 2D adaptive avg pool to 3D
                new_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                setattr(model, name, new_pool)
            
            else:
                # Recursively convert children
                self._convert_to_3d(module)
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Input: (B, C, T, H, W) where T is temporal dimension
        Output: (B, num_classes)
        """
        # Extract features  
        features = self.backbone(x)  # (B, features, T', H', W')
        
        # Global average pooling
        features = self.global_avg_pool(features)  # (B, features, 1, 1, 1)
        features = features.view(features.size(0), -1)  # (B, features)
        
        # Classification
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits


class TSNModel(nn.Module):
    """
    Temporal Segment Networks (TSN) implementation
    From Table III: 74.05% Top-1 accuracy on HRI30
    """
    
    def __init__(self, config: ModelConfig, num_segments: int = 8):
        super(TSNModel, self).__init__()
        self.config = config
        self.num_classes = 30
        self.num_segments = num_segments
        
        # ResNet50 backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classifier = nn.Linear(2048, self.num_classes)
        
        # Consensus function (average)
        self.consensus = nn.AvgPool1d(kernel_size=num_segments, stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TSN
        Input: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Reshape to process each frame independently
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Extract features for each frame
        features = self.backbone(x)  # (B*T, 2048)
        
        # Reshape back
        features = features.view(B, T, -1)  # (B, T, 2048)
        
        # Classify each frame
        frame_logits = self.classifier(features)  # (B, T, num_classes)
        
        # Consensus: average across temporal dimension
        video_logits = torch.mean(frame_logits, dim=1)  # (B, num_classes)
        
        return video_logits


class IRCSNModel(nn.Module):
    """
    Interaction-Reduced Channel-Separated Network (ir-CSN)
    From Table III: 79.17% Top-1 accuracy on HRI30
    """
    
    def __init__(self, config: ModelConfig):
        super(IRCSNModel, self).__init__()
        self.config = config
        self.num_classes = 30
        
        # Simplified implementation - use 3D ResNet
        self.backbone = models.video.r3d_18(pretrained=False)
        
        # Modify for 30 classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TINModel(nn.Module):
    """
    Temporal Interlacing Network (TIN)
    From Table III: 62.10% Top-1 accuracy on HRI30
    """
    
    def __init__(self, config: ModelConfig):
        super(TINModel, self).__init__()
        self.config = config
        self.num_classes = 30
        
        # ResNet50 backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Temporal interlacing components
        self.temporal_conv = nn.Conv1d(2048, 2048, kernel_size=3, padding=1)
        
        # Classification head
        self.classifier = nn.Linear(2048, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal interlacing
        """
        B, C, T, H, W = x.shape
        
        # Process each frame
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, 2048)
        features = features.view(B, T, -1)  # (B, T, 2048)
        
        # Temporal interlacing
        features = features.permute(0, 2, 1)  # (B, 2048, T)
        features = self.temporal_conv(features)  # (B, 2048, T)
        features = features.permute(0, 2, 1)  # (B, T, 2048)
        
        # Global average pooling
        features = torch.mean(features, dim=1)  # (B, 2048)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def create_model(config: ModelConfig) -> nn.Module:
    """
    Factory function to create models based on configuration
    """
    model_name = config.model_name.lower()
    
    if model_name == "slowonly":
        model = SlowOnlyModel(config)
    elif model_name == "tsn":
        model = TSNModel(config)
    elif model_name == "ircsn":
        model = IRCSNModel(config)
    elif model_name == "tin":
        model = TINModel(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


class ModelUtils:
    """Utility functions for model management"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def count_trainable_parameters(model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB"""
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    
    @staticmethod
    def freeze_backbone(model: nn.Module) -> None:
        """Freeze backbone parameters for fine-tuning"""
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
    
    @staticmethod
    def unfreeze_backbone(model: nn.Module) -> None:
        """Unfreeze backbone parameters"""
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = True


class LossFunction:
    """Loss functions for action recognition"""
    
    @staticmethod
    def create_loss_function(
        loss_type: str = "cross_entropy",
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1
    ) -> nn.Module:
        """Create loss function based on type"""
        
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        elif loss_type == "focal":
            return FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


if __name__ == "__main__":
    # Test model creation
    from configs.config import get_config_for_device
    
    config = get_config_for_device()
    model_config = config['model']
    
    # Test SlowOnly model (best performing)
    model = create_model(model_config)
    
    print(f"Created {model_config.model_name} model")
    print(f"Total parameters: {ModelUtils.count_parameters(model):,}")
    print(f"Trainable parameters: {ModelUtils.count_trainable_parameters(model):,}")
    print(f"Model size: {ModelUtils.get_model_size(model):.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 8, 256, 256)  # (B, C, T, H, W)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output example: {output[0, :5]}")