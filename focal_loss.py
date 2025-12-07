#!/usr/bin/env python3
"""
Class-Balanced Focal Loss for Imbalanced Skeleton Action Recognition
Addresses the 40-120 sample class imbalance in HRI30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples by down-weighting easy examples.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (Tensor of shape [num_classes])
            gamma: Focusing parameter (default 2.0)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct class

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss

    Combines:
    1. Focal loss (handles hard examples)
    2. Class-balanced weighting (handles imbalance)

    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    """

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, reduction='mean'):
        """
        Args:
            samples_per_class: List/array of sample counts per class
            beta: Hyperparameter for effective number (0.9999 for CIFAR, 0.999 for ImageNet)
            gamma: Focal loss focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        samples_per_class = np.array(samples_per_class, dtype=np.float32)

        # Calculate effective number
        effective_num = 1.0 - np.power(beta, samples_per_class)

        # Class weights
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize

        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Focal weight
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma

        # Class weight
        class_weight = self.weights[targets]

        # Combined loss
        loss = focal_weight * class_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_samples_per_class(labels):
    """
    Count samples per class from labels array

    Args:
        labels: numpy array of class labels

    Returns:
        samples_per_class: list of counts
    """
    unique, counts = np.unique(labels, return_counts=True)

    # Ensure all classes present (fill missing with 1)
    num_classes = int(labels.max()) + 1
    samples_per_class = np.ones(num_classes, dtype=np.int32)

    for cls, count in zip(unique, counts):
        samples_per_class[cls] = count

    return samples_per_class.tolist()


# Example usage
if __name__ == '__main__':
    # Simulate imbalanced dataset
    samples_per_class = [100, 50, 120, 40, 80]  # Imbalanced

    # Create loss
    criterion = ClassBalancedFocalLoss(samples_per_class, beta=0.9999, gamma=2.0)

    # Test
    logits = torch.randn(32, 5)  # batch_size=32, num_classes=5
    targets = torch.randint(0, 5, (32,))

    loss = criterion(logits, targets)
    print(f"Class-Balanced Focal Loss: {loss.item():.4f}")

    # Compare with standard CE
    ce_loss = F.cross_entropy(logits, targets)
    print(f"Standard CE Loss: {ce_loss.item():.4f}")
