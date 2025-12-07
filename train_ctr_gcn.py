#!/usr/bin/env python3
"""
CTR-GCN Training Script for HRI30
Uses Graph Convolutional Networks for skeleton-based action recognition
Target: 80%+ accuracy with multi-stream fusion
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
from tqdm import tqdm
import math
from focal_loss import ClassBalancedFocalLoss, get_samples_per_class


class Graph:
    """Graph structure for COCO-17 and NTU RGB+D skeletons"""

    def __init__(self, layout='coco', strategy='spatial', num_nodes=17):
        self.layout = layout
        self.strategy = strategy
        self.num_nodes = num_nodes

        if layout == 'ntu-rgb+d':
            # NTU RGB+D 25-joint skeleton
            self.num_nodes = 25
            self.num_node = 25
            self.self_link = [(i, i) for i in range(25)]
            # NTU skeleton connections (0-indexed)
            self.neighbor_link = [
                (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
                (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
                (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)
            ]
            self.center = 20  # Spine center
        elif num_nodes == 17 or layout == 'coco':
            # Full COCO-17 skeleton
            self.num_nodes = 17
            self.num_node = 17
            self.self_link = [(i, i) for i in range(self.num_nodes)]
            self.neighbor_link = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # head
                (0, 5), (0, 6),  # neck to shoulders
                (5, 7), (7, 9),  # left arm
                (6, 8), (8, 10),  # right arm
                (5, 11), (6, 12),  # torso
                (11, 13), (13, 15),  # left leg
                (12, 14), (14, 16),  # right leg
                (5, 6), (11, 12)  # horizontal connections
            ]
            self.center = 0  # nose
        elif num_nodes == 6:
            # Hands-only: 6 joints (shoulders, elbows, wrists)
            self.num_node = 6
            self.self_link = [(i, i) for i in range(6)]
            self.neighbor_link = [
                (0, 2), (2, 4),  # left arm: shoulder -> elbow -> wrist
                (1, 3), (3, 5),  # right arm: shoulder -> elbow -> wrist
                (0, 1),  # shoulders connected
            ]
            self.center = 0  # left shoulder
        else:
            raise ValueError(f"Unsupported layout: {layout}, num_nodes: {num_nodes}")

        self.edge = self.self_link + self.neighbor_link
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        """Generate adjacency matrix (3, V, V) for CTR-GCN"""
        A = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edge:
            A[i, j] = 1
            A[j, i] = 1

        # Create 3-partition adjacency matrix for CTR-GCN
        A_3part = np.zeros((3, self.num_nodes, self.num_nodes))
        A_3part[0] = np.eye(self.num_nodes)  # Self connections
        A_3part[1] = A - np.eye(self.num_nodes)  # Neighbor connections
        A_3part[2] = A_3part[1]  # Outward (symmetric)

        return A_3part


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution Layer"""

    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super().__init__()

        self.num_nodes = A.shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Adjacency matrix
        self.register_buffer('A', torch.FloatTensor(A))

        # Learnable adjacency (for CTR-GCN adaptation)
        if adaptive:
            self.adaptive_A = nn.Parameter(torch.zeros_like(self.A))
        else:
            self.adaptive_A = None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # Get effective adjacency
        A = self.A
        if self.adaptive_A is not None:
            A = A + self.adaptive_A

        # Normalize adjacency
        A = A + torch.eye(V, device=A.device)
        D = A.sum(dim=1, keepdim=True)
        A = A / D

        # Graph convolution
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = torch.matmul(A, x)  # (N, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

        # 1x1 conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class TemporalConv(nn.Module):
    """Temporal Convolution Layer"""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block"""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()

        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class CTRGCN(nn.Module):
    """
    CTR-GCN: Channel-wise Topology Refined Graph Convolutional Network
    Simplified version for quick training
    """

    def __init__(self, num_classes=30, in_channels=2, num_nodes=17, graph_args={'layout': 'coco'}):
        super().__init__()

        # Graph structure (with dynamic num_nodes)
        graph_args_with_nodes = {**graph_args, 'num_nodes': num_nodes}
        self.graph = Graph(**graph_args_with_nodes)
        A = self.graph.A

        # Data normalization (FIXED: use actual num_nodes)
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # ST-GCN blocks
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2),
            STGCNBlock(256, 256, A),
            STGCNBlock(256, 256, A),
        ])

        # Global pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (N, C, T, V) = (N, 2, 60, 17)

        N, C, T, V = x.size()

        # Data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # ST-GCN blocks
        for layer in self.layers:
            x = layer(x)

        # Global pooling
        x = self.gap(x)  # (N, C, 1, 1)
        x = x.view(N, -1)

        # Classifier
        x = self.fc(x)

        return x


class SkeletonDataset(Dataset):
    """Dataset for skeleton sequences with Skeleton Mixup"""

    def __init__(self, data, labels, augment=False, mixup_alpha=0.2):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.data)

    def skeleton_mixup(self, x1, x2, y1, y2):
        """
        Skeleton Mixup: Blend two skeleton sequences

        Critical for reducing 26% overfitting gap!
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_x = lam * x1 + (1 - lam) * x2

        # Return mixed input and labels for mixup loss
        return mixed_x, y1, y2, lam

    def augment_sample(self, x):
        """Apply data augmentation"""
        # x: (C, T, V) = (2, 60, V)

        # Random rotation
        if torch.rand(1) < 0.5:
            theta = (torch.rand(1) - 0.5) * 2 * (20 * math.pi / 180)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)

            # Rotate xy coordinates
            x_rot = x[0] * cos_t - x[1] * sin_t
            y_rot = x[0] * sin_t + x[1] * cos_t
            x = torch.stack([x_rot, y_rot], dim=0)

        # Random scaling
        if torch.rand(1) < 0.5:
            scale = 0.8 + torch.rand(1) * 0.4
            x = x * scale

        # Random temporal shift
        if torch.rand(1) < 0.3:
            shift = int((torch.rand(1) - 0.5) * 10)
            x = torch.roll(x, shift, dims=1)

        # Gaussian noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.02
            x = x + noise

        # Joint dropout (simulate occlusion)
        if torch.rand(1) < 0.2:
            num_drop = int(torch.rand(1) * 3) + 1  # Drop 1-3 joints
            V = x.size(2)
            drop_indices = torch.randperm(V)[:num_drop]
            x[:, :, drop_indices] = 0

        return x

    def __getitem__(self, idx):
        x = self.data[idx]  # (T, V, C)
        y = self.labels[idx]

        # Convert to (C, T, V) for GCN
        x = x.permute(2, 0, 1)  # (2, 60, V)

        # Default values (no mixup)
        y1 = y
        y2 = y
        lam = 1.0

        if self.augment:
            # Standard augmentation
            x = self.augment_sample(x)

            # Skeleton Mixup (50% chance)
            if torch.rand(1) < 0.5:
                idx2 = torch.randint(0, len(self.data), (1,)).item()
                x2 = self.data[idx2].permute(2, 0, 1)
                x2 = self.augment_sample(x2)
                y2 = self.labels[idx2]

                x, y1, y2, lam = self.skeleton_mixup(x, x2, y, y2)

        # Always return consistent structure
        return x, y1, y2, torch.tensor(lam, dtype=torch.float32)


def mixup_criterion(criterion, pred, y_tuple):
    """
    Mixup loss function

    Args:
        criterion: Loss function
        pred: Model predictions
        y_tuple: (y1, y2, lam) from mixup
    """
    y1, y2, lam = y_tuple
    return lam * criterion(pred, y1) + (1 - lam) * criterion(pred, y2)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with Mixup support"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for data, y1, y2, lam in pbar:
        data = data.to(device)
        y1, y2 = y1.to(device), y2.to(device)
        lam = lam.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                output = model(data)
                
                # Mixup loss (element-wise first, then mean)
                loss1 = criterion(output, y1)
                loss2 = criterion(output, y2)
                loss = (lam * loss1 + (1 - lam) * loss2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)

            loss1 = criterion(output, y1)
            loss2 = criterion(output, y2)
            loss = (lam * loss1 + (1 - lam) * loss2).mean()

            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Accuracy (use y1, which is the dominant label if lam > 0.5)
        # Note: For mixup, accuracy is approximate. 
        pred = output.argmax(dim=1)
        correct += pred.eq(y1).sum().item()
        total += data.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, y1, y2, lam in loader: # Unpack 4 values
            data, target = data.to(device), y1.to(device) # Use y1 as the actual target
            # For evaluation, we only need the true target, which is y1 (or y2) when lam=1.0
            
            output = model(data)
            loss = criterion(output, target) # criterion with reduction='none' still returns per-sample loss
            loss = loss.mean() # manually take mean for evaluation loss reporting

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='CTR-GCN Training for HRI30')
    parser.add_argument('--stream', type=str, default='joint',
                       choices=['joint', 'bone', 'motion', 'velocity', 'acceleration', 'hands'],
                       help='Which stream to train on')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    args = parser.parse_args()

    print("=" * 80)
    print(f"🚀 CTR-GCN TRAINING - {args.stream.upper()} STREAM")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    data_dir = Path('/home/ty/human-bahviour/pose_features_large/multistream')
    print(f"\n📂 Loading {args.stream} stream...")

    data = np.load(data_dir / f'{args.stream}_stream.npy')
    labels = np.load(data_dir / 'labels.npy')

    print(f"   Data shape: {data.shape}")
    print(f"   Classes: {len(np.unique(labels))}")

    # Determine number of joints (17 for standard, 6 for hands-only)
    num_joints = data.shape[2]  # (N, T, V, C)
    print(f"   Joints: {num_joints}")

    # Cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'='*80}")

        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Datasets
        train_dataset = SkeletonDataset(X_train, y_train, augment=True)
        val_dataset = SkeletonDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

        # Model (FIXED: dynamic num_joints)
        model = CTRGCN(num_classes=30, in_channels=2, num_nodes=num_joints).to(device)

        if args.stream == 'hands':
            print(f"   ✅ Using 6-joint graph for hands stream")
        else:
            print(f"   ✅ Using 17-joint graph for standard stream")

        # Loss: Class-Balanced Focal Loss (CRITICAL FIX)
        samples_per_class = get_samples_per_class(y_train)
        criterion = ClassBalancedFocalLoss(samples_per_class, beta=0.9999, gamma=2.0, reduction='none').to(device)
        print(f"   ✅ Using Class-Balanced Focal Loss (beta=0.9999, gamma=2.0)")
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        # Training
        best_val_acc = 0
        patience = 0
        max_patience = 15

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                               optimizer, device, scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                # Save best model
                torch.save(model.state_dict(), f'best_model_{args.stream}_fold{fold}.pth')
                print(f"✅ New best: {best_val_acc:.2f}%")
            else:
                patience += 1

            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        fold_scores.append(best_val_acc)
        print(f"\n🎯 Fold {fold+1} Best Accuracy: {best_val_acc:.2f}%")

    # Final results
    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)

    print(f"\n{'='*80}")
    print(f"🏆 FINAL RESULTS - {args.stream.upper()} STREAM")
    print(f"{'='*80}")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Fold Scores: {[f'{s:.2f}%' for s in fold_scores]}")

    # Save results
    results = {
        'stream': args.stream,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'fold_scores': [float(s) for s in fold_scores]
    }

    with open(f'results_{args.stream}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to results_{args.stream}.json")
    print("\n🎯 Next Steps:")
    print("   Train other streams and combine for 80%+ accuracy:")
    print("   python train_ctr_gcn.py --stream bone")
    print("   python train_ctr_gcn.py --stream motion")


if __name__ == '__main__':
    main()
