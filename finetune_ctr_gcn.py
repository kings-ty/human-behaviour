#!/usr/bin/env python3
"""
SOTA Fine-Tuning Script with SkeletonX-Compatible Model
Target: 90%+ Accuracy
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
from tqdm import tqdm
from collections import OrderedDict
import math

# Reuse dataset and loss from existing files
from train_ctr_gcn import SkeletonDataset
from focal_loss import ClassBalancedFocalLoss, get_samples_per_class

# --- DEFINING SKELETONX COMPATIBLE MODEL (Simplified) ---
# This mimics the structure of the checkpoint: l1.gcn1.convs... 

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)
    
class BN_Layer(nn.Module):
    def __init__(self, input_channel):
        super(BN_Layer, self).__init__()
        self.data_bn = nn.BatchNorm1d(input_channel)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        return x

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        self.pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(self.pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x # ReLU done in block

class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1,2], residual=True, residual_kernel_size=1):
        super().__init__()
        self.global_pooling = False # Simplified
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(int((kernel_size - 1) / 2), 0), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0, inplace=True)
            )
        ])
        # Note: SkeletonX checkpoint has complex structure here (convs.0.conv1...). 
        # Since we can't perfectly reverse-engineer the complex MS-TCN from weights alone without code,
        # WE WILL USE A HACK: Load weights into a Dict, and manually compute forward pass using functional API
        # OR, we just map the "Main Branch" weights to a standard TCN.
        pass

# 🚨 STRATEGY SHIFT:
# Since exact architecture matching is risky without source code,
# we will stick to our CTR-GCN but use "Partial Loading" only for layers that match.
# If that fails (as it did), we use **Knowledge Distillation from Scratch**.
# BUT, let's try one last mapping trick:
# The checkpoint has 'l1.gcn1.convs.0.conv1.weight'
# This looks like a specific multi-branch convolution.

# Let's go back to "finetune_ctr_gcn.py" logic but with VERY LOOSE mapping.
# We will look for ANY weight that looks like a convolution kernel (C_out, C_in, 1, 1) 
# and shove it into our model if dimensions match.

def robust_load_weights(model, weight_path, device):
    print(f"   🔄 Brute-force Loading from {weight_path}...")
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model_state = model.state_dict()
    new_state = OrderedDict()
    
    # Collect all available 1x1 conv weights from checkpoint
    # Shape: (Out, In, 1, 1)
    avail_weights = []
    for k, v in state_dict.items():
        if v.ndim == 4 and v.shape[2] == 1 and v.shape[3] == 1:
            avail_weights.append(v)
    
    print(f"   📦 Found {len(avail_weights)} potential 1x1 conv weights in checkpoint")
    
    # Try to fill our model's GCN convs
    filled_count = 0
    used_indices = set()
    
    for k, v in model_state.items():
        if "gcn.conv.weight" in k: # Our spatial convs
            # Find a weight with matching shape
            found = False
            for i, src_w in enumerate(avail_weights):
                if i not in used_indices and src_w.shape == v.shape:
                    new_state[k] = src_w
                    used_indices.add(i)
                    filled_count += 1
                    found = True
                    break
            if not found:
                pass # Init from scratch
                
    print(f"   ✅ Brute-force matched {filled_count} GCN conv layers")
    
    model.load_state_dict(new_state, strict=False)
    return model

# --- MAIN SCRIPT ---

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, y1, y2, lam in tqdm(loader, desc='Fine-tuning', leave=False):
        data = data.to(device)
        y1, y2 = y1.to(device), y2.to(device)
        lam = lam.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = (lam * criterion(output, y1) + (1 - lam) * criterion(output, y2)).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(y1).sum().item()
        total += data.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, y1, y2, lam in loader:
            data, target = data.to(device), y1.to(device)
            output = model(data)
            loss = criterion(output, target).mean()
            total_loss += loss.item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

# Import existing CTRGCN class
from train_ctr_gcn import CTRGCN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, default='joint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=60) 
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--pretrained_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Loading
    data_dir = Path('/home/ty/human-bahviour/pose_features_large/multistream')
    data = np.load(data_dir / f'{args.stream}_stream.npy')
    labels = np.load(data_dir / 'labels.npy')
    num_joints = data.shape[2]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"🚀 STARTING BRUTE-FORCE FINE-TUNING ({args.stream.upper()})")

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n🔹 FOLD {fold + 1}/5")
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_loader = DataLoader(SkeletonDataset(X_train, y_train, augment=True, mixup_alpha=0.2), 
                                batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(SkeletonDataset(X_val, y_val, augment=False), 
                              batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = CTRGCN(num_classes=30, in_channels=2, num_nodes=num_joints).to(device)
        
        # USE BRUTE FORCE LOADER
        model = robust_load_weights(model, args.pretrained_path, device)
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        criterion = ClassBalancedFocalLoss(get_samples_per_class(y_train), beta=0.9999, gamma=2.0, reduction='none').to(device)

        best_acc = 0
        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), f'best_sota_{args.stream}_fold{fold}.pth')
            
            if (epoch+1) % 10 == 0:
                 print(f"   Ep {epoch+1}: {t_acc:.1f}% / {v_acc:.1f}% (Best: {best_acc:.1f}%)")
                 
        print(f"✅ Best: {best_acc:.2f}%")

if __name__ == '__main__':
    main()