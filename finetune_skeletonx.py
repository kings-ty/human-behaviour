#!/usr/bin/env python3
"""
Topology-Aware Transfer Learning for Skeleton Action Recognition
Target: 85%+ Accuracy (Fixed Scaling & Bone Calculation)
"""

import argparse
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import json

from model_skeletonx_v3 import SkeletonX_Model
from focal_loss import ClassBalancedFocalLoss, get_samples_per_class


# ============================================================================
# COCO-17 to NTU-25 Topology Mapping
# ============================================================================
COCO_TO_NTU_MAPPING = {
    0: 3,   # nose -> neck (spine shoulder)
    5: 4,   # left_shoulder -> left_shoulder
    6: 8,   # right_shoulder -> right_shoulder
    7: 5,   # left_elbow -> left_elbow
    8: 9,   # right_elbow -> right_elbow
    9: 6,   # left_wrist -> left_wrist
    10: 10, # right_wrist -> right_wrist
    11: 12, # left_hip -> left_hip
    12: 16, # right_hip -> right_hip
    13: 13, # left_knee -> left_knee
    14: 17, # right_knee -> right_knee
    15: 14, # left_ankle -> left_ankle
    16: 18, # right_ankle -> right_ankle
}

class NTUCompatibleDataset(Dataset):
    def __init__(self, sequences, labels, stream='joint', augment=False, mixup_alpha=0.0):
        self.sequences = sequences
        self.labels = labels
        self.stream = stream  # 'joint', 'bone', 'motion'
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        
        # NTU bone connection pairs (child, parent)
        self.ntu_pairs = (
            (1, 0), (20, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 10),
            (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18)
        )

    def __len__(self):
        return len(self.sequences)

    def _handle_missing_joints(self, seq):
        seq = seq.copy()
        for t in range(seq.shape[0]):
            # If left_hip is missing, replace it with left_knee if available
            if seq[t, 4, 0] == 0 and seq[t, 4, 1] == 0: 
                if not (seq[t, 2, 0] == 0 and seq[t, 2, 1] == 0): seq[t, 4] = seq[t, 2]
            # If right_hip is missing, replace it with right_knee if available
            if seq[t, 3, 0] == 0 and seq[t, 3, 1] == 0: 
                if not (seq[t, 1, 0] == 0 and seq[t, 1, 1] == 0): seq[t, 3] = seq[t, 1]
        return seq

    def _temporal_interpolation(self, seq):
        # Convert to tensor with channel-first and interpolate to (64, 17)
        seq_tensor = torch.from_numpy(seq).float().permute(2, 0, 1).unsqueeze(0)
        seq_interp = F.interpolate(seq_tensor, size=(64, 17), mode='bilinear', align_corners=False)
        return seq_interp.squeeze(0).permute(1, 2, 0).numpy()

    def _channel_expansion(self, seq):
        # Add a zero z-channel to make (x,y,z)
        z_channel = np.zeros((seq.shape[0], seq.shape[1], 1))
        return np.concatenate([seq, z_channel], axis=-1)

    def _topology_mapping(self, seq):
        # Map COCO joints into NTU-25 joint layout and synthesize missing centers
        ntu_seq = np.zeros((seq.shape[0], 25, 3), dtype=np.float32)
        for coco_idx, ntu_idx in COCO_TO_NTU_MAPPING.items():
            ntu_seq[:, ntu_idx, :] = seq[:, coco_idx, :]

        # Pelvis / hip center: average of left and right hip
        left_hip = ntu_seq[:, 12, :]
        right_hip = ntu_seq[:, 16, :]
        ntu_seq[:, 0, :] = (left_hip + right_hip) / 2.0 

        # Shoulder center and spine points
        left_shoulder = ntu_seq[:, 4, :]
        right_shoulder = ntu_seq[:, 8, :]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = ntu_seq[:, 0, :]
        ntu_seq[:, 20, :] = (shoulder_center + hip_center) / 2.0 
        ntu_seq[:, 1, :] = (ntu_seq[:, 0, :] + ntu_seq[:, 20, :]) / 2.0 
        ntu_seq[:, 2, :] = shoulder_center 

        return ntu_seq

    def _apply_augmentation(self, seq):
        """
        🚀 SOTA-Level Augmentation Strategy
        1. Geometric: Rotation + Scaling + Shear (Viewpoint Invariance)
        2. Temporal: Crop & Resize (Speed Invariance)
        3. Structural: Joint Masking (Occlusion Robustness)
        4. Noise: Gaussian Jitter (Sensor Noise)
        """
        C, T, V = seq.shape[2], seq.shape[0], seq.shape[1] # (64, 25, 3)

        # 1. Geometric: Rotation & Shear
        if np.random.rand() < 0.5: 
            # Rotation
            angle = np.random.uniform(-15, 15)
            rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Shear (Simulating camera perspective distortion)
            shear_x = np.random.uniform(-0.1, 0.1)
            shear_y = np.random.uniform(-0.1, 0.1)
            shear_matrix = np.array([[1, shear_x], [shear_y, 1]])
            
            # Apply combined transform to X,Y
            seq[:, :, :2] = seq[:, :, :2] @ rot_matrix.T @ shear_matrix.T

        # 2. Geometric: Scaling
        if np.random.rand() < 0.5: 
            scale = np.random.uniform(0.85, 1.15)
            seq[:, :, :2] *= scale

        # 3. Structural: Joint Masking (DISABLED due to high missing rate in raw data)
        # if np.random.rand() < 0.3:
        #     num_mask = np.random.randint(1, 4)
        #     mask_indices = np.random.choice(V, num_mask, replace=False)
        #     seq[:, mask_indices, :] = 0.0

        # 4. Temporal: Crop & Resize (Speed Variation)
        # Randomly crop a segment and resize back to original length (linear interp)
        if np.random.rand() < 0.3:
            crop_ratio = np.random.uniform(0.8, 1.0)
            crop_len = int(T * crop_ratio)
            start = np.random.randint(0, T - crop_len)
            
            # (T, V, C) -> (C, T, V) for pytorch interpolation
            tensor_seq = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0) 
            cropped = tensor_seq[:, :, start:start+crop_len, :]
            resized = F.interpolate(cropped, size=(T, V), mode='bilinear', align_corners=False)
            seq = resized.squeeze(0).permute(1, 2, 0).numpy()

        # 5. Noise: Gaussian Jitter
        if np.random.rand() < 0.2:
            sigma = 0.01
            seq += np.random.normal(0, sigma, seq.shape)

        return seq

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        seq = self._handle_missing_joints(seq)
        seq = self._temporal_interpolation(seq)
        seq = self._channel_expansion(seq)
        seq = self._topology_mapping(seq) 

        if self.augment:
            seq = self._apply_augmentation(seq)

        seq_tensor = torch.from_numpy(seq).float().permute(2, 0, 1) # (3, 64, 25)

        # Stream branching: support 'joint', 'bone', and 'motion' streams
        if self.stream == 'bone':
            bone_data = torch.zeros_like(seq_tensor)
            for v1, v2 in self.ntu_pairs:
                bone_data[:, :, v1] = seq_tensor[:, :, v1] - seq_tensor[:, :, v2]
            final_data = bone_data  # REMOVED / 4.0 (Data is already small 0~1)
            
        elif self.stream == 'motion':
            motion_data = torch.zeros_like(seq_tensor)
            motion_data[:, :-1, :] = seq_tensor[:, 1:, :] - seq_tensor[:, :-1, :]
            final_data = motion_data * 10.0 # Keep amplification for small motion
        else:
            # Joint stream (normalized)
            final_data = seq_tensor # REMOVED / 3.0 (Data is already 0~1)

        if self.mixup_alpha > 0 and self.augment:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx2 = np.random.randint(len(self.sequences))
            label2 = self.labels[idx2]
        else:
            lam = 1.0
            label2 = label

        return final_data, label, label2, lam

# ============================================================================
# ... Weight Loading / Train / Evaluate functions (unchanged) ...
# ============================================================================
# (strict_load_weights, train_epoch, evaluate functions use the original implementation)
def strict_load_weights(model, weight_path, device='cpu'):
    # ... (original code unchanged) ...
    print(f"\n🔄 Loading pretrained weights from: {weight_path}")
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model_state = model.state_dict()
        new_state = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if 'fc' in name: continue
            if name in model_state:
                if model_state[name].shape == v.shape: new_state[name] = v
        model.load_state_dict(new_state, strict=False)
        print(f"✅ Weights Loaded!")
        return model
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for data, y1, y2, lam in pbar:
        data, y1, y2, lam = data.to(device), y1.to(device), y2.to(device), lam.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = (lam * criterion(output, y1) + (1 - lam) * criterion(output, y2)).mean()
        loss.backward()
        # 🛑 Gradient Clipping (SpaceX Stability Check)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(y1).sum().item()
        total += data.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, y1, y2, lam in tqdm(loader, desc="Validating", leave=False):
            data, target = data.to(device), y1.to(device)
            output = model(data)
            loss = criterion(output, target).mean()
            total_loss += loss.item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

# ============================================================================
# Main Loop (updated)
# ============================================================================
def set_seed(seed=42):
    """🔒 Lock the Random Seed for Exact Reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed Fixed: {seed}")

def main():
    set_seed(42) # Lock it down.
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, default='joint') # joint or bone
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pretrained_path', type=str, default='checkpoints_pretrained/ctrgcn_ntu120_pretrained.pt')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--data_dir', type=str, default='/home/ty/human-bahviour/pose_features_large')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 TOPOLOGY-AWARE TRAINING | Stream: {args.stream} | Device: {device}")

    data_dir = Path(args.data_dir) # Define data_dir from args
    
    # SAFETY CHECK: Prevent Data Leakage from pre-augmented files
    # We must split the ORIGINAL data (2099) first, then augment only the train fold.
    seq_path = data_dir / 'train_sequences_good.npy'
    lbl_path = data_dir / 'train_labels.npy'

    if not seq_path.exists():
         # Fallback for alternative directory structure if needed
         seq_path = data_dir / 'multistream/joint_stream.npy'
         lbl_path = data_dir / 'multistream/labels.npy'

    if (data_dir / 'train_sequences_aug.npy').exists():
        print(f"⚠️  WARNING: Found 'train_sequences_aug.npy'. IGNORING it to prevent Data Leakage.")
        print(f"    We will use On-the-Fly augmentation on the Training Fold only.")

    print(f"📂 Loading Original Data from: {seq_path}")
    data = np.load(seq_path) 
    labels = np.load(lbl_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        if args.fold != -1 and fold != args.fold: continue

        print(f"\n🔹 FOLD {fold + 1}/5")
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        print(f"   📊 Data Split: Train={len(X_train)} (Augmented On-The-Fly), Val={len(X_val)}")
        if len(X_val) != 420 and len(X_val) != 419:
             print(f"   ⚠️  Note: Validation size is {len(X_val)}, expected ~420.")

        # Pass stream info when creating Dataset (scaling / bone calculation decided here)
        train_dataset = NTUCompatibleDataset(X_train, y_train, stream=args.stream, augment=True, mixup_alpha=0.2)
        val_dataset = NTUCompatibleDataset(X_val, y_val, stream=args.stream, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Model
        graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        model = SkeletonX_Model(num_class=30, num_point=25, in_channels=3, graph_args=graph_args).to(device)
        model = strict_load_weights(model, args.pretrained_path, device=device)

        # optimizer configuration (higher weight decay used for motion stream experiments)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.002, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        criterion = ClassBalancedFocalLoss(get_samples_per_class(y_train), beta=0.9999, gamma=2.0, reduction='none').to(device)

        best_acc = 0
        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if v_acc > best_acc:
                best_acc = v_acc
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc': best_acc}, 
                           f'best_topologyaware_{args.stream}_fold{fold}.pth')

            if (epoch+1) % 5 == 0:
                print(f"   Ep {epoch+1}: T={t_acc:.1f}% / V={v_acc:.1f}% (Best: {best_acc:.1f}%)")

        print(f"✅ Fold {fold+1} Finished. Best: {best_acc:.2f}%")

if __name__ == '__main__':
    main()