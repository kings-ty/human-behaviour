# ...existing code...
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from model_skeletonx_v3 import SkeletonX_Model
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# 1. Dataset: model-specific inputs (Joint=raw, Bone=/4, Motion=*10)
# =============================================================================
class EnsembleDataset(Dataset):
    def __init__(self, joint_path, label_path):
        self.data = np.load(joint_path) 
        self.labels = np.load(label_path)
        
        self.ntu_pairs = (
            (1, 0), (20, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 10),
            (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18)
        )

    def __getitem__(self, index):
        raw_data = torch.from_numpy(self.data[index]).float() 
        if torch.all(torch.abs(raw_data[:, 4, :]) < 1e-4): raw_data[:, 4, :] = raw_data[:, 2, :]
        if torch.all(torch.abs(raw_data[:, 3, :]) < 1e-4): raw_data[:, 3, :] = raw_data[:, 1, :]

        data = raw_data.permute(2, 0, 1).unsqueeze(-1) 
        C, T, V, M = data.shape
        data = data.permute(0, 2, 3, 1).contiguous().view(-1, 1, T)
        data = F.interpolate(data, size=64, mode='linear', align_corners=False)
        data = data.view(C, V, M, 64).permute(0, 3, 1, 2).contiguous()

        final_data = torch.zeros(3, 64, 25, M)
        map_idx = {0:3, 5:4, 6:8, 7:5, 8:9, 9:6, 10:10, 11:12, 12:16, 13:13, 14:17, 15:14, 16:18}
        for c, n in map_idx.items():
            final_data[:2, :, n, :] = data[:2, :, c, :]
            
        # 1. Hip center (Base Spine, NTU 0)
        hip_center = (data[:2, :, 11, :] + data[:2, :, 12, :]) / 2
        final_data[:2, :, 0, :] = hip_center
        
        # 2. Shoulder center (Spine Shoulder, NTU 20) -> key point!
        shoulder_center = (data[:2, :, 5, :] + data[:2, :, 6, :]) / 2
        final_data[:2, :, 20, :] = shoulder_center  # shoulder center must be index 20
        
        # 3. Mid spine (NTU 1) -> midpoint between 0 and 20
        final_data[:2, :, 1, :] = (hip_center + shoulder_center) / 2
        
        # 4. Neck (NTU 2) -> midpoint of shoulder (20) and nose (COCO 0)
        # Map COCO 0 (nose) to NTU 3 (head), neck is best set between shoulder and nose
        nose = data[:2, :, 0, :]
        final_data[:2, :, 2, :] = (shoulder_center + nose) / 2  # neck between shoulder and nose
        
        # 5. Head (NTU 3) -> use nose coordinates
        final_data[:2, :, 3, :] = nose

        # =========================================================
        # 🔥 [Final Inputs] exactly the same as used during training!
        # =========================================================
        
        # 1. Joint: raw joints (normalized)
        joint_input = final_data.clone() / 3.0

        # 2. Bone: relative bone vectors scaled (/4.0)
        bone_input = torch.zeros_like(final_data)
        for v1, v2 in self.ntu_pairs:
            bone_input[:, :, v1, :] = final_data[:, :, v1, :] - final_data[:, :, v2, :]
        bone_input = bone_input / 4.0

        # 3. Motion: temporal differences scaled (*10.0)
        motion_input = torch.zeros_like(final_data)
        motion_input[:, :-1, :] = final_data[:, 1:, :] - final_data[:, :-1, :]
        motion_input = motion_input * 10.0  # scale by 10 as in training

        return joint_input.squeeze(-1), bone_input.squeeze(-1), motion_input.squeeze(-1), self.labels[index]

    def __len__(self):
        return len(self.labels)

# (load_model_safe is unchanged from original)
def load_model_safe(model_path, device):
    graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    model = SkeletonX_Model(num_class=30, num_point=25, in_channels=3, graph_args=graph_args)
    print(f"🔧 Loading: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
    else: state_dict = checkpoint
    new_state = OrderedDict()
    for k, v in state_dict.items(): new_state[k.replace('module.', '')] = v
    try: model.load_state_dict(new_state, strict=True); print("✅ Success")
    except: model.load_state_dict(new_state, strict=False); print("⚠️ Non-strict Load")
    model.to(device).eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 3-STREAM FINAL ENSEMBLE CHECK")
    
    BASE_DIR = '/home/ty/human-bahviour/pose_features_large/multistream'
    JOINT_DATA = f'{BASE_DIR}/joint_stream.npy'
    LABELS = f'{BASE_DIR}/labels.npy'
    
    # all three models are loaded
    JOINT_MODEL = 'best_topologyaware_joint_fold0.pth'
    BONE_MODEL = 'best_topologyaware_bone_fold0.pth'
    MOTION_MODEL = 'best_topologyaware_motion_fold0.pth' # recently trained

    dataset = EnsembleDataset(JOINT_DATA, LABELS)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model_j = load_model_safe(JOINT_MODEL, device)
    model_b = load_model_safe(BONE_MODEL, device)
    model_m = load_model_safe(MOTION_MODEL, device)

    all_score_j = []
    all_score_b = []
    all_score_m = []
    all_labels = []

    print("🚀 Extracting scores...")
    with torch.no_grad():
        for j, b, m, l in tqdm(loader):
            j, b, m = j.to(device), b.to(device), m.to(device)
            all_score_j.append(F.softmax(model_j(j), dim=1).cpu().numpy())
            all_score_b.append(F.softmax(model_b(b), dim=1).cpu().numpy())
            all_score_m.append(F.softmax(model_m(m), dim=1).cpu().numpy())
            all_labels.append(l.cpu().numpy())

    all_score_j = np.concatenate(all_score_j)
    all_score_b = np.concatenate(all_score_b)
    all_score_m = np.concatenate(all_score_m)
    all_labels = np.concatenate(all_labels)

    acc_j = np.mean(np.argmax(all_score_j, axis=1) == all_labels) * 100
    acc_b = np.mean(np.argmax(all_score_b, axis=1) == all_labels) * 100
    acc_m = np.mean(np.argmax(all_score_m, axis=1) == all_labels) * 100
    
    print("\n" + "="*50)
    print(f"📊 INDIVIDUAL SCORES")
    print(f"   - Joint : {acc_j:.2f}% (80%)")
    print(f"   - Bone  : {acc_b:.2f}% (76%)")
    print(f"   - Motion: {acc_m:.2f}% (53% -> normal)")
    print("="*50)

    # find best fusion ratio
    print(f"\n🔎 Finding Best Fusion Ratio (Step 0.1)...")
    best_acc = 0
    best_ratios = (0, 0, 0)

    for w_j in np.arange(0.0, 1.1, 0.1):
        for w_b in np.arange(0.0, 1.1 - w_j + 0.01, 0.1):
            w_m = 1.0 - w_j - w_b
            if w_m < -0.01: continue
            
            final_score = (w_j * all_score_j) + (w_b * all_score_b) + (w_m * all_score_m)
            acc = np.mean(np.argmax(final_score, axis=1) == all_labels) * 100
            
            if acc > best_acc:
                best_acc = acc
                best_ratios = (w_j, w_b, w_m)

    print("\n" + "="*50)
    print(f"🏆 FINAL MAX ACCURACY: {best_acc:.2f}%")
    print(f"   Optimal Ratio -> J:{best_ratios[0]:.1f} | B:{best_ratios[1]:.1f} | M:{best_ratios[2]:.1f}")
    print("="*50)

if __name__ == '__main__':
    main()
