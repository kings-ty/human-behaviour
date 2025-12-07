import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from model_skeletonx_v3 import SkeletonX_Model

# =============================================================================
# 1. Test dataset (auto size correction enabled)
# =============================================================================
class TestDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path) # (840, 60, 17, 2)
        
        # [Key] Data size check and correction
        # Training data had mean around 0.6. If test data is smaller (~0.3) scale up by 2.
        data_mean = np.mean(np.abs(self.data))
        print(f"📊 Test Data Mean Size: {data_mean:.4f}")
        
        if data_mean < 0.4:
            print("🚨 Detected SMALL data range (-1~1). Scaling up by 2.0 to match training data (-2~2)...")
            self.data = self.data * 2.0
        else:
            print("✅ Data range matches training data. No scaling needed.")

        self.ntu_pairs = (
            (1, 0), (20, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 10),
            (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18)
        )

    def __getitem__(self, index):
        raw_data = torch.from_numpy(self.data[index]).float() 
        
        # 1. Noise removal / fill missing hips with knees when necessary
        if torch.all(torch.abs(raw_data[:, 4, :]) < 1e-4): raw_data[:, 4, :] = raw_data[:, 2, :]
        if torch.all(torch.abs(raw_data[:, 3, :]) < 1e-4): raw_data[:, 3, :] = raw_data[:, 1, :]

        # 2. Interpolation (60 -> 64 frames)
        data = raw_data.permute(2, 0, 1).unsqueeze(-1) 
        C, T, V, M = data.shape
        data = data.permute(0, 2, 3, 1).contiguous().view(-1, 1, T)
        data = F.interpolate(data, size=64, mode='linear', align_corners=False)
        data = data.view(C, V, M, 64).permute(0, 3, 1, 2).contiguous()

        # 3. Mapping from COCO-17 to NTU-25 (17 -> 25)
        final_data = torch.zeros(3, 64, 25, M)
        map_idx = {0:3, 5:4, 6:8, 7:5, 8:9, 9:6, 10:10, 11:12, 12:16, 13:13, 14:17, 15:14, 16:18}
        for c, n in map_idx.items():
            final_data[:2, :, n, :] = data[:2, :, c, :]
            
        # 1. Hip center (Base Spine, NTU 0)
        hip_center = (data[:2, :, 11, :] + data[:2, :, 12, :]) / 2
        final_data[:2, :, 0, :] = hip_center
        
        # 2. Shoulder center (Spine Shoulder, NTU 20) -> critical keypoint
        shoulder_center = (data[:2, :, 5, :] + data[:2, :, 6, :]) / 2
        final_data[:2, :, 20, :] = shoulder_center  # shoulder center must be index 20
        
        # 3. Mid spine (NTU 1) -> midpoint between hip center (0) and shoulder center (20)
        final_data[:2, :, 1, :] = (hip_center + shoulder_center) / 2
        
        # 4. Neck (NTU 2) -> midpoint of shoulder (20) and nose (COCO 0)
        # Map COCO 0 (nose) to NTU 3 (head); neck set between shoulder and nose
        nose = data[:2, :, 0, :]
        final_data[:2, :, 2, :] = (shoulder_center + nose) / 2  # neck between shoulder and nose
        
        # 5. Head (NTU 3) -> use nose coordinates
        final_data[:2, :, 3, :] = nose

        # =========================================================
        # 🔥 [Final Inputs] validated recipe used in evaluation
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
        motion_input = motion_input * 10.0 

        return joint_input.squeeze(-1), bone_input.squeeze(-1), motion_input.squeeze(-1)

    def __len__(self):
        return len(self.data)

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
    print(f"🎯 GENERATING FINAL SUBMISSION")

    # File paths
    TEST_DATA_PATH = 'test_sequences.npy' 
    JOINT_MODEL = 'best_topologyaware_joint_fold0.pth'
    BONE_MODEL = 'best_topologyaware_bone_fold0.pth'
    MOTION_MODEL = 'best_topologyaware_motion_fold0.pth'

    # Load test data
    try:
        dataset = TestDataset(TEST_DATA_PATH)
        print(f"📂 Loaded Test Data: {len(dataset)} samples")
    except FileNotFoundError:
        print(f"❌ Error: '{TEST_DATA_PATH}' not found!")
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Load models
    model_j = load_model_safe(JOINT_MODEL, device)
    model_b = load_model_safe(BONE_MODEL, device)
    model_m = load_model_safe(MOTION_MODEL, device)

    predictions = []

    print("🚀 Predicting...")
    with torch.no_grad():
        for j, b, m in tqdm(loader):
            j, b, m = j.to(device), b.to(device), m.to(device)

            out_j = F.softmax(model_j(j), dim=1)
            out_b = F.softmax(model_b(b), dim=1)
            out_m = F.softmax(model_m(m), dim=1)

            # 🏆 Golden ratio used: Joint 0.2 : Bone 0.4 : Motion 0.4
            final_score = (0.2 * out_j) + (0.4 * out_b) + (0.4 * out_m)
            
            pred_batch = torch.argmax(final_score, dim=1).cpu().numpy()
            predictions.extend(pred_batch)

    # Save submission
    df = pd.DataFrame({
        'Sample_ID': range(len(predictions)),
        'Label': predictions
    })
    df.to_csv('submission.csv', index=False)
    print("🎉 Done! 'submission.csv' generated.")

if __name__ == '__main__':
    main()