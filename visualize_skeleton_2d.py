import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import random

# ==========================================
# 1. Configuration
# ==========================================
DATA_PATH = 'pose_features_large1/multistream/train_sequences.npy' 

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def main():
    # --- 1. Load Data ---
    if not Path(DATA_PATH).exists():
        alt_path = 'pose_features_large/train_sequences.npy'
        if Path(alt_path).exists():
            path_to_load = alt_path
        else:
            print(f"❌ Error: Data not found at {DATA_PATH}")
            return
    else:
        path_to_load = DATA_PATH

    data = np.load(path_to_load)
    
    # [안전장치] 데이터가 유효한 샘플을 찾을 때까지 반복
    for _ in range(10):
        idx = random.randint(0, len(data)-1)
        seq = data[idx]
        if seq.shape[0] == 3: seq = seq.transpose(1, 2, 0)
        
        # 0이 아닌 값이 절반 이상인 샘플만 통과
        if np.count_nonzero(seq) > (seq.size * 0.5):
            break
    
    print(f"🚀 Analyzing Sample #{idx} (Valid Data Found)")

    # --- 2. Metrics Calculation ---
    # Velocity
    velocity = np.zeros_like(seq)
    velocity[:-1] = seq[1:] - seq[:-1]
    
    # Energy (Simple)
    energy = np.sum(np.linalg.norm(velocity, axis=2), axis=1)

    # [핵심 수정] "가장 에너지가 높은 프레임" 대신 "가장 온전한 프레임" 선택
    # 관절 좌표가 (0,0)이 아닌 개수가 가장 많은 프레임 중 하나를 선택
    valid_joints_per_frame = np.sum(seq.sum(axis=2) > 0, axis=1)
    best_frame_candidates = np.where(valid_joints_per_frame == valid_joints_per_frame.max())[0]
    
    # 후보 중 에너지가 적당히 있는(움직이는) 프레임 선택
    target_frame = best_frame_candidates[len(best_frame_candidates)//2] # 중간쯤 있는 프레임
    
    print(f"📸 Snapshot Frame: {target_frame}")

    # --- 3. Visualization Setup ---
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])

    # [Left] Kinematic Snapshot
    ax_skel = plt.subplot(gs[:, 0])
    ax_skel.set_title(f"Kinematic Snapshot (Frame {target_frame})", fontsize=14, fontweight='bold')
    
    # 좌표 범위 계산 (전체 시퀀스 기준)
    valid_mask = (seq.sum(axis=2) > 0)
    if valid_mask.any():
        all_x = seq[:, :, 0][valid_mask]
        all_y = seq[:, :, 1][valid_mask]
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()
        
        shift_down = (max_y - min_y) * 0.5
        # 여백 20%
        pad_x = (max_x - min_x) * 0.2
        pad_y = (max_y - min_y) * 0.2
        
        ax_skel.set_xlim(min_x - pad_x, max_x + pad_x)
        ax_skel.set_ylim(max_y + pad_y, min_y - pad_y - shift_down)
    
    ax_skel.set_aspect('equal')
    ax_skel.axis('off')

    # Draw Skeleton
    pose = seq[target_frame]
    pose_vel = velocity[target_frame]
    
    # [강제 설정] 선 두께와 점 크기를 무조건 보이게 고정
    for i, j in COCO_PAIRS:
        if pose[i].sum() == 0 or pose[j].sum() == 0: continue
        ax_skel.plot([pose[i,0], pose[j,0]], [pose[i,1], pose[j,1]], 
                     color='black', lw=4, alpha=0.7) # 두께 4로 고정

    # 관절 점 찍기 (선이 안 보여도 점은 보이게)
    ax_skel.scatter(pose[:,0], pose[:,1], s=50, c='black', zorder=3)

    # Draw Arrows (Red)
    # 데이터 스케일 감지
    scale_factor = (max_x - min_x) if valid_mask.any() else 1000
    
    for i in range(17):
        if pose[i].sum() == 0: continue
        v_vec = pose_vel[i]
        
        # 움직임이 미세해도 50배 뻥튀기해서 그림
        if np.linalg.norm(v_vec) > 0:
            draw_vec = v_vec * 20.0 
            ax_skel.arrow(pose[i,0], pose[i,1], draw_vec[0], draw_vec[1], 
                          head_width=scale_factor*0.03, color='#e74c3c', zorder=5)

    # [Right Top] Energy
    ax_energy = plt.subplot(gs[0, 1])
    ax_energy.set_title("Motion Energy Profile", fontsize=12)
    ax_energy.plot(energy, color='#2980b9', lw=2)
    ax_energy.axvline(target_frame, color='red', linestyle='--', label='Snapshot')
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)

    # [Right Bottom] Heatmap
    ax_heat = plt.subplot(gs[1, 1])
    ax_heat.set_title("Joint Velocity Heatmap", fontsize=12)
    vel_mag = np.linalg.norm(velocity, axis=2).T
    im = ax_heat.imshow(vel_mag, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax_heat)
    ax_heat.set_xlabel("Time (Frame)")
    ax_heat.set_ylabel("Joint Index")

    plt.tight_layout()
    plt.savefig("physics_analysis_dashboard_safe.png", dpi=300, bbox_inches='tight')
    print("✅ Dashboard Generated: physics_analysis_dashboard_safe.png")

if __name__ == "__main__":
    main()