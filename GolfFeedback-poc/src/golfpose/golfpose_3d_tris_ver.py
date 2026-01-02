import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from common.model_cross import MixSTE2

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
NPZ_PATH = "outputs/keypoints_2d.npz"
CHECKPOINT_PATH = "golfpose_checkpoints/golfpose_17+0_35.6.bin"  # 17-joint checkpoint
OUTPUT_PATH = "outputs/predicted_3d.npz"

NUM_JOINTS = 17
RECEPTIVE_FIELD = 243   # default GolfPose window
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD NPZ (EXACT FORMAT)
# --------------------------------------------------
data = np.load(NPZ_PATH, allow_pickle=True)

# confirmed format
keypoints_2d = data["positions_2d"].item()
sequence_2d = keypoints_2d["custom"]["sequence"][0]

import numpy as np
import matplotlib.pyplot as plt

SKELETON = [
    (0, 1), (0, 4), (0, 7),
    (7, 8), (8, 9), (9, 10),

    (1, 2), (2, 3),
    (4, 5), (5, 6),

    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
# ---- 2D bbox normalization (CRITICAL) ----
seq = sequence_2d.copy().astype(np.float32)

x = seq[..., 0]
y = seq[..., 1]

cx = (x.max(axis=1) + x.min(axis=1)) / 2.0
cy = (y.max(axis=1) + y.min(axis=1)) / 2.0
scale = np.maximum(
    x.max(axis=1) - x.min(axis=1),
    y.max(axis=1) - y.min(axis=1)
) + 1e-9

seq[..., 0] = (x - cx[:, None]) / scale[:, None]
seq[..., 1] = (y - cy[:, None]) / scale[:, None]

sequence_2d = seq
print("Normalized 2D x min/max:",
      sequence_2d[...,0].min(),
      sequence_2d[...,0].max())
print("Normalized 2D y min/max:",
      sequence_2d[...,1].min(),
      sequence_2d[...,1].max())


d = np.load("outputs/keypoints_2d.npz", allow_pickle=True)
seq = d["positions_2d"].item()["custom"]["sequence"][0]  # (T,17,2)
pts = seq[0]

plt.figure(figsize=(6,6))
plt.scatter(pts[:,0], pts[:,1])
for i,(x,y) in enumerate(pts):
    plt.text(x, y, str(i))

for a,b in SKELETON:
    plt.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]])

plt.gca().invert_yaxis()
plt.title("2D skeleton (frame 0) using repo SKELETON")
plt.show()
# shape: (T, 17, 2)

T = sequence_2d.shape[0]
assert sequence_2d.shape == (T, NUM_JOINTS, 2)

# --------------------------------------------------
# TO TENSOR
# --------------------------------------------------
inputs_2d = torch.from_numpy(sequence_2d).float()
inputs_2d = inputs_2d.unsqueeze(0)  # (1, T, J, 2)

# --------------------------------------------------
# TEMPORAL CHUNKING (same idea as GolfPose)
# --------------------------------------------------
def temporal_chunks(x, receptive_field):
    """
    x: (1, T, J, 2)
    return: (N, receptive_field, J, 2)
    """
    T = x.shape[1]

    if T < receptive_field:
        pad = receptive_field - T
        x = torch.nn.functional.pad(x, (0,0,0,0,0,pad), mode="replicate")
        T = receptive_field

    chunks = []
    stride = receptive_field
    for start in range(0, T - receptive_field + 1, stride):
        chunks.append(x[:, start:start + receptive_field])

    return torch.cat(chunks, dim=0)

inputs_2d = temporal_chunks(inputs_2d, RECEPTIVE_FIELD)

# --------------------------------------------------
# LOAD MODEL (NO CLUB)
# --------------------------------------------------
model = MixSTE2(
    num_frame=RECEPTIVE_FIELD,
    num_joints=NUM_JOINTS,
    in_chans=2,
    embed_dim_ratio=512,
    depth=8,
    num_heads=8,
    mlp_ratio=2.0,
    qkv_bias=True,
    drop_path_rate=0.0
)

model = nn.DataParallel(model).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_pos"], strict=False)
model.eval()

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
with torch.no_grad():
    inputs_2d = inputs_2d.to(DEVICE)
    pred_3d = model(inputs_2d)

# pred_3d shape: (N, F, 17, 3)
pred_3d = pred_3d.cpu().numpy()

# --------------------------------------------------
# REASSEMBLE TO (T, 17, 3)
# --------------------------------------------------
output_3d = np.zeros((T, NUM_JOINTS, 3))
cursor = 0

for i in range(pred_3d.shape[0]):
    length = min(RECEPTIVE_FIELD, T - cursor)
    output_3d[cursor:cursor + length] = pred_3d[i, :length]
    cursor += length

# --------------------------------------------------
# SAVE
# --------------------------------------------------
np.savez_compressed(
    OUTPUT_PATH,
    positions_3d=output_3d,          # key GolfDataset looks for
    metadata=data.get("metadata", {})  # optional but safe
)

print("Saved 3D prediction:", OUTPUT_PATH)
print("Key: positions_3d")
print("Shape:", output_3d.shape)