import numpy as np

KEYPOINTS_2D_NPZ = "outputs/keypoints_2d.npz"
PREDICTED_3D_NPZ = "outputs/predicted_3d_denorm.npz"   # contains positions_3d
OUT_NPZ          = "outputs/keypoints_2d_with3d.npz"

# ----------------------------
# Load 2D NPZ
# ----------------------------
k2d = np.load(KEYPOINTS_2D_NPZ, allow_pickle=True)
positions_2d = k2d["positions_2d"].item()

metadata = k2d["metadata"] if "metadata" in k2d.files else {}
keypoint_scores = k2d["keypoint_scores"] if "keypoint_scores" in k2d.files else None

# Grab 2D sequence shape (T, J, 2) from the known location
seq2d = positions_2d["custom"]["sequence"][0]
T, J, _ = seq2d.shape

# ----------------------------
# Load predicted 3D
# ----------------------------
p3d = np.load(PREDICTED_3D_NPZ, allow_pickle=True)
if "positions_3d" not in p3d.files:
    raise KeyError(f"{PREDICTED_3D_NPZ} does not contain 'positions_3d'. Found: {p3d.files}")

pred3d = p3d["positions_3d"]  # expected shape (T, J, 3)

# ----------------------------
# Validate shapes
# ----------------------------
if pred3d.ndim != 3 or pred3d.shape[2] != 3:
    raise ValueError(f"pred3d must be (T, J, 3). Got: {pred3d.shape}")

if pred3d.shape[0] != T:
    raise ValueError(f"Frame mismatch: 2D has T={T}, 3D has T={pred3d.shape[0]}")

if pred3d.shape[1] != J:
    raise ValueError(f"Joint mismatch: 2D has J={J}, 3D has J={pred3d.shape[1]}")

# ----------------------------
# Wrap 3D into SAME dict format as 2D
# ----------------------------
positions_3d = {
    "custom": {
        "sequence": pred3d.astype(np.float32)   # <-- NOT a list
    }
}


# ----------------------------
# Save merged NPZ
# ----------------------------
save_kwargs = {
    "positions_2d": positions_2d,
    "positions_3d": positions_3d,
    "metadata": metadata
}

if keypoint_scores is not None:
    save_kwargs["keypoint_scores"] = keypoint_scores

np.savez_compressed(OUT_NPZ, **save_kwargs)

print("Merged file saved:", OUT_NPZ)
print("Contains:", list(save_kwargs.keys()))
print("2D shape:", seq2d.shape)
print("3D shape:", pred3d.shape)
