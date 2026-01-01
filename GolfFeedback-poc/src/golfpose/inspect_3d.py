import numpy as np
from paths import DATA_DIR
NPY_PATH =  "predicted_3d.npy"

# -----------------------------
# LOAD
# -----------------------------
pred_3d = np.load(NPY_PATH)

# -----------------------------
# BASIC INFO
# -----------------------------
print("Loaded:", NPY_PATH)
print("Type:", type(pred_3d))
print("Dtype:", pred_3d.dtype)
print("Shape:", pred_3d.shape)

T, J, C = pred_3d.shape

print("\n=== INTERPRETATION ===")
print(f"Frames (T): {T}")
print(f"Joints (J): {J}")
print(f"Coordinates per joint: {C} (x, y, z)")

# -----------------------------
# CHECK ROOT JOINT
# -----------------------------
root = pred_3d[:, 0]
print("\nRoot joint (joint 0) stats:")
print("  min:", root.min(axis=0))
print("  max:", root.max(axis=0))

# -----------------------------
# PER-JOINT MOTION MAGNITUDE
# -----------------------------
motion = np.linalg.norm(pred_3d[1:] - pred_3d[:-1], axis=-1)
mean_motion = motion.mean(axis=0)

print("\nMean motion magnitude per joint:")
for j in range(J):
    print(f"  Joint {j:02d}: {mean_motion[j]:.4f}")

# -----------------------------
# SAMPLE FRAME
# -----------------------------
frame_id = 0
print(f"\nSample frame {frame_id}:")
print(pred_3d[frame_id])
