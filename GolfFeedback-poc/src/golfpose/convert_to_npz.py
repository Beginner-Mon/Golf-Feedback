"""
Convert Step-2 2D keypoints (.npy) into GolfPose-compatible .npz

Input:
- outputs/keypoints_2d.npy   (T, 22, 3)  [x, y, confidence]

Output:
- outputs/golfpose_2d.npz
    positions_2d = {
        "G5": {
            "swing": (T, 22, 2)  # normalized camera space
        }
    }
"""

import os
import numpy as np
import cv2

# ============================
# CONFIG
# ============================
KEYPOINT_NPY = 'outputs/keypoints_2d.npy'
IMAGE_DIR = 'golfswing/images/S1/S1_Swing_01.2120309'

SUBJECT = 'G5'
ACTION = 'swing'
CAMERA = 0   # GolfPose assumes camera index exists

OUTPUT_NPZ = 'GolfFeedback-poc\src\golfpose\data\golf\detected/golfpose_2d.npz'
# ============================


# ----------------------------
# Load keypoints
# ----------------------------
keypoints = np.load(KEYPOINT_NPY)  # (T, 22, 3)
assert keypoints.ndim == 3 and keypoints.shape[2] == 3, 'Invalid keypoint shape'

T, J, _ = keypoints.shape
print(f'[INFO] Loaded keypoints: {keypoints.shape}')

# ----------------------------
# Load image size
# ----------------------------
image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.png'))
])

assert len(image_files) >= 1, 'No images found to infer resolution'

img = cv2.imread(image_files[0])
H, W = img.shape[:2]
print(f'[INFO] Image resolution: {W} x {H}')

# ----------------------------
# Drop confidence
# ----------------------------
keypoints_xy = keypoints[..., :2].astype(np.float32)  # (T, 22, 2)

# ----------------------------
# Normalize to camera space
# GolfPose expects:
#   x, y ∈ [-1, 1], origin at image center
# ----------------------------
keypoints_xy[..., 0] = (keypoints_xy[..., 0] / W - 0.5) * 2.0
keypoints_xy[..., 1] = (keypoints_xy[..., 1] / H - 0.5) * 2.0

# ----------------------------
# Build dataset dictionary
# ----------------------------
positions_2d = {
    SUBJECT: {
        ACTION: keypoints_xy
    }
}

# ----------------------------
# Save
# ----------------------------
os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
np.savez_compressed(OUTPUT_NPZ, positions_2d=positions_2d)

print('\n[SUCCESS]')
print('Saved:', OUTPUT_NPZ)
print('Subject:', SUBJECT)
print('Action:', ACTION)
print('Shape:', keypoints_xy.shape)
