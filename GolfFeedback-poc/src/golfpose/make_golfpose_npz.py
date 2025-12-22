import numpy as np
import os

# Load same source used for 2D
kpts = np.load('outputs/keypoints_2d.npy')  # (T,22,3)

T, J, _ = kpts.shape

# Dummy 3D (z = 0)
kpts_3d = np.zeros((T, J, 3), dtype=np.float32)
kpts_3d[..., :2] = kpts[..., :2]

subject = 'G5'
action = 'swing'

# 🔑 ADD CAMERA DIMENSION
positions_3d = {
    subject: {
        action: [kpts_3d]   # list = cameras
    }
}

os.makedirs('golfswing', exist_ok=True)

np.savez(
    'golfswing/data_3d_golf_gt.npz',
    positions_3d=positions_3d,
    subjects=[subject]
)

print('✅ Saved golfswing/data_3d_golf_gt.npz')
print('3D shape:', kpts_3d.shape)
print('With camera dim:', len(positions_3d[subject][action]))
