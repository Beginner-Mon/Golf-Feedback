import numpy as np
import os

# Load Step 2 output
kpts = np.load('outputs/keypoints_2d.npy')  # (T,22,3)

# Keep (x,y)
kpts_2d = kpts[..., :2]  # (T,22,2)

subject = 'G5'
action = 'swing'

# 🔑 ADD CAMERA DIMENSION
positions_2d = {
    subject: {
        action: [kpts_2d]   # list = cameras
    }
}

metadata = {
    'layout_name': 'golf',
    'num_joints': 22,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11],
        [2, 4, 6, 8, 10, 12]
    ]
}

os.makedirs('golfswing', exist_ok=True)

np.savez(
    'golfswing/data_2d_golf_gt.npz',
    positions_2d=positions_2d,
    metadata=metadata,
    subjects=[subject]
)

print('✅ Saved golfswing/data_2d_golf_gt.npz')
print('2D shape:', kpts_2d.shape)
print('With camera dim:', len(positions_2d[subject][action]))
