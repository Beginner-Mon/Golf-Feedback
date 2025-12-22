"""
GolfPose Step 2 – Temporal 2D Keypoint Extraction (CORRECT)

Input:
- Image sequence folder (one swing)

Output:
- outputs/keypoints_2d.npy  → (T, 22, 3)
"""

import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import Pose2DInferencer

# ---------------- CONFIG ----------------
IMAGE_DIR = 'golfswing/images/S1/S1_Swing_01.2120309'

DET_CONFIG = 'configs/mmdet/golfpose_detector_1cls_yolox_s.py'
DET_CHECKPOINT = 'golfpose_checkpoints/golfpose_detector_1cls_yolox_s.pth'

POSE_CONFIG = 'configs/mmpose/golfpose_golfer_hrnetw48.py'
POSE_CHECKPOINT = 'golfpose_checkpoints/golfpose_golfer_hrnetw48.pth'

DEVICE = 'cuda'   # or 'cpu'
OUTPUT_DIR = 'outputs'
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Load models ----------------
print('[INFO] Loading detector...')
det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)

print('[INFO] Loading 2D pose model...')
pose_inferencer = Pose2DInferencer(
    model=POSE_CONFIG,
    weights=POSE_CHECKPOINT,
    device=DEVICE
)

# ---------------- Load frames ----------------
image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.png'))
])

assert len(image_files) > 0, 'No images found'

print(f'[INFO] Found {len(image_files)} frames')

# ---------------- Process frames ----------------
all_keypoints = []

for idx, img_path in enumerate(image_files):
    print(f'[INFO] Frame {idx+1}/{len(image_files)}')

    # ---- Step 1: Detect golfer ----
    det_result = inference_detector(det_model, img_path)
    instances = det_result.pred_instances

    if len(instances) == 0:
        raise RuntimeError(f'No golfer detected in {img_path}')

    bboxes = instances.bboxes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    bbox = bboxes[scores.argmax()]   # best golfer

    # ---- Step 2: 2D Pose ----
    result = next(
        pose_inferencer(
            img_path,
            bboxes=[bbox.tolist()],
            return_datasamples=True
        )
    )

    pred = result['predictions'][0]
    inst = pred.pred_instances

    keypoints = inst.keypoints[0]          # (22, 2)
    kp_scores = inst.keypoint_scores[0]    # (22,)

    # (22,3)
    keypoints_2d = np.concatenate(
        [keypoints, kp_scores[:, None]],
        axis=1
    )

    all_keypoints.append(keypoints_2d)

# ---------------- Save output ----------------
all_keypoints = np.stack(all_keypoints, axis=0)

out_path = os.path.join(OUTPUT_DIR, 'keypoints_2d.npy')
np.save(out_path, all_keypoints)

print('[SUCCESS]')
print('Saved:', out_path)
print('Shape:', all_keypoints.shape)
