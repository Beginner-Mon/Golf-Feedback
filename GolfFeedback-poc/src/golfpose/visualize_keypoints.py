import os
import numpy as np
import cv2

# ---------------- CONFIG ----------------
IMAGE_DIR = 'golfswing/images/test'
KEYPOINT_FILE = 'outputs/keypoints_2d.npy'
OUTPUT_VIDEO = 'outputs/vis_2d_keypoints.mp4'

CONF_THRESHOLD = 0.3
FPS = 30   # adjust if needed
# ---------------------------------------

# Load data
keypoints_2d = np.load(KEYPOINT_FILE)  # (T, 22, 3)

image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.png'))
])

assert len(image_files) == keypoints_2d.shape[0], 'Frame count mismatch'

# Read first frame to get size
first_img = cv2.imread(image_files[0])
H, W = first_img.shape[:2]

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

# -------- Skeleton (corrected GolfPose layout) --------
# SKELETON = [
#     # Right leg
#     (0, 1), (1, 2), (2, 3),
#     # Left leg
#     (0, 4), (4, 5), (5, 6),

#     # Spine & head
#     (0, 7), (7, 8), (8, 9), (9, 10),

#     # Left arm
#     (8, 11), (11, 12), (12, 13),

#     # Right arm
#     (8, 14), (14, 15), (15, 16),

#     # Club
#     (13, 17), (16, 17),
#     (17, 18),
#     (18, 19), (18, 20), (18, 21)
# ]

SKELETON = [
    (0, 1), (0, 4), (0, 7),
    (7, 8), (8, 9), (9, 10),

    (1, 2), (2, 3),
    (4, 5), (5, 6),

    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
# -------- Process all frames --------
for idx, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    kps = keypoints_2d[idx]

    # Draw keypoints
    for x, y, c in kps:
        if c > CONF_THRESHOLD:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Draw skeleton
    for i, j in SKELETON:
        if kps[i, 2] > CONF_THRESHOLD and kps[j, 2] > CONF_THRESHOLD:
            p1 = (int(kps[i, 0]), int(kps[i, 1]))
            p2 = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.line(img, p1, p2, (255, 0, 0), 2)

    video.write(img)

    if idx % 50 == 0:
        print(f'[INFO] Processed frame {idx}/{len(image_files)}')

video.release()
print('[SUCCESS] Video saved to:', OUTPUT_VIDEO)
