from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv
import os

# ---------------- CONFIG ----------------
CONFIG_FILE = 'configs/mmdet/golfpose_detector_2cls_yolox_s.py'
CHECKPOINT_FILE = 'golfpose_checkpoints/golfpose_detector_2cls_yolox_s.pth'
IMAGE_FILE = 'golfswing\images\S1\S1_Swing_05.2120309\S1_Swing_05.2120309_000006.jpg'
OUTPUT_FILE = 'detector_result.jpg'
DEVICE = 'cuda:0'
# --------------------------------------

# Load image
image = mmcv.imread(IMAGE_FILE)

# Init model
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

# Inference
result = inference_detector(model, IMAGE_FILE)

# Visualizer
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta

# Draw + save
visualizer.add_datasample(
    name='result',
    image=image,
    data_sample=result,
    draw_gt=False,
    out_file=OUTPUT_FILE,
    pred_score_thr=0.3
)

print(f"YOLOX inference complete. Output saved to {OUTPUT_FILE}")
