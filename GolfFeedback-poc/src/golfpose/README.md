# GolfPose – 3D Golf Swing Pose Inference Module

This module integrates **GolfPose**, a state-of-the-art 3D human pose estimation framework for golf swings, into the **Golf Feedback Proof-of-Concept** project.

It provides a **single-camera 3D pose inference pipeline** based on 2D keypoints and is intended for downstream analysis and feedback generation.

---

## 📌 Background

This work is **adapted from the official GolfPose project**, introduced in ICPR 2024:

- 📄 **Paper**:  
  *GolfPose: Realtime 3D Human Pose Estimation for Golf Swing*  
  https://minghanlee.github.io/papers/ICPR_2024_GolfPose.pdf

- 💻 **Original GitHub Repository**:  
  https://github.com/MingHanLee/GolfPose

All core model ideas, architecture, and evaluation metrics originate from the original authors.

---

## 📁 Location in Repository

This module lives inside the Golf Feedback project:

GolfFeedback-poc/
└── src/
└── golfpose/
├── common/
├── configs/
├── golfpose_3d.py
├── environment.yml
└── README.md


## 🚀 What This Module Does

- Accepts **single-camera 2D pose keypoints**
- Runs **3D human pose inference** using pretrained GolfPose models
- Outputs 3D joint positions suitable for:
  - Swing analysis
  - Motion metrics
  - Feedback and visualization

This implementation is **inference-focused** and simplifies parts of the original multi-camera pipeline.

---

## ⚙️ Environment Setup

### 1. Create Conda environment

bash
conda env create -f environment.yml
conda activate golfpose
2. Install remaining dependencies
bash
Sao chép mã
pip install timm opencv-python mmcv mmdet mmpose
Refer to the original GolfPose repository for detailed dependency versions if needed.

▶️ Running 3D Inference
Example Command
bash
Sao chép mã
python golfpose_3d.py \
  -k gt \
  -d golf \
  -str G5 \
  -ste G5 \
  -c golfpose_checkpoints \
  --evaluate golfpose_17+5_39.2_32.3_62.8.bin \
  -f 243 \
  -s 243 \
  -gpu 0 \
  -club 5
  
  
📚 References

If you use this module in research or reports, please cite:

M. Lee et al., GolfPose: Realtime 3D Human Pose Estimation for Golf Swing,
ICPR 2024.

And reference the original implementation:

https://github.com/MingHanLee/GolfPose
