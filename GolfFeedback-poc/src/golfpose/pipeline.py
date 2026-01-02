import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import mmcv
from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from einops import rearrange
from common.model_cross import MixSTE2


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: Video -> 3D Pose"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--pose-config", type=str, required=True, help="MMPose config")
    parser.add_argument("--pose-checkpoint", type=str, required=True, help="MMPose checkpoint")
    parser.add_argument("--golfpose-checkpoint", type=str, 
                        default="golfpose_checkpoints/golfpose_17+0_35.6.bin",
                        help="GolfPose 3D model checkpoint")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    return parser.parse_args()


def extract_frames(video_path, out_dir, skip=1):
    """Extract frames from video"""
    print(f"[1/5] Extracting frames from video...")
    
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_paths = []
    frame_idx = 0
    saved_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_idx += 1
        
        frame_idx += 1
    
    cap.release()
    
    print(f"   Extracted {len(frame_paths)} frames (skipped every {skip})")
    print(f"   Original FPS: {fps:.2f}, Total frames: {total_frames}")
    
    return frame_paths, fps


def detect_person(frame_paths, det_conf):
    """Detect person in first frame and return bbox"""
    print(f"[2/5] Detecting person...")
    
    img = cv2.imread(frame_paths[0])
    h, w, _ = img.shape
    
    model = YOLO("yolov8n.pt")
    results = model(img, conf=det_conf, classes=[0])
    r = results[0]
    
    if r.boxes is None or len(r.boxes) == 0:
        raise RuntimeError("No person detected in first frame")
    
    box = r.boxes.xyxy[0].cpu().numpy()
    score = float(r.boxes.conf[0].cpu().numpy())
    x1, y1, x2, y2 = map(int, box)
    
    # Add padding
    pad = 0.1
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w - 1, int(x2 + pad * bw))
    y2 = min(h - 1, int(y2 + pad * bh))
    
    bbox = np.array([x1, y1, x2, y2], dtype=float)
    
    print(f"   Detected bbox: [{x1}, {y1}, {x2}, {y2}], score={score:.3f}")
    
    return bbox


def estimate_2d_poses(frame_paths, bbox, pose_config, pose_checkpoint, device):
    """Run 2D pose estimation on all frames"""
    print(f"[3/5] Estimating 2D poses...")
    
    register_all_modules()
    
    model = init_model(pose_config, pose_checkpoint, device=device)
    bbox_input = bbox[None, :]  # (1, 4)
    
    all_keypoints = []
    
    for idx, img_path in enumerate(frame_paths):
        results = inference_topdown(model, img_path, bbox_input)
        pred = results[0].pred_instances
        xy = pred.keypoints[0]
        score = pred.keypoint_scores[0]
        keypoints = np.concatenate([xy, score[:, None]], axis=1)
        all_keypoints.append(keypoints)
        
        if (idx + 1) % 50 == 0 or idx == len(frame_paths) - 1:
            print(f"   Processed {idx + 1}/{len(frame_paths)} frames")
    
    keypoints_2d = np.stack(all_keypoints, axis=0)
    return keypoints_2d


def normalize_2d_poses(keypoints_xy):
    """Normalize 2D poses using bbox"""
    seq = keypoints_xy.copy().astype(np.float32)
    
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
    
    return seq


def temporal_chunks(x, receptive_field):
    """Split sequence into temporal chunks"""
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


def predict_3d_poses(keypoints_2d, checkpoint_path, device):
    """Predict 3D poses using GolfPose"""
    print(f"[4/5] Predicting 3D poses...")
    
    NUM_JOINTS = 17
    RECEPTIVE_FIELD = 243
    
    # Normalize
    keypoints_xy = keypoints_2d[:, :, :2]
    normalized = normalize_2d_poses(keypoints_xy)
    
    T = normalized.shape[0]
    
    # To tensor
    inputs_2d = torch.from_numpy(normalized).float()
    inputs_2d = inputs_2d.unsqueeze(0)
    inputs_2d = temporal_chunks(inputs_2d, RECEPTIVE_FIELD)
    
    # Load model
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
    
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_pos"], strict=False)
    model.eval()
    
    # Inference
    with torch.no_grad():
        inputs_2d = inputs_2d.to(device)
        pred_3d = model(inputs_2d)
    
    pred_3d = pred_3d.cpu().numpy()
    
    # Reassemble
    output_3d = np.zeros((T, NUM_JOINTS, 3))
    cursor = 0
    
    for i in range(pred_3d.shape[0]):
        length = min(RECEPTIVE_FIELD, T - cursor)
        output_3d[cursor:cursor + length] = pred_3d[i, :length]
        cursor += length
    
    print(f"   3D poses shape: {output_3d.shape}")
    
    return output_3d


def scale_3d_poses(pred_3d):
    """Scale 3D poses to realistic dimensions"""
    BONES = [
        (0,1),(1,2),(2,3),
        (0,4),(4,5),(5,6),
        (0,7),(7,8),(8,9),(9,10),
        (8,11),(11,12),(12,13),
        (8,14),(14,15),(15,16)
    ]
    
    lengths = []
    for a, b in BONES:
        v = pred_3d[:, b] - pred_3d[:, a]
        lengths.append(np.linalg.norm(v, axis=1))
    
    current = float(np.mean(np.stack(lengths, axis=1)))
    target = 0.30  # meters
    scale = target / (current + 1e-9)
    
    return pred_3d * scale


def save_outputs(out_dir, keypoints_2d, pred_3d, fps):
    """Save all outputs"""
    print(f"[5/5] Saving outputs...")
    
    T, J, _ = keypoints_2d.shape
    keypoints_xy = keypoints_2d[:, :, :2]
    keypoints_score = keypoints_2d[:, :, 2]
    
    # Scale 3D poses
    pred_3d_scaled = scale_3d_poses(pred_3d)
    
    # Prepare data structures
    positions_2d = {
        "custom": {
            "sequence": [keypoints_xy]
        }
    }
    
    positions_3d = {
        "custom": {
            "sequence": pred_3d_scaled.astype(np.float32)
        }
    }
    
    metadata = {
        "layout_name": "custom_mmpose",
        "num_joints": J,
        "fps": fps,
        "keypoints_symmetry": [
            list(range(0, J // 2)),
            list(range(J // 2, J))
        ]
    }
    
    # Save combined output
    output_path = os.path.join(out_dir, "pose_reconstruction.npz")
    np.savez_compressed(
        output_path,
        positions_2d=positions_2d,
        positions_3d=positions_3d,
        metadata=metadata,
        keypoint_scores=keypoints_score
    )
    
    print(f"   Saved to: {output_path}")
    print(f"   - 2D poses: {keypoints_xy.shape}")
    print(f"   - 3D poses: {pred_3d_scaled.shape}")
    print(f"   - FPS: {fps:.2f}")
    
    return output_path


def main():
    args = parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*60)
    print("Video to 3D Pose Reconstruction Pipeline")
    print("="*60)
    
    # Step 1: Extract frames
    frame_paths, fps = extract_frames(args.video, args.out_dir, args.skip_frames)
    
    # Step 2: Detect person
    bbox = detect_person(frame_paths, args.det_conf)
    
    # Step 3: 2D pose estimation
    keypoints_2d = estimate_2d_poses(
        frame_paths, bbox, args.pose_config, args.pose_checkpoint, args.device
    )
    
    # Step 4: 3D pose prediction
    pred_3d = predict_3d_poses(keypoints_2d, args.golfpose_checkpoint, args.device)
    
    # Step 5: Save outputs
    output_path = save_outputs(args.out_dir, keypoints_2d, pred_3d, fps)
    
    print("="*60)
    print("Pipeline completed successfully!")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()