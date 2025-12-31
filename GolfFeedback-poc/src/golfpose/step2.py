import argparse
import os
import numpy as np
import torch
import mmcv
from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description="Step2: 2D Pose Estimation (MMPose 1.x)")
    parser.add_argument("--seq-dir", type=str, required=True, help="Directory containing image sequence")
    parser.add_argument("--bbox", type=str, default="bbox.npy", help="Path to bbox.npy from Step1")
    parser.add_argument("--pose-config", type=str, required=True, help="Pose config file")
    parser.add_argument("--pose-checkpoint", type=str, required=True, help="Pose checkpoint file")
    parser.add_argument("--out-dir", type=str, default="outputs_step2", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Register modules (required for legacy MMPose)
    register_all_modules()

    # 2. Load bbox
    raw_bbox = np.load(args.bbox, allow_pickle=True)

    # Normalize to numpy array (1, 4)
    if isinstance(raw_bbox, np.ndarray) and raw_bbox.shape == (4,):
        base_bbox = raw_bbox.astype(float)[None, :]  # (1, 4)
    elif isinstance(raw_bbox, (list, np.ndarray)):
        b = raw_bbox[0]
        base_bbox = np.array(b, dtype=float)[None, :]
    else:
        raise RuntimeError("Unsupported bbox.npy format")

    # 3. Load model
    print(f"[INFO] Loading model from {args.pose_config}")
    model = init_model(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device
    )

    # 4. Load image sequence
    img_files = sorted([
        os.path.join(args.seq_dir, f)
        for f in os.listdir(args.seq_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if len(img_files) == 0:
        raise RuntimeError("No images found")

    print(f"[INFO] Running inference on {len(img_files)} frames...")

    all_keypoints = []

    # 5. Inference loop
    for idx, img_path in enumerate(img_files):
        # ALWAYS pass bbox as (1, 4)
        results = inference_topdown(model, img_path, base_bbox)

        pred = results[0].pred_instances
        xy = pred.keypoints[0]          # (J, 2)
        score = pred.keypoint_scores[0] # (J,)

        keypoints = np.concatenate([xy, score[:, None]], axis=1)
        all_keypoints.append(keypoints)

        if (idx + 1) % 10 == 0:
            print(f" Processed {idx + 1}/{len(img_files)} frames")

    # 6. Save output
    keypoints_2d = np.stack(all_keypoints, axis=0)
    out_path = os.path.join(args.out_dir, "keypoints_2d.npy")
    np.save(out_path, keypoints_2d)

    print(f"[INFO] Saved keypoints to {out_path}")
    print(f"[INFO] Output shape: {keypoints_2d.shape}")
    print("[INFO] Step 2 completed successfully")


if __name__ == "__main__":
    main()
