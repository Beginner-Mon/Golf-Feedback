import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Step1: Person Detection")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    img_path = args.image
    out_dir = args.out_dir
    conf_thr = args.conf

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Load image
    # -----------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w, _ = img.shape
    print(f"[INFO] Image loaded: {w}x{h}")

    # -----------------------------
    # Load YOLOv8 (pretrained)
    # -----------------------------
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # pretrained on COCO

    # -----------------------------
    # Run detection (person only)
    # -----------------------------
    results = model(img, conf=conf_thr, classes=[0])  # class 0 = person
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        raise RuntimeError("No person detected in the image")

    # -----------------------------
    # Take TOP-1 person
    # -----------------------------
    box = r.boxes.xyxy[0].cpu().numpy()
    score = float(r.boxes.conf[0].cpu().numpy())

    x1, y1, x2, y2 = map(int, box)

    # Optional padding (recommended)
    pad = 0.1
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w - 1, int(x2 + pad * bw))
    y2 = min(h - 1, int(y2 + pad * bh))

    bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

    print(f"[INFO] Detected person bbox: {bbox.tolist()}, score={score:.3f}")

    # -----------------------------
    # Save bbox (Step2 compatible)
    # -----------------------------
    bbox_path = os.path.join(out_dir, "bbox.npy")
    np.save(bbox_path, bbox)
    print(f"[INFO] Saved bbox to {bbox_path}")

    # -----------------------------
    # Visualization (same role as before)
    # -----------------------------
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"person {score:.2f}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    vis_path = os.path.join(out_dir, "detector_result.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"[INFO] Saved visualization to {vis_path}")

    print("[INFO] Step1 completed successfully")


if __name__ == "__main__":
    main()
