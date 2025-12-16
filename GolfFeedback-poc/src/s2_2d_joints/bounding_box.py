from ultralytics import YOLO
import cv2
import argparse
import os

# COCO skeleton (pairs of keypoint indices)
COCO_SKELETON = [
    (5, 7), (7, 9),     # Left arm
    (6, 8), (8, 10),    # Right arm
    (5, 6),            # Shoulders
    (5, 11), (6, 12),  # Torso
    (11, 12),
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)   # Right leg
]

def detect_image_from_frame(frame_image, model_path):
    """
    Detect person keypoints for a single image (frame)
    Prints joints instead of saving the image.
    """
    model = YOLO(model_path)
    results = model(frame_image)

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints

        if boxes is None or keypoints is None:
            continue

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if model.names[cls] != "person":
                continue

            kpts = keypoints.xy[i]  # (17, 2)
            print("Person joints:\n", kpts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLO pose model path")
    parser.add_argument("--output", type=str, default="outputs", help="Output folder")
    args = parser.parse_args()

    detect_image_from_frame(args.image, args.model, args.output)
