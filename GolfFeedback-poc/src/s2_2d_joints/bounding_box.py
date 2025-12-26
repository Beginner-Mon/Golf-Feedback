from ultralytics import YOLO
import cv2
import argparse
import os
import numpy as np

# COCO skeleton (pairs of keypoint indices)
COCO_SKELETON = [
    (5, 7), (7, 9),     # Left arm
    (6, 8), (8, 10),    # Right arm
    (5, 6),             # Shoulders
    (5, 11), (6, 12),   # Torso
    (11, 12),
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)   # Right leg
]

# -------------------------------------------------------------------
# 1. Detect only — return raw data
# -------------------------------------------------------------------
def detect_image_from_frame(frame_image, model_path):
    model = YOLO(model_path)
    image = cv2.imread(frame_image)

    if image is None:
        raise ValueError(f"Cannot read image: {frame_image}")

    results = model(image)
    persons = []

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue

        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            if model.names[cls] != "person":
                continue

            kpts = r.keypoints.xy[i].cpu().numpy()  # (17, 2)
            persons.append({"keypoints": kpts})

    return image, persons

# -------------------------------------------------------------------
# 2. Visualize & save
#    - overlay: image + keypoints
#    - keypoints only: WHITE background, BLUE lines
# -------------------------------------------------------------------
def visualize_keypoints(image, persons, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)

    h, w = image.shape[:2]

    overlay_img = image.copy()

    # WHITE background
    keypoints_only_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    for person in persons:
        kpts = person["keypoints"]

        # joints
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(overlay_img, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.circle(keypoints_only_img, (int(x), int(y)), 4, (255, 0, 0), -1)

        # skeleton
        for i, j in COCO_SKELETON:
            if (
                kpts[i][0] > 0 and kpts[i][1] > 0 and
                kpts[j][0] > 0 and kpts[j][1] > 0
            ):
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))

                cv2.line(overlay_img, pt1, pt2, (0, 255, 0), 2)
                cv2.line(keypoints_only_img, pt1, pt2, (255, 0, 0), 2)

    cv2.imwrite(os.path.join(output_dir, f"{image_name}_overlay.jpg"), overlay_img)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_keypoints.jpg"), keypoints_only_img)

# -------------------------------------------------------------------
# 3. Main — orchestration only
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    image, persons = detect_image_from_frame(args.image, args.model)

    image_name = os.path.splitext(os.path.basename(args.image))[0]

    visualize_keypoints(
        image=image,
        persons=persons,
        output_dir=args.output,
        image_name=image_name
    )

    print(f"[✓] Done. Saved to {args.output}")
