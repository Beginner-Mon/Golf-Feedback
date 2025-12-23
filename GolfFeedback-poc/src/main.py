# main.py
from s1_image_sequencing.test_video import process_video
from s2_2d_joints.metrics_calculate import calculate_all_metrics

from ultralytics import YOLO

VIDEO_PATH = "../du.mp4"
YOLO_MODEL_PATH = "../models/yolov8n-pose.pt"
METRIC_FILTER = {
    0: ["LOWER-ANGLE", "SHOULDER-ANGLE", "SPINE-ANGLE", "STANCE-RATIO", "UPPER-TILT"],
    1: ["HEAD-LOC", "HIP-LINE", "HIP-ROTATION", "HIP-SHIFTED",
        "LEFT-ARM-ANGLE", "RIGHT-ARM-ANGLE", "SHOULDER-ANGLE",
        "SHOULDER-LOC", "SPINE-ANGLE"],
    2: ["HEAD-LOC", "HIP-ANGLE", "HIP-LINE", "HIP-ROTATION",
        "HIP-SHIFTED", "LEFT-ARM-ANGLE", "SHOULDER-ANGLE",
        "SHOULDER-LOC", "UPPER-TILT"],
    3: ["HEAD-LOC", "HIP-ANGLE", "HIP-LINE", "HIP-ROTATION",
        "HIP-SHIFTED", "LEFT-LEG-ANGLE", "RIGHT-ARM-ANGLE",
        "RIGHT-DISTANCE", "RIGHT-LEG-ANGLE", "SHOULDER-ANGLE",
        "SHOULDER-LOC"],
    4: ["HEAD-LOC", "HIP-ANGLE", "HIP-HANGING-BACK", "HIP-LINE",
        "HIP-ROTATION", "HIP-SHIFTED", "RIGHT-ARM-ANGLE",
        "RIGHT-ARMPIT-ANGLE", "SHOULDER-HANGING-BACK", "SPINE-ANGLE"],
    5: ["HEAD-LOC", "HIP-HANGING-BACK", "HIP-LINE", "HIP-SHIFTED",
        "LEFT-ARM-ANGLE", "LEFT-LEG-ANGLE", "RIGHT-ARM-ANGLE",
        "SHOULDER-ANGLE", "SHOULDER-HANGING-BACK", "SPINE-ANGLE"],
    6: ["HEAD-LOC", "HIP-ANGLE", "HIP-LINE", "HIP-SHIFTED",
        "LEFT-ARM-ANGLE", "RIGHT-LEG-ANGLE", "SHOULDER-ANGLE",
        "SPINE-ANGLE", "WEIGHT-SHIFT"],
    7: ["FINISH-ANGLE", "HIP-ANGLE", "HIP-LINE", "HIP-SHIFTED",
        "SHOULDER-ANGLE", "SPINE-ANGLE"],
}

# Step 1: Extract event frames
event_frames = process_video(VIDEO_PATH)  # list of (event_name, frame_image)

# Step 2: Load YOLO once
yolo_model = YOLO(YOLO_MODEL_PATH)

k_address = None   # Will store keypoints of the first frame (Address)

print("\n=== Processing Event Frames ===")

for idx, (event_name, frame_img) in enumerate(event_frames):
    print(f"\nEvent: {event_name}")

    # Run pose estimation
    results = yolo_model(frame_img)

    for r in results:
        boxes = r.boxes
        keypoints_list = r.keypoints

        if boxes is None or keypoints_list is None:
            continue

        # Loop through detections
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if yolo_model.names[cls] != "person":
                continue

            kpts = keypoints_list.xy[i]  # (17,2)
            keypoints = [(float(x), float(y)) for x, y in kpts]

            # ------------------------------
            # 1. Save Address keypoints
            # ------------------------------
            if k_address is None:
                k_address = keypoints
                print("Address pose captured.")
                

            # ------------------------------
            # 2. Compute metrics using Address
            # ------------------------------
            metrics = calculate_all_metrics(keypoints, k_address)

            # ------------------------------
            # 3. Print metrics
            # ------------------------------
            needed = METRIC_FILTER[idx]

            print("\nFiltered Metrics:")
            for name, value in metrics.items():
                if name in needed:
                    print(f"{idx}-{name}: {value:.2f}")

