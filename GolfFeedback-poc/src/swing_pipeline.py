import sys
import os
import torch
from ultralytics import YOLO
from s1_image_sequencing.test_video import process_video
from s2_2d_joints.metrics_calculate import calculate_all_metrics
from s3_NAM_model.nam.models import NAM, get_num_units
from s3_NAM_model.nam.trainer import LitNAM
from s3_NAM_model.nam.data import NAMDataset
from s3_NAM_model.process_metrics import get_ideal_values, load_config_from_dicts
from paths import MODELS_DIR, DATA_DIR, S3_DIR

YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-pose.pt"

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
COCO_KEYPOINTS = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# ============================================================
# Feedback logic (NOW INLINE)
# ============================================================
def generate_feedback(current: float, ideal: float, tol_ratio: float = 0.05) -> str:
    """
    Compare current metric to NAM-ideal value.
    """
    if ideal == 0:
        return "NO_IDEAL"

    delta = current - ideal
    tol = abs(ideal) * tol_ratio

    if abs(delta) <= tol:
        return "GOOD"
    elif delta > 0:
        return "TOO_HIGH"
    else:
        return "TOO_LOW"



def get_event_frames(video_path: str):
    """
    Stage 1:
    Input  : video path
    Output : list[(event_name, frame_img)]
    """
    return process_video(video_path)


def get_2d_joints(event_frames):
    """
    Returns joints in strict COCO order:
    joints[event_idx]["joints"] -> List[[x, y]] length = 17
    """

    yolo_model = YOLO(YOLO_MODEL_PATH)

    joints_by_event = {}
    k_address = None

    for idx, (event_name, frame_img) in enumerate(event_frames):

        results = yolo_model(frame_img)

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            for i, box in enumerate(r.boxes):
                if yolo_model.names[int(box.cls[0])] != "person":
                    continue

                # YOLOv8 already outputs COCO-ordered keypoints
                kpts = r.keypoints.xy[i].tolist()  # shape (17, 2)

                # Safety: enforce length
                if len(kpts) != 17:
                    continue

                joints = [
                    [float(x), float(y)] for x, y in kpts
                ]

                if k_address is None:
                    k_address = joints

                joints_by_event[idx] = {
                    "event": event_name,
                    "joints": joints,        # COCO order
                    "k_address": k_address,  # internal use only
                }

    return joints_by_event


def get_metrics(joints_by_event):
    """
    Stage 3:
    Input  : joints_by_event
    Output : metrics_by_event
    """

    # ------------------------------
    # Load NAM
    # ------------------------------
    config = load_config_from_dicts(S3_DIR / "output/BS/0/hparams.yaml")
    config.data_path = "s3_NAM_model/faceon_cleaned.csv"

    dataset = NAMDataset(
        config,
        data_path=config.data_path,
        features_columns=config.features_columns,
        targets_column=config.targets_column,
    )

    model = NAM(
        config=config,
        name=config.experiment_name,
        num_inputs=len(dataset[0][0]),
        num_units=get_num_units(config, dataset.features),
    )

    litmodel = LitNAM(config, model)
    litmodel.load_state_dict(
        torch.load(
            S3_DIR / "output/BS/0/checkpoints/epoch=09-val_loss=1270.3485.ckpt"
        )["state_dict"]
    )

    ideal_values = get_ideal_values(litmodel.model, dataset)

    # ------------------------------
    # Compute metrics
    # ------------------------------
    metrics_by_event = {}

    for idx, data in joints_by_event.items():
        joints = data["joints"]
        k_address = data["k_address"]
        event_name = data["event"]

        metrics = calculate_all_metrics(joints, k_address)
        needed = METRIC_FILTER[idx]

        event_metrics = {}

        for metric_name in needed:
            if metric_name not in metrics:
                continue

            ideal_key = f"{idx}-{metric_name}"
            if ideal_key not in ideal_values:
                continue

            current = metrics[metric_name]
            ideal = ideal_values[ideal_key]

            event_metrics[metric_name] = {
                "current": round(current, 2),
                "ideal": round(ideal, 2),
                "delta": round(current - ideal, 2),
                "feedback": generate_feedback(current, ideal),
            }

        metrics_by_event[idx] = {
            "event": event_name,
            "metrics": event_metrics,
        }

    return metrics_by_event

def main():
    video_path = DATA_DIR / "du.mp4"

    # ------------------------------
    # Stage 1: Event frames
    # ------------------------------
    event_frames = get_event_frames(video_path)
    print(f"[Stage 1] Extracted {len(event_frames)} event frames")

    # ------------------------------
    # Stage 2: 2D joints
    # ------------------------------
    joints_by_event = get_2d_joints(event_frames)
    print(f"[Stage 2] Detected joints for {len(joints_by_event)} events")

    # ------------------------------
    # Stage 3: Metrics + feedback
    # ------------------------------
    metrics_by_event = get_metrics(joints_by_event)
    print(f"[Stage 3] Computed metrics for {len(metrics_by_event)} events")

    # ------------------------------
    # Print results
    # ------------------------------
    for idx, data in metrics_by_event.items():
        print(f"\nEvent {idx} - {data['event']}")
        for metric_name, values in data["metrics"].items():
            print(
                f"  {metric_name}: "
                f"current={values['current']}, "
                f"ideal={values['ideal']}, "
                f"delta={values['delta']}, "
                f"feedback={values['feedback']}"
            )


if __name__ == "__main__":
    main()
