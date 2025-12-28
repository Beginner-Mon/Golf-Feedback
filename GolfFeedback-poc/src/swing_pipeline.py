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


# ============================================================
# Core pipeline
# ============================================================
def analyze_swing(video_path: str) -> dict:
    """
    Core swing analysis pipeline.
    Reused by CLI and FastAPI.
    """

    # ------------------------------
    # Load NAM
    # ------------------------------
    config = load_config_from_dicts("s3_NAM_model/output/BS/0/hparams.yaml")
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
            "s3_NAM_model/output/BS/0/checkpoints/epoch=09-val_loss=1270.3485.ckpt"
        )["state_dict"]
    )

    ideal_values = get_ideal_values(litmodel.model, dataset)

    # ------------------------------
    # Video + YOLO
    # ------------------------------
    event_frames = process_video(video_path)
    yolo_model = YOLO(YOLO_MODEL_PATH)

    k_address = None
    results_by_event = {}

    for idx, (event_name, frame_img) in enumerate(event_frames):

        results = yolo_model(frame_img)

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            for i, box in enumerate(r.boxes):
                if yolo_model.names[int(box.cls[0])] != "person":
                    continue

                kpts = r.keypoints.xy[i]
                keypoints = [(float(x), float(y)) for x, y in kpts]

                if k_address is None:
                    k_address = keypoints

                metrics = calculate_all_metrics(keypoints, k_address)
                needed = METRIC_FILTER[idx]

                event_feedback = {}

                for metric_name in needed:
                    if metric_name not in metrics:
                        continue

                    ideal_key = f"{idx}-{metric_name}"
                    if ideal_key not in ideal_values:
                        continue

                    current = metrics[metric_name]
                    ideal = ideal_values[ideal_key]

                    event_feedback[metric_name] = {
                        "current": round(current, 2),
                        "ideal": round(ideal, 2),
                        "delta": round(current - ideal, 2),
                        "feedback": generate_feedback(current, ideal),
                    }

                results_by_event[idx] = {
                    "event": event_name,
                    "metrics": event_feedback,
                }

    return results_by_event


# ============================================================
# Local testing
# ============================================================
if __name__ == "__main__":
    output = analyze_swing("../du.mp4")
    for idx, data in output.items():
        print(f"\nEvent {idx} - {data['event']}")
        for m, v in data["metrics"].items():
            print(m, v)
