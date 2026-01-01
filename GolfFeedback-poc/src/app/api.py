from fastapi import APIRouter, UploadFile, File, Form,Query
import base64
import cv2
import shutil
import uuid
import os
import numpy as np
import math

from swing_pipeline import get_event_frames, get_2d_joints, get_metrics
from paths import GP_DIR
PREDICTED_3D_PATH = GP_DIR / "predicted_3d_denorm.npz"
router = APIRouter()

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

@router.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    return_frames: bool = Form(False),
    return_joints: bool = Form(False),
):
    """
    Unified swing analysis API.

    - metrics: ALWAYS returned
    - frames : optional
    - joints : optional
    """

    temp_path = f"/tmp/{uuid.uuid4()}.mp4"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:

        event_frames = get_event_frames(temp_path)

        joints_by_event = get_2d_joints(event_frames)

        metrics_by_event = get_metrics(joints_by_event)

        events = {}

        for idx, (event_name, frame_img) in enumerate(event_frames):
            event_obj = {
                "event": event_name,
                "metrics": metrics_by_event[idx]["metrics"],
            }

            if return_frames:
                event_obj["image"] = encode_image(frame_img)

            if return_joints:
                event_obj["joints"] = joints_by_event[idx]["joints"]

            events[idx] = event_obj

        return {
            "status": "success",
            "events": events,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/analyze/components")
async def analyze_video_components(
    file: UploadFile = File(...)
):
    """
    Legacy API:
    Returns 3 top-level components:
    - event_frames
    - joints
    - metrics
    """

    temp_path = f"/tmp/{uuid.uuid4()}.mp4"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        event_frames = get_event_frames(temp_path)

        event_frames_payload = {
            idx: {
                "event": event_name,
                "image": encode_image(frame_img),
            }
            for idx, (event_name, frame_img) in enumerate(event_frames)
        }

        joints_by_event = get_2d_joints(event_frames)

        joints_payload = {
            idx: {
                "event": data["event"],
                "joints": data["joints"],   # COCO order, length=17
            }
            for idx, data in joints_by_event.items()
        }

        metrics_by_event = get_metrics(joints_by_event)

        metrics_payload = {
            idx: {
                "event": data["event"],
                "metrics": data["metrics"],
            }
            for idx, data in metrics_by_event.items()
        }

        return {
            "status": "success",
            "event_frames": event_frames_payload,
            "joints": joints_payload,
            "metrics": metrics_payload,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/analyze/3d")
async def get_predicted_3d(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=300),
):
    """
    Paginated access to 3D pose prediction.
    Expected NPZ structure:
    positions_3d -> ndarray (T, J, 3)
    """

    if not os.path.exists(PREDICTED_3D_PATH):
        return {
            "status": "error",
            "message": "predicted 3d npz not found"
        }

    try:
        npz = np.load(PREDICTED_3D_PATH)

        if "positions_3d" not in npz:
            raise ValueError("positions_3d key not found")

        pred_3d = npz["positions_3d"]

        # --- Validate array ---
        if not isinstance(pred_3d, np.ndarray):
            raise ValueError("positions_3d is not a numpy array")

        if pred_3d.ndim != 3 or pred_3d.shape[2] != 3:
            raise ValueError(
                f"Invalid shape {pred_3d.shape}, expected (T, J, 3)"
            )

    except Exception as e:
        return {
            "status": "error",
            "message": "Invalid NPZ structure",
            "detail": str(e),
        }

    # --- Pagination ---
    total_frames = pred_3d.shape[0]
    total_pages = math.ceil(total_frames / page_size)

    if page > total_pages:
        return {
            "status": "error",
            "message": "page out of range",
            "total_pages": total_pages,
        }

    start = (page - 1) * page_size
    end = min(start + page_size, total_frames)

    frames_payload = [
        {
            "frame": i,
            "joints_3d": pred_3d[i].tolist(),
        }
        for i in range(start, end)
    ]

    return {
        "status": "success",
        "meta": {
            "joints": pred_3d.shape[1],
            "total_frames": total_frames,
        },
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_frames": total_frames,
            "total_pages": total_pages,
        },
        "data": frames_payload,
    }
