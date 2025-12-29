from fastapi import APIRouter, UploadFile, File, Form
import base64
import cv2
import shutil
import uuid
import os
from swing_pipeline import get_event_frames, get_2d_joints, get_metrics

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
