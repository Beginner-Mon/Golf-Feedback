from fastapi import APIRouter, UploadFile, File
import shutil
import uuid
import os

from swing_pipeline import analyze_swing

router = APIRouter()


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_name = f"/tmp/{uuid.uuid4()}.mp4"

    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = analyze_swing(temp_name)

    os.remove(temp_name)

    return {
        "status": "success",
        "results": results
    }
