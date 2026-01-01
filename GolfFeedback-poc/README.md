# GolfFeedback-POC

A comprehensive golf swing analysis pipeline combining computer vision, pose detection, and machine learning models to provide real-time feedback on golf swings.

## Project Overview

This project processes golf swing videos through multiple stages to extract key metrics, detect events, and provide AI-powered feedback. It consists of both a backend API and a frontend interface.

## Folder Structure

### `/data`
Contains datasets used for training and testing:
- **CaddieSet.csv** - Caddie-related dataset
- **dtl_data.csv** - DTL (Down The Line) video analysis data
- **faceon_data.csv** - Face-on video analysis data

### `/models`
Pre-trained ML models:
- **yolov8n-pose.pt** - YOLOv8 Nano pose detection model for identifying golfer joints and body keypoints

### `/src` - Source Code

#### **`app/`** - FastAPI Backend
The main backend API application that serves the golf analysis pipeline:
- **main.py** - FastAPI application setup with CORS middleware configured for frontend communication (http://localhost:3000)
- **api.py** - API routes and endpoints for swing analysis

#### **`golfpose/`** - Pose Detection Module
Core golf pose detection pipeline using MMPose and MMDetection frameworks:
- **golfpose_3d.py** - 3D pose reconstruction
- **make_golfpose_npz.py** / **make_golfpose_2d_npz.py** - Data preprocessing for pose models
- **step1.py** / **step2.py** - Multi-stage pose detection pipeline
- **visualize_keypoints.py** - Visualization tools for detected keypoints
- `common/` - Shared utilities including custom datasets, camera calibration, model definitions
- `configs/` - Configuration files for MMDetection and MMPose models
- `work_dirs/` - Pre-trained model weights and checkpoints

#### **`s1_image_sequencing/`** - Video Frame Processing
Sequences and analyzes video frames using deep learning:
- **preprocess_video.py** - Extracts frames from video input
- **model.py** / **MobileNetV2.py** - Neural network for frame classification
- **dataloader.py** - Data loading utilities
- **test_video.py** - Testing module for video analysis
- `event_frames/` - Output directory for key event frames

#### **`s2_2d_joints/`** - 2D Joint Analysis
Processes 2D joint detections and metrics:
- **bounding_box.py** - Bounding box extraction and processing
- **metrics_calculate.py** - Calculates golf swing metrics from joint positions

#### **`s3_NAM_model/`** - Neural Additive Models
Interpretable ML models for performance prediction:
- **main.py** - NAM model training and inference
- **process_metrics.py** - Metric processing pipeline
- `nam/` - Custom NAM implementation
- `output/` - Model outputs and results (BS, DA, etc.)

#### **Root Source Files**
- **paths.py** - Central path management (see below)
- **swing_pipeline.py** - Main orchestration of the entire analysis pipeline
- **caddie_preprocess.py** - Preprocessing for Caddie dataset

## paths.py Overview

`paths.py` is a centralized path configuration module that defines all directory paths used throughout the project:

```python
ROOT = Path(__file__).resolve().parents[1]  # GolfFeedback-poc root directory

DATA_DIR = ROOT / "data"       # Path to datasets
MODELS_DIR = ROOT / "models"   # Path to pre-trained models
SRC_DIR = ROOT / "src"         # Path to source code
S3_DIR = SRC_DIR / "s3_NAM_model"  # Path to NAM model module
```

**Benefits:**
- Single source of truth for all paths
- Avoids hardcoded path strings throughout the codebase
- Makes code more portable and maintainable
- Enables easy path changes without modifying multiple files

## Getting Started

### Backend Setup

1. **Navigate to the source directory:**
   ```bash
   cd src
   ```

2. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Run the development server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   - The API will be available at `http://localhost:8000`
   - Automatic reload enabled for development
   - Interactive API docs available at `http://localhost:8000/docs`

### Frontend Integration

The frontend is configured to communicate with the backend at `http://localhost:3000`. Make sure the Next.js frontend is also running during development.

## Environment Setup

A virtual environment (`.venv`) is recommended. Activate it before running the backend:

**Windows (PowerShell):**
```bash
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

## Requirements

See `requirements.txt` for all Python package dependencies. Key packages include:
- FastAPI - Web API framework
- MMPose / MMDetection - Pose and object detection frameworks
- YOLOv8 - Object detection
- TensorFlow / PyTorch - Deep learning frameworks
- OpenCV - Computer vision
