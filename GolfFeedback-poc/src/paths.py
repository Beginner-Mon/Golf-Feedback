from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SRC_DIR = ROOT / "src"
S3_DIR = SRC_DIR / "s3_NAM_model"
