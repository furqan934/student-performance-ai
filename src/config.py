from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

MODEL_DIR = PROJECT_ROOT / "models"

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

MODEL_PATH = CHECKPOINT_DIR / "student_performance_model.pth"

DATASET_PATH = RAW_DATA_DIR / "student-performance.csv"

INPUT_SIZE = 58