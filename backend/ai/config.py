from pathlib import Path

IMAGE_SIZE = 256
GRID_SIZE = 7
NUM_BOXES = 1
NUM_CLASSES = 9
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 30
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
ARTIFACTS_DIR = Path("backend/artifacts")
WEIGHTS_PATH = ARTIFACTS_DIR / "weights" / "emotion_detector.pt"
LABELS = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "natural",
    "sad",
    "sleepy",
    "surprised",
]
DATASET_DIR = Path("9 Facial Expressions you need")
