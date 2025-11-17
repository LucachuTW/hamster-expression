# Hamster Expression

Modular facial emotion detection stack built from scratch on top of the **9 Facial Expressions You Need** dataset.  
The backend exposes a FastAPI service powered by a custom YOLO-style grid detector, and the frontend is a lightweight web UI that streams camera frames and reacts with themed artwork per detected emotion.

## Project Layout

```
.
├── 9 Facial Expressions you need/    # Dataset (images + YOLO formatted labels)
├── backend/
│   ├── ai/                           # Training code, model, datasets, utils
│   ├── app/                          # FastAPI application
│   └── artifacts/weights/            # Stores exported checkpoints
├── frontend/                         # Standalone UI (HTML/CSS/JS + assets)
└── requirements.txt                  # Python dependencies
```

## Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training the Detector

The detector mimics a YOLOv1 workflow (single bounding box per grid cell, custom loss) and can be trained directly on the provided dataset.

```bash
python backend/ai/train.py \
  --data-dir "9 Facial Expressions you need" \
  --epochs 50 \
  --image-size 256 \
  --lr 1e-4
```

The best checkpoint is stored at `backend/artifacts/weights/emotion_detector.pt` (override with `--weights`).

## FastAPI Backend

Start the inference API after training:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints:

| Method | Path                  | Description                                  |
| ------ | --------------------- | -------------------------------------------- |
| GET    | `/health`             | Health probe                                  |
| POST   | `/api/predict`        | Multipart image upload prediction             |
| POST   | `/api/predict/stream` | Base64 frame predictions for the web client   |

Logs are emitted with timestamps and component names for easy tracing.

## Frontend UI

Serve the static files using any HTTP server (must allow camera permissions):

```bash
python -m http.server 5500 --directory frontend
```

Then open `http://localhost:5500` in a browser. The interface shows:

- Live user camera feed on the left.
- The latest predicted emotion and confidence.
- Emotion-specific artwork on the right (stored under `frontend/assets/emotions`).

The browser streams JPEG frames to `/api/predict/stream` every second; the backend responds with bounding boxes and the dominant emotion without any YOLO dependency.

## Docker Helpers

- `tools/Dockerfile` packages the full stack (PyTorch runtime + project files).  
- `tools/run_docker.sh` rebuilds the image when sources change (Docker caching keeps it fast) and spawns an interactive container with the repo mounted.

```bash
./tools/run_docker.sh
# inside the container, start backend/frontend manually when ready
```

## Development Notes

- All code, logs, and identifiers are in English as requested.
- The backend and frontend run independently (API vs. static site) and communicate purely via HTTP.
- The AI stack is fully custom (data loader, model, loss, training script, predictor, and service integration).
