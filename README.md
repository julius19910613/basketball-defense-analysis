# Basketball Defense Analysis

Hybrid basketball video analysis with player tracking, R(2+1)D action
classification, lightweight motion features, and optional Ollama VLM review.

The project started from a SpaceJam basketball action recognition baseline and
now includes a FastAPI service for asynchronous video analysis plus a
Mac-friendly training script for fine-tuning the R(2+1)D model.

## What It Does

- Tracks players in a basketball video with YOLOv8 + ByteTrack or OpenCV legacy
  trackers.
- Splits each player track into short 16-frame clips.
- Classifies each clip into 10 basketball action labels with R(2+1)D.
- Computes simple trajectory and box-motion features.
- Optionally asks a local Ollama VLM to review low-confidence clips.
- Fuses model, motion, and VLM evidence into a final action decision.
- Writes JSON analysis results and optional annotated videos.

Supported action labels:

```text
0 block
1 pass
2 run
3 dribble
4 shoot
5 ball in hand
6 defense
7 pick
8 no_action
9 walk
```

## Current Architecture

```text
app/main.py                 FastAPI application
app/analysis/router.py      Async API endpoints
app/analysis/service.py     End-to-end analysis orchestration
app/analysis/tracking.py    YOLO/ByteTrack and OpenCV tracking
app/analysis/inference.py   R(2+1)D clip inference
app/analysis/vlm.py         Ollama VLM verification client
app/analysis/fusion.py      Final decision and temporal smoothing
app/video/writer.py         Annotated video output
train_mac.py                Mac/CPU/MPS-friendly training entrypoint
dataset.py                  SpaceJam video dataset loader
tests/                      Offline regression tests
```

There is also a legacy standalone script, `hybrid_analysis.py`, but new work
should prefer the structured API under `app/`.

## Data And Model Files

Large runtime artifacts are intentionally not tracked by git.

Expected local layout:

```text
dataset/
  annotation_dict.json
  augmented_annotation_dict.json
  examples/
    *.mp4

model_checkpoints/
  r2plus1d_augmented-2/
  r2plus1d_v3/

analysis_outputs/
output_videos/
histories/
```

Notes:

- `dataset/`, `model_checkpoints/`, `analysis_outputs/`, `output_videos/`, and
  local histories are runtime data.
- `train_mac.py` can train with only `dataset/annotation_dict.json`; if
  `dataset/augmented_annotation_dict.json` is missing, it falls back to the
  original annotations.
- The API model loader uses `BASKETBALL_MODEL_PATH`,
  `BASKETBALL_START_EPOCH`, `BASKETBALL_LR`, and
  `BASKETBALL_BASE_MODEL_NAME` to locate checkpoint weights.

## Setup

Use the existing virtual environment when working in this repository:

```bash
source ./venv/bin/activate
```

For a fresh environment, install the project dependencies needed by the current
API and training paths:

```bash
python -m pip install torch torchvision opencv-contrib-python fastapi uvicorn \
  pydantic-settings ultralytics scikit-learn tqdm easydict numpy
```

The historical `requirements.txt` is from the original upstream project and is
not yet a complete lockfile for the current FastAPI + YOLO + Ollama workflow.

Copy the sample environment file when you want to override defaults:

```bash
cp .env.example .env
```

Important environment variables use the `BASKETBALL_` prefix:

```text
BASKETBALL_MODEL_PATH=model_checkpoints/r2plus1d_augmented-2/
BASKETBALL_TRACKER_TYPE=YOLO
BASKETBALL_VLM_MODE=low-confidence
BASKETBALL_OLLAMA_MODEL=qwen3-vl:4b
BASKETBALL_OLLAMA_HOST=http://127.0.0.1:11434
BASKETBALL_OUTPUT_DIR=analysis_outputs
BASKETBALL_VIDEO_OUTPUT_DIR=output_videos
```

## Start The API

```bash
./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

Interactive API docs:

```text
http://127.0.0.1:8765/docs
```

## Run A Video Analysis

Start an asynchronous analysis task:

```bash
curl -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "low-confidence",
    "max_frames": 180,
    "generate_video": true,
    "tracker_conf_thres": 0.3,
    "tracker_iou_thres": 0.6,
    "tracker_min_appear_ratio": 0.02,
    "tracker_min_appear_abs": 5
  }'
```

Poll the returned task id:

```bash
curl http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

Useful request options:

- `vlm_mode`: `off`, `low-confidence`, or `always`
- `boxes_file`: optional JSON boxes file for OpenCV tracker initialization
- `max_frames`: cap frames for faster smoke tests
- `generate_video`: write an annotated MP4 when true
- `vid_stride`: lower values produce more clips
- `low_confidence` and `high_confidence`: override fusion thresholds

The service writes JSON reports under `analysis_outputs/` and annotated videos
under `output_videos/`. Static files are served from:

```text
/static/outputs
/static/videos
```

## Ollama VLM Review

The VLM is optional. The non-LLM R(2+1)D model remains the primary classifier;
the VLM is used as a bounded reviewer for uncertain clips.

To enable VLM review locally:

```bash
ollama serve
ollama pull qwen3-vl:4b
```

Then run API requests with:

```json
{"vlm_mode": "low-confidence"}
```

Fusion behavior:

- If VLM is off or unavailable, keep the R(2+1)D prediction.
- If VLM agrees, raise confidence conservatively.
- If R(2+1)D confidence is high, keep it during conflicts and mark review.
- If R(2+1)D confidence is low and VLM is confident, allow VLM override.
- Isolated low-confidence temporal outliers can be smoothed to neighboring
  labels.

## Training

`train_mac.py` is the current training entrypoint for local CPU/MPS/CUDA runs.
It is designed to survive long-running local jobs:

- auto-selects MPS, CUDA, or CPU unless `--device` is provided
- uses small default batches for 16 GB Mac setups
- skips corrupted/unreadable dataset samples instead of aborting the epoch
- supports resume from `latest.pt` or `best.pt`
- saves `best.pt` only when validation accuracy improves
- supports `--save-best-only` to avoid epoch checkpoints and `latest.pt`

Example:

```bash
./venv/bin/python train_mac.py \
  --device mps \
  --batch-size 2 \
  --epochs 20 \
  --lr 1e-4 \
  --layers layer3 layer4 fc \
  --annotation-path dataset/annotation_dict.json \
  --augmented-path dataset/augmented_annotation_dict.json \
  --video-dir dataset/examples/ \
  --augmented-dir dataset/examples/ \
  --model-dir model_checkpoints/r2plus1d_v3/ \
  --history-path histories/history_r2plus1d_v3.txt \
  --num-workers 0
```

Resume:

```bash
./venv/bin/python train_mac.py \
  --resume model_checkpoints/r2plus1d_v3/latest.pt \
  --model-dir model_checkpoints/r2plus1d_v3/ \
  --history-path histories/history_r2plus1d_v3.txt
```

Check a training history file:

```bash
./venv/bin/python scripts/check_training.py \
  --history-path histories/history_r2plus1d_v3.txt
```

## Testing

The regression tests are offline and mock the expensive model/video paths where
possible. They do not require SpaceJam data, a live Ollama server, or real
checkpoint files.

```bash
./venv/bin/python -m pytest tests -q
```

Current expected result:

```text
37 passed
```

## Safety Notes

- API video paths are checked so requests cannot read outside the repository
  root through path traversal.
- Analysis tasks run asynchronously and can be polled through the status
  endpoint.
- `pytest.ini` limits test discovery to `tests/`, so manual experiment scripts
  are not collected accidentally.
- Runtime outputs and datasets are ignored by git to avoid committing large or
  machine-specific files.

## Credits

The action recognition baseline and dataset context come from the SpaceJam
basketball action dataset by Simone Francia and the original R(2+1)D
basketball-action-recognition work. This repository extends that baseline with
a local service, YOLO/ByteTrack tracking, Ollama VLM verification, checkpoint
resilience, and focused regression tests.
