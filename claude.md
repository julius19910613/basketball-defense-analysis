# Claude Development Constraints for Basketball-Defense-Analysis

This file defines project-specific coding standards, architecture constraints, and guidelines for Claude-based AI developers working on this repository.

## 1. Project Context & Stack

* **Domain**: Spatio-Temporal classification of basketball actions (Shoot, Dribble, Defence, Pass, Block, Run, Walk, Pick, etc.) from video inputs, with hybrid verification via local VLM (Ollama).
* **Tech Stack**:
  * **Core DL Framework**: PyTorch (`torch`, `torch.nn`, `torch.optim`) & Torchvision (`torchvision.models.video`).
  * **Video Processing & Tracking**: OpenCV (`cv2`), `vidaug` (for video data augmentation).
  * **API Framework**: FastAPI with Uvicorn ASGI server.
  * **Data Validation**: Pydantic v2 models and `pydantic-settings` for configuration.
  * **Data & Numerical Ops**: NumPy, Pandas, Scikit-Learn.
  * **UI/Visualization**: Matplotlib, Jupyter Notebooks (for error analysis).

## 2. Architecture Constraints

### FastAPI Service (`app/`)
* **Domain-Driven Layout**: Code under `app/` is organized by domain (`analysis/`, `models/`, `video/`), not by layer type. Each domain folder contains its own `router.py`, `schemas.py`, `service.py`, and supporting modules.
* **Layered Responsibility**:
  * **Routers** (`router.py`): Handle HTTP only — request parsing, response serialization, dependency injection. Never contain business logic.
  * **Services** (`service.py`): Orchestrate business logic, call domain-specific modules. This is the only layer that coordinates multiple sub-modules.
  * **Domain modules** (`tracking.py`, `inference.py`, `fusion.py`, etc.): Pure functions and classes with no HTTP awareness. Must be testable in isolation.
* **Dependency Injection**: Use FastAPI's `Depends()` for all shared resources (settings, device, model). Never import global singletons directly in routers.
* **Schemas**: All API request/response bodies must be Pydantic v2 `BaseModel` subclasses defined in `schemas.py`. Never return raw dicts from endpoints.
* **Configuration**: All tuneable parameters live in `app/config.py` as a `pydantic-settings` `BaseSettings` class, loaded from environment variables or `.env` files.

### Legacy Scripts (project root)
* **Do Not Modify** `main.py`, `train.py`, `inference.py`, `dataset.py`, `augment_videos.py`, `C3D.py` unless explicitly requested. These are training/demo utilities outside the API surface.
* **Shared Utilities**: `utils/checkpoints.py` and `utils/metrics.py` are shared by both legacy scripts and the FastAPI app. Changes here must not break either consumer.

## 3. PyTorch & Model Guidelines

* **Device Agnosticism**: Always write device-agnostic code. Auto-detect `cuda` first, then `mps`, then CPU.
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
* **Memory Management**:
  * 3D-CNN training uses high GPU memory. Maintain small batch sizes (default: `batch_size = 8`).
  * Always use `torch.no_grad()` during inference to prevent OOM errors.
* **Fine-Tuning**: When working with pretrained models (like `r2plus1d_18`), freeze backbone weights first and only unfreeze target layers (e.g., `layer3`, `layer4`, `fc`) unless full training is explicitly requested.
* **Model Lifecycle**: The R(2+1)D model is loaded once at app startup via FastAPI lifespan and shared across requests. Never reload the model per-request.

## 4. Data Pipelines

* **Video Frame Constraints**: Clips are strictly constrained to 16 frames. Asserts must be checked when converting video inputs.
* **OpenCV Safety**: Ensure proper release of OpenCV resources (`cap.release()`) in all video reading pipelines. Use try/finally blocks.
* **Data Consistency**: Data splitting must use fixed random seeds (`torch.Generator().manual_seed(1)`) for reproducible validation splits.

## 5. API Design Rules

* **Versioned Endpoints**: All API routes are prefixed with `/api/v1/`.
* **Error Handling**: Use FastAPI `HTTPException` with appropriate status codes. Never let unhandled exceptions leak to the client.
* **Sync for CPU-bound**: Video tracking and model inference are CPU-bound operations. Use synchronous `def` endpoints — FastAPI will run them in a thread pool automatically.
* **File Responses**: Use `FileResponse` for serving generated videos and JSON files.

## 6. Formatting & Documentation Rules

* **Docstrings**: All new classes, functions, and modules must contain Google-style docstrings describing input shapes, output shapes, and behavior.
* **Type Hints**: All function signatures must include type hints. Use `from __future__ import annotations` for forward references.
* **Code Clarity**: Retain inline comments and respect Python PEP 8 formatting. Avoid placeholder code or Mock objects in production code.

## 7. Testing

* **Test Location**: All tests live in `tests/` at the project root.
* **Offline Tests**: Unit tests must not require the SpaceJam dataset, a live Ollama server, or GPU. Use synthetic data and mocked models.
* **API Tests**: Use `fastapi.testclient.TestClient` for synchronous endpoint tests.
