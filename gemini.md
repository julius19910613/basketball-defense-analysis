# Gemini Development Constraints for Basketball-Action-Recognition

This file defines project-specific coding standards, architecture constraints, and guidelines for AI developers working on this repository.

## 1. Project Context & Stack
* **Domain**: Spatio-Temporal classification of basketball actions (e.g., Shoot, Dribble, Defence, Pass, Block, Run, Walk, Pick, etc.) from video inputs.
* **Tech Stack**:
  * **Core DL Framework**: PyTorch (`torch`, `torch.nn`, `torch.optim`) & Torchvision (`torchvision.models.video`).
  * **Video Processing & Tracking**: OpenCV (`cv2`), `vidaug` (for video data augmentation).
  * **Data & Numerical Ops**: NumPy, Pandas, Scikit-Learn (specifically `confusion_matrix`, `metrics`).
  * **UI/Visualization**: Matplotlib, Jupyter Notebooks (for error analysis).

## 2. Coding & Architectural Constraints

### PyTorch & Model Guidelines
* **Device Agnosticism**: Always write device-agnostic code. Auto-detect `cuda` first, then CPU.
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
* **Memory Management**: 
  * 3D-CNN training uses high GPU memory. Maintain small batch sizes (default: `batch_size = 8`).
  * Always use `torch.no_grad()` (or `torch.set_grad_enabled(False)`) during inference, validation, and testing to prevent CUDA Out Of Memory (OOM) errors.
* **Fine-Tuning**: When working with pretrained models (like `r2plus1d_18`), freeze backbone weights first and only unfreeze target layers (e.g., `layer3`, `layer4`, `fc`) unless full training is explicitly requested.

### Data Pipelines (`dataset.py`)
* **Video Frame Constraints**: Clips are strictly constrained to 16 frames. Asserts must be checked when converting video inputs to tensors/arrays.
* **OpenCV Safety**: Ensure proper release of OpenCV resources (`cap.release()`) in all video reading pipelines to prevent memory leaks and file lock errors.
* **Data Consistency**: Data splitting must use fixed random seeds (e.g., `torch.Generator().manual_seed(1)`) to ensure reproducible validation splits.

### Output & Metrics (`utils/`)
* **Metrics**: Evaluation should report Accuracy, F1-Score, Precision, and Recall using `utils/metrics.py`.
* **Checkpointing**: Training runs must implement checkpointing (epoch-based) and log history using the helpers in `utils/checkpoints.py`. Do not write custom logging logic directly in `train.py`.

## 3. Formatting & Documentation Rules
* **Docstrings**: All new classes, functions, and datasets must contain detailed Google-style or Sphinx-style docstrings describing input shapes, output shapes, and behavior.
* **Code Clarity**: Retain inline comments and respect Python PEP 8 formatting standards. Avoid placeholder code or Mock objects.
