from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from app.analysis.schemas import ModelPrediction

# Label configuration from hybrid_analysis.py
LABELS: Dict[int, str] = {
    0: "block",
    1: "pass",
    2: "run",
    3: "dribble",
    4: "shoot",
    5: "ball in hand",
    6: "defense",
    7: "pick",
    8: "no_action",
    9: "walk",
}

LABEL_TO_ID = {value: key for key, value in LABELS.items()}


def inference_batch(batch: torch.Tensor) -> torch.Tensor:
    """Prepare a batch of clips for R(2+1)D model inference.

    Converts BGR→RGB, normalizes pixel values to [0, 1] (matching training
    preprocessing in dataset.py VideoToTensor), and permutes to (B, C, T, H, W).

    Args:
        batch: Tensor of shape (B, T, H, W, C) with BGR uint8 pixel values.

    Returns:
        Tensor of shape (B, C, T, H, W) with RGB float32 values in [0, 1].
    """
    # BGR → RGB: reverse the channel dimension
    batch = batch.flip(-1)
    # Normalize to [0, 1] to match training preprocessing (dataset.py L191: frames /= 255)
    batch = batch.float() / 255.0
    # (B, T, H, W, C) → (B, C, T, H, W)
    return batch.permute(0, 4, 1, 2, 3)


def predict_player_clips(
    model: torch.nn.Module,
    player_clips: Dict[int, Sequence[np.ndarray]],
    device: torch.device | None = None,
    batch_size: int = 8,
) -> Dict[int, List[ModelPrediction]]:
    """Run model inference on pre-cropped video windows for all players.

    Args:
        model: Loaded PyTorch R(2+1)D model.
        player_clips: Dictionary mapping player index to list of clip arrays.
        device: Target compute device. Auto-detected if None.
        batch_size: Inference batch size.

    Returns:
        Dictionary mapping player index to list of ModelPrediction objects.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions: Dict[int, List[ModelPrediction]] = {}

    for player, clips in player_clips.items():
        predictions: List[ModelPrediction] = []
        for start in range(0, len(clips), batch_size):
            batch_np = np.asarray(clips[start : start + batch_size])
            batch = inference_batch(torch.FloatTensor(batch_np)).to(device)
            with torch.no_grad():
                outputs = model(batch)
                softmax = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            for row in softmax:
                action_id = int(np.argmax(row))
                probabilities = {
                    LABELS[idx]: float(prob)
                    for idx, prob in enumerate(row[: len(LABELS)])
                }
                predictions.append(
                    ModelPrediction(
                        action_id=action_id,
                        action=LABELS[action_id],
                        confidence=float(row[action_id]),
                        probabilities=probabilities,
                    )
                )
        all_predictions[player] = predictions

    return all_predictions
