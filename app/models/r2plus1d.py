from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

import sys
import os
# Ensure project root is in sys.path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.checkpoints import load_weights
from app.config import Settings


def build_r2plus1d_model(
    settings: Settings,
    device: torch.device | None = None,
) -> nn.Module:
    """Build and return a R(2+1)D-18 model with checkpoint weights loaded.

    Args:
        settings: Application settings containing model_path, base_model_name,
            start_epoch, lr, and num_classes.
        device: Target device. Auto-detected if None.

    Returns:
        The loaded model in eval mode on the target device.
    """
    from easydict import EasyDict

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.video.r2plus1d_18(weights=None, progress=False)
    model.fc = nn.Linear(model.fc.in_features, settings.num_classes, bias=True)

    checkpoint_args = EasyDict(
        {
            "base_model_name": settings.base_model_name,
            "start_epoch": settings.start_epoch,
            "lr": settings.lr,
            "model_path": settings.model_path,
        }
    )
    model = load_weights(model, checkpoint_args)
    model = model.to(device)
    model.eval()
    return model
