from __future__ import annotations

from typing import Annotated

import torch
from fastapi import Depends

from app.config import Settings, get_settings
from app.models.r2plus1d import build_r2plus1d_model
from app.analysis.service import AnalysisService

# We'll use a global var to hold the loaded model created during app lifespan
_GLOBAL_MODEL: torch.nn.Module | None = None
_GLOBAL_DEVICE: torch.device | None = None


def init_globals(model: torch.nn.Module, device: torch.device) -> None:
    """Initialize global dependencies during FastAPI lifespan."""
    global _GLOBAL_MODEL, _GLOBAL_DEVICE
    _GLOBAL_MODEL = model
    _GLOBAL_DEVICE = device


def get_device() -> torch.device:
    if _GLOBAL_DEVICE is None:
        raise RuntimeError("Global device not initialized. Check app lifespan.")
    return _GLOBAL_DEVICE


def get_model() -> torch.nn.Module:
    if _GLOBAL_MODEL is None:
        raise RuntimeError("Global model not initialized. Check app lifespan.")
    return _GLOBAL_MODEL


def get_analysis_service(
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[torch.nn.Module, Depends(get_model)],
    device: Annotated[torch.device, Depends(get_device)],
) -> AnalysisService:
    return AnalysisService(settings=settings, model=model, device=device)
