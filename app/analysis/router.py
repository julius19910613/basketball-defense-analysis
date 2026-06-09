from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings
from app.analysis.schemas import AnalysisRequest, AnalysisResponse
from app.analysis.service import AnalysisService

# We must import from the DI module where we inject settings and model
from app.dependencies import get_settings, get_analysis_service

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/run", response_model=AnalysisResponse)
def run_analysis(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
    settings: Settings = Depends(get_settings),
) -> AnalysisResponse:
    """Run a hybrid video analysis blockingly (CPU-bound)."""
    video_path = request.video_path
    real_cwd = os.path.realpath(os.getcwd())
    real_video_path = os.path.realpath(video_path)
    if not real_video_path.startswith(real_cwd):
        raise HTTPException(status_code=400, detail="Access denied: Invalid video path.")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
    
    if request.boxes_file and not os.path.exists(request.boxes_file):
        raise HTTPException(status_code=400, detail=f"Boxes file not found: {request.boxes_file}")

    try:
        return service.run_analysis(request)
    except Exception as e:
        import logging
        logging.getLogger("app.analysis.router").error("Analysis failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
