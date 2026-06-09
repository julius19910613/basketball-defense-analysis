from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Size2D(BaseModel):
    width: int
    height: int


class ModelPrediction(BaseModel):
    action_id: int
    action: str
    confidence: float
    probabilities: Dict[str, float]


class MotionFeatures(BaseModel):
    avg_center_speed: float
    max_center_speed: float
    avg_box_area: float
    area_change_ratio: float


class VLMDecisionResponse(BaseModel):
    action: Optional[str]
    confidence: float
    reason: str
    visible_ball: Optional[bool]
    needs_review: bool
    raw_response: str
    available: bool


class FinalDecisionResponse(BaseModel):
    action_id: int
    action: str
    confidence: float
    source: str
    needs_review: bool
    reason: str


class AnalysisRecordResponse(BaseModel):
    player: int
    clip_index: int
    start_frame: int
    end_frame: int
    r2plus1d: ModelPrediction
    motion: MotionFeatures
    vlm: Optional[VLMDecisionResponse]
    final: FinalDecisionResponse


class AnalysisSummaryResponse(BaseModel):
    clip_count: int
    action_counts: Dict[str, int]
    needs_review_count: int
    source_counts: Dict[str, int]


class AnalysisRequest(BaseModel):
    """Payload for running a new analysis pipeline."""
    video_path: str = Field(..., description="Path to the video file to analyze.")
    vlm_mode: str = Field(
        default="low-confidence", 
        description="VLM verification mode: off | low-confidence | always"
    )
    boxes_file: Optional[str] = Field(
        default=None, 
        description="Path to an optional JSON file containing initial bounding boxes."
    )
    max_frames: Optional[int] = Field(
        default=None, 
        description="Optional maximum number of frames to process."
    )
    generate_video: bool = Field(
        default=True,
        description="If True, generates and saves an annotated output video."
    )


class AnalysisResponse(BaseModel):
    """Full payload returned from a successful analysis."""
    video: str
    created_at_unix: float
    runtime_seconds: float
    frame_size: Size2D
    seq_length: int
    vid_stride: int
    vlm_mode: str
    ollama_model: Optional[str]
    records: list[AnalysisRecordResponse]
    summary: AnalysisSummaryResponse
