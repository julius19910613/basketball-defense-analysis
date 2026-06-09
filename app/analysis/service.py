from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch

from app.config import Settings
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisRecordResponse,
    AnalysisSummaryResponse,
    Size2D,
)
from app.analysis.tracking import extract_tracked_frames, crop_windows
from app.analysis.inference import predict_player_clips
from app.analysis.motion import compute_motion_features
from app.analysis.vlm import OllamaVLMVerifier
from app.analysis.fusion import fuse_decision, should_call_vlm, apply_temporal_smoothing, summarize_records
from app.video.writer import write_annotated_video


class AnalysisService:
    """Orchestrates the hybrid analysis pipeline."""

    def __init__(self, settings: Settings, model: torch.nn.Module, device: torch.device):
        self.settings = settings
        self.model = model
        self.device = device
        
    def run_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Run the full hybrid analysis pipeline blockingly."""
        started_at = time.time()
        
        # 1. Video Tracking
        video_frames, player_boxes, width, height, colors = extract_tracked_frames(
            video_path=request.video_path,
            tracker_type=self.settings.tracker_type,
            headless=True,
            boxes_file=request.boxes_file,
            max_frames=request.max_frames,
        )
        
        # 2. Window Cropping
        player_clips = crop_windows(
            video_frames,
            player_boxes,
            seq_length=self.settings.seq_length,
            vid_stride=self.settings.vid_stride,
        )
        
        # 3. Model Inference
        predictions = predict_player_clips(
            model=self.model,
            player_clips=player_clips,
            device=self.device,
            batch_size=self.settings.batch_size,
        )
        
        # 4. VLM Initialization
        verifier: Optional[OllamaVLMVerifier] = None
        if request.vlm_mode != "off":
            verifier = OllamaVLMVerifier(
                model=self.settings.ollama_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
                image_width=self.settings.vlm_image_width,
            )

        # 5. Fusion & Verification
        output_records: List[Dict[str, Any]] = []
        final_prediction_ids: Dict[int, Dict[int, int]] = {}
        vlm_used_count = 0

        for player, player_predictions in predictions.items():
            final_prediction_ids[player] = {}
            for clip_index, prediction in enumerate(player_predictions):
                motion = compute_motion_features(
                    player_boxes,
                    player=player,
                    clip_index=clip_index,
                    seq_length=self.settings.seq_length,
                    vid_stride=self.settings.vid_stride,
                )
                
                vlm_decision = None
                if verifier and should_call_vlm(
                    request.vlm_mode,
                    prediction,
                    self.settings.low_confidence,
                    vlm_used_count,
                    self.settings.max_vlm_clips,
                ):
                    from app.analysis.vlm import select_keyframes
                    frames = select_keyframes(
                        player_clips[player][clip_index], 
                        max_frames=self.settings.vlm_frames
                    )
                    vlm_decision = verifier.verify(frames, prediction, motion)
                    vlm_used_count += 1

                final = fuse_decision(
                    prediction,
                    vlm_decision,
                    high_confidence=self.settings.high_confidence,
                    low_confidence=self.settings.low_confidence,
                )
                final_prediction_ids[player][clip_index] = final.action_id
                
                output_records.append({
                    "player": player,
                    "clip_index": clip_index,
                    "start_frame": clip_index * self.settings.vid_stride,
                    "end_frame": clip_index * self.settings.vid_stride + self.settings.seq_length - 1,
                    "r2plus1d": prediction,
                    "motion": motion,
                    "vlm": vlm_decision,
                    "final": final,
                })

        # 6. Temporal Smoothing
        apply_temporal_smoothing(output_records, final_prediction_ids, self.settings.smoothing_confidence)
        
        # Build Response
        summary_dict = summarize_records(output_records)
        
        response = AnalysisResponse(
            video=request.video_path,
            created_at_unix=started_at,
            runtime_seconds=time.time() - started_at,
            frame_size=Size2D(width=width, height=height),
            seq_length=self.settings.seq_length,
            vid_stride=self.settings.vid_stride,
            vlm_mode=request.vlm_mode,
            ollama_model=self.settings.ollama_model if request.vlm_mode != "off" else None,
            records=[AnalysisRecordResponse(**r) for r in output_records],
            summary=AnalysisSummaryResponse(**summary_dict),
        )

        analysis_id = str(uuid4().hex)

        # 8. Video Generation (Write video first to avoid orphan JSON on failure)
        if request.generate_video:
            import cv2
            fps = 30.0
            cap = cv2.VideoCapture(request.video_path)
            if cap.isOpened():
                val = cap.get(cv2.CAP_PROP_FPS)
                if val is not None and val > 0:
                    fps = val
                cap.release()

            video_name = os.path.basename(request.video_path).split(".")[0]
            video_output_path = os.path.join(self.settings.video_output_dir, f"{video_name}_{analysis_id}.mp4")
            write_annotated_video(
                video_path=video_output_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=final_prediction_ids,
                colors=colors,
                frame_width=width,
                frame_height=height,
                vid_stride=self.settings.vid_stride,
                fps=fps,
            )

        # 7. Persistence
        os.makedirs(self.settings.output_dir, exist_ok=True)
        json_path = os.path.join(self.settings.output_dir, f"{analysis_id}.json")
        with open(json_path, "w") as fp:
            fp.write(response.model_dump_json(indent=2))

        return response
