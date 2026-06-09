from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from app.analysis.inference import LABELS


def write_annotated_video(
    video_path: str,
    video_frames: Sequence[np.ndarray],
    player_boxes: Sequence[Sequence[Sequence[float]]],
    predictions: Dict[int, Dict[int, int] | Sequence[int]],
    colors: Sequence[Tuple[int, int, int]],
    frame_width: int,
    frame_height: int,
    vid_stride: int,
    fps: float = 30.0,
) -> None:
    """Render bounding boxes and action labels onto video frames and save to disk.

    Args:
        video_path: Path where the .mp4 file will be saved.
        video_frames: Sequence of raw BGR numpy frames.
        player_boxes: Per-frame per-player sequence of [x, y, w, h] boxes.
        predictions: Dictionary mapping player index to dict or list of action IDs per clip.
        colors: Sequence of BGR colors for each player's bounding box.
        frame_width: Output video width.
        frame_height: Output video height.
        vid_stride: Number of frames per clip inference stride.
        fps: Frames per second of the output video.
    """
    output_dir = os.path.dirname(video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (frame_width, frame_height),
    )
    
    for frame_index, raw_frame in enumerate(video_frames):
        frame = raw_frame.copy()
        for player in predictions.keys():
            if frame_index >= len(player_boxes) or player >= len(player_boxes[frame_index]) or player < 0:
                continue
            box = player_boxes[frame_index][player]
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            color = colors[player % len(colors)]
            cv2.rectangle(frame, p1, p2, color, 2, 1)

            player_preds = predictions.get(player)
            if player_preds is not None:
                if isinstance(player_preds, dict):
                    target_clip = frame_index // vid_stride
                    max_clip = max(player_preds.keys()) if player_preds else 0
                    clip_index = min(target_clip, max_clip)
                    action_id = player_preds.get(clip_index)
                else:
                    target_clip = frame_index // vid_stride
                    max_clip = len(player_preds) - 1
                    clip_index = min(target_clip, max_clip)
                    action_id = player_preds[clip_index] if 0 <= clip_index < len(player_preds) else None

                if action_id is not None:
                    action = LABELS[action_id]
                    cv2.putText(
                        frame,
                        action,
                        (p1[0] - 10, p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
        out.write(frame)
        
    out.release()
