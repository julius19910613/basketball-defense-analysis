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
    predictions: Dict[int, Sequence[int]],
    colors: Sequence[Tuple[int, int, int]],
    frame_width: int,
    frame_height: int,
    vid_stride: int,
) -> None:
    """Render bounding boxes and action labels onto video frames and save to disk.

    Args:
        video_path: Path where the .mp4 file will be saved.
        video_frames: Sequence of raw BGR numpy frames.
        player_boxes: Per-frame per-player sequence of [x, y, w, h] boxes.
        predictions: Dictionary mapping player index to list of action IDs per clip.
        colors: Sequence of BGR colors for each player's bounding box.
        frame_width: Output video width.
        frame_height: Output video height.
        vid_stride: Number of frames per clip inference stride.
    """
    output_dir = os.path.dirname(video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        10,
        (frame_width, frame_height),
    )
    
    for frame_index, raw_frame in enumerate(video_frames):
        frame = raw_frame.copy()
        for player in range(len(player_boxes[0])):
            box = player_boxes[frame_index][player]
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            color = colors[player % len(colors)]
            cv2.rectangle(frame, p1, p2, color, 2, 1)

            clip_index = frame_index // vid_stride
            if clip_index < len(predictions[player]):
                action = LABELS[predictions[player][clip_index]]
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
