from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import json
import numpy as np


TRACKER_TYPES = {
    "BOOSTING": lambda: cv2.legacy.TrackerBoosting_create(),
    "MIL": lambda: cv2.legacy.TrackerMIL_create(),
    "KCF": lambda: cv2.legacy.TrackerKCF_create(),
    "TLD": lambda: cv2.legacy.TrackerTLD_create(),
    "MEDIANFLOW": lambda: cv2.legacy.TrackerMedianFlow_create(),
    "GOTURN": lambda: cv2.legacy.TrackerGOTURN_create(),
    "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create(),
    "CSRT": lambda: cv2.legacy.TrackerCSRT_create(),
}


def create_tracker_by_name(tracker_type: str) -> cv2.legacy.Tracker:
    """Create an OpenCV legacy tracker by name.

    Args:
        tracker_type: One of BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT.

    Returns:
        An initialized tracker instance.

    Raises:
        ValueError: If the tracker type is unknown.
    """
    tracker_key = tracker_type.upper()
    if tracker_key not in TRACKER_TYPES:
        available = ", ".join(TRACKER_TYPES)
        raise ValueError(f"Unknown tracker '{tracker_type}'. Available: {available}")
    return TRACKER_TYPES[tracker_key]()


def default_headless_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """Return default bounding boxes scaled to the given video dimensions.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of (x, y, w, h) tuples for two default player boxes.
    """
    scale_x = width / 1280.0
    scale_y = height / 720.0
    return [
        (int(350 * scale_x), int(100 * scale_y), int(150 * scale_x), int(400 * scale_y)),
        (int(600 * scale_x), int(120 * scale_y), int(150 * scale_x), int(400 * scale_y)),
    ]


def read_boxes_file(path: str) -> List[Tuple[int, int, int, int]]:
    """Read bounding boxes from a JSON file.

    Accepts either ``{"boxes": [[x, y, w, h], ...]}`` or ``[[x, y, w, h], ...]``.

    Args:
        path: Path to a JSON file.

    Returns:
        List of (x, y, w, h) tuples.
    """
    with open(path, "r") as fp:
        payload = json.load(fp)
    boxes = payload["boxes"] if isinstance(payload, dict) else payload
    return [tuple(int(v) for v in box) for box in boxes]


def extract_tracked_frames(
    video_path: str,
    tracker_type: str = "CSRT",
    headless: bool = True,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    boxes_file: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Tuple[
    List[np.ndarray],
    List[Tuple[Tuple[float, float, float, float], ...]],
    int,
    int,
    List[Tuple[int, int, int]],
]:
    """Extract video frames with multi-object tracking.

    Args:
        video_path: Path to the input video file.
        tracker_type: OpenCV tracker algorithm name.
        headless: If True, skip GUI-based ROI selection.
        boxes: Explicit initial bounding boxes. Takes priority over boxes_file.
        boxes_file: Path to a JSON file with initial bounding boxes.
        max_frames: Optional cap on the number of frames to process.

    Returns:
        Tuple of (video_frames, player_boxes, width, height, colors).
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        success, first_frame = cap.read()
        if not success:
            raise RuntimeError(f"Failed to read video: {video_path}")

        height, width = first_frame.shape[:2]

        if boxes is not None:
            init_boxes = boxes
        elif boxes_file:
            init_boxes = read_boxes_file(boxes_file)
        elif headless:
            init_boxes = default_headless_boxes(width, height)
        else:
            init_boxes = []
            while True:
                box = cv2.selectROI("HybridMultiTracker", first_frame, fromCenter=False, showCrosshair=True)
                init_boxes.append(tuple(int(v) for v in box))
                print("Press q to quit selecting boxes and start tracking")
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break

        colors: List[Tuple[int, int, int]] = [(255, 0, 0), (0, 0, 255), (0, 180, 0), (255, 160, 0)]
        while len(colors) < len(init_boxes):
            colors.append(tuple(int(x) for x in np.random.randint(0, 256, size=3).tolist()))

        trackers = cv2.legacy.MultiTracker_create()
        for box in init_boxes:
            trackers.add(create_tracker_by_name(tracker_type), first_frame, box)

        video_frames: List[np.ndarray] = []
        player_boxes: List[Tuple[Tuple[float, float, float, float], ...]] = []

        frame = first_frame
        is_first = True
        while True:
            raw_frame = frame.copy()
            if is_first:
                player_boxes.append(tuple(tuple(float(v) for v in box) for box in init_boxes))
                is_first = False
            else:
                success, tracked_boxes = trackers.update(frame)
                if not success:
                    fallback = player_boxes[-1] if player_boxes else tuple(tuple(float(v) for v in box) for box in init_boxes)
                    player_boxes.append(fallback)
                else:
                    player_boxes.append(tuple(tuple(float(v) for v in box) for box in tracked_boxes))
            video_frames.append(raw_frame)

            if max_frames is not None and len(video_frames) >= max_frames:
                break

            success, frame = cap.read()
            if not success:
                break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    return video_frames, player_boxes, width, height, colors[: len(init_boxes)]


def crop_video(
    clip: Sequence[np.ndarray],
    crop_window: Sequence[Sequence[Sequence[float]]],
    player: int = 0,
    output_size: Tuple[int, int] = (128, 176),
) -> List[np.ndarray]:
    """Crop and resize player regions from a clip of frames.

    Args:
        clip: Sequence of video frames (H, W, 3).
        crop_window: Per-frame per-player bounding boxes.
        player: Player index to crop.
        output_size: (width, height) of the output crop.

    Returns:
        List of resized cropped frames with shape (height, width, 3).
    """
    video: List[np.ndarray] = []
    w_out, h_out = output_size
    for idx, frame in enumerate(clip):
        x, y, w, h = [int(v) for v in crop_window[idx][player]]
        cropped = frame[max(y, 0): max(y + h, 0), max(x, 0): max(x + w, 0)]
        try:
            resized = cv2.resize(cropped, dsize=(w_out, h_out), interpolation=cv2.INTER_NEAREST)
        except cv2.error:
            resized = video[idx - 1] if (idx > 0 and video) else np.zeros((h_out, w_out, 3), dtype=np.uint8)
        video.append(resized)
    return video


def crop_windows(
    video_frames: Sequence[np.ndarray],
    player_boxes: Sequence[Sequence[Sequence[float]]],
    seq_length: int = 16,
    vid_stride: int = 8,
) -> Dict[int, List[np.ndarray]]:
    """Split video into overlapping windows and crop each player.

    Args:
        video_frames: Full list of video frames.
        player_boxes: Per-frame per-player bounding boxes.
        seq_length: Number of frames per clip window.
        vid_stride: Stride between clip windows.

    Returns:
        Dict mapping player index to list of clip arrays, each with shape
        (seq_length, H, W, 3).

    Raises:
        ValueError: If inputs are empty.
    """
    if not video_frames or not player_boxes:
        raise ValueError("Cannot crop windows from empty video or empty player boxes")

    player_count = len(player_boxes[0])
    player_frames: Dict[int, List[np.ndarray]] = {p: [] for p in range(player_count)}
    n_clips = max(1, math.ceil((len(video_frames) - seq_length) / vid_stride) + 1)

    for clip_idx in range(n_clips):
        start = clip_idx * vid_stride
        end = start + seq_length
        clip = list(video_frames[start:end])
        crop_win = list(player_boxes[start:end])

        if len(clip) < seq_length:
            remaining = seq_length - len(clip)
            clip.extend([np.zeros_like(video_frames[0]) for _ in range(remaining)])
            last_boxes = crop_win[-1] if crop_win else player_boxes[-1]
            crop_win.extend([last_boxes for _ in range(remaining)])

        for player in range(player_count):
            player_frames[player].append(np.asarray(crop_video(clip, crop_win, player)))

    return player_frames
