from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from torchvision import models

from utils.checkpoints import load_weights


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


@dataclass
class ModelPrediction:
    action_id: int
    action: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class MotionFeatures:
    avg_center_speed: float
    max_center_speed: float
    avg_box_area: float
    area_change_ratio: float


@dataclass
class VLMDecision:
    action: Optional[str]
    confidence: float
    reason: str
    visible_ball: Optional[bool]
    needs_review: bool
    raw_response: str
    available: bool


@dataclass
class FinalDecision:
    action_id: int
    action: str
    confidence: float
    source: str
    needs_review: bool
    reason: str


def build_checkpoint_args(
    model_path: str,
    start_epoch: int,
    lr: float,
    base_model_name: str,
) -> EasyDict:
    return EasyDict(
        {
            "base_model_name": base_model_name,
            "start_epoch": start_epoch,
            "lr": lr,
            "model_path": model_path,
        }
    )


def build_r2plus1d_model(
    checkpoint_args: EasyDict,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.video.r2plus1d_18(weights=None, progress=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
    model = load_weights(model, checkpoint_args)
    model = model.to(device)
    model.eval()
    return model


def predict_player_clips(
    model: torch.nn.Module,
    player_clips: Dict[int, Sequence[np.ndarray]],
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> Dict[int, List[ModelPrediction]]:
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


def create_tracker_by_name(tracker_type: str) -> cv2.legacy.Tracker:
    tracker_key = tracker_type.upper()
    if tracker_key not in TRACKER_TYPES:
        available = ", ".join(TRACKER_TYPES)
        raise ValueError(f"Unknown tracker '{tracker_type}'. Available trackers: {available}")
    return TRACKER_TYPES[tracker_key]()


def default_headless_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    scale_x = width / 1280.0
    scale_y = height / 720.0
    return [
        (
            int(350 * scale_x),
            int(100 * scale_y),
            int(150 * scale_x),
            int(400 * scale_y),
        ),
        (
            int(600 * scale_x),
            int(120 * scale_y),
            int(150 * scale_x),
            int(400 * scale_y),
        ),
    ]


def read_boxes_file(path: str) -> List[Tuple[int, int, int, int]]:
    with open(path, "r") as fp:
        payload = json.load(fp)
    boxes = payload["boxes"] if isinstance(payload, dict) else payload
    return [tuple(int(value) for value in box) for box in boxes]


def extract_tracked_frames(
    video_path: str,
    tracker_type: str = "CSRT",
    headless: bool = True,
    boxes_file: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[Tuple[Tuple[float, float, float, float], ...]], int, int, List[Tuple[int, int, int]]]:
    cap = cv2.VideoCapture(video_path)
    success, first_frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError(f"Failed to read video: {video_path}")

    height, width = first_frame.shape[:2]
    if boxes_file:
        boxes = read_boxes_file(boxes_file)
    elif headless:
        boxes = default_headless_boxes(width, height)
    else:
        boxes = []
        while True:
            box = cv2.selectROI("HybridMultiTracker", first_frame, fromCenter=False, showCrosshair=True)
            boxes.append(tuple(int(value) for value in box))
            print("Press q to quit selecting boxes and start tracking")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    colors = [(255, 0, 0), (0, 0, 255), (0, 180, 0), (255, 160, 0)]
    while len(colors) < len(boxes):
        colors.append(tuple(int(value) for value in np.random.randint(0, 255, size=3)))

    trackers = cv2.legacy.MultiTracker_create()
    for box in boxes:
        trackers.add(create_tracker_by_name(tracker_type), first_frame, box)

    video_frames: List[np.ndarray] = []
    player_boxes: List[Tuple[Tuple[float, float, float, float], ...]] = []

    frame = first_frame
    while True:
        raw_frame = frame.copy()
        success, tracked_boxes = trackers.update(frame)
        if not success:
            player_boxes.append(tuple(tuple(float(value) for value in box) for box in boxes))
        else:
            player_boxes.append(tuple(tuple(float(value) for value in box) for box in tracked_boxes))
        video_frames.append(raw_frame)
        if max_frames is not None and len(video_frames) >= max_frames:
            break

        success, frame = cap.read()
        if not success:
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    return video_frames, player_boxes, width, height, colors[: len(boxes)]


def crop_video(
    clip: Sequence[np.ndarray],
    crop_window: Sequence[Sequence[Sequence[float]]],
    player: int = 0,
    output_size: Tuple[int, int] = (128, 176),
) -> List[np.ndarray]:
    video: List[np.ndarray] = []
    width, height = output_size
    for index, frame in enumerate(clip):
        x, y, w, h = [int(value) for value in crop_window[index][player]]
        cropped_frame = frame[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)]
        try:
            resized_frame = cv2.resize(
                cropped_frame,
                dsize=(width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        except cv2.error:
            resized_frame = video[index - 1] if video else np.zeros((height, width, 3), dtype=np.uint8)
        video.append(resized_frame)
    return video


def crop_windows(
    video_frames: Sequence[np.ndarray],
    player_boxes: Sequence[Sequence[Sequence[float]]],
    seq_length: int = 16,
    vid_stride: int = 8,
) -> Dict[int, List[np.ndarray]]:
    if not video_frames or not player_boxes:
        raise ValueError("Cannot crop windows from an empty video or empty player boxes")

    player_count = len(player_boxes[0])
    player_frames: Dict[int, List[np.ndarray]] = {player: [] for player in range(player_count)}
    n_clips = len(video_frames) // vid_stride

    for clip_index in range(n_clips):
        start = clip_index * vid_stride
        end = start + seq_length
        clip = list(video_frames[start:end])
        crop_window = list(player_boxes[start:end])
        if len(clip) < seq_length:
            remaining = seq_length - len(clip)
            clip.extend([np.zeros_like(video_frames[0]) for _ in range(remaining)])
            last_boxes = crop_window[-1] if crop_window else player_boxes[-1]
            crop_window.extend([last_boxes for _ in range(remaining)])

        for player in range(player_count):
            player_frames[player].append(np.asarray(crop_video(clip, crop_window, player)))

    return player_frames


def inference_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch.permute(0, 4, 1, 2, 3)


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


def compute_motion_features(
    player_boxes: Sequence[Sequence[Sequence[float]]],
    player: int,
    clip_index: int,
    seq_length: int,
    vid_stride: int,
) -> MotionFeatures:
    start = clip_index * vid_stride
    window = player_boxes[start : start + seq_length]
    boxes = [np.asarray(frame_boxes[player], dtype=float) for frame_boxes in window]
    if not boxes:
        return MotionFeatures(0.0, 0.0, 0.0, 0.0)

    centers = np.asarray([(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in boxes])
    areas = np.asarray([max(box[2], 0.0) * max(box[3], 0.0) for box in boxes])
    if len(centers) > 1:
        speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    else:
        speeds = np.asarray([0.0])

    first_area = float(areas[0]) if float(areas[0]) else 1.0
    return MotionFeatures(
        avg_center_speed=float(np.mean(speeds)),
        max_center_speed=float(np.max(speeds)),
        avg_box_area=float(np.mean(areas)),
        area_change_ratio=float((areas[-1] - areas[0]) / first_area),
    )


def select_keyframes(clip: np.ndarray, max_frames: int = 5) -> List[np.ndarray]:
    if len(clip) == 0:
        return []
    frame_count = min(max_frames, len(clip))
    indices = np.linspace(0, len(clip) - 1, frame_count, dtype=int)
    return [clip[int(index)] for index in indices]


def encode_frames_jpeg(frames: Iterable[np.ndarray], max_width: int = 384) -> List[str]:
    encoded: List[str] = []
    for frame in frames:
        image = frame
        if image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(
                image,
                (max_width, int(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            encoded.append(base64.b64encode(buffer).decode("ascii"))
    return encoded


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in VLM response")
    return json.loads(match.group(0))


class OllamaVLMVerifier:
    def __init__(
        self,
        model: str = "qwen3-vl:4b",
        host: str = "http://127.0.0.1:11434",
        timeout: float = 45.0,
        image_width: int = 224,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.image_width = image_width

    def verify(
        self,
        frames: Sequence[np.ndarray],
        prediction: ModelPrediction,
        motion: MotionFeatures,
    ) -> VLMDecision:
        images = encode_frames_jpeg(frames, max_width=self.image_width)
        if not images:
            return VLMDecision(
                action=None,
                confidence=0.0,
                reason="No frames were available for VLM verification.",
                visible_ball=None,
                needs_review=True,
                raw_response="",
                available=False,
            )

        prompt = self._build_prompt(prediction, motion)
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": prompt,
            "images": images,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 220},
        }
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return VLMDecision(
                action=None,
                confidence=0.0,
                reason=f"Ollama VLM HTTP error {exc.code}: {detail}",
                visible_ball=None,
                needs_review=True,
                raw_response=detail,
                available=False,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return VLMDecision(
                action=None,
                confidence=0.0,
                reason=f"Ollama VLM unavailable: {exc}",
                visible_ball=None,
                needs_review=True,
                raw_response="",
                available=False,
            )

        raw = str(body.get("response") or body.get("thinking") or "")
        if not raw:
            raw = json.dumps(body)
        try:
            parsed = extract_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return VLMDecision(
                action=None,
                confidence=0.0,
                reason=f"VLM returned non-JSON response: {exc}",
                visible_ball=None,
                needs_review=True,
                raw_response=raw,
                available=True,
            )

        action = normalize_action(parsed.get("action"))
        confidence = clamp_float(parsed.get("confidence"), 0.0, 1.0, default=0.0)
        return VLMDecision(
            action=action,
            confidence=confidence,
            reason=str(parsed.get("reason", "")),
            visible_ball=parse_optional_bool(parsed.get("visible_ball")),
            needs_review=bool(parsed.get("needs_review", False)) or action is None,
            raw_response=raw,
            available=True,
        )

    def _build_prompt(self, prediction: ModelPrediction, motion: MotionFeatures) -> str:
        labels = ", ".join(LABELS.values())
        return (
            "You are verifying a basketball single-player action from a short sequence "
            "of cropped frames. Choose exactly one action from this label set: "
            f"{labels}.\n"
            "Return only compact JSON with keys: action, confidence, reason, "
            "visible_ball, needs_review.\n"
            f"R(2+1)D prediction: {prediction.action} "
            f"confidence={prediction.confidence:.3f}.\n"
            "Motion features: "
            f"avg_center_speed={motion.avg_center_speed:.2f}, "
            f"max_center_speed={motion.max_center_speed:.2f}, "
            f"area_change_ratio={motion.area_change_ratio:.3f}.\n"
            "Use the visual evidence first. If unsure, set needs_review=true."
        )


def normalize_action(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace("-", "_")
    aliases = {
        "ball_in_hand": "ball in hand",
        "ball hand": "ball in hand",
        "no action": "no_action",
        "none": "no_action",
        "defence": "defense",
    }
    cleaned = aliases.get(cleaned, cleaned)
    return cleaned if cleaned in LABEL_TO_ID else None


def parse_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def fuse_decision(
    prediction: ModelPrediction,
    vlm: Optional[VLMDecision],
    high_confidence: float,
    low_confidence: float,
) -> FinalDecision:
    if vlm is None or not vlm.available or vlm.action is None:
        needs_review = prediction.confidence < low_confidence
        return FinalDecision(
            action_id=prediction.action_id,
            action=prediction.action,
            confidence=prediction.confidence,
            source="r2plus1d",
            needs_review=needs_review,
            reason="VLM was not used or unavailable.",
        )

    if vlm.action == prediction.action:
        fused_confidence = max(prediction.confidence, (prediction.confidence + vlm.confidence) / 2)
        return FinalDecision(
            action_id=prediction.action_id,
            action=prediction.action,
            confidence=fused_confidence,
            source="r2plus1d+vlm",
            needs_review=vlm.needs_review and fused_confidence < high_confidence,
            reason=f"VLM agreed: {vlm.reason}",
        )

    if prediction.confidence >= high_confidence and prediction.confidence >= vlm.confidence:
        return FinalDecision(
            action_id=prediction.action_id,
            action=prediction.action,
            confidence=prediction.confidence,
            source="r2plus1d_confident_conflict",
            needs_review=True,
            reason=f"VLM disagreed with lower/equal confidence: {vlm.reason}",
        )

    if vlm.confidence >= prediction.confidence or prediction.confidence < low_confidence:
        action_id = LABEL_TO_ID[vlm.action]
        return FinalDecision(
            action_id=action_id,
            action=vlm.action,
            confidence=vlm.confidence,
            source="vlm_override",
            needs_review=vlm.needs_review,
            reason=f"VLM overrode low-confidence model prediction: {vlm.reason}",
        )

    return FinalDecision(
        action_id=prediction.action_id,
        action=prediction.action,
        confidence=prediction.confidence,
        source="r2plus1d_conflict",
        needs_review=True,
        reason=f"R(2+1)D retained despite VLM disagreement: {vlm.reason}",
    )


def should_call_vlm(
    mode: str,
    prediction: ModelPrediction,
    low_confidence: float,
    used_count: int,
    max_vlm_clips: int,
) -> bool:
    if mode == "off":
        return False
    if used_count >= max_vlm_clips:
        return False
    if mode == "always":
        return True
    return prediction.confidence < low_confidence


def build_output_record(
    player: int,
    clip_index: int,
    seq_length: int,
    vid_stride: int,
    prediction: ModelPrediction,
    motion: MotionFeatures,
    vlm: Optional[VLMDecision],
    final: FinalDecision,
) -> Dict[str, Any]:
    return {
        "player": player,
        "clip_index": clip_index,
        "start_frame": clip_index * vid_stride,
        "end_frame": clip_index * vid_stride + seq_length - 1,
        "r2plus1d": asdict(prediction),
        "motion": asdict(motion),
        "vlm": asdict(vlm) if vlm is not None else None,
        "final": asdict(final),
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


def run_hybrid_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    started_at = time.time()
    if args.detector != "tracker":
        raise NotImplementedError("The hybrid pipeline currently supports tracker mode only.")

    video_frames, player_boxes, width, height, colors = extract_tracked_frames(
        args.video,
        tracker_type=args.tracker,
        headless=args.headless,
        boxes_file=args.boxes_file,
        max_frames=args.max_frames,
    )
    player_clips = crop_windows(
        video_frames,
        player_boxes,
        seq_length=args.seq_length,
        vid_stride=args.vid_stride,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_args = build_checkpoint_args(
        model_path=args.model_path,
        start_epoch=args.start_epoch,
        lr=args.lr,
        base_model_name=args.base_model_name,
    )
    model = build_r2plus1d_model(checkpoint_args, device=device)
    predictions = predict_player_clips(
        model,
        player_clips,
        device=device,
        batch_size=args.batch_size,
    )

    verifier = None
    if args.vlm_mode != "off":
        verifier = OllamaVLMVerifier(
            model=args.ollama_model,
            host=args.ollama_host,
            timeout=args.ollama_timeout,
            image_width=args.vlm_image_width,
        )

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
                seq_length=args.seq_length,
                vid_stride=args.vid_stride,
            )
            vlm_decision: Optional[VLMDecision] = None
            if verifier and should_call_vlm(
                args.vlm_mode,
                prediction,
                args.low_confidence,
                vlm_used_count,
                args.max_vlm_clips,
            ):
                frames = select_keyframes(player_clips[player][clip_index], max_frames=args.vlm_frames)
                vlm_decision = verifier.verify(frames, prediction, motion)
                vlm_used_count += 1

            final = fuse_decision(
                prediction,
                vlm_decision,
                high_confidence=args.high_confidence,
                low_confidence=args.low_confidence,
            )
            final_prediction_ids[player][clip_index] = final.action_id
            output_records.append(
                build_output_record(
                    player,
                    clip_index,
                    args.seq_length,
                    args.vid_stride,
                    prediction,
                    motion,
                    vlm_decision,
                    final,
                )
            )

    apply_temporal_smoothing(output_records, final_prediction_ids, args.smoothing_confidence)

    payload = {
        "video": args.video,
        "created_at_unix": started_at,
        "runtime_seconds": time.time() - started_at,
        "frame_size": {"width": width, "height": height},
        "seq_length": args.seq_length,
        "vid_stride": args.vid_stride,
        "vlm_mode": args.vlm_mode,
        "ollama_model": args.ollama_model if args.vlm_mode != "off" else None,
        "records": output_records,
        "summary": summarize_records(output_records),
    }

    write_json(args.json_output, payload)
    if args.video_output:
        write_annotated_video(
            args.video_output,
            video_frames,
            player_boxes,
            final_prediction_ids,
            colors,
            frame_width=width,
            frame_height=height,
            vid_stride=args.vid_stride,
        )
    return payload


def apply_temporal_smoothing(
    records: List[Dict[str, Any]],
    final_prediction_ids: Dict[int, Dict[int, int]],
    confidence_threshold: float,
) -> None:
    by_player: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_player.setdefault(int(record["player"]), []).append(record)

    for player, player_records in by_player.items():
        player_records.sort(key=lambda item: int(item["clip_index"]))
        for index in range(1, len(player_records) - 1):
            previous_final = player_records[index - 1]["final"]
            current_final = player_records[index]["final"]
            next_final = player_records[index + 1]["final"]

            stable_neighbors = previous_final["action"] == next_final["action"]
            isolated_low_confidence = (
                current_final["action"] != previous_final["action"]
                and float(current_final["confidence"]) < confidence_threshold
            )
            if stable_neighbors and isolated_low_confidence:
                smoothed_action = previous_final["action"]
                smoothed_action_id = LABEL_TO_ID[smoothed_action]
                current_final.update(
                    {
                        "action_id": smoothed_action_id,
                        "action": smoothed_action,
                        "confidence": max(
                            float(previous_final["confidence"]),
                            float(next_final["confidence"]),
                            float(current_final["confidence"]),
                        ),
                        "source": f"{current_final['source']}+temporal_smoother",
                        "needs_review": True,
                        "reason": (
                            current_final["reason"]
                            + " Isolated low-confidence label smoothed by neighboring windows."
                        ),
                    }
                )
                final_prediction_ids[player][int(player_records[index]["clip_index"])] = smoothed_action_id


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    review_count = 0
    by_source: Dict[str, int] = {}
    for record in records:
        final = record["final"]
        counts[final["action"]] = counts.get(final["action"], 0) + 1
        by_source[final["source"]] = by_source.get(final["source"], 0) + 1
        if final["needs_review"]:
            review_count += 1
    return {
        "clip_count": len(records),
        "action_counts": counts,
        "needs_review_count": review_count,
        "source_counts": by_source,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid basketball action analysis with R(2+1)D and optional Ollama VLM verification."
    )
    parser.add_argument("--video", default="examples/lebron_shoots.mp4")
    parser.add_argument("--json-output", default="analysis_outputs/lebron_shoots_hybrid.json")
    parser.add_argument("--video-output", default="output_videos/lebron_shoots_hybrid.mp4")
    parser.add_argument("--detector", choices=["tracker"], default="tracker")
    parser.add_argument("--tracker", default="CSRT")
    parser.add_argument("--boxes-file", default=None, help="Optional JSON list of initial boxes [x, y, w, h].")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for fast service/demo runs.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--vid-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", default="model_checkpoints/r2plus1d_augmented-2/")
    parser.add_argument("--base-model-name", default="r2plus1d_multiclass")
    parser.add_argument("--start-epoch", type=int, default=19)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--vlm-mode", choices=["off", "low-confidence", "always"], default="low-confidence")
    parser.add_argument("--ollama-model", default="qwen3-vl:4b")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=float, default=45.0)
    parser.add_argument("--vlm-frames", type=int, default=1)
    parser.add_argument("--vlm-image-width", type=int, default=224)
    parser.add_argument("--max-vlm-clips", type=int, default=8)
    parser.add_argument("--low-confidence", type=float, default=0.55)
    parser.add_argument("--high-confidence", type=float, default=0.75)
    parser.add_argument("--smoothing-confidence", type=float, default=0.6)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    payload = run_hybrid_analysis(args)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
