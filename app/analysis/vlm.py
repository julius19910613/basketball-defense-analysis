from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from app.analysis.inference import LABEL_TO_ID, LABELS
from app.analysis.schemas import ModelPrediction, MotionFeatures, VLMDecisionResponse


def normalize_action(value: Any) -> Optional[str]:
    """Normalize VLM text output to a valid model action label."""
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
    """Safely parse a boolean from varying JSON representations."""
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
    """Safely parse and clamp a float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def select_keyframes(clip: np.ndarray, max_frames: int = 5) -> List[np.ndarray]:
    """Select evenly spaced keyframes from a video clip."""
    if len(clip) == 0:
        return []
    frame_count = min(max_frames, len(clip))
    indices = np.linspace(0, len(clip) - 1, frame_count, dtype=int)
    return [clip[int(index)] for index in indices]


def encode_frames_jpeg(frames: Iterable[np.ndarray], max_width: int = 384) -> List[str]:
    """Encode numpy frames to base64 JPEG strings for Ollama."""
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
    """Extract a JSON object from a potentially noisy VLM text response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in VLM response")
    return json.loads(match.group(0))


class OllamaVLMVerifier:
    """Client for verifying actions against a local Ollama VLM."""

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
    ) -> VLMDecisionResponse:
        """Call Ollama to verify a low-confidence model prediction."""
        images = encode_frames_jpeg(frames, max_width=self.image_width)
        if not images:
            return VLMDecisionResponse(
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
            return VLMDecisionResponse(
                action=None,
                confidence=0.0,
                reason=f"Ollama VLM HTTP error {exc.code}: {detail}",
                visible_ball=None,
                needs_review=True,
                raw_response=detail,
                available=False,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return VLMDecisionResponse(
                action=None,
                confidence=0.0,
                reason=f"Ollama VLM unavailable: {exc}",
                visible_ball=None,
                needs_review=True,
                raw_response="",
                available=False,
            )

        raw = str(body.get("response") or "")
        if not raw:
            raw = json.dumps(body)
            
        try:
            parsed = extract_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return VLMDecisionResponse(
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
        return VLMDecisionResponse(
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
