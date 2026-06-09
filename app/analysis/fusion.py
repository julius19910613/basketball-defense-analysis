from __future__ import annotations

from typing import Any, Dict, List, Sequence, Optional

from app.analysis.inference import LABEL_TO_ID
from app.analysis.schemas import FinalDecisionResponse, ModelPrediction, VLMDecisionResponse


def fuse_decision(
    prediction: ModelPrediction,
    vlm: Optional[VLMDecisionResponse],
    high_confidence: float,
    low_confidence: float,
) -> FinalDecisionResponse:
    """Fuse R(2+1)D prediction with optional VLM verification."""
    if vlm is None or not vlm.available or vlm.action is None:
        needs_review = prediction.confidence < low_confidence
        return FinalDecisionResponse(
            action_id=prediction.action_id,
            action=prediction.action,
            confidence=prediction.confidence,
            source="r2plus1d",
            needs_review=needs_review,
            reason="VLM was not used or unavailable.",
        )

    if vlm.action == prediction.action:
        fused_confidence = max(prediction.confidence, (prediction.confidence + vlm.confidence) / 2)
        return FinalDecisionResponse(
            action_id=prediction.action_id,
            action=prediction.action,
            confidence=fused_confidence,
            source="r2plus1d+vlm",
            needs_review=vlm.needs_review and fused_confidence < high_confidence,
            reason=f"VLM agreed: {vlm.reason}",
        )

    if prediction.confidence >= high_confidence and prediction.confidence >= vlm.confidence:
        return FinalDecisionResponse(
            action_id=prediction.action_id,
            action=prediction.action,
            confidence=prediction.confidence,
            source="r2plus1d_confident_conflict",
            needs_review=True,
            reason=f"VLM disagreed with lower/equal confidence: {vlm.reason}",
        )

    if vlm.confidence >= prediction.confidence or prediction.confidence < low_confidence:
        action_id = LABEL_TO_ID[vlm.action]
        return FinalDecisionResponse(
            action_id=action_id,
            action=vlm.action,
            confidence=vlm.confidence,
            source="vlm_override",
            needs_review=vlm.needs_review,
            reason=f"VLM overrode low-confidence model prediction: {vlm.reason}",
        )

    return FinalDecisionResponse(
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
    """Determine if the VLM verifier should be called for a specific prediction."""
    if mode == "off":
        return False
    if used_count >= max_vlm_clips:
        return False
    if mode == "always":
        return True
    return prediction.confidence < low_confidence


def apply_temporal_smoothing(
    records: List[Dict[str, Any]],
    final_prediction_ids: Dict[int, Dict[int, int]],
    confidence_threshold: float,
) -> None:
    """Apply temporal smoothing to fix isolated low-confidence predictions."""
    by_player: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_player.setdefault(int(record["player"]), []).append(record)

    for player, player_records in by_player.items():
        player_records.sort(key=lambda item: int(item["clip_index"]))
        for index in range(1, len(player_records) - 1):
            previous_final = player_records[index - 1]["final"]
            current_final = player_records[index]["final"]
            next_final = player_records[index + 1]["final"]

            stable_neighbors = previous_final.action == next_final.action
            isolated_low_confidence = (
                current_final.action != previous_final.action
                and float(current_final.confidence) < confidence_threshold
            )
            if stable_neighbors and isolated_low_confidence:
                smoothed_action = previous_final.action
                smoothed_action_id = LABEL_TO_ID[smoothed_action]
                current_final.action_id = smoothed_action_id
                current_final.action = smoothed_action
                current_final.confidence = max(
                    float(previous_final.confidence),
                    float(next_final.confidence),
                    float(current_final.confidence),
                )
                current_final.source = f"{current_final.source}+temporal_smoother"
                current_final.needs_review = True
                current_final.reason = (
                    current_final.reason
                    + " Isolated low-confidence label smoothed by neighboring windows."
                )
                final_prediction_ids[player][int(player_records[index]["clip_index"])] = smoothed_action_id


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics for a list of analysis records."""
    counts: Dict[str, int] = {}
    review_count = 0
    by_source: Dict[str, int] = {}
    for record in records:
        final = record["final"]
        counts[final.action] = counts.get(final.action, 0) + 1
        by_source[final.source] = by_source.get(final.source, 0) + 1
        if final.needs_review:
            review_count += 1
    return {
        "clip_count": len(records),
        "action_counts": counts,
        "needs_review_count": review_count,
        "source_counts": by_source,
    }
