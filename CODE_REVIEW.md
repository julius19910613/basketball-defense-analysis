# Code Review Findings (by Claude Sonnet 4.6 Thinking)

## P0 — Runtime Crashes

### Bug #5: fusion.py L103 — dict vs object access
- `apply_temporal_smoothing` reads `final["action"]` as dict key, but `final` is a `FinalDecisionResponse` Pydantic object
- Must use `final.action` instead of `final["action"]`

### Bug #3: tracking.py L141-143 — tracker failure fallback
- When `trackers.update()` fails (`success=False`), falls back to `init_boxes` (initial boxes)
- Should fall back to last-known tracked positions instead

## P1 — Data Loss / Silent Errors

### Bug #4: tracking.py L218 — off-by-one n_clips
- `n_clips = len(video_frames) // vid_stride` silently drops last partial window
- Should use `ceil` or `(len(video_frames) - seq_length) // vid_stride + 1`

### Bug #7: vlm.py L170 — thinking fallback
- Falls back to `body.get("thinking")` which is chain-of-thought text, not JSON
- Should only use the actual model response field

## P2 — Robustness

### Bug #2: service.py L117 — frame index off-by-one
- `start_frame` calculation may be off vs actual video timestamps

### Bug #1: main.py L42-48 — module-level get_settings()
- `get_settings()` called at module import time before .env fully loaded
- StaticFiles uses relative paths — breaks if launched from different directory

## P3 — Quality

### Bug #6: writer.py L41 — hardcoded FPS=10
- Should respect original video FPS

---

## Test Plan

### Unit Tests
1. tracking: crop_windows boundary, tracker failure fallback, read_boxes_file JSON formats
2. fusion: all 5 branches of fuse_decision, apply_temporal_smoothing with 1/2/multi clips, VLM unavailable
3. inference: predict_player_clips with zero clips, batch spanning, softmax mismatch guard
4. vlm: normalize_action aliases, extract_json_object noisy text, OllamaVLMVerifier HTTP error/timeout

### Integration Tests
5. service: run_analysis with synthetic video + mocked tracker/model
6. API: POST /api/v1/analysis/run — missing file (400), valid (200), service exception (500)

### Error Scenario Tests
7. VLM timeout, model load failure, empty video, oversized video
