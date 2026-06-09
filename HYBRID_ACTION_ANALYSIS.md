# Hybrid Basketball Action Analysis

This project now has a sidecar hybrid pipeline that keeps the original
`main.py` demo intact while adding a more auditable analysis flow:

1. OpenCV CSRT tracking creates per-player bounding-box tracks.
2. Each track is split into 16-frame windows with stride 8.
3. The existing R(2+1)D checkpoint predicts the action and softmax scores.
4. Simple trajectory features are computed from each player box sequence.
5. Ollama VLM verification is called only for selected windows.
6. Conservative fusion produces a final label, review flag, JSON report, and
   optional annotated video.

## Why This Shape

The deployed non-LLM action model is the primary classifier because it is fast,
deterministic, and trained for the 10-label basketball action space. The Ollama
VLM is used as a reviewer, not a detector and not a full-video classifier. This
keeps latency bounded and makes failures easier to inspect.

The pipeline intentionally lives in `hybrid_analysis.py` instead of modifying
`main.py`. The original demo has global args, import-time side effects, and
hard-coded demo boxes. The sidecar keeps that behavior stable.

## Current Execution

Run without VLM:

```bash
./venv/bin/python hybrid_analysis.py --vlm-mode off
```

Run with low-confidence VLM review:

```bash
./venv/bin/python hybrid_analysis.py --vlm-mode low-confidence --max-vlm-clips 8
```

Use a custom initial box file:

```bash
./venv/bin/python hybrid_analysis.py --boxes-file boxes.json
```

`boxes.json` can be either a JSON list:

```json
[[350, 100, 150, 400], [600, 120, 150, 400]]
```

or an object:

```json
{"boxes": [[350, 100, 150, 400], [600, 120, 150, 400]]}
```

## Outputs

Default outputs:

- `analysis_outputs/lebron_shoots_hybrid.json`
- `output_videos/lebron_shoots_hybrid.mp4`

The JSON report contains per-window records:

- frame range
- player id
- R(2+1)D action and probabilities
- motion features
- optional VLM decision
- final fused decision
- review flag

## Fusion Policy

- If VLM is off or unavailable, keep the R(2+1)D result.
- If VLM agrees, raise confidence conservatively.
- If R(2+1)D is high confidence, keep it even during conflicts and mark review.
- If R(2+1)D is low confidence and VLM is confident, allow VLM override.
- If a low-confidence label is isolated between two matching neighbor windows,
  smooth it to the neighbor label and mark it for review.

## Validation

Offline unit tests do not require SpaceJam, a live Ollama server, or the
checkpoint:

```bash
./venv/bin/python -m unittest discover -s tests
```

Optional end-to-end smoke test:

```bash
./venv/bin/python hybrid_analysis.py --vlm-mode off --json-output /tmp/hybrid.json --video-output /tmp/hybrid.mp4
```

Optional real Ollama smoke test:

```bash
ollama ps
./venv/bin/python hybrid_analysis.py --vlm-mode always --max-vlm-clips 1 --json-output /tmp/hybrid_vlm.json --video-output ""
```

## Known Limits

- The current tracker mode still needs either the demo headless boxes or a box
  file. It is not a general multi-player detector yet.
- The SpaceJam dataset is unavailable from the original public Drive link, so
  this is an inference/analysis pipeline, not a retraining pipeline.
- VLM output is constrained to the 10 known labels and may abstain by producing
  invalid/low-confidence output. The fusion layer treats that as review-needed,
  not as a new class.
