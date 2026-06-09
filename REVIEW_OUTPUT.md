# Review Findings

The following bugs and findings were identified in the codebase:

- **CRITICAL** | `app/analysis/tracking.py:189` | `crop_video` fallback `video[idx-1]` when `idx==0` and `cv2.resize` fails accesses `video[-1]`; on empty list raises `IndexError`, on non-empty silently uses wrong frame
- **HIGH** | `app/analysis/motion.py:31` | No bounds check on player index; `IndexError` if `player >= len(frame_boxes)` for any frame in the window
- **HIGH** | `app/video/writer.py:47` | Iterates `range(len(player_boxes[0]))` but indexes `predictions[player]` — `KeyError` if predictions dict keys don't cover 0..N-1
- **HIGH** | `app/video/writer.py:54-55` | `clip_index = frame_index // vid_stride` can exceed `len(predictions[player])` for trailing frames, causing `IndexError`
- **HIGH** | `app/analysis/router.py:24-25` | No path sanitization on user-supplied `video_path`; allows arbitrary filesystem reads via path traversal
- **HIGH** | `app/models/r2plus1d.py:7` | Bare import from `utils.checkpoints` fails unless CWD is project root; needs relative/absolute package import
- **MEDIUM** | `app/analysis/fusion.py:124` | `apply_temporal_smoothing` indexes `final_prediction_ids[player]` by `clip_index` int value, but list built by append — breaks if players have non-contiguous/non-zero-based clip indices
- **MEDIUM** | `app/video/writer.py:41` | FPS hardcoded to 10; should use source video FPS when available
- **MEDIUM** | `app/analysis/router.py:32-35` | Raw exception message forwarded to client in HTTP 500; leaks internal paths and stack info
- **MEDIUM** | `app/analysis/tracking.py:130` | `np.random.randint` yields `np.int64`; color tuple of `np.int64` may cause cv2 type errors
- **MEDIUM** | `app/analysis/service.py:146-149` | No cleanup on partial failure — JSON persisted before video write; if video write crashes, orphan JSON remains
