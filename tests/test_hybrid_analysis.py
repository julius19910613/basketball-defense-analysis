import unittest
import numpy as np

from app.analysis.schemas import ModelPrediction, VLMDecisionResponse, FinalDecisionResponse
from app.analysis.vlm import extract_json_object, normalize_action
from app.analysis.fusion import fuse_decision, should_call_vlm, apply_temporal_smoothing
from app.analysis.tracking import crop_windows


class HybridAnalysisTest(unittest.TestCase):
    def make_prediction(self, action="dribble", confidence=0.4):
        action_ids = {
            "block": 0,
            "pass": 1,
            "run": 2,
            "dribble": 3,
            "shoot": 4,
            "ball in hand": 5,
            "defense": 6,
            "pick": 7,
            "no_action": 8,
            "walk": 9,
        }
        return ModelPrediction(
            action_id=action_ids[action],
            action=action,
            confidence=confidence,
            probabilities={action: confidence},
        )

    def test_extract_json_object_accepts_wrapped_json(self):
        parsed = extract_json_object('Here is the result: {"action": "shoot", "confidence": 0.7}')
        self.assertEqual(parsed["action"], "shoot")
        self.assertEqual(parsed["confidence"], 0.7)

    def test_normalize_action_maps_aliases(self):
        self.assertEqual(normalize_action("ball_in_hand"), "ball in hand")
        self.assertEqual(normalize_action("no action"), "no_action")
        self.assertEqual(normalize_action("defence"), "defense")
        self.assertIsNone(normalize_action("guarding"))

    def test_fuse_uses_r2plus1d_when_vlm_unavailable(self):
        prediction = self.make_prediction("shoot", 0.8)
        final = fuse_decision(prediction, None, high_confidence=0.75, low_confidence=0.55)
        self.assertEqual(final.action, "shoot")
        self.assertEqual(final.source, "r2plus1d")
        self.assertFalse(final.needs_review)

    def test_fuse_vlm_override_for_low_confidence_prediction(self):
        prediction = self.make_prediction("dribble", 0.35)
        vlm = VLMDecisionResponse(
            action="defense",
            confidence=0.72,
            reason="No ball is visible and stance is defensive.",
            visible_ball=False,
            needs_review=False,
            raw_response="{}",
            available=True,
        )
        final = fuse_decision(prediction, vlm, high_confidence=0.75, low_confidence=0.55)
        self.assertEqual(final.action, "defense")
        self.assertEqual(final.source, "vlm_override")

    def test_should_call_vlm_respects_mode_and_limit(self):
        low_prediction = self.make_prediction("dribble", 0.4)
        high_prediction = self.make_prediction("shoot", 0.9)
        self.assertTrue(should_call_vlm("low-confidence", low_prediction, 0.55, 0, 2))
        self.assertFalse(should_call_vlm("low-confidence", high_prediction, 0.55, 0, 2))
        self.assertFalse(should_call_vlm("off", low_prediction, 0.55, 0, 2))
        self.assertFalse(should_call_vlm("always", low_prediction, 0.55, 2, 2))

    def test_crop_windows_pads_short_tail(self):
        frames = [np.full((32, 32, 3), fill_value=index, dtype=np.uint8) for index in range(10)]
        boxes = [[(0, 0, 16, 16)]] * 10
        windows = crop_windows(frames, boxes, seq_length=4, vid_stride=3)
        self.assertEqual(len(windows[0]), 3)
        self.assertEqual(windows[0][0].shape, (4, 176, 128, 3))

    def test_temporal_smoothing_replaces_isolated_low_confidence_label(self):
        records = [
            {
                "player": 0,
                "clip_index": 0,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.7, source="r2plus1d", needs_review=False, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 1,
                "final": FinalDecisionResponse(
                    action_id=3, action="dribble", confidence=0.3, source="r2plus1d", needs_review=True, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 2,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.8, source="r2plus1d", needs_review=False, reason=""
                ),
            },
        ]
        predictions = {0: {0: 6, 1: 3, 2: 6}}
        apply_temporal_smoothing(records, predictions, confidence_threshold=0.6)
        self.assertEqual(records[1]["final"].action, "defense")
        self.assertEqual(predictions[0], {0: 6, 1: 6, 2: 6})

    def test_tracker_failure_fallback(self):
        from unittest.mock import patch, MagicMock
        with patch('cv2.VideoCapture') as mock_vc, \
             patch('cv2.legacy.MultiTracker_create') as mock_mt:
            
            mock_cap = MagicMock()
            mock_cap.read.side_effect = [
                (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                (False, None)
            ]
            mock_vc.return_value = mock_cap
            
            mock_tracker = MagicMock()
            mock_tracker.update.side_effect = [
                (True, [(10, 10, 20, 20)]),
                (False, []),
                (True, [(30, 30, 20, 20)]),
            ]
            mock_mt.return_value = mock_tracker
            
            from app.analysis.tracking import extract_tracked_frames
            frames, player_boxes, w, h, colors = extract_tracked_frames(
                video_path="dummy.mp4",
                tracker_type="CSRT",
                headless=True,
                boxes=[(5, 5, 20, 20)]
            )
            
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(player_boxes), 3)
            self.assertEqual(player_boxes[0], ((5.0, 5.0, 20.0, 20.0),))
            self.assertEqual(player_boxes[1], ((10.0, 10.0, 20.0, 20.0),))
            self.assertEqual(player_boxes[2], ((10.0, 10.0, 20.0, 20.0),))

    def test_crop_windows_n_clips_boundary(self):
        frames_17 = [np.zeros((10, 10, 3)) for _ in range(17)]
        boxes_17 = [[(0, 0, 5, 5)]] * 17
        windows_17 = crop_windows(frames_17, boxes_17, seq_length=16, vid_stride=8)
        self.assertEqual(len(windows_17[0]), 2)

        frames_25 = [np.zeros((10, 10, 3)) for _ in range(25)]
        boxes_25 = [[(0, 0, 5, 5)]] * 25
        windows_25 = crop_windows(frames_25, boxes_25, seq_length=16, vid_stride=8)
        self.assertEqual(len(windows_25[0]), 3)

    def test_vlm_verifier_only_uses_response_field(self):
        import json
        from unittest.mock import patch, MagicMock
        from app.analysis.vlm import OllamaVLMVerifier
        from app.analysis.schemas import MotionFeatures
        
        verifier = OllamaVLMVerifier(model="test-model", host="http://localhost:11434")
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": '{"action": "shoot", "confidence": 0.95, "reason": "visible shot"}',
                "thinking": 'Let me think about this. The player is shooting...'
            }).encode('utf-8')
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            
            motion = MotionFeatures(
                avg_center_speed=1.0, max_center_speed=2.0, avg_box_area=100.0, area_change_ratio=1.0
            )
            prediction = self.make_prediction("shoot", 0.95)
            
            frames = [np.zeros((128, 176, 3), dtype=np.uint8)]
            vlm_decision = verifier.verify(frames, prediction, motion)
            
            self.assertTrue(vlm_decision.available)
            self.assertEqual(vlm_decision.action, "shoot")
            self.assertEqual(vlm_decision.confidence, 0.95)
            self.assertEqual(vlm_decision.reason, "visible shot")
            
            mock_resp.read.return_value = json.dumps({
                "thinking": '{"action": "shoot", "confidence": 0.95}'
            }).encode('utf-8')
            
            vlm_decision2 = verifier.verify(frames, prediction, motion)
            self.assertIsNone(vlm_decision2.action)

    def test_lifespan_mounts_static_directories_with_absolute_paths(self):
        from unittest.mock import patch, MagicMock
        from app.main import lifespan
        from fastapi import FastAPI
        
        app = FastAPI()
        
        with patch('app.main.build_r2plus1d_model'), \
             patch('app.main.init_globals'), \
             patch('os.makedirs') as mock_makedirs, \
             patch('os.path.isdir', return_value=True), \
             patch('os.path.abspath') as mock_abspath, \
             patch.object(app, 'mount') as mock_mount, \
             patch('app.main.get_settings') as mock_get_settings:
            
            mock_settings = MagicMock()
            mock_settings.output_dir = "rel_output"
            mock_settings.video_output_dir = "rel_video_output"
            mock_get_settings.return_value = mock_settings
            
            mock_abspath.side_effect = lambda x: f"/abs/{x}"
            
            import anyio
            async def run_lifespan():
                async with lifespan(app):
                    pass
            
            anyio.run(run_lifespan)
            
            mock_abspath.assert_any_call("rel_output")
            mock_abspath.assert_any_call("rel_video_output")
            mock_makedirs.assert_any_call("/abs/rel_output", exist_ok=True)
            mock_makedirs.assert_any_call("/abs/rel_video_output", exist_ok=True)
            
            self.assertEqual(mock_mount.call_count, 2)
            first_call_args = mock_mount.call_args_list[0]
            second_call_args = mock_mount.call_args_list[1]
            
            self.assertEqual(first_call_args[0][0], "/static/outputs")
            self.assertEqual(first_call_args[0][1].directory, "/abs/rel_output")
            self.assertEqual(second_call_args[0][0], "/static/videos")
            self.assertEqual(second_call_args[0][1].directory, "/abs/rel_video_output")

    def test_crop_video_resize_failure_idx_zero(self):
        import cv2
        from unittest.mock import patch
        from app.analysis.tracking import crop_video

        clip = [np.zeros((10, 10, 3), dtype=np.uint8)]
        crop_window = [[(0, 0, 5, 5)]]
        
        with patch('cv2.resize', side_effect=cv2.error("Mocked resize error")):
            result = crop_video(clip, crop_window, player=0, output_size=(128, 176))
            
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (176, 128, 3))
        self.assertTrue(np.all(result[0] == 0))

    def test_motion_features_invalid_player_index(self):
        from app.analysis.motion import compute_motion_features
        player_boxes = [[[(0.0, 0.0, 10.0, 10.0)]]]
        with self.assertRaises(IndexError):
            compute_motion_features(
                player_boxes=player_boxes,
                player=2,
                clip_index=0,
                seq_length=1,
                vid_stride=1,
            )

    def test_write_annotated_video_player_count_mismatch(self):
        from app.video.writer import write_annotated_video
        import tempfile
        import shutil
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            video_path = os.path.join(temp_dir, "test_out.mp4")
            video_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
            player_boxes = [[
                (10, 10, 20, 20),
                (30, 30, 20, 20),
                (50, 50, 20, 20),
            ]]
            predictions = {
                0: {0: 1},
                2: {0: 3},
                5: {0: 4},
            }
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            write_annotated_video(
                video_path=video_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=predictions,
                colors=colors,
                frame_width=100,
                frame_height=100,
                vid_stride=8,
                fps=30.0,
            )
            self.assertTrue(os.path.exists(video_path))
        finally:
            shutil.rmtree(temp_dir)

    def test_writer_clip_index_overflow(self):
        from app.video.writer import write_annotated_video
        import tempfile
        import shutil
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            video_path = os.path.join(temp_dir, "test_overflow.mp4")
            video_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(20)]
            player_boxes = [[(10, 10, 20, 20)]] * 20
            
            predictions_list = {
                0: [1, 2]
            }
            
            predictions_dict = {
                0: {0: 1, 1: 3}
            }
            
            colors = [(255, 0, 0)]
            
            write_annotated_video(
                video_path=video_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=predictions_list,
                colors=colors,
                frame_width=100,
                frame_height=100,
                vid_stride=8,
                fps=30.0,
            )
            self.assertTrue(os.path.exists(video_path))
            
            write_annotated_video(
                video_path=video_path,
                video_frames=video_frames,
                player_boxes=player_boxes,
                predictions=predictions_dict,
                colors=colors,
                frame_width=100,
                frame_height=100,
                vid_stride=8,
                fps=30.0,
            )
            self.assertTrue(os.path.exists(video_path))
        finally:
            shutil.rmtree(temp_dir)

    def test_path_traversal_video_path(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from unittest.mock import patch, MagicMock
        
        with patch('app.main.build_r2plus1d_model', return_value=MagicMock()):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/analysis/run",
                    json={
                        "video_path": "../suspicious_file.mp4",
                        "vlm_mode": "off",
                    }
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Access denied", response.json()["detail"])

    def test_temporal_smoothing_non_contiguous_indices(self):
        records = [
            {
                "player": 0,
                "clip_index": 0,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.7, source="r2plus1d", needs_review=False, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 4,
                "final": FinalDecisionResponse(
                    action_id=3, action="dribble", confidence=0.3, source="r2plus1d", needs_review=True, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 8,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.8, source="r2plus1d", needs_review=False, reason=""
                ),
            },
        ]
        predictions = {0: {0: 6, 4: 3, 8: 6}}
        apply_temporal_smoothing(records, predictions, confidence_threshold=0.6)
        self.assertEqual(records[1]["final"].action, "defense")
        self.assertEqual(predictions[0][4], 6)

    def test_video_capture_release_on_error(self):
        from unittest.mock import patch, MagicMock
        import cv2
        from app.analysis.service import AnalysisService
        from app.analysis.schemas import AnalysisRequest
        from app.config import Settings

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = Exception("Mock CAP error")

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('app.analysis.service.extract_tracked_frames') as mock_etf, \
             patch('app.analysis.service.crop_windows') as mock_cw, \
             patch('app.analysis.service.predict_player_clips') as mock_ppc, \
             patch('app.analysis.service.write_annotated_video') as mock_wav:

             mock_etf.return_value = ([], {}, 100, 100, [])
             mock_cw.return_value = {}
             mock_ppc.return_value = {}

             settings = Settings(video_output_dir="dummy_out")
             service = AnalysisService(settings=settings, model=MagicMock(), device="cpu")

             request = AnalysisRequest(
                 video_path="dummy.mp4",
                 vlm_mode="off",
                 generate_video=True
             )

             with self.assertRaises(Exception) as context:
                 service.run_analysis(request)

             self.assertIn("Mock CAP error", str(context.exception))
             mock_cap.release.assert_called_once()

    def test_fuse_vlm_action_unknown_label(self):
        from app.analysis.fusion import fuse_decision
        from app.analysis.schemas import ModelPrediction, VLMDecisionResponse

        prediction = ModelPrediction(
            action_id=6,
            action="defense",
            confidence=0.5,
            probabilities={"defense": 0.5}
        )
        vlm = VLMDecisionResponse(
            available=True,
            action="unknown_vlm_action_name",
            confidence=0.9,
            needs_review=False,
            reason="VLM proposed an action not in LABEL_TO_ID",
            visible_ball=False,
            raw_response="{}",
        )

        decision = fuse_decision(
            prediction=prediction,
            vlm=vlm,
            high_confidence=0.8,
            low_confidence=0.4
        )

        self.assertEqual(decision.action_id, prediction.action_id)
        self.assertEqual(decision.action, prediction.action)
        self.assertEqual(decision.confidence, prediction.confidence)
        self.assertEqual(decision.source, "r2plus1d")
        self.assertIn("VLM returned unknown action label", decision.reason)

    def test_path_traversal_symlink(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from unittest.mock import patch, MagicMock

        with patch('app.main.build_r2plus1d_model', return_value=MagicMock()):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/analysis/run",
                    json={
                        "video_path": "/tmp/outside_file.mp4",
                        "vlm_mode": "off",
                    }
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Access denied", response.json()["detail"])

    def test_end_frame_clamped(self):
        from unittest.mock import patch, MagicMock
        import numpy as np
        from app.analysis.service import AnalysisService
        from app.analysis.schemas import AnalysisRequest
        from app.config import Settings
        from app.analysis.schemas import ModelPrediction

        dummy_frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]

        with patch('app.analysis.service.extract_tracked_frames') as mock_etf, \
             patch('app.analysis.service.crop_windows') as mock_cw, \
             patch('app.analysis.service.predict_player_clips') as mock_ppc:

             mock_etf.return_value = (dummy_frames, [((0.0, 0.0, 10.0, 10.0),)] * 5, 100, 100, [])
             mock_cw.return_value = {0: [np.zeros((16, 10, 10, 3))]}
             mock_ppc.return_value = {0: [ModelPrediction(action_id=0, action="run", confidence=0.9, probabilities={"run": 0.9})]}

             settings = Settings(seq_length=16, vid_stride=8)
             service = AnalysisService(settings=settings, model=MagicMock(), device="cpu")

             request = AnalysisRequest(
                 video_path="dummy.mp4",
                 vlm_mode="off",
                 generate_video=False
             )

             response = service.run_analysis(request)
             self.assertEqual(len(response.records), 1)
             self.assertEqual(response.records[0].end_frame, 4)

    def test_writer_unknown_action_id(self):
        from unittest.mock import patch, MagicMock
        import numpy as np
        from app.video.writer import write_annotated_video

        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        boxes = [[(10.0, 10.0, 20.0, 20.0)]]
        predictions = {0: {0: 999}}
        colors = [(255, 0, 0)]

        with patch('cv2.VideoWriter') as mock_writer, \
             patch('cv2.putText') as mock_put_text:

             mock_out = MagicMock()
             mock_writer.return_value = mock_out

             write_annotated_video(
                 video_path="dummy_out.mp4",
                 video_frames=frames,
                 player_boxes=boxes,
                 predictions=predictions,
                 colors=colors,
                 frame_width=100,
                 frame_height=100,
                 vid_stride=8
             )

             mock_put_text.assert_called()
             called_args = mock_put_text.call_args[0]
             self.assertEqual(called_args[1], "unknown")

    def test_temporal_smoothing_dict_not_list(self):
        from app.analysis.fusion import apply_temporal_smoothing
        from app.analysis.schemas import FinalDecisionResponse

        records = [
            {
                "player": 0,
                "clip_index": 0,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.7, source="r2plus1d", needs_review=False, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 1,
                "final": FinalDecisionResponse(
                    action_id=3, action="dribble", confidence=0.3, source="r2plus1d", needs_review=True, reason=""
                ),
            },
            {
                "player": 0,
                "clip_index": 2,
                "final": FinalDecisionResponse(
                    action_id=6, action="defense", confidence=0.8, source="r2plus1d", needs_review=False, reason=""
                ),
            },
        ]
        predictions = {0: {0: 6, 1: 3, 2: 6}}
        apply_temporal_smoothing(records, predictions, confidence_threshold=0.6)
        self.assertEqual(records[1]["final"].action, "defense")
        self.assertEqual(predictions[0][1], 6)


if __name__ == "__main__":
    unittest.main()
