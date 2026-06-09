from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from hybrid_analysis import run_hybrid_analysis


DEFAULT_VIDEO = "/Users/ppt/Documents/301f7b05051c278427b8dd3794447137.mp4"
DEFAULT_BOXES = "service_inputs/301f7b05051c278427b8dd3794447137_boxes.json"
DEFAULT_JSON = "analysis_outputs/301f7b05051c278427b8dd3794447137_hybrid.json"
DEFAULT_VIDEO_OUTPUT = "output_videos/301f7b05051c278427b8dd3794447137_hybrid.mp4"


def make_analysis_args(query: dict[str, list[str]]) -> SimpleNamespace:
    def one(name: str, default: str) -> str:
        return query.get(name, [default])[0]

    def one_int(name: str, default: int | None) -> int | None:
        value = query.get(name, [None])[0]
        return int(value) if value not in (None, "") else default

    return SimpleNamespace(
        video=one("video", DEFAULT_VIDEO),
        json_output=one("json_output", DEFAULT_JSON),
        video_output=one("video_output", DEFAULT_VIDEO_OUTPUT),
        detector="tracker",
        tracker=one("tracker", "CSRT"),
        boxes_file=one("boxes_file", DEFAULT_BOXES),
        headless=True,
        seq_length=one_int("seq_length", 16),
        vid_stride=one_int("vid_stride", 8),
        batch_size=one_int("batch_size", 8),
        model_path=one("model_path", "model_checkpoints/r2plus1d_augmented-2/"),
        base_model_name=one("base_model_name", "r2plus1d_multiclass"),
        start_epoch=one_int("start_epoch", 19),
        lr=float(one("lr", "0.0001")),
        vlm_mode=one("vlm_mode", "low-confidence"),
        ollama_model=one("ollama_model", "qwen3-vl:4b"),
        ollama_host=one("ollama_host", "http://127.0.0.1:11434"),
        ollama_timeout=float(one("ollama_timeout", "45")),
        vlm_frames=one_int("vlm_frames", 1),
        vlm_image_width=one_int("vlm_image_width", 224),
        max_vlm_clips=one_int("max_vlm_clips", 4),
        low_confidence=float(one("low_confidence", "0.55")),
        high_confidence=float(one("high_confidence", "0.75")),
        smoothing_confidence=float(one("smoothing_confidence", "0.6")),
        max_frames=one_int("max_frames", 360),
    )


class HybridServiceHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.write_html()
            return
        if parsed.path == "/analyze":
            self.handle_analyze(parse_qs(parsed.query))
            return
        if parsed.path == "/summary":
            self.write_existing_summary()
            return
        super().do_GET()

    def handle_analyze(self, query: dict[str, list[str]]) -> None:
        args = make_analysis_args(query)
        force = query.get("force", ["0"])[0] in {"1", "true", "yes"}
        if os.path.exists(args.json_output) and not force:
            with open(args.json_output) as fp:
                payload = json.load(fp)
        else:
            payload = run_hybrid_analysis(args)
        self.write_json(
            {
                "summary": payload["summary"],
                "json_url": "/" + args.json_output,
                "video_url": "/" + args.video_output if args.video_output else None,
                "video": payload["video"],
                "frame_size": payload["frame_size"],
                "seq_length": payload["seq_length"],
                "vid_stride": payload["vid_stride"],
                "vlm_mode": payload["vlm_mode"],
            }
        )

    def write_existing_summary(self) -> None:
        if not os.path.exists(DEFAULT_JSON):
            self.write_json({"error": "No analysis output yet. Open /analyze first."}, status=404)
            return
        with open(DEFAULT_JSON) as fp:
            payload = json.load(fp)
        self.write_json(payload["summary"])

    def write_html(self) -> None:
        body = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Hybrid Basketball Analysis</title></head>
<body>
  <h1>Hybrid Basketball Analysis Service</h1>
  <p>Default input: /Users/ppt/Documents/301f7b05051c278427b8dd3794447137.mp4</p>
  <ul>
    <li><a href="/analyze">Run or view default analysis</a></li>
    <li><a href="/summary">Summary JSON</a></li>
    <li><a href="/analysis_outputs/301f7b05051c278427b8dd3794447137_hybrid.json">Full JSON</a></li>
    <li><a href="/output_videos/301f7b05051c278427b8dd3794447137_hybrid.mp4">Annotated video</a></li>
  </ul>
</body>
</html>
"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve hybrid basketball analysis outputs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), HybridServiceHandler)
    print(f"Hybrid service running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
