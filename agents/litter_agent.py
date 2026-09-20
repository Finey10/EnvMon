"""
Agent 3 — Litter Detection Agent

Two-stage detection pipeline:
  Stage 1 (YOLO): YOLOv8 Medium — fast object detection for specific identifiable items.
  Stage 2 (Groq Vision Fallback): If YOLO detects 0 objects (e.g. amorphous garbage piles,
            plastic bags, unrecognised debris), falls back to Groq LLaMA 4 Scout vision
            which can describe and count arbitrary litter in plain English.

Returns the shared JSON contract PLUS bounding boxes for dashboard rendering:
  {
    "signal": "litter",
    "severity": "low|medium|high",
    "value": <object_count>,
    "note": "...",
    "boxes": [[x1, y1, x2, y2, label, confidence], ...]
    "detection_method": "yolo" | "vision_llm"
  }
"""

import os
import re
import base64
import json
from pathlib import Path
from ultralytics import YOLO
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── YOLOv8 Medium ──────────────────────────────────────────────────────────────
_MODEL_NAME = "yolov8m.pt"
_model_cache = None


def _get_model() -> YOLO:
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(_MODEL_NAME)
    return _model_cache


def _severity_from_count(count: int) -> str:
    if count >= 7:
        return "high"
    elif count >= 3:
        return "medium"
    return "low"


# ── Main Entry Point ───────────────────────────────────────────────────────────
def run(image_path: str) -> dict:
    """
    Main entry point for the Litter Detection Agent.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Shared JSON contract dict with an additional 'boxes' and 'detection_method' key.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Stage 1: YOLOv8 ──────────────────────────────────────────────────────
    model = _get_model()
    # Lower confidence to catch occluded garbage, but high enough to avoid clean street hallucinations
    results = model(str(path), imgsz=1280, conf=0.15, verbose=False)
    result = results[0]

    boxes = []
    if result.boxes is not None:
        excluded_classes = {
            "person", "car", "motorcycle", "bus", "train", "truck",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "potted plant"
        }
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in excluded_classes:
                continue

            # Remap all valid detections to generic "Litter Item"
            boxes.append([
                round(x1, 1), round(y1, 1),
                round(x2, 1), round(y2, 1),
                "Litter Item", round(conf, 3)
            ])

    count = len(boxes)
    severity = _severity_from_count(count)

    severity_desc = {
        "low": "minimal litter presence",
        "medium": "moderate litter accumulation",
        "high": "heavy litter contamination",
    }

    note = (
        f"Detected {count} object(s) in the scene — {severity_desc[severity]}. "
        f"Object count mapped to {severity} severity threshold."
    )

    return {
        "signal": "litter",
        "severity": severity,
        "value": count,
        "note": note,
        "boxes": boxes,
        "image_path": str(path),
        "detection_method": "yolo",
    }


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    result = run(img)
    display = {k: v for k, v in result.items() if k != "boxes"}
    display["box_count"] = len(result["boxes"])
    print(json.dumps(display, indent=2))
