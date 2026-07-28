"""
MediaPipe face detector (Tasks API).

DEVIATION FROM SPEC: the Day 6 task targeted the legacy
``mp.solutions.face_detection`` API, but mediapipe's Python-3.13 wheels
(0.10.30+) removed the entire ``mp.solutions`` package. This implementation uses
the current Tasks API (``mediapipe.tasks.python.vision.FaceDetector``) with the
BlazeFace short-range model. The InferenceModel contract and output format are
unchanged — only the mediapipe calls differ. Other consequences of the switch:
  - short-range model (the Tasks API's standard face bundle) rather than the
    legacy ``model_selection=1`` full-range model;
  - the Tasks API returns ABSOLUTE-pixel boxes, so there is no relative->absolute
    conversion to do;
  - a one-time ~230 KB ``.tflite`` model bundle is downloaded to
    settings.models_cache_dir (same auto-download pattern as ultralytics weights).
"""

import logging
import os
import time
import urllib.request
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from api.config import settings
from workers.inference.base import InferenceModel, InferenceResult

logger = logging.getLogger(__name__)

_MODEL_VERSION = "mediapipe-face-v0.10.35"
# Google-hosted BlazeFace short-range detector bundle (pinned version 1).
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_FILENAME = "blaze_face_short_range.tflite"


class FaceDetector(InferenceModel):
    """MediaPipe (Tasks API) BlazeFace short-range face detector."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._min_confidence = min_detection_confidence
        self._model: Optional[mp_vision.FaceDetector] = None

    def _ensure_model_file(self) -> str:
        """Return the local ``.tflite`` path, downloading it once if missing."""
        os.makedirs(settings.models_cache_dir, exist_ok=True)
        path = os.path.join(settings.models_cache_dir, _MODEL_FILENAME)
        if not os.path.exists(path):
            logger.info("downloading MediaPipe face model to %s", path)
            urllib.request.urlretrieve(_MODEL_URL, path)
        return path

    def load(self) -> None:
        """Create the Tasks-API detector, fetching the model bundle if needed."""
        model_path = self._ensure_model_file()
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=self._min_confidence,
        )
        self._model = mp_vision.FaceDetector.create_from_options(options)
        logger.info("MediaPipe face detector loaded (%s)", _MODEL_VERSION)

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_path: str, **options: object) -> InferenceResult:
        """Detect faces in the image and return normalized detections."""
        if self._model is None:
            raise RuntimeError("FaceDetector.predict called before load()")
        if not os.path.exists(image_path):
            raise ValueError(f"could not read image {image_path}")

        # Post-filter on top of the model's own min_detection_confidence (set at
        # load). Default 0.0 = no extra filtering unless the caller asks for it.
        threshold = float(options.get("confidence_threshold", 0.0))

        # create_from_file decodes the image; the Tasks API returns absolute
        # pixel bounding boxes, so no relative->absolute conversion is needed.
        mp_image = mp.Image.create_from_file(image_path)

        start = time.perf_counter()
        result = self._model.detect(mp_image)
        inference_ms = int((time.perf_counter() - start) * 1000)

        detections = self._extract_detections(result, threshold)
        logger.info(
            "MediaPipe face inference on %s: %d detection(s) in %d ms",
            image_path,
            len(detections),
            inference_ms,
        )
        return InferenceResult(
            detections=detections,
            inference_ms=inference_ms,
            model_version=_MODEL_VERSION,
        )

    @staticmethod
    def _extract_detections(result: object, threshold: float) -> list[dict]:
        """Convert Tasks-API detections into our normalized detection dicts."""
        detections: list[dict] = []
        for detection in getattr(result, "detections", None) or []:
            score = float(detection.categories[0].score)
            if score < threshold:
                continue
            box = detection.bounding_box  # absolute pixels
            x1 = max(0, round(box.origin_x))
            y1 = max(0, round(box.origin_y))
            x2 = round(box.origin_x + box.width)
            y2 = round(box.origin_y + box.height)
            detections.append(
                {
                    "label": "face",
                    "confidence": round(score, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections
