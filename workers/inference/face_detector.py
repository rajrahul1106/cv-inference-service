"""
YOLOv8 face detector.

Uses a YOLOv8-nano model fine-tuned for face detection (the
``arnabdhar/YOLOv8-Face-Detection`` weights on Hugging Face), wrapped behind the
InferenceModel interface. This follows the same pattern as
workers/inference/yolo_detector.py — ultralytics does the forward pass and
returns absolute-pixel xyxy boxes.

Replaces the previous MediaPipe BlazeFace implementation, which was optimized
for close-up faces: it missed angled/multiple faces, emitted a tail of low-score
false positives, and reported square boxes in its own padded coordinate space
rather than true image pixels.
"""

import logging
import time
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from api.config import settings
from workers.inference.base import InferenceModel, InferenceResult

logger = logging.getLogger(__name__)

# Match YOLO's default confidence threshold.
_DEFAULT_CONFIDENCE = 0.25
_MODEL_VERSION = "yolov8n-face-v1.0"
# Pretrained face weights (single class: FACE), ~6 MB.
_HF_REPO_ID = "arnabdhar/YOLOv8-Face-Detection"
_HF_FILENAME = "model.pt"


class FaceDetector(InferenceModel):
    """YOLOv8-nano face detector."""

    def __init__(self, min_detection_confidence: float = _DEFAULT_CONFIDENCE) -> None:
        self._min_confidence = min_detection_confidence
        self._model: Optional[YOLO] = None

    def _ensure_weights(self) -> str:
        """Return the local weights path, downloading them once if missing."""
        return hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_HF_FILENAME,
            cache_dir=settings.models_cache_dir,
        )

    def load(self) -> None:
        """Fetch the face weights (first use per machine) and load the model."""
        logger.info("loading YOLO face weights from %s", _HF_REPO_ID)
        self._model = YOLO(self._ensure_weights())
        logger.info("YOLO face weights loaded (%s)", _MODEL_VERSION)

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_path: str, **options: object) -> InferenceResult:
        """Detect faces in the image and return normalized detections."""
        if self._model is None:
            raise RuntimeError("FaceDetector.predict called before load()")

        confidence = float(
            options.get("confidence_threshold", self._min_confidence)
        )

        start = time.perf_counter()
        # verbose=False keeps ultralytics from printing to stdout (we log ourselves).
        results = self._model.predict(source=image_path, conf=confidence, verbose=False)
        inference_ms = int((time.perf_counter() - start) * 1000)

        detections = self._extract_detections(results)
        logger.info(
            "YOLO face inference on %s: %d detection(s) in %d ms",
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
    def _extract_detections(results: list) -> list[dict]:
        """Flatten ultralytics Results into ``{label, confidence, bbox}`` dicts."""
        detections: list[dict] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        # The model's single class is "FACE"; normalize casing.
                        "label": "face",
                        "confidence": round(float(box.conf[0]), 4),
                        # Pixel coordinates, rounded to ints (SPEC §6 example).
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    }
                )
        return detections
