"""
Fire and smoke detector (custom-trained YOLOv8s).

Wraps ``ultralytics.YOLO`` behind the InferenceModel interface, exactly like
workers/inference/yolo_detector.py. The weights are trained in-house on a
12,733-image Roboflow fire/smoke dataset (2 classes: fire, smoke) and live in
the repo at models/yolov8n_fire.pt, so no download is needed at load time.

Class names are read from the loaded model rather than hardcoded, so retrained
weights with different or additional classes work without code changes.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from workers.inference.base import InferenceModel, InferenceResult

logger = logging.getLogger(__name__)

# Match YOLO's default confidence threshold.
_DEFAULT_CONFIDENCE = 0.25
# yolov8s architecture, custom-trained (the filename's "n" is historical).
_MODEL_VERSION = "yolov8s-fire-custom-v1"
# Resolved from this file so the worker finds the weights regardless of its CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHTS = str(_PROJECT_ROOT / "models" / "yolov8n_fire.pt")


class FireDetector(InferenceModel):
    """Custom-trained YOLOv8s fire/smoke detector."""

    def __init__(self, weights_path: str = _DEFAULT_WEIGHTS) -> None:
        self._weights_path = weights_path
        self._model: Optional[YOLO] = None

    def load(self) -> None:
        """Load the custom fire weights from disk (once per worker process)."""
        logger.info("loading fire weights from %s", self._weights_path)
        self._model = YOLO(self._weights_path)
        logger.info(
            "fire weights loaded (%s), classes=%s",
            _MODEL_VERSION,
            list(self._model.names.values()),
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_path: str, **options: object) -> InferenceResult:
        """Detect fire/smoke in the image and return normalized detections."""
        if self._model is None:
            raise RuntimeError("FireDetector.predict called before load()")

        confidence = float(options.get("confidence_threshold", _DEFAULT_CONFIDENCE))

        start = time.perf_counter()
        # verbose=False keeps ultralytics from printing to stdout (we log ourselves).
        results = self._model.predict(source=image_path, conf=confidence, verbose=False)
        inference_ms = int((time.perf_counter() - start) * 1000)

        detections = self._extract_detections(results)
        logger.info(
            "fire inference on %s: %d detection(s) in %d ms",
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
            names = result.names  # class-id -> label ({0: 'fire', 1: 'smoke'})
            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": names[class_id],
                        "confidence": round(float(box.conf[0]), 4),
                        # Pixel coordinates, rounded to ints (SPEC §6 example).
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    }
                )
        return detections
