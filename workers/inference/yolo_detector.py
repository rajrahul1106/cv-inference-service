"""
YOLOv8 object detector (ultralytics).

Wraps ``ultralytics.YOLO`` behind the InferenceModel interface. Weights load
once per process (via the model registry); ``predict()`` runs a single forward
pass and normalizes ultralytics' output into our detection dicts.
"""

import logging
import time
from typing import Optional

from ultralytics import YOLO

from workers.inference.base import InferenceModel, InferenceResult

logger = logging.getLogger(__name__)

# YOLO's own default confidence threshold.
_DEFAULT_CONFIDENCE = 0.25
# Semantic version identifying the model architecture + weights.
_MODEL_VERSION = "yolov8n-cocopretrained-v1"


class YOLODetector(InferenceModel):
    """COCO-pretrained YOLOv8-nano detector."""

    def __init__(self, weights_path: str = "yolov8n.pt") -> None:
        self._weights_path = weights_path
        self._model: Optional[YOLO] = None

    def load(self) -> None:
        """Instantiate the ultralytics model.

        On first use per machine, ultralytics auto-downloads the ~6 MB nano
        weights, so this needs network access once.
        """
        logger.info("loading YOLO weights from %s", self._weights_path)
        self._model = YOLO(self._weights_path)
        logger.info("YOLO weights loaded (%s)", _MODEL_VERSION)

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_path: str, **options: object) -> InferenceResult:
        """Detect objects in the image and return normalized detections."""
        if self._model is None:
            raise RuntimeError("YOLODetector.predict called before load()")

        confidence = float(options.get("confidence_threshold", _DEFAULT_CONFIDENCE))

        start = time.perf_counter()
        # verbose=False keeps ultralytics from printing to stdout (we log ourselves).
        results = self._model.predict(source=image_path, conf=confidence, verbose=False)
        inference_ms = int((time.perf_counter() - start) * 1000)

        detections = self._extract_detections(results)
        logger.info(
            "YOLO inference on %s: %d detection(s) in %d ms",
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
            names = result.names  # class-id -> label
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
