"""
Fire/smoke detector — PLACEHOLDER.

There is no real fire-trained model wired in yet. This composes the base
COCO-pretrained YOLOv8 detector (workers/inference/yolo_detector.py) and keeps
only detections whose label is fire/smoke. COCO contains no such classes, so
``predict()`` returns an empty list today — but the emptiness emerges from a
real inference pass, and swapping ``weights_path`` for genuinely fire-trained
weights (with fire/smoke classes) makes this work with **zero code changes**.

The placeholder status is flagged three ways: this docstring, the
``model_version`` string, and a warning logged at load time. Do not present this
as a working fire detector until real weights are trained or sourced.
"""

import logging

from workers.inference.base import InferenceModel, InferenceResult
from workers.inference.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

# Labels a real fire model would emit; base COCO has none of these.
_FIRE_LABELS = {"fire", "smoke"}
_MODEL_VERSION = "yolov8n-fire-placeholder-v1"


class FireDetector(InferenceModel):
    """Placeholder fire/smoke detector built on base YOLOv8 + label filtering."""

    def __init__(self, weights_path: str = "yolov8n.pt") -> None:
        self._weights_path = weights_path
        self._detector = YOLODetector(weights_path)

    def load(self) -> None:
        self._detector.load()
        logger.warning(
            "FireDetector loaded PLACEHOLDER weights (%s): base COCO has no "
            "fire/smoke classes, so detections will be empty until a real fire "
            "model is trained or sourced",
            self._weights_path,
        )

    def is_loaded(self) -> bool:
        return self._detector.is_loaded()

    def predict(self, image_path: str, **options: object) -> InferenceResult:
        """Run base detection and keep only fire/smoke labels (empty on COCO)."""
        result = self._detector.predict(image_path, **options)
        fire = [d for d in result.detections if d["label"] in _FIRE_LABELS]
        logger.info(
            "fire detection on %s: %d fire/smoke detection(s) in %d ms",
            image_path,
            len(fire),
            result.inference_ms,
        )
        return InferenceResult(
            detections=fire,
            inference_ms=result.inference_ms,
            model_version=_MODEL_VERSION,
        )
