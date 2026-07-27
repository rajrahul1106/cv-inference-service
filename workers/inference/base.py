"""
The inference model contract.

Every model the workers serve (YOLO today; MediaPipe face + fire on Day 7)
implements ``InferenceModel``: ``load()`` once per process, then ``predict()``
per image. Keeping this interface narrow lets the model registry
(workers/model_registry.py) treat all models uniformly and keeps the task code
(workers/tasks.py) model-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Normalized output of a single ``predict()`` call.

    ``detections`` is a list of ``{"label": str, "confidence": float,
    "bbox": [x1, y1, x2, y2]}`` dicts (bbox in pixel coordinates).
    """

    detections: list[dict]
    inference_ms: int
    model_version: str


class InferenceModel(ABC):
    """Abstract base for a loadable, predicting computer-vision model."""

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Called once per worker process."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: str, **options: object) -> InferenceResult:
        """Run inference on the image at ``image_path`` and return results."""
        raise NotImplementedError

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True once ``load()`` has completed successfully."""
        raise NotImplementedError
