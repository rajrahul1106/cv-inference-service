"""
Per-process model registry (singleton cache).

Models are expensive to load (~1-3 s) but cheap to reuse, so each worker process
loads a given model once and caches it here, keyed by model_type. Celery prefork
gives every worker child its own module state, so this dict is naturally
per-process — exactly the caching granularity we want, and it means loading
happens inside the forked child (no sharing torch models across fork()).

Loading is lazy: a model type is loaded the first time get_model() is asked for
it, then reused for the life of the process. See the Day 5 plan for why lazy
beats eager (worker_process_init) and per-task loading.

CLAUDE.md flags this singleton as author-owned; implemented here per the Day 5
task's explicit spec.
"""

import logging
import time

from workers.inference.base import InferenceModel
from workers.inference.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

# model_type -> loaded model, populated lazily by get_model().
_MODELS: dict[str, InferenceModel] = {}

# model_type -> the class that implements it. Extended on Day 7 (face, fire).
_MODEL_CLASSES: dict[str, type[InferenceModel]] = {
    "yolo": YOLODetector,
}


def get_model(model_type: str) -> InferenceModel:
    """Return the loaded model for ``model_type``, loading + caching on first use.

    The first call for a given type in a process pays the load cost; subsequent
    calls return the cached instance. Raises ``KeyError`` for unknown types.
    """
    cached = _MODELS.get(model_type)
    if cached is not None:
        return cached

    try:
        model_cls = _MODEL_CLASSES[model_type]
    except KeyError:
        raise KeyError(
            f"unknown model_type {model_type!r}; "
            f"known types: {sorted(_MODEL_CLASSES)}"
        ) from None

    logger.info("loading model for type %r", model_type)
    start = time.perf_counter()
    model = model_cls()
    model.load()
    _MODELS[model_type] = model
    logger.info(
        "loaded model %r in %d ms", model_type, int((time.perf_counter() - start) * 1000)
    )
    return model


def clear_registry() -> None:
    """Drop all cached models. Intended for tests that need clean process state."""
    _MODELS.clear()
