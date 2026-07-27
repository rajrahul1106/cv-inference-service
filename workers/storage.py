"""
Image handling utilities for the worker: download an input image, ensure the
working directories exist, and clean up temporary files after inference.
"""

import logging
import os
from urllib.parse import urlparse
from uuid import uuid4

import requests

from api.config import settings

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 30
# content-type -> file extension for the image formats we accept.
_EXTENSION_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
_KNOWN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Servers/CDNs frequently serve images under a generic type; fall back to the
# URL's extension in that case rather than rejecting a real image.
_GENERIC_CONTENT_TYPES = {"application/octet-stream", "binary/octet-stream", ""}


def ensure_dirs() -> None:
    """Create the upload and annotated directories if they don't exist."""
    for path in (settings.upload_dir, settings.annotated_dir):
        os.makedirs(path, exist_ok=True)


def _resolve_extension(url: str, content_type: str) -> str:
    """Return the file extension to save under, or raise if this isn't an image.

    Prefers the content-type; when the server sends a generic type, falls back to
    a known image extension in the URL path. Preserves the URL's own extension
    when the content-type is itself a known image type.
    """
    url_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if content_type in _EXTENSION_BY_TYPE:
        return url_ext if url_ext in _KNOWN_EXTENSIONS else _EXTENSION_BY_TYPE[content_type]
    if content_type in _GENERIC_CONTENT_TYPES and url_ext in _KNOWN_EXTENSIONS:
        return url_ext
    raise ValueError(
        f"not an image: content-type {content_type!r}, url extension {url_ext!r} "
        f"for {url}"
    )


def download_image(url: str, dest_dir: str, max_size_mb: int = 10) -> str:
    """Download an image to ``dest_dir`` and return the saved file path.

    Validates content-type (must be a known image) and content-length (must be
    under ``max_size_mb``) before reading the body, and re-checks the size while
    streaming in case the header lied or was absent. Raises ``TimeoutError`` on
    timeout and ``ValueError`` on any validation/transport failure.
    """
    max_bytes = max_size_mb * 1024 * 1024

    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS, stream=True)
    except requests.Timeout as exc:
        raise TimeoutError(f"timed out downloading {url}") from exc
    except requests.RequestException as exc:
        raise ValueError(f"failed to download {url}: {exc}") from exc

    with response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ValueError(f"download {url} failed: {exc}") from exc

        content_type = (
            response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        )
        extension = _resolve_extension(url, content_type)

        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            raise ValueError(
                f"image at {url} is {content_length} bytes, "
                f"exceeds {max_size_mb} MB limit"
            )

        dest_path = os.path.join(dest_dir, f"{uuid4()}{extension}")
        downloaded = 0
        with open(dest_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    fh.close()
                    os.remove(dest_path)
                    raise ValueError(
                        f"image at {url} exceeds {max_size_mb} MB limit while streaming"
                    )
                fh.write(chunk)

    logger.info("downloaded %s -> %s (%d bytes)", url, dest_path, downloaded)
    return dest_path


def cleanup_image(path: str) -> None:
    """Best-effort delete of a temporary image. Logs but never raises."""
    try:
        os.remove(path)
        logger.info("cleaned up %s", path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.error("failed to clean up %s: %s", path, exc)
