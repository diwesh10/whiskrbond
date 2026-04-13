"""
image_utils.py
--------------
Handles loading images from:
  - Local file paths (jpg, jpeg, png, webp, gif)
  - HTTP/HTTPS URLs
  - PIL Image objects directly

All images are returned as RGB PIL Images.
"""

import io
import requests
from pathlib import Path
from PIL import Image
from typing import Union


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

REQUEST_TIMEOUT = 15  # seconds
REQUEST_HEADERS = {
    "User-Agent": "WhiskerBond/1.0 (Pet Breed Identifier; +https://github.com/whiskerbond)",
}


def load_image(source: Union[str, Path]) -> Image.Image:
    """
    Load an image from a file path or URL and return an RGB PIL Image.

    Args:
        source: Local file path or HTTP(S) URL

    Returns:
        PIL.Image.Image in RGB mode

    Raises:
        FileNotFoundError: If local file doesn't exist
        ValueError: If URL returns non-image content or unsupported format
        requests.RequestException: On network errors
    """
    source = str(source)

    if source.startswith("http://") or source.startswith("https://"):
        return _load_from_url(source)
    else:
        return _load_from_path(source)


def _load_from_path(path: str) -> Image.Image:
    """Load image from local filesystem."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    suffix = p.suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    img = Image.open(p)
    return img.convert("RGB")


def _load_from_url(url: str) -> Image.Image:
    """Download and load image from a URL."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError(f"Timed out downloading image from: {url}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP {e.response.status_code} downloading image from: {url}")

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type and not url.lower().endswith(
        tuple(SUPPORTED_EXTENSIONS)
    ):
        raise ValueError(
            f"URL does not appear to be an image. Content-Type: {content_type}"
        )

    img = Image.open(io.BytesIO(response.content))
    return img.convert("RGB")


def validate_image(img: Image.Image) -> None:
    """
    Basic sanity checks on a loaded image.
    Raises ValueError if something looks wrong.
    """
    w, h = img.size
    if w < 32 or h < 32:
        raise ValueError(f"Image too small: {w}×{h}px. Minimum 32×32px required.")
    if w > 8000 or h > 8000:
        raise ValueError(
            f"Image very large: {w}×{h}px. Please use images under 8000×8000px."
        )
