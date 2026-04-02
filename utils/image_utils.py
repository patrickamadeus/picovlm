import io
import os
from typing import Any

import numpy as np
from PIL import Image


def _normalize_numpy_image(image: np.ndarray) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported numpy image shape: {image.shape}.")
    if image.dtype == np.uint8:
        return image
    arr = image
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0 if float(np.nanmax(arr)) <= 1.0 else 255.0)
        if float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
    else:
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def coerce_image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(_normalize_numpy_image(image)).convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as img:
            return img.convert("RGB")
    if isinstance(image, dict):
        for key in ("image", "array", "bytes", "path"):
            if image.get(key) is not None:
                return coerce_image_to_pil(image[key])
    raise ValueError(f"Unsupported image type: {type(image)}")
