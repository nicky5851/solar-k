"""E001 classical baseline: a dark-feature map on the preprocessed image.

Method: morphological black top-hat. Filaments are thin, dark features a
few pixels wide; black top-hat isolates dark structures smaller than its
structuring element while ignoring slower background variation -- which is
exactly what's left after preprocess.py's limb-darkening correction, so no
extra background subtraction step is needed here.
"""
from __future__ import annotations

import cv2
import numpy as np


def dark_feature_map(processed: np.ndarray, disk_mask: np.ndarray, kernel_radius: int = 9) -> np.ndarray:
    """``processed`` is preprocess.preprocess_image's float32 output in ~[0,1]."""
    img_u8 = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_radius + 1, 2 * kernel_radius + 1))
    blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel)
    probs = blackhat.astype(np.float32) / 255.0
    probs[~disk_mask] = 0.0
    return probs
