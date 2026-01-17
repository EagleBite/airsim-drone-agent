from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image

def bgr_to_jpeg_data_url(img_bgr: np.ndarray, quality: int = 85) -> str:
    """BGR numpy -> JPEG base64 data URL"""
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("期望输入为 BGR 三通道图像 (H,W,3)")

    img_rgb = img_bgr[:, :, ::-1].astype(np.uint8)
    pil_img = Image.fromarray(img_rgb)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"