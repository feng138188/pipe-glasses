"""
MobileSAM 管道检测（升级版）
===============================
114ms GPU 推理 vs 原版 SAM 1500ms，13x 加速。
模型 39MB，可导出 ONNX 跑在手机上。

原理：用户点击管道 → MobileSAM 像素级分割 → 拟合圆柱轴线
"""

import sys, os
import math
import numpy as np
import cv2
from typing import Optional

# MobileSAM 依赖
sys.path.insert(0, "/tmp/mobilesam_repo")
from mobile_sam import sam_model_registry, SamPredictor
import torch

_sam = None


def load_sam():
    global _sam
    if _sam is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  加载 MobileSAM (device={device})...")
        model = sam_model_registry["vit_t"](checkpoint="/tmp/mobile_sam.pth")
        model.to(device).eval()
        _sam = SamPredictor(model)
        print(f"  MobileSAM 就绪 ({'GPU' if device=='cuda' else 'CPU'})")
    return _sam


def detect_pipe_with_tap(image_bgr: np.ndarray,
                         tap_x: int, tap_y: int) -> Optional[dict]:
    """
    点击管道 → MobileSAM 分割 → 拟合轴线。

    返回: {
        "line": [x1, y1, x2, y2],
        "angle_deg": 3.2,
        "diameter_px": 45,
        "confidence": 0.95,
        "box": [x1, y1, x2, y2],
    } 或 None
    """
    h, w = image_bgr.shape[:2]
    tap_x = max(0, min(w - 1, tap_x))
    tap_y = max(0, min(h - 1, tap_y))

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor = load_sam()
    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[tap_x, tap_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    if scores[0] < 0.5:
        return None

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]
    score = float(scores[best_idx])

    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return None

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect

    if rw < rh:
        rw, rh = rh, rw
        angle += 90

    angle_rad = math.radians(angle)
    half_len = rw / 2
    x1 = cx - half_len * math.cos(angle_rad)
    y1 = cy - half_len * math.sin(angle_rad)
    x2 = cx + half_len * math.cos(angle_rad)
    y2 = cy + half_len * math.sin(angle_rad)

    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1

    return {
        "line": [int(x1), int(y1), int(x2), int(y2)],
        "angle_deg": round(math.degrees(math.atan2(y2 - y1, x2 - x1)), 1),
        "diameter_px": round(rh, 1),
        "confidence": round(score, 2),
        "box": [int(cx - rw / 2), int(cy - rh / 2), int(cx + rw / 2), int(cy + rh / 2)],
    }
