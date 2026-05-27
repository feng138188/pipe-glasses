"""
Grounding DINO 管道检测器
=========================
零样本检测 — 用文字 "pipe" 提示即可在画面中找到管道，
无需任何训练数据。

模型: IDEA-Research/grounding-dino-tiny (轻量版, ~60MB)
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple, List

# 延迟加载（首次推理时自动下载模型）
_pipe = None


def _load_model():
    global _pipe
    if _pipe is not None:
        return _pipe

    from transformers import pipeline

    print("  正在加载 Grounding DINO 模型（首次需下载 ~60MB）...")
    _pipe = pipeline(
        model="IDEA-Research/grounding-dino-tiny",
        task="zero-shot-object-detection",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("  模型加载完成")
    return _pipe


def detect_pipes(image: np.ndarray,
                 confidence: float = 0.25) -> List[dict]:
    """
    在图像中检测管道/管材。

    返回: [{"box": (x1,y1,x2,y2), "score": 0.85}, ...]
    """
    from PIL import Image

    model = _load_model()
    h, w = image.shape[:2]

    # Grounding DINO 在 transformers 5.x 需要 PIL Image
    pil_img = Image.fromarray(image[..., ::-1])  # BGR → RGB

    results = model(
        pil_img,
        candidate_labels=["pipe", "tube", "cylinder"],
        threshold=confidence,
    )

    boxes = []
    for r in results:
        box = r["box"]
        x1 = int(box["xmin"] * w)
        y1 = int(box["ymin"] * h)
        x2 = int(box["xmax"] * w)
        y2 = int(box["ymax"] * h)
        boxes.append({
            "box": (x1, y1, x2, y2),
            "score": r["score"],
            "label": r["label"],
        })

    return boxes


def fit_line_in_box(image: np.ndarray,
                    box: Tuple[int, int, int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    在检测框内用 Canny + Hough 找最长的近似水平线段。

    返回: ((x1,y1), (x2,y2)) 或 None
    """
    import cv2

    x1, y1, x2, y2 = box
    # 扩大一点边框
    margin = 10
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                            minLineLength=60, maxLineGap=20)
    if lines is None:
        return None

    best, best_len = None, 0
    for line in lines:
        bx1, by1, bx2, by2 = line[0]
        angle = math.degrees(math.atan2(abs(by2 - by1), abs(bx2 - bx1)))
        if angle > 30:
            continue
        ln = math.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)
        if ln > best_len:
            best_len = ln
            # 转回原图坐标
            best = ((bx1 + x1, by1 + y1), (bx2 + x1, by2 + y1))

    return best


def detect_pipe_line(image: np.ndarray,
                     confidence: float = 0.25) -> Tuple[Optional[Tuple], Optional[dict]]:
    """
    一站式检测：DINO 找管道 → 框内拟合管线。

    返回: (line, detection_info)
      line: ((x1,y1), (x2,y2)) 或 None
      info: {"box": (x1,y1,x2,y2), "score": 0.85} 或 None
    """
    boxes = detect_pipes(image, confidence)
    if not boxes:
        return None, None

    # 取最高置信度
    for det in sorted(boxes, key=lambda b: b["score"], reverse=True):
        line = fit_line_in_box(image, det["box"])
        if line is not None:
            return line, det

    return None, boxes[0]
