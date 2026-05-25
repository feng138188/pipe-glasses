"""
DINO 管道检测 API 服务
======================
GPU 端运行，接收手机发来的图像帧，返回 DINO 检测到的管道位置。

启动: python dino_server.py --port 8777
"""

import io
import time
import argparse
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Pipe DINO Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 延迟加载 DINO 模型
dino_pipe = None


def get_dino():
    global dino_pipe
    if dino_pipe is None:
        from transformers import pipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  加载 DINO 模型 (device={device})...")
        dino_pipe = pipeline(
            model="IDEA-Research/grounding-dino-tiny",
            task="zero-shot-object-detection",
            device=device,
        )
        print("  模型就绪")
    return dino_pipe


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    接收 JPEG 图像，返回管道检测结果。

    请求: POST /detect  (multipart form, field "file" = JPEG binary)
    响应: {
        "detected": true/false,
        "box": [x1, y1, x2, y2],      // DINO 检测框（像素坐标）
        "score": 0.85,                  // 置信度
        "time_ms": 580                  // 推理耗时
    }
    """
    t0 = time.time()
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = image.size

    pipe = get_dino()
    results = pipe(image, candidate_labels=["pipe", "tube", "cylinder"], threshold=0.2)

    elapsed = (time.time() - t0) * 1000

    if not results:
        return {"detected": False, "box": None, "score": 0, "time_ms": round(elapsed)}

    # 取最高分
    best = max(results, key=lambda r: r["score"])
    box = best["box"]
    return {
        "detected": True,
        "box": [
            int(box["xmin"] * w), int(box["ymin"] * h),
            int(box["xmax"] * w), int(box["ymax"] * h),
        ],
        "score": round(best["score"], 3),
        "label": best["label"],
        "time_ms": round(elapsed),
    }


@app.post("/sam")
async def sam_detect(file: UploadFile = File(...), x: int = 320, y: int = 240):
    """
    SAM 分割: 用户点击管道 → 精准分割 → 拟合轴线。

    请求: POST /sam  (multipart form, file=JPEG, x=tap_x, y=tap_y)
    响应: {
        "detected": true,
        "line": [x1, y1, x2, y2],
        "angle_deg": 3.2,
        "diameter_px": 45,
        "confidence": 0.95,
        "time_ms": 1200
    }
    """
    t0 = time.time()
    raw = await file.read()
    image_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"detected": False, "error": "invalid image"}

    from sam_detect import detect_pipe_with_tap
    result = detect_pipe_with_tap(image_bgr, x, y)

    elapsed = (time.time() - t0) * 1000
    if result is None:
        return {"detected": False, "time_ms": round(elapsed)}
    result["time_ms"] = round(elapsed)
    result["detected"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"管道 API 启动: http://{args.host}:{args.port}")
    print("  端点: /detect  /sam  /ellipse  /health")
    get_dino()
    uvicorn.run(app, host=args.host, port=args.port)


@app.post("/ellipse")
async def ellipse_detect(file: UploadFile = File(...),
                         x1: int = 0, y1: int = 0,
                         x2: int = 640, y2: int = 480):
    """
    椭圆法管道检测: 框选管口 → 拟合椭圆 → 推算管道轴向。

    请求: POST /ellipse  (file=JPEG, x1,y1,x2,y2=ROI框坐标)
    响应: {detected, angle_deg, view_angle_deg, line, ellipse}
    """
    t0 = time.time()
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"detected": False, "error": "invalid image"}

    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    roi = (x1, y1, x2 - x1, y2 - y1)
    if roi[2] < 10 or roi[3] < 10:
        return {"detected": False, "error": "ROI too small"}

    # 椭圆检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi_gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for cnt in contours:
        if len(cnt) < 5: continue
        area = cv2.contourArea(cnt)
        if area < 100: continue
        e = cv2.fitEllipse(cnt)
        (_, _), (a, b), _ = e
        ratio = min(a, b) / max(a, b)
        if ratio > 0.3 and area > best_area:
            best_area = area
            best = e

    if best is None:
        return {"detected": False, "time_ms": round((time.time() - t0) * 1000)}

    (ecx, ecy), (ea, eb), eang = best
    ecx += x1; ecy += y1

    # 推算管道轴向
    import math
    if ea > eb:
        pipe_ang_rad = math.radians(eang + 90)
        major, minor = ea, eb
    else:
        pipe_ang_rad = math.radians(eang)
        major, minor = eb, ea
    ratio = min(1.0, max(0.0, minor / major))
    view_angle = math.degrees(math.asin(ratio))

    hl = major * 1.5
    lx1 = ecx - hl * math.cos(pipe_ang_rad)
    ly1 = ecy - hl * math.sin(pipe_ang_rad)
    lx2 = ecx + hl * math.cos(pipe_ang_rad)
    ly2 = ecy + hl * math.sin(pipe_ang_rad)
    if lx1 > lx2: lx1, ly1, lx2, ly2 = lx2, ly2, lx1, ly1
    slope_angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))

    elapsed = (time.time() - t0) * 1000
    return {
        "detected": True,
        "angle_deg": round(slope_angle, 1),
        "view_angle_deg": round(view_angle, 1),
        "line": [int(lx1), int(ly1), int(lx2), int(ly2)],
        "ellipse": {"cx": int(ecx), "cy": int(ecy),
                     "a": int(ea), "b": int(eb), "angle": round(eang, 1)},
        "apparent_diam_px": round(major, 1),
        "time_ms": round(elapsed),
    }
