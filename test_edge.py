"""
侧面管道边缘检测 — 适合近距侧面图
====================================
对于占画面大部分的侧面管道，传统 Canny+Hough 边缘检测反而比 SAM 更准更快。

用法：
  python test_edge.py 照片.jpg
  鼠标拖拽框选管道区域，按空格确认
"""

import sys, math
import cv2
import numpy as np


class ROISelector:
    def __init__(self, image):
        self.image = image
        self.roi = None
        self.drawing = False
        self.start = (0, 0)
        self.window = "Drag ROI around pipe  |  SPACE=confirm  q=quit"

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            disp = self.image.copy()
            cv2.rectangle(disp, self.start, (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window, disp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1 = min(self.start[0], x)
            y1 = min(self.start[1], y)
            x2 = max(self.start[0], x)
            y2 = max(self.start[1], y)
            self.roi = (x1, y1, x2 - x1, y2 - y1)
            disp = self.image.copy()
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(self.window, disp)

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        cv2.putText(self.image, "Drag box around pipe, then SPACE",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(self.window, self.image)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                return None
            elif key == ord(' ') and self.roi is not None:
                cv2.destroyWindow(self.window)
                return self.roi


def detect_pipe_edge_in_roi(image, roi):
    """在 ROI 内找管道边缘线"""
    x, y, w, h = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y + h, x:x + w]

    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_gray = clahe.apply(roi_gray)

    # 高斯模糊
    roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # Canny 边缘
    edges = cv2.Canny(roi_gray, 40, 120)

    # HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                            minLineLength=100, maxLineGap=30)

    if lines is None:
        return None

    # 过滤近似水平线（±30°）
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))))
        if angle < 30:
            candidates.append((x1 + x, y1 + y, x2 + x, y2 + y))  # 转全局坐标

    if not candidates:
        return None

    # 选最长
    best = max(candidates, key=lambda l: math.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2))

    # 计算坡度和管径
    dx = best[2] - best[0]
    dy = best[3] - best[1]
    angle_deg = math.degrees(math.atan2(dy, dx))
    length_px = math.sqrt(dx ** 2 + dy ** 2)

    # 估算管径（取垂线方向的边缘间距）
    nx = -dy / length_px
    ny = dx / length_px
    widths = []
    for t in [0.2, 0.4, 0.6, 0.8]:
        cx = int(best[0] + dx * t)
        cy = int(best[1] + dy * t)
        # 沿法线扫描
        for step in range(5, min(w, h) // 2):
            px1 = int(cx + nx * step)
            py1 = int(cy + ny * step)
            px2 = int(cx - nx * step)
            py2 = int(cy - ny * step)
            # 检查是否在 ROI 和图像范围内
            if (x <= px1 < x + w and y <= py1 < y + h
                    and x <= px2 < x + w and y <= py2 < y + h
                    and 0 <= px1 < image.shape[1] and 0 <= py1 < image.shape[0]
                    and 0 <= px2 < image.shape[1] and 0 <= py2 < image.shape[0]):
                ey1, ex1 = py1 - y, px1 - x
                ey2, ex2 = py2 - y, px2 - x
                if (0 <= ey1 < edges.shape[0] and 0 <= ex1 < edges.shape[1]
                        and 0 <= ey2 < edges.shape[0] and 0 <= ex2 < edges.shape[1]):
                    if edges[ey1, ex1] > 0 and edges[ey2, ex2] > 0:
                        widths.append(step * 2)
                        break

    diam_px = np.median(widths) if widths else 0

    return {
        "line": list(best),
        "angle_deg": round(angle_deg, 1),
        "length_px": round(length_px),
        "diameter_px": round(diam_px),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python test_edge.py <照片.jpg>")
        sys.exit(1)

    path = sys.argv[1]
    image = cv2.imread(path)
    if image is None:
        print(f"无法读取: {path}")
        sys.exit(1)

    # 缩放显示
    h, w = image.shape[:2]
    max_disp = 1200
    scale = 1.0
    if max(w, h) > max_disp:
        scale = max_disp / max(w, h)
        disp = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        disp = image.copy()

    selector = ROISelector(disp)
    roi_disp = selector.run()

    if roi_disp is None:
        print("取消")
        return

    # 映射回原图
    if scale != 1.0:
        x, y, rw, rh = roi_disp
        roi = (int(x / scale), int(y / scale), int(rw / scale), int(rh / scale))
    else:
        roi = roi_disp

    print(f"\nROI: {roi}")
    print("检测中...")
    result = detect_pipe_edge_in_roi(image, roi)

    if result is None:
        print("✗ 未检测到管道边缘")
        return

    x1, y1, x2, y2 = result["line"]
    print(f"\n{'=' * 40}")
    print(f"  管道角度:  {result['angle_deg']}°")
    print(f"  管道长度:  {result['length_px']} px")
    print(f"  管径估算:  {result['diameter_px']} px")
    print(f"{'=' * 40}")

    # 画结果
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(image, (x1, y1), 6, (0, 255, 0), -1)
    cv2.circle(image, (x2, y2), 6, (0, 255, 0), -1)
    cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)

    # 水平参考虚线
    my = (y1 + y2) // 2
    cv2.line(image, (0, my), (w, my), (150, 150, 150), 1)
    cv2.putText(image, f"Angle: {result['angle_deg']} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Diam: {result['diameter_px']:.0f} px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_path = path.rsplit('.', 1)[0] + '_edge_result.jpg'
    cv2.imwrite(out_path, image)
    print(f"\n结果: {out_path}")

    cv2.imshow("Result - Press any key", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
