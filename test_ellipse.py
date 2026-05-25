"""
椭圆法管道检测 — 适合管口可见的半正面视角
=============================================
原理: 圆形管口在透视下呈椭圆 → 椭圆长轴方向 = 管道轴向投影
      椭圆短/长轴比 = 管道与视线夹角

用法:
  python test_ellipse.py 照片.jpg
  鼠标框选管口区域，按空格确认
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
        self.window = "Drag box around PIPE OPENING | SPACE=ok q=quit"

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
        cv2.putText(self.image, "框住管口(圆形开口部分)，按SPACE",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(self.window, self.image)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'): return None
            if key == ord(' ') and self.roi is not None:
                cv2.destroyWindow(self.window)
                return self.roi


def detect_pipe_from_opening(image, roi):
    """
    在管口ROI内拟合椭圆 → 推管道轴向。

    返回: {
        "line": [x1,y1,x2,y2],  # 管道轴线
        "angle_deg": float,      # 倾斜角
        "view_angle_deg": float, # 视线与管道夹角 (0=正对, 90=侧面)
        "ellipse": ((cx,cy), (a,b), angle),  # 拟合椭圆
    }
    """
    x, y, w, h = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y + h, x:x + w]

    # Canny 边缘
    edges = cv2.Canny(roi_gray, 40, 120)

    # 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 找最大的近似圆形轮廓（管口）
    best_ellipse = None
    best_area = 0
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (ecx, ecy), (ea, eb), eang = ellipse
        ratio = min(ea, eb) / max(ea, eb)
        # 至少接近椭圆（不是极端细长）
        if ratio > 0.3 and area > best_area:
            best_area = area
            best_ellipse = ellipse

    if best_ellipse is None:
        return None

    (ecx, ecy), (ea, eb), eang = best_ellipse
    # 转回全局坐标
    ecx += x
    ecy += y

    # ---- 从椭圆推算管道轴向 ----
    # 管道轴线方向 = 椭圆长轴的垂直方向
    # 椭圆 angle 是长轴与水平线的夹角（OpenCV 坐标系）
    if ea > eb:
        major_len, minor_len = ea, eb
        pipe_angle_rad = math.radians(eang + 90)  # 垂直方向
    else:
        major_len, minor_len = eb, ea
        pipe_angle_rad = math.radians(eang)

    # 视线与管道夹角 = arcsin(minor/major)
    ratio = minor_len / major_len
    ratio = min(1.0, max(0.0, ratio))
    view_angle = math.degrees(math.asin(ratio))

    # 管道轴线端点（沿主轴方向延伸）
    half_len = major_len * 1.5
    x1 = ecx - half_len * math.cos(pipe_angle_rad)
    y1 = ecy - half_len * math.sin(pipe_angle_rad)
    x2 = ecx + half_len * math.cos(pipe_angle_rad)
    y2 = ecy + half_len * math.sin(pipe_angle_rad)

    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1

    slope_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    # ---- 高度/距离估算 ----
    # 管口在画面中的像素直径
    apparent_diam_px = major_len
    # 若已知实际管径(mm)和焦距(px)，可算距离
    # distance_m = (real_diam_mm / 1000) * focal_px / apparent_diam_px
    # height = camera_h - distance * sin(pitch_angle)

    return {
        "line": [int(x1), int(y1), int(x2), int(y2)],
        "angle_deg": round(slope_angle, 1),
        "view_angle_deg": round(view_angle, 1),
        "ellipse": best_ellipse,
        "ellipse_center": (int(ecx), int(ecy)),
        "apparent_diam_px": round(apparent_diam_px, 1),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python test_ellipse.py <照片.jpg>")
        sys.exit(1)

    path = sys.argv[1]
    image = cv2.imread(path)
    if image is None:
        print(f"无法读取: {path}")
        sys.exit(1)

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

    if scale != 1.0:
        rx, ry, rw, rh = roi_disp
        roi = (int(rx / scale), int(ry / scale), int(rw / scale), int(rh / scale))
    else:
        roi = roi_disp

    result = detect_pipe_from_opening(image, roi)
    if result is None:
        print("✗ 未找到管口椭圆，请确保框中有清晰的管口边缘")
        return

    print(f"\n{'='*50}")
    print(f"  管道坡度:    {result['angle_deg']}°")
    print(f"  视线夹角:    {result['view_angle_deg']}°")
    print(f"  管口像素直径: {result['apparent_diam_px']} px")
    print()
    print(f"  ---- 高度估算（需输入管径和俯仰角）----")
    print(f"  实际管径 DN300 → 距离≈{100*800/result['apparent_diam_px']:.1f}m")
    print(f"  实际管径 DN500 → 距离≈{150*800/result['apparent_diam_px']:.1f}m")
    print(f"  公式: 高度=眼高-距离*sin(俯仰角)")
    print(f"  例: 眼高1.6m, 俯仰-5°, DN300→高度≈{1.6 - (100*800/result['apparent_diam_px'])*math.sin(math.radians(-5)):.1f}m")
    if result['view_angle_deg'] < 15:
        print(f"  ⚠ 几乎正对管口 → 坡度不准，建议侧面拍")
    elif result['view_angle_deg'] > 60:
        print(f"  ✓ 接近侧面视角，角度较准确")
    print(f"{'='*50}")

    # 画结果
    x1, y1, x2, y2 = [int(v) for v in result["line"]]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(image, (x1, y1), 6, (0, 255, 0), -1)
    cv2.circle(image, (x2, y2), 6, (0, 255, 0), -1)

    # 画椭圆
    (ecx, ecy), (ea, eb), eang = result["ellipse"]
    ecx, ecy = int(ecx + roi[0]), int(ecy + roi[1])
    ea, eb = int(ea), int(eb)
    cv2.ellipse(image, ((ecx, ecy), (ea, eb), eang), (255, 0, 0), 2)
    cv2.circle(image, (ecx, ecy), 4, (0, 0, 255), -1)

    cv2.rectangle(image, (int(roi[0]), int(roi[1])),
                  (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 255), 2)

    dia = result['apparent_diam_px']
    dist300 = 100 * 800 / dia  # DN300 example
    cv2.putText(image, f"Angle: {result['angle_deg']} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"View: {result['view_angle_deg']} deg  Diam: {dia:.0f}px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(image, f"Dist(DN300): {dist300:.1f}m", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    out_path = path.rsplit('.', 1)[0] + '_ellipse_result.jpg'
    cv2.imwrite(out_path, image)
    print(f"结果: {out_path}")
    print(f"蓝圈=管口椭圆  绿线=管道轴线  黄框=ROI")

    cv2.imshow("Result - any key to close", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
