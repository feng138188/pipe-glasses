"""
本地测试 SAM — 命令行版
===========================
拍好的管道照片，直接切。

用法:
  python test_sam_local.py pipe_photo.jpg
  然后鼠标点击画面中的管道，按任意键继续
"""

import sys
import cv2
import numpy as np

from sam_detect import detect_pipe_with_tap


class ClickSelector:
    def __init__(self, image):
        self.image = image
        self.point = None
        self.window = "Click pipe  |  q=quit  c=clear"

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)
            # 画十字
            disp = self.image.copy()
            cv2.drawMarker(disp, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(self.window, disp)

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        disp = self.image.copy()
        cv2.putText(disp, "Click on the pipe", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(self.window, disp)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                return None
            elif key == ord('c'):
                self.point = None
                disp = self.image.copy()
                cv2.putText(disp, "Click on the pipe", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(self.window, disp)
            elif self.point is not None and key != 255:
                cv2.destroyWindow(self.window)
                return self.point


def main():
    if len(sys.argv) < 2:
        print("用法: python test_sam_local.py <管道照片.jpg>")
        sys.exit(1)

    path = sys.argv[1]
    image = cv2.imread(path)
    if image is None:
        print(f"无法读取图片: {path}")
        sys.exit(1)

    h, w = image.shape[:2]
    # 保持显示比例
    max_disp = 1200
    if max(w, h) > max_disp:
        scale = max_disp / max(w, h)
        disp_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        disp_img = image.copy()

    selector = ClickSelector(disp_img)
    disp_point = selector.run()

    if disp_point is None:
        print("取消")
        sys.exit(0)

    # 把显示坐标映射回原图坐标
    if max(w, h) > max_disp:
        scale = max_disp / max(w, h)
        tap_x = int(disp_point[0] / scale)
        tap_y = int(disp_point[1] / scale)
    else:
        tap_x, tap_y = disp_point

    print(f"\n点击位置: ({tap_x}, {tap_y})")
    print("MobileSAM 推理中...")

    result = detect_pipe_with_tap(image, tap_x, tap_y)

    if result is None:
        print("✗ SAM 未检测到管道")
        sys.exit(1)

    print(f"\n检测结果:")
    print(f"  管轴线: ({result['line'][0]},{result['line'][1]}) → ({result['line'][2]},{result['line'][3]})")
    print(f"  画面角度: {result['angle_deg']}°")
    print(f"  管径像素: {result['diameter_px']} px")
    print(f"  置信度: {result['confidence']}")

    # 可视化
    (x1, y1), (x2, y2) = result['line'][:2], result['line'][2:]
    bx1, by1, bx2, by2 = result['box']

    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(image, (x1, y1), 6, (0, 255, 0), -1)
    cv2.circle(image, (x2, y2), 6, (0, 255, 0), -1)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    cv2.drawMarker(image, (tap_x, tap_y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

    cv2.putText(image, f"Angle: {result['angle_deg']} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Diam: {result['diameter_px']:.0f} px", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Conf: {result['confidence']:.0%}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = path.rsplit('.', 1)[0] + '_sam_result.jpg'
    cv2.imwrite(out_path, image)
    print(f"\n结果已保存: {out_path}")
    print(f"红线=点击位置  绿线=SAM拟合轴线  蓝框=分割框")

    cv2.imshow("SAM Result", image)
    print("按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
