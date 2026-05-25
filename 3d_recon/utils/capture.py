"""
多视角采集工具
=================
手机/眼镜绕管道拍摄多帧，用于 3D 重建。

使用方法:
  python utils/capture.py --camera 0 --output ./images/ --count 20
"""

import cv2
import os
import argparse
import time


def capture_sequence(camera_id: int, output_dir: str,
                     count: int = 20, interval: float = 0.3):
    """
    从摄像头连续采集帧。

    理想采集方式：绕管道缓慢移动，每次移动 ~10°
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    print("=" * 50)
    print("  多视角管道采集")
    print("=" * 50)
    print(f"\n  将采集 {count} 帧到 {output_dir}/")
    print(f"  请绕管道缓慢移动摄像头 (~10°/帧)")
    print(f"  按 SPACE 开始采集，按 q 退出")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            for i in range(count):
                ret, frame = cap.read()
                if not ret:
                    break
                path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(path, frame)
                cv2.putText(frame, f"Captured: {i+1}/{count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Capture", frame)
                cv2.waitKey(1)
                time.sleep(interval)
            print(f"  ✓ 已采集 {count} 帧")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="./captured_frames")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.3)
    args = parser.parse_args()

    capture_sequence(args.camera, args.output, args.count, args.interval)
