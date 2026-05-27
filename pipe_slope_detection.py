"""
水管坡度检测 - 纯摄像头方案
============================
仅使用摄像头 + 内置IMU实现水管坡度和高度检测

核心原理：
1. 坡度 = 水管在画面中的角度 - 头部倾斜角（IMU提供）
2. 高度 = 通过管径像素宽度作为比例尺推算距离，再结合俯仰角计算

依赖：
pip install opencv-python numpy

使用方法：
python pipe_slope_detection.py --image path/to/image.jpg
python pipe_slope_detection.py --video path/to/video.mp4
python pipe_slope_detection.py --camera 0
"""

import cv2
import numpy as np
import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from height_measurement import PipeHeightEstimator


# ============================================================
# 数据结构
# ============================================================

@dataclass
class IMUData:
    """IMU数据（模拟）"""
    pitch: float = 0.0   # 俯仰角（度），正值=低头
    roll: float = 0.0    # 横滚角（度）
    yaw: float = 0.0     # 偏航角（度）
    timestamp: float = 0.0


@dataclass
class PipeDetectionResult:
    """水管检测结果"""
    detected: bool
    slope_angle: float       # 真实坡度角度（度），已补偿头部倾斜
    slope_percent: float     # 坡度百分比
    height: float            # 估计高度（米）
    confidence: float        # 置信度 0~1
    line_start: Tuple[int, int]
    line_end: Tuple[int, int]
    raw_image_angle: float   # 画面中的原始角度（未补偿）
    imu_pitch: float         # IMU俯仰角
    imu_roll: float          # IMU横滚角


# ============================================================
# IMU模拟器（实际设备替换为真实IMU读取）
# ============================================================

class IMUSimulator:
    """
    IMU模拟器
    
    在实际AR眼镜上，替换为设备SDK的IMU接口：
    - RealWear: 使用Android SensorManager
    - 手机: 使用加速度计+陀螺仪
    """
    
    def __init__(self):
        self.pitch = 0.0  # 模拟俯仰角
        self.roll = 0.0
        
    def get_data(self) -> IMUData:
        """获取当前IMU数据"""
        return IMUData(
            pitch=self.pitch,
            roll=self.roll,
            yaw=0.0,
            timestamp=time.time()
        )
    
    def set_pitch(self, pitch: float):
        """模拟设置俯仰角（用于测试）"""
        self.pitch = pitch
    
    def set_roll(self, roll: float):
        """模拟设置横滚角（用于测试）"""
        self.roll = roll


# ============================================================
# 核心检测器
# ============================================================

class PipeSlopeDetector:
    """
    水管坡度检测器 - 纯摄像头方案
    
    只需要：
    1. RGB摄像头图像
    2. IMU俯仰角（补偿头部倾斜）
    3. 已知管径（用于高度估计）
    """
    
    def __init__(self,
                 pipe_diameter_mm: float = 100.0,
                 camera_focal_length_px: float = 800.0,
                 eye_height_m: float = 1.6,
                 use_dino: bool = False):
        """
        参数:
            pipe_diameter_mm: 管道直径（毫米），如DN100=100mm
            camera_focal_length_px: 相机焦距（像素），需标定
            eye_height_m: 眼镜佩戴高度（米）
            use_dino: 使用 Grounding DINO 零样本检测（需 GPU 否则慢）
        """
        self.pipe_diameter_mm = pipe_diameter_mm
        self.pipe_diameter_m = pipe_diameter_mm / 1000.0
        self.focal_length = camera_focal_length_px
        self.eye_height = eye_height_m
        self.use_dino = use_dino
        self._dino_loaded = False

        # 高度估计器（管径比例尺法）
        self.height_estimator = PipeHeightEstimator(
            pipe_diameter_mm=pipe_diameter_mm,
            focal_length_px=camera_focal_length_px,
            eye_height_m=eye_height_m
        )

        # IMU
        self.imu = IMUSimulator()
        
        # 检测参数
        self.min_line_length = 180
        self.max_line_gap = 60

        # 平滑滤波
        self.slope_history = []
        self.height_history = []
        self.smooth_window = 10  # 滑动平均窗口

        # 跟踪状态 — 锁定后窄带跟踪，不全局搜索
        self.prev_line = None       # (x1,y1,x2,y2) 上一帧
        self.track_lost = 0         # 连续丢帧计数
        self.track_max_lost = 60    # 丢帧上限，超限后重搜
        self.track_band = 60        # 跟踪带宽（像素，y方向±）
        self.tap_point = None       # 用户点击的坐标 (x, y)，用于手动锁定

    def clear_history(self):
        """清除平滑历史（切换检测目标时调用）"""
        self.slope_history.clear()
        self.height_history.clear()
        self.prev_line = None
        self.track_lost = 0
        self.tap_point = None
        
    def set_pipe_diameter(self, diameter_mm: float):
        """设置管径（语音输入后调用）"""
        self.pipe_diameter_mm = diameter_mm
        self.pipe_diameter_m = diameter_mm / 1000.0
        self.height_estimator.pipe_diameter_m = self.pipe_diameter_m
        print(f"管径已设置: DN{int(diameter_mm)}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # CLAHE自适应直方图均衡化（应对光照变化）
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_pipe_lines(self, gray: np.ndarray,
                          roi: Optional[Tuple[int, int, int, int]] = None
                          ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        检测水管边缘线。跟踪模式下只搜 ROI 区域。
        """
        # ROI：只裁剪边缘图（简化：用 mask）
        edges = cv2.Canny(gray, 80, 200, apertureSize=3)
        if roi is not None:
            rx, ry, rw, rh = roi
            mask = np.zeros(edges.shape, dtype=np.uint8)
            mask[ry:ry+rh, rx:rw] = 255
            edges = cv2.bitwise_and(edges, edges, mask=mask)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        if lines is None:
            return []

        result = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 25:
                result.append(((x1, y1), (x2, y2)))
        return result
    
    def find_main_pipe_line(self, lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                            image: Optional[np.ndarray] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """选最可能是管道的线段。有 tap 点优先，否则用长度×颜色分数排序。"""
        if not lines:
            return None
        if self.tap_point is not None:
            tx, ty = self.tap_point
            best, best_dist = None, float("inf")
            for (x1, y1), (x2, y2) in lines:
                ln = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if ln < 100:
                    continue
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                d = math.sqrt((mx - tx) ** 2 + (my - ty) ** 2)
                if d < best_dist:
                    best_dist = d
                    best = ((x1, y1), (x2, y2))
            self.tap_point = None
            if best is not None:
                return best

        # 颜色+长度评分选线
        best = None
        best_score = 0
        for (x1, y1), (x2, y2) in lines:
            ln = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if ln < 100:
                continue
            cs = self.pipe_color_score(image, ((x1, y1), (x2, y2))) if image is not None else 0.0
            score = ln * (1.0 + cs)  # 长度 × 颜色分数
            if score > best_score:
                best_score = score
                best = ((x1, y1), (x2, y2))
        return best
    
    def pipe_color_score(self, image: np.ndarray,
                         line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """
        评估线段是否像管道：管道颜色均匀，两侧颜色不同。

        返回 0~1 分数，越高越像管道。
        """
        (x1, y1), (x2, y2) = line
        h, w = image.shape[:2]
        ln = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if ln < 50:
            return 0.0

        dx = (x2 - x1) / ln
        dy = (y2 - y1) / ln
        nx, ny = -dy, dx  # 法线

        step = max(1, int(ln / 10))
        offset = max(8, int(self.pipe_diameter_mm / 5))

        colors_inner = []
        colors_outer = []
        for i in range(0, int(ln), step):
            t = i / ln
            mx = int(x1 + (x2 - x1) * t)
            my = int(y1 + (y2 - y1) * t)

            # 法线正方向（下方/内侧）采样
            px = int(mx + nx * offset)
            py = int(my + ny * offset)
            if 0 <= px < w and 0 <= py < h:
                colors_inner.append(image[py, px].astype(np.float32))

            # 法线反方向（上方/外侧）采样
            px2 = int(mx - nx * offset)
            py2 = int(my - ny * offset)
            if 0 <= px2 < w and 0 <= py2 < h:
                colors_outer.append(image[py2, px2].astype(np.float32))

        if len(colors_inner) < 3 or len(colors_outer) < 3:
            return 0.0

        inner = np.array(colors_inner)
        outer = np.array(colors_outer)

        # 内侧颜色方差（越低越均匀→越像管道）
        inner_var = np.mean(np.std(inner, axis=0)) / 100.0
        uniformity = max(0, 1.0 - inner_var)

        # 内外颜色差异（越大越像真实边缘）
        inner_mean = np.mean(inner, axis=0)
        outer_mean = np.mean(outer, axis=0)
        contrast = np.linalg.norm(inner_mean - outer_mean) / 255.0

        # 综合分数
        score = uniformity * 0.5 + contrast * 0.5
        return min(1.0, score)

    def calculate_slope_with_imu(self, line: Tuple[Tuple[int, int], Tuple[int, int]],
                                  imu: IMUData, image_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        计算真实坡度（含IMU补偿）

        补偿顺序：
        1. Roll补偿 — 将线段绕图像中心旋转 -roll，消除歪头影响
        2. Pitch补偿 — 减去头部俯仰角

        返回: (角度-度, 坡度百分比)
        """
        (x1, y1), (x2, y2) = line
        iw, ih = image_size
        cx, cy = iw / 2.0, ih / 2.0

        # Roll补偿：将线端点绕图像中心旋转 -roll
        if abs(imu.roll) > 0.01:
            roll_rad = math.radians(-imu.roll)
            cos_a, sin_a = math.cos(roll_rad), math.sin(roll_rad)

            for pt in [(x1, y1), (x2, y2)]:
                dx, dy = pt[0] - cx, pt[1] - cy
                nx = dx * cos_a - dy * sin_a + cx
                ny = dx * sin_a + dy * cos_a + cy
                if pt == (x1, y1):
                    x1, y1 = nx, ny
                else:
                    x2, y2 = nx, ny

        # 确保从左到右
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # 画面中的角度（图像y轴向下为正）
        dx = x2 - x1
        dy = y2 - y1
        image_angle = math.degrees(math.atan2(dy, dx))

        # Pitch补偿：减去头部俯仰角
        # pitch正值=低头，低头时画面中物体会显得向上倾斜
        real_angle = image_angle - imu.pitch

        # 坡度百分比
        slope_percent = math.tan(math.radians(real_angle)) * 100

        return real_angle, slope_percent
    
    def estimate_height(self, image: np.ndarray,
                        line: Tuple[Tuple[int, int], Tuple[int, int]],
                        imu: IMUData) -> float:
        """
        利用管径作为比例尺估计高度

        原理：
        1. 管径的像素宽度 → 推算距离（通过高度估计器）
        2. 距离 + 俯仰角 → 推算高度差
        3. 眼镜高度 - 高度差 = 水管高度
        """
        (x1, y1), (x2, y2) = line
        pipe_center_y = (y1 + y2) / 2

        # 沿管道线段方向提取ROI，测量管径像素宽度
        pipe_width_px = self._measure_pipe_width_along_line(image, line)

        # 使用高度估计器计算
        result = self.height_estimator.estimate_height(
            pipe_center_y=pipe_center_y,
            image_height=image.shape[0],
            imu_pitch=imu.pitch,
            pipe_width_px=pipe_width_px
        )

        return max(0, result.height)

    def _measure_pipe_width_along_line(self, image: np.ndarray,
                                        line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """
        沿管道线段方向，在多个采样点处测量管径像素宽度，取中位数
        """
        (x1, y1), (x2, y2) = line
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 1:
            return 0.0

        # 管道法线方向的单位向量（垂直于线段）
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        nx, ny = -dy, dx  # 法线方向

        h, w = edges.shape
        widths = []
        num_samples = 5
        search_radius = max(20, int(0.15 * length))

        for i in range(num_samples):
            # 沿线段的采样点
            t = (i + 0.5) / num_samples
            cx = int(x1 + (x2 - x1) * t)
            cy = int(y1 + (y2 - y1) * t)

            # 沿法线双向扫描找边缘点
            edge_points = []
            for step in range(-search_radius, search_radius + 1):
                px = int(cx + nx * step)
                py = int(cy + ny * step)
                if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                    edge_points.append(step)

            if len(edge_points) >= 2:
                ww = max(edge_points) - min(edge_points)
                if 5 < ww < search_radius * 2 * 0.9:
                    widths.append(ww)

        return float(np.median(widths)) if widths else 0.0
    
    def smooth_value(self, value: float, history: list) -> float:
        """滑动平均平滑"""
        history.append(value)
        if len(history) > self.smooth_window:
            history.pop(0)
        return sum(history) / len(history)
    
    def detect(self, image: np.ndarray) -> PipeDetectionResult:
        """
        主检测函数 — DINO 优先 + 传统跟踪回退

        DINO 模式：用 Grounding DINO 语义识别管道，找到后在框内拟合管线。
        传统模式：Canny+Hough 边缘检测 + 窄带跟踪。
        """
        imu_data = self.imu.get_data()
        gray = self.preprocess(image)
        h, w = gray.shape

        main_line = None

        # ---- DINO 检测（每15帧或跟踪丢失时） ----
        if self.use_dino:
            need_dino = (
                self.prev_line is None
                or self.track_lost > self.track_max_lost // 3
                or self.track_lost % 15 == 0
            )
            if need_dino:
                try:
                    from dino_detector import detect_pipe_line
                    dino_line, dino_info = detect_pipe_line(image)
                    if dino_line is not None:
                        main_line = dino_line
                        self.track_lost = 0
                except Exception as e:
                    print(f"  DINO 检测异常: {e}")

        # ---- 传统跟踪模式 ----
        if main_line is None:
            roi = None
            if self.prev_line is not None and self.track_lost < self.track_max_lost:
                px1, py1, px2, py2 = self.prev_line
                center_y = int((py1 + py2) / 2)
                band_top = max(0, center_y - self.track_band)
                band_bot = min(h, center_y + self.track_band)
                roi = (0, band_top, w, band_bot - band_top)

            lines = self.detect_pipe_lines(gray, roi)
            main_line = self.find_main_pipe_line(lines, image)

        # ---- 更新跟踪状态 ----
        if main_line is None:
            if self.prev_line is not None:
                self.track_lost += 1
            return PipeDetectionResult(
                detected=False, slope_angle=0.0, slope_percent=0.0,
                height=0.0, confidence=0.0,
                line_start=(0, 0), line_end=(0, 0),
                raw_image_angle=0.0, imu_pitch=imu_data.pitch, imu_roll=imu_data.roll
            )

        (x1, y1), (x2, y2) = main_line
        self.prev_line = (x1, y1, x2, y2)
        self.track_lost = 0

        # 计算坡度（含IMU补偿：pitch + roll）
        image_size = (image.shape[1], image.shape[0])
        real_angle, slope_percent = self.calculate_slope_with_imu(main_line, imu_data, image_size)
        
        # 估计高度（使用管径比例尺法）
        height = self.estimate_height(image, main_line, imu_data)
        
        # 平滑
        smooth_angle = self.smooth_value(real_angle, self.slope_history)
        smooth_height = self.smooth_value(height, self.height_history)
        
        # 计算画面原始角度
        (x1, y1), (x2, y2) = main_line
        raw_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # 置信度
        line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        confidence = min(line_length / 400, 1.0)
        
        return PipeDetectionResult(
            detected=True,
            slope_angle=smooth_angle,
            slope_percent=math.tan(math.radians(smooth_angle)) * 100,
            height=smooth_height,
            confidence=confidence,
            line_start=(x1, y1),
            line_end=(x2, y2),
            raw_image_angle=raw_angle,
            imu_pitch=imu_data.pitch,
            imu_roll=imu_data.roll
        )


# ============================================================
# AR显示
# ============================================================

class AROverlay:
    """AR叠加显示"""
    
    def __init__(self):
        self.target_slope_min = 2.0   # 目标坡度范围（度）
        self.target_slope_max = 5.0
        self.target_height_min = 0.5  # 目标高度范围（米）
        self.target_height_max = 1.5
    
    def set_targets(self, slope_min: float, slope_max: float, 
                    height_min: float, height_max: float):
        """设置目标范围"""
        self.target_slope_min = slope_min
        self.target_slope_max = slope_max
        self.target_height_min = height_min
        self.target_height_max = height_max
    
    def draw(self, image: np.ndarray, result: PipeDetectionResult) -> np.ndarray:
        """绘制AR叠加"""
        output = image.copy()
        h, w = image.shape[:2]
        
        if result.detected:
            # 绘制水管检测线（绿色）
            cv2.line(output, result.line_start, result.line_end, (0, 255, 0), 3)
            cv2.circle(output, result.line_start, 6, (0, 200, 255), -1)
            cv2.circle(output, result.line_end, 6, (0, 200, 255), -1)
            
            # 绘制水平参考线（虚线效果）
            mid_y = (result.line_start[1] + result.line_end[1]) // 2
            for x in range(0, w, 20):
                cv2.line(output, (x, mid_y), (x + 10, mid_y), (100, 100, 100), 1)
        
        # 信息面板（半透明黑色背景）
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 280, 180
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # 文字信息
        y = panel_y + 25
        
        if result.detected:
            # 坡度
            slope_color = self._get_status_color(
                abs(result.slope_angle), 
                self.target_slope_min, 
                self.target_slope_max
            )
            cv2.putText(output, f"坡度: {result.slope_angle:.1f} ({result.slope_percent:.1f}%)",
                       (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, slope_color, 2)
            
            y += 30
            # 高度
            height_color = self._get_status_color(
                result.height,
                self.target_height_min,
                self.target_height_max
            )
            cv2.putText(output, f"高度: {result.height:.2f}m",
                       (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, height_color, 2)
            
            y += 30
            # 置信度
            cv2.putText(output, f"置信度: {result.confidence:.0%}",
                       (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            y += 30
            # IMU信息
            cv2.putText(output, f"IMU: pitch={result.imu_pitch:.1f} roll={result.imu_roll:.1f}",
                       (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            y += 35
            # 状态判断
            slope_ok = self.target_slope_min <= abs(result.slope_angle) <= self.target_slope_max
            height_ok = self.target_height_min <= result.height <= self.target_height_max
            
            if slope_ok and height_ok:
                status = "PASS"
                status_color = (0, 255, 0)
            else:
                issues = []
                if not slope_ok:
                    issues.append("坡度")
                if not height_ok:
                    issues.append("高度")
                status = f"调整: {'+'.join(issues)}"
                status_color = (0, 0, 255)
            
            cv2.putText(output, status, (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            cv2.putText(output, "未检测到水管", (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 30
            cv2.putText(output, "请对准水管", (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 目标范围提示（底部）
        target_text = f"目标: 坡度{self.target_slope_min:.0f}-{self.target_slope_max:.0f} | 高度{self.target_height_min:.1f}-{self.target_height_max:.1f}m"
        cv2.putText(output, target_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        return output
    
    def _get_status_color(self, value: float, min_val: float, max_val: float) -> Tuple[int, int, int]:
        """根据值是否在范围内返回颜色"""
        if min_val <= value <= max_val:
            return (0, 255, 0)  # 绿色-合格
        elif value < min_val * 0.8 or value > max_val * 1.2:
            return (0, 0, 255)  # 红色-严重偏离
        else:
            return (0, 255, 255)  # 黄色-接近边界


# ============================================================
# 主程序
# ============================================================

def process_video(detector: PipeSlopeDetector, overlay: AROverlay, 
                  source, output_path: Optional[str] = None):
    """处理视频/摄像头"""
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"无法打开视频源: {source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("=" * 50)
    print("AI管道施工眼镜 - 纯摄像头方案")
    print("=" * 50)
    print(f"管径: DN{int(detector.pipe_diameter_mm)}")
    print(f"目标坡度: {overlay.target_slope_min}°~{overlay.target_slope_max}°")
    print(f"目标高度: {overlay.target_height_min}m~{overlay.target_height_max}m")
    print("-" * 50)
    print("按键:")
    print("  q - 退出")
    print("  w/s - 模拟抬头/低头（IMU俯仰角±1°）")
    print("  a/d - 模拟左歪/右歪头（IMU横滚角±1°）")
    print("  p - 切换管径 (DN50/100/150/200)")
    print("=" * 50)
    
    pipe_sizes = [50, 100, 150, 200]
    pipe_idx = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        result = detector.detect(frame)
        
        # AR叠加
        output = overlay.draw(frame, result)
        
        # 显示
        cv2.imshow("AI Pipe Glasses", output)
        
        if writer:
            writer.write(output)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            detector.imu.set_pitch(detector.imu.pitch - 1)
            print(f"  IMU俯仰: {detector.imu.pitch:.1f}°  横滚: {detector.imu.roll:.1f}°")
        elif key == ord('s'):
            detector.imu.set_pitch(detector.imu.pitch + 1)
            print(f"  IMU俯仰: {detector.imu.pitch:.1f}°  横滚: {detector.imu.roll:.1f}°")
        elif key == ord('a'):
            detector.imu.set_roll(detector.imu.roll - 1)
            print(f"  IMU俯仰: {detector.imu.pitch:.1f}°  横滚: {detector.imu.roll:.1f}°")
        elif key == ord('d'):
            detector.imu.set_roll(detector.imu.roll + 1)
            print(f"  IMU俯仰: {detector.imu.pitch:.1f}°  横滚: {detector.imu.roll:.1f}°")
        elif key == ord('p'):
            pipe_idx = (pipe_idx + 1) % len(pipe_sizes)
            detector.set_pipe_diameter(pipe_sizes[pipe_idx])
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def process_image(detector: PipeSlopeDetector, overlay: AROverlay, 
                  image_path: str, output_path: Optional[str] = None):
    """处理单张图片"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    result = detector.detect(image)
    output = overlay.draw(image, result)
    
    print("\n检测结果:")
    print(f"  检测到水管: {result.detected}")
    if result.detected:
        print(f"  真实坡度: {result.slope_angle:.2f}°")
        print(f"  坡度百分比: {result.slope_percent:.2f}%")
        print(f"  估计高度: {result.height:.2f}m")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  画面原始角度: {result.raw_image_angle:.2f}°")
        print(f"  IMU补偿: {result.imu_pitch:.2f}°")
    
    if output_path:
        cv2.imwrite(output_path, output)
        print(f"\n结果已保存: {output_path}")
    
    cv2.imshow("AI Pipe Glasses", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AI管道施工眼镜 - 纯摄像头坡度检测")
    parser.add_argument("--image", type=str, help="输入图片路径")
    parser.add_argument("--video", type=str, help="输入视频路径")
    parser.add_argument("--camera", type=int, default=-1, help="摄像头ID")
    parser.add_argument("--url", type=str, help="IP摄像头/手机流URL (如 http://192.168.1.x:8080/video)")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--diameter", type=float, default=100, help="管径(mm)，如100=DN100")
    parser.add_argument("--slope-min", type=float, default=2.0, help="目标最小坡度(度)")
    parser.add_argument("--slope-max", type=float, default=5.0, help="目标最大坡度(度)")
    parser.add_argument("--height-min", type=float, default=0.5, help="目标最小高度(米)")
    parser.add_argument("--height-max", type=float, default=1.5, help="目标最大高度(米)")
    parser.add_argument("--imu-pitch", type=float, default=0.0, help="模拟IMU俯仰角(度)")
    parser.add_argument("--imu-roll", type=float, default=0.0, help="模拟IMU横滚角(度)")
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = PipeSlopeDetector(pipe_diameter_mm=args.diameter)
    detector.imu.set_pitch(args.imu_pitch)
    detector.imu.set_roll(args.imu_roll)
    
    # 创建AR叠加
    overlay = AROverlay()
    overlay.set_targets(args.slope_min, args.slope_max, args.height_min, args.height_max)
    
    # 处理
    if args.image:
        process_image(detector, overlay, args.image, args.output)
    elif args.video:
        process_video(detector, overlay, args.video, args.output)
    elif args.camera >= 0:
        process_video(detector, overlay, args.camera, args.output)
    elif args.url:
        process_video(detector, overlay, args.url, args.output)
    else:
        print("AI管道施工眼镜 - 纯摄像头方案")
        print("=" * 40)
        print("\n使用方法:")
        print(f"  python {__file__} --camera 0")
        print(f"  python {__file__} --image pipe.jpg")
        print(f"  python {__file__} --video pipe.mp4")
        print(f"  python {__file__} --url http://192.168.1.100:8080/video")
        print(f"\n参数:")
        print(f"  --diameter 100    管径DN100")
        print(f"  --slope-min 2     最小坡度2°")
        print(f"  --slope-max 5     最大坡度5°")
        print(f"  --imu-pitch 5     模拟低头5°")
        print(f"  --imu-roll 3      模拟歪头3°")
        print(f"  --url URL         手机IP摄像头流")
        print(f"\n手机测试:")
        print(f"  1. 手机安装 'IP摄像头' App")
        print(f"  2. 打开App，记录WiFi地址如 http://192.168.1.100:8080")
        print(f"  3. python {__file__} --url http://192.168.1.100:8080/video")


if __name__ == "__main__":
    main()
