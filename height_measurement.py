"""
水管高度测量模块 - 纯摄像头方案
================================
利用管径作为比例尺，结合IMU俯仰角估计水管高度

原理：
1. 已知管径 D（工人输入）
2. 摄像头拍到管道，测量管径的像素宽度 W_px
3. 距离 = (D × 焦距) / W_px
4. 高度 = 眼镜高度 - 距离 × sin(俯仰角)

依赖：
pip install opencv-python numpy
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HeightResult:
    """高度测量结果"""
    height: float           # 水管高度（米）
    distance: float         # 水管距离（米）
    pipe_width_px: float    # 管径像素宽度
    confidence: float       # 置信度
    method: str             # 使用的方法


class PipeHeightEstimator:
    """
    水管高度估计器 - 纯摄像头方案
    
    不需要深度传感器，只需要：
    1. 摄像头图像
    2. 已知管径
    3. 相机焦距（标定）
    4. IMU俯仰角
    """
    
    def __init__(self,
                 pipe_diameter_mm: float = 100.0,
                 focal_length_px: float = 800.0,
                 eye_height_m: float = 1.6,
                 vertical_fov_deg: float = 60.0):
        """
        参数:
            pipe_diameter_mm: 管道外径（毫米）
            focal_length_px: 相机焦距（像素），需要标定
            eye_height_m: 眼镜佩戴高度（米）
            vertical_fov_deg: 相机垂直视场角（度）
        """
        self.pipe_diameter_m = pipe_diameter_mm / 1000.0
        self.focal_length = focal_length_px
        self.eye_height = eye_height_m
        self.vertical_fov = vertical_fov_deg
        
    def measure_pipe_width(self, image: np.ndarray, pipe_region: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        测量水管在图像中的像素宽度
        
        参数:
            image: BGR图像
            pipe_region: 水管区域 (x, y, w, h)，如果为None则自动检测
            
        返回:
            管径像素宽度
        """
        if pipe_region is not None:
            x, y, w, h = pipe_region
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # 转灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 在垂直方向上找水管的上下边缘
        # 对每列求边缘点，找到两条平行边缘的间距
        
        col_widths = []
        for col in range(edges.shape[1]):
            edge_points = np.where(edges[:, col] > 0)[0]
            if len(edge_points) >= 2:
                # 取最远的两个边缘点的距离
                width = edge_points[-1] - edge_points[0]
                if 10 < width < edges.shape[0] * 0.8:  # 合理范围
                    col_widths.append(width)
        
        if col_widths:
            # 取中位数作为管径宽度（更鲁棒）
            return float(np.median(col_widths))
        
        return 0.0
    
    def estimate_distance(self, pipe_width_px: float) -> float:
        """
        通过管径像素宽度估计距离
        
        公式: distance = (实际直径 × 焦距) / 像素宽度
        
        参数:
            pipe_width_px: 管径像素宽度
            
        返回:
            估计距离（米）
        """
        if pipe_width_px <= 0:
            return 0.0
        
        distance = (self.pipe_diameter_m * self.focal_length) / pipe_width_px
        return distance
    
    def estimate_height(self, 
                        pipe_center_y: float,
                        image_height: int,
                        imu_pitch: float = 0.0,
                        pipe_width_px: float = 0.0) -> HeightResult:
        """
        估计水管高度
        
        参数:
            pipe_center_y: 水管中心的y坐标（像素）
            image_height: 图像高度（像素）
            imu_pitch: IMU俯仰角（度），正值=低头
            pipe_width_px: 管径像素宽度（用于距离估计）
            
        返回:
            HeightResult
        """
        # 方法选择
        if pipe_width_px > 0:
            # 方法A：管径比例尺法（更准确）
            distance = self.estimate_distance(pipe_width_px)
            method = "管径比例尺"
        else:
            # 方法B：默认距离估计
            distance = 2.0  # 假设2米
            method = "默认距离"
        
        # 计算水管相对于视线中心的角度
        image_center_y = image_height / 2
        pixel_offset = pipe_center_y - image_center_y  # 正值=水管在画面下方
        
        # 像素偏移转角度
        angle_per_pixel = self.vertical_fov / image_height
        view_angle = pixel_offset * angle_per_pixel  # 正值=向下看
        
        # 总俯仰角 = 画面中的角度 + 头部俯仰角
        total_angle = view_angle + imu_pitch
        
        # 高度差 = 距离 × tan(总俯仰角)
        # 正的total_angle表示向下看，所以高度差为正（水管比眼睛低）
        height_diff = distance * math.tan(math.radians(total_angle))
        
        # 水管高度 = 眼睛高度 - 高度差
        pipe_height = self.eye_height - height_diff
        pipe_height = max(0, pipe_height)  # 不能为负
        
        # 置信度：距离越近越准
        if distance > 0:
            confidence = min(1.0, 3.0 / distance)  # 3m内置信度较高
        else:
            confidence = 0.3
        
        return HeightResult(
            height=pipe_height,
            distance=distance,
            pipe_width_px=pipe_width_px,
            confidence=confidence,
            method=method
        )


# ============================================================
# 相机标定工具
# ============================================================

class CameraCalibrator:
    """
    简易相机标定
    
    使用已知尺寸的物体（如管道）进行标定
    """
    
    def __init__(self):
        self.focal_length = None
    
    def calibrate_with_known_object(self, 
                                     actual_size_m: float,
                                     pixel_size: float,
                                     known_distance_m: float) -> float:
        """
        使用已知物体标定焦距
        
        参数:
            actual_size_m: 物体实际尺寸（米）
            pixel_size: 物体在图像中的像素大小
            known_distance_m: 物体到相机的已知距离（米）
            
        返回:
            焦距（像素）
        """
        # focal_length = (pixel_size × distance) / actual_size
        self.focal_length = (pixel_size * known_distance_m) / actual_size_m
        return self.focal_length
    
    def calibrate_interactive(self):
        """
        交互式标定指引
        
        步骤：
        1. 将已知直径的管道放在已知距离处
        2. 拍照
        3. 测量管道在图像中的像素宽度
        4. 计算焦距
        """
        print("=" * 50)
        print("相机焦距标定")
        print("=" * 50)
        print("\n步骤：")
        print("1. 准备一根已知直径的管道")
        print("2. 将管道放在已知距离处（如1米）")
        print("3. 用眼镜/手机拍照")
        print("4. 测量管道在图像中的像素宽度")
        print()
        
        try:
            diameter = float(input("管道直径(mm): "))
            distance = float(input("管道距离(m): "))
            pixel_width = float(input("管道像素宽度(px): "))
            
            focal = self.calibrate_with_known_object(
                diameter / 1000.0, pixel_width, distance
            )
            
            print(f"\n标定结果: 焦距 = {focal:.1f} 像素")
            print(f"请将此值设置到检测器中")
            
            return focal
        except ValueError:
            print("输入无效")
            return None


# ============================================================
# 演示
# ============================================================

def demo():
    """演示高度测量"""
    print("=" * 50)
    print("水管高度测量演示 - 纯摄像头方案")
    print("=" * 50)
    
    # 创建估计器
    estimator = PipeHeightEstimator(
        pipe_diameter_mm=100,   # DN100
        focal_length_px=800,    # 需要标定
        eye_height_m=1.6        # 眼镜高度
    )
    
    # 模拟不同场景
    scenarios = [
        {"name": "水管在视线正前方", "pipe_y": 240, "img_h": 480, "pitch": 0, "width_px": 40},
        {"name": "水管在脚下（低头看）", "pipe_y": 400, "img_h": 480, "pitch": 15, "width_px": 80},
        {"name": "水管在头顶（抬头看）", "pipe_y": 100, "img_h": 480, "pitch": -10, "width_px": 30},
        {"name": "水管很近", "pipe_y": 300, "img_h": 480, "pitch": 5, "width_px": 120},
        {"name": "水管较远", "pipe_y": 260, "img_h": 480, "pitch": 2, "width_px": 20},
    ]
    
    for s in scenarios:
        result = estimator.estimate_height(
            pipe_center_y=s["pipe_y"],
            image_height=s["img_h"],
            imu_pitch=s["pitch"],
            pipe_width_px=s["width_px"]
        )
        
        print(f"\n场景: {s['name']}")
        print(f"  IMU俯仰角: {s['pitch']}°")
        print(f"  管径像素宽度: {s['width_px']}px")
        print(f"  估计距离: {result.distance:.2f}m")
        print(f"  估计高度: {result.height:.2f}m")
        print(f"  置信度: {result.confidence:.0%}")
        print(f"  方法: {result.method}")


if __name__ == "__main__":
    demo()
