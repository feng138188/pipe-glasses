"""
合成图像测试 - 无需实地验证
==============================
自动生成不同角度/高度的水管图像，验证检测算法精度

使用方法：
python test_synthetic.py              # 运行全部测试
python test_synthetic.py --visual     # 显示可视化结果
python test_synthetic.py --angle 3.5  # 测试指定角度
"""

import cv2
import numpy as np
import math
import argparse
from typing import Tuple

# 导入检测器
from pipe_slope_detection import PipeSlopeDetector, AROverlay, PipeDetectionResult


class SyntheticPipeGenerator:
    """
    合成水管图像生成器
    
    生成已知角度和位置的水管图像，用于验证算法精度
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
    
    def generate_pipe_image(self, 
                            angle_deg: float = 3.0,
                            pipe_y_center: int = 240,
                            pipe_thickness: int = 30,
                            pipe_color: Tuple[int, int, int] = (180, 180, 180),
                            background_color: Tuple[int, int, int] = (50, 50, 50),
                            add_noise: bool = True,
                            add_texture: bool = True) -> np.ndarray:
        """
        生成一张包含水管的合成图像
        
        参数:
            angle_deg: 水管角度（度），正值=右端低
            pipe_y_center: 水管中心y坐标
            pipe_thickness: 水管粗细（像素）
            pipe_color: 水管颜色 BGR
            background_color: 背景颜色 BGR
            add_noise: 是否添加噪声
            add_texture: 是否添加纹理
            
        返回:
            合成图像 (BGR)
        """
        # 创建背景
        image = np.full((self.height, self.width, 3), background_color, dtype=np.uint8)
        
        # 添加背景纹理（模拟工地环境）
        if add_texture:
            noise = np.random.randint(0, 30, (self.height, self.width, 3), dtype=np.uint8)
            image = cv2.add(image, noise)
        
        # 计算水管端点
        angle_rad = math.radians(angle_deg)
        half_width = self.width // 2
        
        # 水管从左到右
        x1 = 50
        x2 = self.width - 50
        
        # 根据角度计算y偏移
        dx = x2 - x1
        dy = int(dx * math.tan(angle_rad))
        
        y1 = pipe_y_center - dy // 2
        y2 = pipe_y_center + dy // 2
        
        # 绘制水管（用粗线模拟）
        cv2.line(image, (x1, y1), (x2, y2), pipe_color, pipe_thickness)
        
        # 绘制水管边缘线（上下两条线）
        offset_y = pipe_thickness // 2
        # 上边缘
        cv2.line(image, (x1, y1 - offset_y), (x2, y2 - offset_y), 
                (pipe_color[0] - 30, pipe_color[1] - 30, pipe_color[2] - 30), 2)
        # 下边缘
        cv2.line(image, (x1, y1 + offset_y), (x2, y2 + offset_y), 
                (pipe_color[0] - 30, pipe_color[1] - 30, pipe_color[2] - 30), 2)
        
        # 添加高斯噪声
        if add_noise:
            noise = np.random.normal(0, 5, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_test_set(self, angles: list = None) -> list:
        """
        生成一组测试图像
        
        参数:
            angles: 角度列表，默认 [-5, -3, -1, 0, 1, 3, 5, 8, 10]
            
        返回:
            [(角度, 图像), ...] 列表
        """
        if angles is None:
            angles = [-5, -3, -1, 0, 1, 2, 3, 4, 5, 8, 10]
        
        test_set = []
        for angle in angles:
            image = self.generate_pipe_image(angle_deg=angle)
            test_set.append((angle, image))
        
        return test_set


def run_accuracy_test(visual: bool = False):
    """
    运行精度测试
    
    生成已知角度的图像，检测后对比误差
    """
    print("=" * 60)
    print("  AI管道施工眼镜 - 合成图像精度测试")
    print("  （无需实地，电脑上直接验证）")
    print("=" * 60)
    
    # 创建生成器和检测器
    generator = SyntheticPipeGenerator()
    detector = PipeSlopeDetector(pipe_diameter_mm=100)
    overlay = AROverlay()
    
    # 测试角度
    test_angles = [-8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 10]
    
    print(f"\n测试 {len(test_angles)} 个角度...")
    print(f"{'真实角度':>10} | {'检测角度':>10} | {'误差':>8} | {'状态':>6}")
    print("-" * 50)
    
    errors = []
    results_for_display = []
    
    for true_angle in test_angles:
        # 生成合成图像
        image = generator.generate_pipe_image(angle_deg=true_angle)

        # 清除历史，避免跨测试用例污染
        detector.clear_history()

        # 检测
        result = detector.detect(image)
        
        if result.detected:
            error = abs(result.slope_angle - true_angle)
            errors.append(error)
            status = "✓" if error < 2.0 else "✗"
            print(f"{true_angle:>10.1f}° | {result.slope_angle:>10.1f}° | {error:>6.2f}° | {status:>6}")
        else:
            print(f"{true_angle:>10.1f}° | {'未检测到':>10} | {'--':>8} | {'✗':>6}")
        
        results_for_display.append((true_angle, image, result))
    
    # 统计
    if errors:
        print("-" * 50)
        print(f"\n精度统计:")
        print(f"  平均误差: {np.mean(errors):.2f}°")
        print(f"  最大误差: {np.max(errors):.2f}°")
        print(f"  标准差:   {np.std(errors):.2f}°")
        print(f"  检测率:   {len(errors)}/{len(test_angles)} ({len(errors)/len(test_angles)*100:.0f}%)")
        
        # 判断是否满足需求
        if np.mean(errors) < 1.0:
            print(f"\n  ✓ 精度满足要求（平均误差<1°）")
        else:
            print(f"\n  ✗ 精度需要优化（平均误差≥1°）")
    
    # 可视化
    if visual and results_for_display:
        print("\n显示可视化结果（按任意键切换，q退出）...")
        for true_angle, image, result in results_for_display:
            output = overlay.draw(image, result)
            
            # 添加真实角度标注
            cv2.putText(output, f"TRUE: {true_angle:.1f} deg", (400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Synthetic Test", output)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()


def run_imu_compensation_test(visual: bool = False):
    """
    测试IMU补偿效果
    
    模拟工人低头/抬头时，验证补偿是否正确
    """
    print("\n" + "=" * 60)
    print("  IMU补偿测试")
    print("  模拟工人低头/抬头，验证坡度补偿")
    print("=" * 60)
    
    generator = SyntheticPipeGenerator()
    detector = PipeSlopeDetector(pipe_diameter_mm=100)
    
    # 真实坡度固定为3°
    true_slope = 3.0
    
    # 模拟不同的头部倾斜
    head_pitches = [-10, -5, -2, 0, 2, 5, 10, 15]
    
    print(f"\n真实坡度: {true_slope}°")
    print(f"{'头部俯仰':>10} | {'画面角度':>10} | {'补偿后':>8} | {'误差':>6} | {'状态':>4}")
    print("-" * 55)
    
    for pitch in head_pitches:
        # 画面中的角度 = 真实坡度 + 头部俯仰角
        # （低头时，水管在画面中看起来更倾斜）
        apparent_angle = true_slope + pitch
        
        # 生成对应的合成图像
        image = generator.generate_pipe_image(angle_deg=apparent_angle)
        
        # 设置IMU
        detector.imu.set_pitch(pitch)
        
        # 清除历史（避免平滑影响）
        detector.clear_history()
        
        # 检测
        result = detector.detect(image)
        
        if result.detected:
            error = abs(result.slope_angle - true_slope)
            status = "✓" if error < 2.0 else "✗"
            print(f"{pitch:>10.1f}° | {apparent_angle:>10.1f}° | {result.slope_angle:>6.1f}° | {error:>5.2f}° | {status:>4}")
        else:
            print(f"{pitch:>10.1f}° | {apparent_angle:>10.1f}° | {'N/A':>8} | {'--':>6} | {'✗':>4}")
    
    print(f"\n说明:")
    print(f"  头部俯仰 > 0 = 低头")
    print(f"  头部俯仰 < 0 = 抬头")
    print(f"  补偿后的值应该接近真实坡度 {true_slope}°")


def run_height_test():
    """
    测试高度估计
    """
    print("\n" + "=" * 60)
    print("  高度估计测试")
    print("=" * 60)
    
    from height_measurement import PipeHeightEstimator
    
    estimator = PipeHeightEstimator(
        pipe_diameter_mm=100,
        focal_length_px=800,
        eye_height_m=1.6
    )
    
    # 模拟不同高度的水管
    # 水管越低 → 在画面中越靠下 → pipe_center_y越大
    test_cases = [
        {"name": "水管在1.5m（接近眼睛高度）", "pipe_y": 250, "pitch": 2, "width": 40},
        {"name": "水管在1.0m", "pipe_y": 320, "pitch": 10, "width": 50},
        {"name": "水管在0.5m（较低）", "pipe_y": 380, "pitch": 20, "width": 60},
        {"name": "水管在0.3m（很低）", "pipe_y": 420, "pitch": 30, "width": 80},
    ]
    
    print(f"\n{'场景':<25} | {'估计高度':>8} | {'置信度':>6} | {'距离':>6}")
    print("-" * 60)
    
    for case in test_cases:
        result = estimator.estimate_height(
            pipe_center_y=case["pipe_y"],
            image_height=480,
            imu_pitch=case["pitch"],
            pipe_width_px=case["width"]
        )
        print(f"{case['name']:<25} | {result.height:>6.2f}m | {result.confidence:>5.0%} | {result.distance:>5.2f}m")


def run_roll_compensation_test(visual: bool = False):
    """
    测试Roll补偿效果

    模拟工人歪头时，验证Roll补偿是否正确
    """
    print("\n" + "=" * 60)
    print("  Roll补偿测试")
    print("  模拟工人歪头，验证横滚补偿")
    print("=" * 60)

    generator = SyntheticPipeGenerator()
    detector = PipeSlopeDetector(pipe_diameter_mm=100)

    true_slope = 3.0
    roll_angles = [-15, -10, -5, -2, 0, 2, 5, 10, 15]

    print(f"\n真实坡度: {true_slope}°")
    print(f"{'头部横滚':>8} | {'画面角度':>10} | {'补偿后':>8} | {'误差':>6} | {'状态':>4}")
    print("-" * 55)

    for roll in roll_angles:
        # 歪头时，水平管道在画面中呈现的角度 ≈ true_slope + roll（近似）
        apparent_angle = true_slope + roll

        image = generator.generate_pipe_image(angle_deg=apparent_angle)
        detector.imu.set_pitch(0.0)
        detector.imu.set_roll(roll)
        detector.clear_history()

        result = detector.detect(image)

        if result.detected:
            error = abs(result.slope_angle - true_slope)
            status = "✓" if error < 2.0 else "✗"
            print(f"{roll:>8.1f}° | {apparent_angle:>10.1f}° | {result.slope_angle:>6.1f}° | {error:>5.2f}° | {status:>4}")
        else:
            print(f"{roll:>8.1f}° | {apparent_angle:>10.1f}° | {'N/A':>8} | {'--':>6} | {'✗':>4}")

    print(f"\n说明:")
    print(f"  横滚 > 0 = 右歪头")
    print(f"  横滚 < 0 = 左歪头")
    print(f"  补偿后的值应该接近真实坡度 {true_slope}°")


def run_single_angle_test(angle: float, visual: bool = False):
    """测试单个角度"""
    print(f"\n测试角度: {angle}°")
    
    generator = SyntheticPipeGenerator()
    detector = PipeSlopeDetector(pipe_diameter_mm=100)
    overlay = AROverlay()
    
    image = generator.generate_pipe_image(angle_deg=angle)
    result = detector.detect(image)
    
    print(f"  检测到: {result.detected}")
    if result.detected:
        print(f"  检测角度: {result.slope_angle:.2f}°")
        print(f"  误差: {abs(result.slope_angle - angle):.2f}°")
    
    if visual:
        output = overlay.draw(image, result)
        cv2.putText(output, f"TRUE: {angle:.1f} deg", (400, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Test", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="合成图像测试 - 无需实地验证")
    parser.add_argument("--visual", action="store_true", help="显示可视化")
    parser.add_argument("--angle", type=float, help="测试指定角度")
    parser.add_argument("--imu-test", action="store_true", help="运行IMU补偿测试")
    parser.add_argument("--height-test", action="store_true", help="运行高度测试")
    parser.add_argument("--roll-test", action="store_true", help="运行Roll补偿测试")
    parser.add_argument("--all", action="store_true", help="运行全部测试")

    args = parser.parse_args()

    if args.angle is not None:
        run_single_angle_test(args.angle, args.visual)
    elif args.imu_test:
        run_imu_compensation_test(args.visual)
    elif args.height_test:
        run_height_test()
    elif args.roll_test:
        run_roll_compensation_test(args.visual)
    elif args.all:
        run_accuracy_test(args.visual)
        run_imu_compensation_test(args.visual)
        run_roll_compensation_test(args.visual)
        run_height_test()
    else:
        # 默认运行全部
        run_accuracy_test(args.visual)
        run_imu_compensation_test(args.visual)
        run_roll_compensation_test(args.visual)
        run_height_test()


if __name__ == "__main__":
    main()
