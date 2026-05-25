"""
圆柱体拟合 — 从 3D 点云提取管道参数
======================================
输入: PLY 点云
输出: 管道轴线、坡度角度、直径、高度

方法: RANSAC 圆柱体拟合 → 提取轴线方向和位置
"""

import numpy as np
import math
from typing import Optional, Tuple


def fit_cylinder_ransac(points: np.ndarray,
                        radius_range: Tuple[float, float] = (0.05, 1.5),
                        max_iterations: int = 1000,
                        distance_threshold: float = 0.02) -> Optional[dict]:
    """
    RANSAC 圆柱体拟合。

    参数:
        points: (N, 3) 点云坐标
        radius_range: 管径搜索范围 (min, max) 米
        max_iterations: RANSAC 迭代次数
        distance_threshold: 内点距离阈值（米）

    返回:
        {
            "axis_point": (3,)    — 轴线上一点
            "axis_direction": (3,) — 轴线单位方向向量
            "radius": float       — 半径 (米)
            "inliers": (M, 3)     — 内点
            "slope_angle_deg": float — 坡度角度
            "slope_percent": float   — 坡度百分比
        }
        或 None
    """
    if len(points) < 50:
        return None

    best_model = None
    best_inlier_count = 0

    for _ in range(max_iterations):
        # 随机采样 2 个点定义轴线方向
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 0.01:
            continue
        direction /= length

        # 随机采样 1 个点估计半径
        p3 = points[np.random.randint(len(points))]
        # p3 到轴线 (p1, direction) 的距离
        v = p3 - p1
        dist = np.linalg.norm(np.cross(v, direction))

        if dist < radius_range[0] or dist > radius_range[1]:
            continue

        # 统计内点
        all_v = points - p1
        all_dist = np.linalg.norm(np.cross(all_v, direction), axis=1)
        inlier_mask = np.abs(all_dist - dist) < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = {
                "axis_point": p1.copy(),
                "axis_direction": direction.copy(),
                "radius": float(dist),
                "inliers": points[inlier_mask],
            }

    if best_model is None or best_inlier_count < 20:
        return None

    # 用内点精细拟合
    inliers = best_model["inliers"]
    # 重新估计轴线（PCA 第一主成分）
    centroid = np.mean(inliers, axis=0)
    cov = np.cov((inliers - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    best_direction = eigenvectors[:, -1]  # 最大特征值的方向

    # 确保方向合理（接近水平）
    if abs(best_direction[2]) > abs(best_direction[0]):
        best_direction = -best_direction

    # 重新估计半径（中位数距离）
    v = inliers - centroid
    radii = np.linalg.norm(np.cross(v, best_direction), axis=1)
    best_radius = float(np.median(radii))

    # 计算坡度
    # 投影到重力方向（假设 Z 轴向上）
    dx, dy, dz = best_direction
    horizontal_proj = math.sqrt(dx**2 + dy**2)
    if horizontal_proj > 1e-6:
        slope_angle = math.degrees(math.atan2(dz, horizontal_proj))
    else:
        slope_angle = 90.0

    slope_percent = math.tan(math.radians(slope_angle)) * 100

    return {
        "axis_point": centroid.tolist(),
        "axis_direction": best_direction.tolist(),
        "radius": best_radius,
        "diameter": best_radius * 2,
        "slope_angle_deg": round(slope_angle, 2),
        "slope_percent": round(slope_percent, 2),
        "inlier_count": len(inliers),
        "inlier_ratio": round(len(inliers) / len(points), 3),
    }


def load_ply(path: str) -> np.ndarray:
    """加载 PLY 点云，返回 (N, 3) numpy 数组"""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def detect_pipe_3d(ply_path: str) -> Optional[dict]:
    """
    一站式: PLY 点云 → 管道 3D 参数。

    返回: 同 fit_cylinder_ransac 的返回值
    """
    points = load_ply(ply_path)
    if len(points) == 0:
        return None

    print(f"  点云: {len(points)} 点")
    result = fit_cylinder_ransac(points)

    if result:
        print(f"  管道直径: {result['diameter']*1000:.0f}mm")
        print(f"  坡度: {result['slope_angle_deg']:.1f}° ({result['slope_percent']:.1f}%)")
        print(f"  内点比例: {result['inlier_ratio']}")
    else:
        print("  ✗ 未检测到管道")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python cylinder_fit.py <points3D.ply>")
        sys.exit(1)
    result = detect_pipe_3d(sys.argv[1])
    if result:
        for k, v in result.items():
            if k != "inliers":
                print(f"  {k}: {v}")
