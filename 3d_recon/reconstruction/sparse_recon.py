"""
稀疏重建 — COLMAP 封装
=========================
输入: 多视角 JPEG 帧目录
输出: 稀疏 3D 点云 (PLY) + 相机位姿
"""

import subprocess
import os
import shutil
from pathlib import Path
from typing import Optional


def run_colmap(image_dir: str, output_dir: str,
               colmap_bin: str = "colmap") -> bool:
    """
    对一组图像执行 COLMAP 稀疏重建。

    参数:
        image_dir: 输入图像目录
        output_dir: 输出目录（会创建 sparse/0/）
        colmap_bin: COLMAP 可执行文件路径

    返回:
        成功/失败
    """
    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.abspath(output_dir)
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    print("=" * 50)
    print("  COLMAP 稀疏重建")
    print("=" * 50)

    # 1. 特征提取
    print("\n[1/4] 特征提取...")
    cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_image_size", "2000",
    ]
    subprocess.run(cmd, check=True)

    # 2. 特征匹配
    print("[2/4] 特征匹配...")
    cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "1",
    ]
    subprocess.run(cmd, check=True)

    # 3. 稀疏重建
    print("[3/4] 稀疏重建 (SFM)...")
    cmd = [
        colmap_bin, "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    ]
    subprocess.run(cmd, check=True)

    # 4. 导出点云
    print("[4/4] 导出点云...")
    sparse0 = os.path.join(sparse_dir, "0")
    if not os.path.exists(sparse0):
        print("  ✗ 重建失败：无稀疏模型")
        return False

    ply_path = os.path.join(output_dir, "points3D.ply")
    cmd = [
        colmap_bin, "model_converter",
        "--input_path", sparse0,
        "--output_path", ply_path,
        "--output_type", "PLY",
    ]
    subprocess.run(cmd, check=True)

    print(f"  ✓ 重建完成: {ply_path}")
    return True


def extract_camera_poses(sparse_dir: str) -> list:
    """
    从 COLMAP 稀疏模型中提取相机位姿。

    返回: [{"image": "frame001.jpg", "R": (3,3), "t": (3,)}, ...]
    """
    import numpy as np

    cameras_bin = os.path.join(sparse_dir, "0", "cameras.bin")
    images_bin = os.path.join(sparse_dir, "0", "images.bin")

    if not os.path.exists(images_bin):
        return []

    # COLMAP 二进制读取
    poses = []
    try:
        from pycolmap import Reconstruction
        recon = Reconstruction(sparse_dir + "/0")
        for image_id, image in recon.images.items():
            R = image.rotation_matrix()
            t = image.translation()
            poses.append({
                "image": image.name,
                "R": R.tolist(),
                "t": t.tolist(),
            })
    except ImportError:
        print("  提示: pip install pycolmap 可读取相机位姿")

    return poses


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python reconstruction/sparse_recon.py <图像目录> <输出目录>")
        sys.exit(1)
    run_colmap(sys.argv[1], sys.argv[2])
