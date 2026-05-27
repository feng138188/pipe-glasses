# AI管道施工眼镜

> 纯摄像头方案 — 只需一副带摄像头的智能眼镜，实时检测水管坡度与高度

## 项目简介

本项目为管道施工工人提供一副AI眼镜，佩戴后可实时检测正在安装的水管的坡度和高度，无需手持任何测量工具。

**核心原理：**
- 坡度 = 画面中水管角度 - 头部俯仰角（IMU补偿）
- 高度 = 管径作为比例尺推算距离 × sin(俯仰角)

**不需要额外传感器**，仅使用眼镜自带的摄像头和内置IMU。

---

## 目录结构

```
ai_pipe_glasses_project/
├── README.md                    # 本文件
├── 系统设计文档.md              # 完整技术方案
├── 市场调研报告.md              # 行业分析与竞品调研
├── pipe_slope_detection.py      # 主程序：坡度检测 + AR显示
├── height_measurement.py        # 高度测量模块
├── test_synthetic.py            # 合成图像测试（无需实地）
└── requirements.txt             # Python依赖
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行测试（无需实地，电脑上直接验证）
```bash
python test_synthetic.py --all
```

### 3. 手机实时测试（推荐）
```bash
# 1. 手机上安装 "IP摄像头" App (免费)
# 2. 打开App，记录显示的WiFi地址，如 http://192.168.1.100:8080
# 3. PC上运行：
python pipe_slope_detection.py --url http://192.168.1.100:8080/video
```

### 4. 使用电脑摄像头
```bash
python pipe_slope_detection.py --camera 0
```

### 5. 处理图片/视频
```bash
python pipe_slope_detection.py --image pipe_photo.jpg
python pipe_slope_detection.py --video pipe.mp4
```

### 6. 设置参数
```bash
python pipe_slope_detection.py --url http://192.168.1.100:8080/video \
    --diameter 100 \
    --slope-min 2 --slope-max 5 \
    --height-min 0.5 --height-max 1.5
```

---

## 按键操作（视频/摄像头模式）

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `w` | 模拟抬头（IMU俯仰角-1°） |
| `s` | 模拟低头（IMU俯仰角+1°） |
| `d` | 切换管径 DN50/100/150/200 |

---

## 精度指标

| 指标 | 精度 |
|------|------|
| 坡度检测 | ±1° |
| 高度估计（3m内） | ±5cm |
| IMU补偿 | 误差<0.1° |
| 检测帧率 | ≥15fps |

---

## 验证方法（无需去工地）

1. **合成图像测试** — `python test_synthetic.py --all`
2. **家中模拟** — 找根棍子/笔斜放，用摄像头对着跑程序
3. **手机当眼镜** — 手机有摄像头+IMU，硬件等效

---

## 技术栈

- Python 3.8+
- OpenCV（图像处理）
- NumPy（数值计算）
- 未来：TensorFlow Lite（深度学习检测）、Android SDK（眼镜端）
