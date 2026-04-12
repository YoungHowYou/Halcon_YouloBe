# Halcon_YouloBe

HALCON 算法扩展包，集成 OpenVINO 深度学习推理、OpenCV 图像处理算子和 EXIF 元数据读写功能，补充 HALCON 原生未提供的算法能力。

## 功能概览

### 深度学习推理（OpenVINO）

| 算子 | 说明 |
|------|------|
| `OpenvinoLoadModel` | 加载 OpenVINO 模型（.xml/.bin），支持 CPU / GPU / NPU 设备 |
| `OpenvinoInfer` | 通用模型推理 |
| `yolo_seg_detect` | YOLOv8 实例分割检测，支持滑动窗口、NMS、掩膜输出 |

### OpenCV 特征检测与匹配

| 算子 | 说明 |
|------|------|
| `cv_orb_detect` | ORB 特征检测与描述子计算 |
| `cv_akaze_detect` | AKAZE 特征检测（旋转不变性） |
| `cv_bf_knn_match` | 暴力 KNN 匹配 + Lowe's ratio 筛选 |
| `cv_estimate_affine_partial2d` | RANSAC 估计仿射变换（平移 + 旋转 + 缩放） |

### 图像处理

| 算子 | 说明 |
|------|------|
| `CLAHE_image` | 自适应直方图均衡化（CLAHE） |
| `remap` | 基于坐标映射的几何变换 |
| `PNGIn` | PNG 编码（可控压缩等级） |
| `PNGOut` | PNG 解码 |
| `add_roi` / `sub_A_roi` / `sub_B_roi` / `mul_roi` / `div_A_roi` / `div_B_roi` | ROI 区域算术运算 |

### EXIF 元数据

| 算子 | 说明 |
|------|------|
| `write_image_exif` | 写入 EXIF 信息：GPS 坐标、相机参数、光圈快门 ISO 等 |

## 项目结构

```
Halcon_YouloBe/
├── source/                  # 源码
│   ├── Halcon_OpenVino.cpp  # 主要实现（OpenVINO、OpenCV、EXIF）
│   └── Halcon_YouloBe.c     # C 接口封装
├── include/                 # 头文件
├── def/                     # HALCON 算子定义文件
├── 3rd/                     # 第三方依赖（OpenCV、OpenVINO、Exiv2）
├── examples/                # 示例程序（.hdev）
├── doc/                     # 文档
├── bin/                     # 编译输出
└── CMakeLists.txt
```

## 依赖

- **HALCON** — 需设置 `HALCONROOT` 环境变量
- **OpenCV 4.5.3** — 已包含在 `3rd/opencv/`
- **OpenVINO** — 已包含在 `3rd/openvino/`
- **Exiv2** — 已包含在 `3rd/exiv/`

## 编译

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

编译产物输出到 `bin/` 目录，包含 C / C++ / .NET 三种接口的 DLL。

## 使用方式

扩展包算子采用**字典传参**模式：

```halcon
* 1. 加载模型
OpenvinoLoadModel (ModelPath, DeviceType, ModelHandle)

* 2. 创建配置字典，设置输入
create_dict (Dict)
set_dict_object (Image, Dict, 'Image')
set_dict_tuple (Dict, 'Handle', ModelHandle)
set_dict_tuple (Dict, 'ConfidenceThreshold', 0.5)

* 3. 执行推理
yolo_seg_detect (Dict)

* 4. 获取结果
get_dict_tuple (Dict, 'ClassIDs', ClassIDs)
get_dict_object (Masks, Dict, 'Masks')
```

更多示例见 [examples/](examples/) 目录。

## 许可证

[MIT License](LICENSE) — Copyright (c) 2025 YoungHowYou
