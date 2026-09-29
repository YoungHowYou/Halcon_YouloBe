# Halcon_YouloBe

HALCON 算法扩展包，集成 OpenVINO 深度学习推理（YOLO26 实例分割）、OpenCV 图像处理算子和 EXIF 元数据读写功能，补充 HALCON 原生未提供的算法能力。

## 功能概览

### 深度学习推理（OpenVINO 2026.3）

| 算子 | 说明 |
|------|------|
| `OpenvinoLoadModel` | 加载 OpenVINO 模型（.xml/.bin），支持 CPU / GPU / NPU 设备 |
| `OpenvinoInfer` | 通用模型推理 |
| `yolo_seg_detect` | YOLO26 实例分割检测，支持滑动窗口、NMS、掩膜输出。**自动识别两种导出格式**：<br>• `end2end=True`（NMS-free，TopK 已烘焙进图，`[1,300,D]`）<br>• `end2end=False`（原始输出 `[1,D,8400]`，算子内做类别 argmax + 置信度阈值 + NMS） |

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
| `PNGIn` / `PNGOut` | PNG 编解码（可控压缩等级） |
| `add_roi` / `sub_A_roi` / `sub_B_roi` / `mul_roi` / `div_A_roi` / `div_B_roi` | ROI 区域算术运算 |

### EXIF 元数据

| 算子 | 说明 |
|------|------|
| `write_image_exif` | 写入 EXIF 信息：GPS 坐标、相机参数、光圈快门 ISO 等（支持 PNG/JPEG） |

## 依赖管理：vcpkg

本项目使用 [vcpkg](https://vcpkg.io/) manifest 模式管理依赖（`vcpkg.json`）：

| 依赖 | 版本 | 说明 |
|------|------|------|
| OpenVINO | 2026.3.0 | `cpu,gpu,ir,auto,hetero,npu` features（不含 ONNX/TF 前端，体积更小） |
| OpenCV | 4.12.0 | 默认 features（含 dnn/calib3d/features2d） |
| exiv2 | 0.28.8 | `png` feature（EXIF 写 PNG 需要） |

> 旧的 `3rd/` 预编译库已移除，全部依赖由 vcpkg 构建。

### 编译（Windows）

前置：已安装 vcpkg 且设置 `VCPKG_ROOT` 环境变量（CMake 会自动拾取工具链，无需显式指定）。

```powershell
cmake -B build
cmake --build build --config Release      # Debug 同理
```

编译产物输出到 `bin/`，包含 C / C++ / .NET 三种接口的 DLL，以及全部运行时依赖
（`openvino*.dll` 插件、`opencv*.dll`、`tbb12.dll`、`exiv2.dll` 等，由 vcpkg applocal
与 POST_BUILD 步骤自动拷贝）。

> **中文系统首次构建 OpenVINO 的注意事项**：OpenVINO 的 GPU 插件构建脚本
> （`kernels_db_gen.py` 等）未指定文件编码，在 GBK 代码页机器上会报
> `UnicodeDecodeError`。需设置：
> ```powershell
> $env:PYTHONUTF8='1'
> $env:VCPKG_KEEP_ENV_VARS='PYTHONUTF8'   # 必须！vcpkg 默认会剥离 Python 环境变量
> $env:VCPKG_MAX_CONCURRENCY='4'          # 内存 ≤16GB 的机器建议限制并发，避免链接 OOM
> ```
> 另外 OpenVINO 全量构建约需 20-40 分钟（首次），之后命中 vcpkg 二进制缓存秒过。

### 编译（Linux）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

需要 HALCON 环境变量（`HALCONROOT` / `HALCONEXAMPLES` / `HALCONARCH`），vcpkg 的 lib 目录
会自动烧进 RPATH，HDevelop 无需 `LD_LIBRARY_PATH` 即可加载。

## 脱离 HALCON 复用 yolo_core

YOLO26-seg 推理核心在 `yolo_core/` 静态库（`yolo::Model`，OpenVINO + OpenCV，
**不依赖 HALCON**），扩展包 supply 层只是它的薄封装。其他 C++ 项目可直接复用：

```bash
# 1) 安装到任意前缀（库 + 头文件 + CMake package config）
cmake --install build --config Release --prefix D:/yolo_core

# 2) 消费方 CMakeLists.txt
find_package(yolo_core CONFIG REQUIRED)   # 自动 find_dependency OpenCV/OpenVINO
target_link_libraries(app PRIVATE yolo::yolo_core)

# 3) 配置消费方（安装前缀 + 本仓库的 vcpkg 依赖树）
cmake -B build -DCMAKE_PREFIX_PATH="D:/yolo_core;<本仓库>/build/vcpkg_installed/x64-windows"
```

```cpp
#include <yolo/yolo.hpp>
yolo::Model m;
std::string err;
m.load("best.xml", "CPU", err);                 // 加载（CPU/GPU/NPU）
yolo::InferParams p;                            // conf/nms/mask/滑窗步长/窗口尺寸
std::vector<yolo::Detection> dets;
m.detect(bgr_img, p, dets, err);                // 8UC1/8UC3 输入，原图绝对坐标+掩膜输出
```

## 激活扩展包

`bin/` 必须能从进程搜索到依赖 DLL，因此除了 `HALCONEXTENSIONS` 还需要把包目录加进 `PATH`：

```powershell
[Environment]::SetEnvironmentVariable('HALCONEXTENSIONS', 'D:\desk\source\Halcon_Extension\Halcon_YouloBe', 'Machine')  # 反斜杠路径
# PATH 追加（机器级）：
#   D:\desk\source\Halcon_Extension\Halcon_YouloBe\bin
```

重开终端 / HDevelop 后生效。

## 使用方式

扩展包算子采用**字典传参**模式：

```halcon
* 1. 加载模型
create_dict (DictHandle)
set_dict_tuple (DictHandle, '工程路径', ModelPath)
set_dict_tuple (DictHandle, '使用设备', 'CPU')   * CPU / GPU / NPU
OpenvinoLoadModel (DictHandle, ModelHandle)

* 2. 设置输入与阈值
set_dict_object (Image, DictHandle, 'InputImage')
set_dict_tuple (DictHandle, 'ConfThreshold', 0.25)
set_dict_tuple (DictHandle, 'NMSThreshold', 0.45)   * 仅 raw（end2end=False）模型需要

* 3. 执行推理
yolo_seg_detect (ModelHandle, DictHandle)

* 4. 获取结果
get_dict_tuple (DictHandle, 'ClassIDs', ClassIDs)
get_dict_tuple (DictHandle, 'Confidences', Confidences)
get_dict_object (Masks, DictHandle, 'ClassLabelImage')
get_dict_object (Boxes, DictHandle, 'BoundingBoxes')
```

更多示例见 [examples/](examples/) 目录。

## YOLO26 与 NMS 说明

YOLO26 架构上是 **NMS-free**（one-to-one 分配 + 端到端 head），与 v8/v11 需要外部 NMS 不同：

- **`end2end=True` 导出**（YOLO26 唯一原生支持的端到端方式）：TopK（300）已烘焙进图，
  输出 `[1,300,D]`，无需 NMS。旧版 OpenVINO（≤2025.x）在 GPU 上执行该 TopK 存在
  **性能/结果问题（即所谓"NMS bug"）**——OpenVINO **2026.1 已修复**
  （PR [#34539](https://github.com/openvinotoolkit/openvino/pull/34539)，
  GPU TopK 重写为 radix 内核；2026.2/2026.3 持续优化，并修复了 CPU TopK 的 NaN 排序问题）。
  本项目使用 2026.3.0，该问题已闭环。
- **`end2end=False` 导出**：输出原始 `[1,D,8400]`，必须自己做 NMS——本算子已内置
  （窗口内 `cv::dnn::NMSBoxes`，滑窗模式额外全局 NMS）。
- Ultralytics 侧 `nms=True + dynamic=True` 导出静默丢后处理的 bug 已于 2026-08 修复
  （[#25843](https://github.com/ultralytics/ultralytics/pull/25843)），与本项目的
  end2end 路径无关；重新导出模型建议用 ≥8.4.35。

## 项目结构

```
Halcon_YouloBe/
├── source/                  # 源码
│   ├── Halcon_OpenVino.cpp  # 主要实现（OpenVINO、OpenCV、EXIF）
│   └── Halcon_YouloBe.c     # C 接口封装
├── include/                 # 头文件
├── def/                     # HALCON 算子定义文件
├── cmake/                   # 构建辅助脚本（OpenVINO 插件 DLL 拷贝）
├── yolo_core/               # YOLO26-seg 推理核心静态库（yolo::Model，HALCON-free，可独立安装复用）
├── examples/                # 示例程序（.hdev）与测试模型
├── doc/                     # HTML 帮助
├── help/                    # 算子签名数据库（构建时自动同步，HALCON 调用校验依赖）
├── bin/                     # 编译输出（含全部运行时 DLL）
├── vcpkg.json               # vcpkg 依赖清单
└── CMakeLists.txt
```

## 许可证

[MIT License](LICENSE) — Copyright (c) 2025 YoungHowYou
