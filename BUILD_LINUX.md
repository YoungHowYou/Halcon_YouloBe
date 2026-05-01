# Linux 构建与部署指南

本工程是一个 HALCON 扩展包（C / C++ 共享库），通过 OpenVINO 推理 YOLO 模型，并使用 OpenCV、exiv2 处理图像和元数据。Windows 流程保持不变，本文档只描述 **Linux 上从零到能在 HDevelop 中调用** 的完整步骤。

---

## 1. 前置环境检查

### 1.1 HALCON 环境变量
确认下列变量已导出（通常 `source /opt/halcon/.profile_halcon` 或 MVTec 安装脚本会做这件事）：

```bash
echo $HALCONROOT       # 例：/home/njsc/MVTec/HALCON-24.11-Progress-Steady
echo $HALCONEXAMPLES   # 例：$HALCONROOT/examples
echo $HALCONARCH       # 例：x64-linux 或 aarch64-linux
echo $LD_LIBRARY_PATH  # 必须包含 $HALCONROOT/lib/$HALCONARCH
```

任意一项为空都先补上，CMake 用 `HALCONEXAMPLES` 找 `UseHALCON.cmake`，用 `HALCONROOT` + `HALCONARCH` 定位 `libhalconcpp.so`。

### 1.2 工具链
```bash
sudo apt install build-essential cmake pkg-config
cmake --version    # 需要 ≥ 3.16
g++ --version      # 需要 ≥ 9（OpenVINO 2024 推荐）
```

> 当前 CMakeLists.txt 已把非 .NET 路径的最低版本降到 3.16，3.28.x 可直接使用。

---

## 2. 安装第三方依赖

CMake 在 Linux 分支会**先**查找 `3rd/<lib>/lib/linux/` 下是否有预编译产物，找不到再回退到系统 `find_package`。两种方式任选其一。

### 2.1 方案 A：系统包（推荐，先跑通流程）

#### OpenCV
```bash
sudo apt install libopencv-dev
# 验证
pkg-config --modversion opencv4
```

#### exiv2
```bash
sudo apt install libexiv2-dev
# 验证
pkg-config --modversion exiv2
```

#### OpenVINO（Intel 官方仓库）
按 Intel 官方文档（https://docs.openvino.ai/2024/get-started/install-openvino.html）二选一：

**APT 方式（最省心）**
```bash
# 添加 Intel APT 源（按官方文档复制最新指令）
sudo apt install openvino
```

**Tarball 方式（不污染系统）**
```bash
cd /opt
sudo tar -xzf ~/Downloads/l_openvino_toolkit_*.tgz
sudo mv l_openvino_toolkit_* intel/openvino_2024
# 每次构建/运行前都要 source 一下：
source /opt/intel/openvino_2024/setupvars.sh
```

`setupvars.sh` 会把 OpenVINO 的 `cmake/` 目录写入 `OpenVINO_DIR`，让 `find_package(OpenVINO)` 直接命中。

### 2.2 方案 B：把预编译产物放进 `3rd/`

如果产线机器没有外网或不想动系统，把 Linux 版本的 `.so` 放到对应目录：

```
3rd/opencv/lib/linux/    libopencv_*.so*       # 与 include/opencv2 配套
3rd/openvino/lib/linux/  libopenvino*.so*      # 与 include/openvino 配套
3rd/exiv/lib/linux/      libexiv2.so*          # 与 include/exiv2 配套
```

CMake 检测到这些 `.so` 后会优先用，不会再走 `find_package`。注意：**Windows 的 `.lib` 不能用在 Linux 上**，必须是 ELF `.so`，且 ABI（GCC 版本/C++ 标准）要与本机一致。

---

## 3. 构建

```bash
cd $HALCONROOT/Halcon_Extension/Halcon_YouloBe

# 用单独目录，避免污染原有 Windows 的 build/
mkdir -p build-linux && cd build-linux

# 如果用了 OpenVINO tarball，先 source 一次
source /opt/intel/openvino_2024/setupvars.sh   # 视情况

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

构建产物会落在仓库根目录的 `bin/` 下，期望看到：

```
bin/libHalcon_YouloBe.so     # 主扩展包
bin/libHalcon_YouloBec.so    # C 接口（HALCON_AddExtensionPackage 生成的 *cint 目标）
bin/libHalcon_YouloBecpp.so  # C++ 接口（*cppint 目标）
```

构建结束时 CMake 会**自动**把这三个 `.so` 软链到
`lib/${HALCON_ARCHITECTURE}/`（例如 `lib/x64-linux/`），不再需要手工 `ln -sf`。

构建过程中关注 CMake 的状态行确认走对了来源：

```
-- Using bundled OpenCV from .../3rd/opencv/lib/linux       # 走方案 B
-- Using system OpenCV 4.x.y (...)                          # 走方案 A
-- Using bundled OpenVINO from ...
-- Using system OpenVINO
```

### 3.1 Debug 构建
单配置生成器（默认 Make/Ninja）下用 `-DCMAKE_BUILD_TYPE=Debug` 切换；Linux 分支不区分 debug/release 库目录（OpenCV/OpenVINO 在 Linux 上通常没有 `d` 后缀）。

---

## 4. 注册到 HALCON

HALCON 通过环境变量 `HALCONEXTENSIONS` 找扩展包。两种部署方式：

### 4.1 直接用源码目录（开发期推荐）

```bash
# 把本工程根目录加到 HALCONEXTENSIONS（多个用 ":" 分隔）
export HALCONEXTENSIONS=$HOME/MVTec/HALCON-24.11-Progress-Steady/Halcon_Extension/Halcon_YouloBe:$HALCONEXTENSIONS
```

CMake 已经把 `lib/$HALCONARCH/libHalcon_YouloBe*.so` 软链准备好，HALCON 启动时
直接按这个路径加载，**不再需要手动 `ln -sf` 或额外设置 `LD_LIBRARY_PATH`**：
`.so` 内部的 RPATH 已包含 `bin/` 与 `3rd/<lib>/lib/linux/` 三个目录，运行时
自动定位 OpenCV / OpenVINO / exiv2。

> 如果以后把 `bin/` 移走或单独打包发布，记得把对应的 `3rd/*/lib/linux/*.so*`
> 一起带走（保持 `lib/x64-linux/` 与 `3rd/` 的相对位置不变即可）。

### 4.2 安装到 HALCON 标准目录

打成正式扩展包后，把整个 `Halcon_YouloBe/` 目录拷贝到 `$HALCONROOT/extension_packages/` 即可，HALCON 启动时会自动扫描。

### 4.3 在 HDevelop 验证

```hdev
* 任选一个本扩展包暴露的算子，例如：
OpenvinoLoadModel(...)
```

如果加载失败，检查顺序：
1. `ldd bin/libHalcon_YouloBe.so` 看是否所有依赖（opencv, openvino, exiv2, halconcpp）都能解析。
2. `echo $HALCONEXTENSIONS` 是否包含本工程目录。
3. HDevelop 的 *Edit → Preferences → Paths* 里也可以查看实际加载的扩展包列表。

---

## 5. 运行期常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `cannot open shared object file: libopenvino.so.xxxx` | OpenVINO 未在 `LD_LIBRARY_PATH` | `source setupvars.sh`；或把 OpenVINO 库目录加到 `LD_LIBRARY_PATH` |
| `undefined symbol: _ZN2cv...` | 编译时与运行时 OpenCV 版本不一致 | 卸载多余 OpenCV，确保 `ldd` 指向同一份 |
| HDevelop 启动时报扩展包加载失败 | `.so` 没装到 `lib/$HALCONARCH/` | 见 §4.1 软链步骤 |
| `INT64` / `UINT64` 编译报错 | 头文件里包含顺序问题 | `Halcon_Def.h` 已为非 Windows 平台 typedef，确认源文件 `#include "Halcon_Def.h"` 在最前 |
| GPU/NPU 推理结果异常 | OpenVINO 默认精度被降级 | 代码已对 FP32 模型强制 ACCURACY 模式（见 `Halcon_Def.h` 中 `load_model`） |

---

## 6. 跨平台代码改动备忘

本次为 Linux 适配做的改动，方便后续维护：

- `include/Halcon_YouloBe.h`：`Test_EXPORTS_API` 在非 Windows 改用 `__attribute__((visibility("default")))`。
- `include/Halcon_Def.h`：
  - `<windows.h>` / `<conio.h>` 用 `_WIN32` 宏卫起来。
  - 为非 Windows 平台 `typedef` 了 `INT64` / `UINT64`，避免改散落在源文件里的旧称谓。
- `CMakeLists.txt`：
  - 非 .NET 路径的最低 CMake 版本降到 3.16；`HALCON_DOTNET` 在非 Windows 默认 `OFF`。
  - HALCON 库目录按 `HALCONARCH` 自动选择。
  - 三方库 Windows 走 `3rd/*/lib/{debug,release}/*.lib`；Linux 优先
    `3rd/*/lib/linux/`，回退到系统 `find_package`。**头文件**也按平台分发：
    Linux 优先用 `3rd/*/include-linux`，回退到 `3rd/*/include`。
  - Linux 启用 `CXX_VISIBILITY_PRESET hidden`，仅导出 `Test_EXPORTS_API` 标记的接口。
  - Linux 强制 `cxx_std_14`（项目里 bundled 的 exiv2 0.27 仍依赖
    `std::auto_ptr`，C++17 默认会编译失败）。
  - Linux 把 `bin/`、`3rd/<lib>/lib/linux/` 通过 `$ORIGIN` 相对路径写入 RPATH，
    `--disable-new-dtags` 确保用 `RPATH` 而非 `RUNPATH`，HDevelop 加载本扩展时
    无需额外 `LD_LIBRARY_PATH`。
  - 三个目标（主、`*cint`、`*cppint`）的 POST_BUILD 都会把对应 `.so` 软链到
    `lib/${HALCON_ARCHITECTURE}/`，HALCON 直接按官方目录结构发现扩展包。

- `source/Halcon_OpenVino.cpp`：
  - 非 Windows 上 `HPGetPPar` 的输出是 `Hcpar const*`，把这几处局部变量改成
    `const Hcpar*`，并在构造 `HTuple` 时 `const_cast` 回去（HTuple 构造器只
    读源数据，cast 安全）。
  - `WideCharToMultiByte` / `MultiByteToWideChar` 仅 Windows 提供，仅在
    `_WIN32` 下编译；Linux 直接走 UTF-8 字节序，不需要这层转换。
- `include/Halcon_Def.h`：
  - 用 `EXIV2_TEST_VERSION(0,28,0)` 定义统一的 `ExivImagePtr` /
    `ExivValuePtr` 别名。Windows 的 exiv2 0.28 用 `UniquePtr`，Linux 上常见
    的 0.27.x 仍是 `AutoPtr`，源码中只用别名即可两边都过。

如果未来要再加一个第三方库，按相同的模式扩展 Linux 分支即可。
