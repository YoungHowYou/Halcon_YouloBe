#pragma once
#if defined(_WIN32) || defined(_WIN64)
  #include <windows.h>
  #include <conio.h>
#endif
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "HalconCpp.h"
#include "HDevThread.h"
#include <string>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstring>  // for std::memcpy
#include <cstdint>
#include <exiv2/exiv2.hpp>
#include "Halcon_YouloBe.h"

#if !defined(_WIN32) && !defined(_WIN64)
  // Windows-specific integer type aliases for portability on Linux/macOS
  typedef int64_t  INT64;
  typedef uint64_t UINT64;
#endif

// exiv2 0.28 以上把 Image::AutoPtr / Value::AutoPtr 改名为 UniquePtr。
// 这里统一对外暴露 ExivImagePtr / ExivValuePtr，源码不再直接用 UniquePtr/AutoPtr。
// 注意：0.28 起 EXIV2_TEST_VERSION 宏已删除，统一改用 EXIV2_VERSION + EXIV2_MAKE_VERSION 判断。
#include <exiv2/version.hpp>
#if defined(EXIV2_VERSION) && (EXIV2_VERSION >= EXIV2_MAKE_VERSION(0, 28, 0))
  using ExivImagePtr = Exiv2::Image::UniquePtr;
  using ExivValuePtr = Exiv2::Value::UniquePtr;
#else
  using ExivImagePtr = Exiv2::Image::AutoPtr;
  using ExivValuePtr = Exiv2::Value::AutoPtr;
#endif

using namespace std;
using namespace HalconCpp;

// YOLO26-seg 推理核心在 yolo_core 库（yolo/yolo.hpp），此处不再保留模型类实现