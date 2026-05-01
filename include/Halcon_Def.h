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

// exiv2 0.28 以上把 Image::AutoPtr / Value::AutoPtr 改名为 UniquePtr，
// Linux 上常见的发行版（含 Ubuntu 24.04）仍是 0.27.x，仅有 AutoPtr。
// 这里统一对外暴露 ExivImagePtr / ExivValuePtr，源码不再直接用 UniquePtr/AutoPtr。
#if defined(EXIV2_TEST_VERSION) && EXIV2_TEST_VERSION(0,28,0)
  using ExivImagePtr = Exiv2::Image::UniquePtr;
  using ExivValuePtr = Exiv2::Value::UniquePtr;
#else
  using ExivImagePtr = Exiv2::Image::AutoPtr;
  using ExivValuePtr = Exiv2::Value::AutoPtr;
#endif

using namespace std;
using namespace HalconCpp;

// YOLO 检测结果结构体
struct YOLODetection
{
    float x1, y1, x2, y2;  // 边界框坐标（相对于原始图像）
    float confidence;       // 置信度
    int class_id;          // 类别ID
    float mask_coeffs[32]; // 分割掩码系数（最多32个）
    bool has_mask;         // 是否有分割掩码
};

class OpenVINOModel
{
private:
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Shape input_shape; // 保存输入形状供后续使用

public:
    int input_width;  // 修正命名规范
    int input_height; // 修正拼写错误

    bool load_model(HTuple hv_Path, HTuple hv_Device)
    {
        try
        {
            std::string model_path = hv_Path.S().Text();
            std::string device = hv_Device.S().Text();

            // 读取模型，检测其原生输入精度
            auto model = core.read_model(model_path);
            auto model_input_type = model->input().get_element_type();

            ov::AnyMap config;
            if (model_input_type == ov::element::f32)
            {
                // FP32 模型：强制 GPU 以 FP32 精度推理，防止 GPU 自动降精度为 FP16
                // Intel GPU 默认将 FP32 运算降为 FP16，导致 YOLO 坐标值溢出（检测结果错乱）
                config[ov::hint::inference_precision.name()] = ov::element::f32;
                config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
            }
            // FP16 模型：不设置 inference_precision，让模型以原生 FP16 精度运行

            compiled_model = core.compile_model(model, device, config);

            // 获取并保存输入形状
            input_shape = compiled_model.input().get_shape();
            input_width = static_cast<int>(input_shape[3]);
            input_height = static_cast<int>(input_shape[2]);

            // 创建推理请求
            infer_request = compiled_model.create_infer_request();

            return true;
        }
        catch (const std::exception &e)
        {
            return false;
        }
    }

    // 完整推理流程 - 兼容 CPU/GPU/NPU
    bool infer(const cv::Mat &input_image, ov::Tensor & output_tensor)
    {
        try
        {
            // 直接传原始图给 blobFromImage，由它一次完成 resize + normalize + swapRB
            // 省去手动 cv::resize 的中间 Mat 分配和数据复制
            cv::Mat blob;
            cv::dnn::blobFromImage(
                input_image,
                blob,
                1.0 / 255.0,
                cv::Size(input_width, input_height),
                cv::Scalar(0, 0, 0),
                true,  // swapRB: BGR->RGB
                false,
                CV_32F
            );

            // ==========================================
            // 修复: 使用 get_input_tensor() + memcpy 兼容 NPU/GPU
            // ==========================================
            ov::Tensor input_tensor = infer_request.get_input_tensor();
            std::memcpy(input_tensor.data<float>(), blob.ptr<float>(), blob.total() * sizeof(float));

            // 执行推理
            infer_request.infer();

            // ==========================================
            // 修复: 深拷贝输出 Tensor 确保数据在 Host 内存
            // ==========================================
            auto tensor = infer_request.get_output_tensor();
            output_tensor = ov::Tensor(tensor.get_element_type(), tensor.get_shape());
            tensor.copy_to(output_tensor);

            return true;
        }
        catch (const std::exception &e)
        {
            return false;
        }
    }

    // YOLO 实例分割推理 - 兼容 CPU/GPU/NPU
    bool infer_yolo_seg(const cv::Mat &input_image,
                        ov::Tensor &det_output,
                        ov::Tensor &proto_output,
                        float &scale_x, float &scale_y)
    {
        try
        {
            scale_x = (float)input_image.cols / input_width;
            scale_y = (float)input_image.rows / input_height;

            // 直接传原始图给 blobFromImage，一次完成 resize + normalize + swapRB
            cv::Mat blob;
            cv::dnn::blobFromImage(
                input_image,
                blob,
                1.0 / 255.0,
                cv::Size(input_width, input_height),
                cv::Scalar(0, 0, 0),
                true,  // swapRB: BGR->RGB
                false,
                CV_32F
            );

            // memcpy 兼容 NPU/GPU（set_input_tensor 在某些后端不触发同步）
            ov::Tensor input_tensor = infer_request.get_input_tensor();
            std::memcpy(input_tensor.data<float>(), blob.ptr<float>(), blob.total() * sizeof(float));

            infer_request.infer();

            // 按固定索引取输出并深拷贝，强制触发 GPU->Host 内存同步
            // 输出0: [1, N, 38] 检测结果；输出1: [1, 32, H, W] 分割原型
            for (size_t i = 0; i < 2; i++)
            {
                auto tensor = infer_request.get_output_tensor(i);
                auto shape  = tensor.get_shape();
                ov::Tensor &dst = (shape.size() == 3) ? det_output : proto_output;
                dst = ov::Tensor(tensor.get_element_type(), shape);
                std::memcpy(dst.data<float>(), tensor.data<float>(), tensor.get_byte_size());
            }

            return true;
        }
        catch (const std::exception &e)
        {
            return false;
        }
    }
    // 释放所有模型资源
    void Release()
    {
        // 重置推理请求（释放内部张量和内存）
        infer_request = ov::InferRequest();
        // 释放编译模型（释放设备端内存）
        compiled_model = ov::CompiledModel();
        // 对于Core，通常不重置（可复用）
        // 如果确定不再需要，也可以重置
        // core = ov::Core();
    }
};