#pragma once
#include <windows.h>
#include <stdio.h>
#include <conio.h>
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
#include < algorithm >
#include <cstring>  // for std::memcpy
#include <exiv2/exiv2.hpp>
#include "Halcon_YouloBe.h"

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

    // 修改1: 强制 GPU 使用 FP32 推理精度，防止 YOLO 坐标溢出
    bool load_model(HTuple hv_Path, HTuple hv_Device)
    {
        try
        {
            // 获取模型路径和设备
            std::string model_path = hv_Path.S().Text();
            std::string device = hv_Device.S().Text();
            
            // ==========================================
            // 关键修复：强制 GPU 使用 FP32 推理精度，防止 YOLO 坐标溢出
            // GPU 默认 FP16 会导致 NMS 后处理节点数值溢出
            // ==========================================
            ov::AnyMap config;
            config[ov::hint::inference_precision.name()] = ov::element::f32;
            config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
            
            compiled_model = core.compile_model(model_path, device, config);

            // 获取并保存输入形状
            input_shape = compiled_model.input().get_shape();
            input_width = static_cast<int>(input_shape[3]);
            input_height = static_cast<int>(input_shape[2]);

            // 创建推理请求
            infer_request = compiled_model.create_infer_request();

            // std::cout << "模型加载成功: " << input_width << "x" << input_height << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            // std::cerr << "模型加载失败: " << e.what() << std::endl;
            return false;
        }
    }

    // 修改4: 完整推理流程 - 兼容 CPU/GPU/NPU
    bool infer(const cv::Mat &input_image, ov::Tensor & output_tensor)
    {
        try
        {
            // 预处理：调整尺寸并创建blob
            cv::Mat resized, blob;
            cv::resize(input_image, resized, cv::Size(input_width, input_height));

            cv::dnn::blobFromImage(
                resized,
                blob,
                1.0 / 255.0, // 归一化到[0,1]
                cv::Size(input_width, input_height),
                cv::Scalar(0, 0, 0),
                true,  // swapRB: BGR->RGB
                false, // crop
                CV_32F // 输出类型
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
            // 记录原始尺寸
            int orig_w = input_image.cols;
            int orig_h = input_image.rows;
            scale_x = (float)orig_w / input_width;
            scale_y = (float)orig_h / input_height;

            // 预处理：调整尺寸并创建blob
            cv::Mat resized, blob;
            cv::resize(input_image, resized, cv::Size(input_width, input_height));

            cv::dnn::blobFromImage(
                resized,
                blob,
                1.0 / 255.0, // 归一化到[0,1]
                cv::Size(input_width, input_height),
                cv::Scalar(0, 0, 0),
                true,  // swapRB: BGR->RGB
                false, // crop
                CV_32F // 输出类型
            );

            // ==========================================
            // 修复1: 使用 get_input_tensor() + memcpy 兼容 NPU/GPU
            // ==========================================
            ov::Tensor input_tensor = infer_request.get_input_tensor();
            std::memcpy(input_tensor.data<float>(), blob.ptr<float>(), blob.total() * sizeof(float));

            // 执行推理
            infer_request.infer();

            // ==========================================
            // 修复2: 强制内存同步，解决 GPU 输出全0
            // 使用 memcpy 强制触发 Host-Device 内存同步
            // ==========================================
            auto outputs = compiled_model.outputs();
            for (size_t i = 0; i < outputs.size(); i++) {
                auto tensor = infer_request.get_output_tensor(i);
                auto shape = tensor.get_shape();
                
                if (shape.size() == 3) {
                    // [1, 300, 38] - 检测结果
                    det_output = ov::Tensor(tensor.get_element_type(), shape);
                    // 强制使用 memcpy 触发数据同步 (tensor.data() 会强制同步 GPU->Host)
                    std::memcpy(det_output.data<float>(), tensor.data<float>(), tensor.get_byte_size());
                } else if (shape.size() == 4) {
                    // [1, 32, 160, 160] - 分割原型
                    proto_output = ov::Tensor(tensor.get_element_type(), shape);
                    // 强制使用 memcpy 触发数据同步
                    std::memcpy(proto_output.data<float>(), tensor.data<float>(), tensor.get_byte_size());
                }
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