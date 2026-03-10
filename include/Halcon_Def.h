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
#include <exiv2/exiv2.hpp>
#include "Halcon_YouloBe.h"

using namespace std;
using namespace HalconCpp;


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

    // 修改1: 修正拼写，正确处理FP16配置
    bool load_model(HTuple hv_Path, HTuple hv_Device)
    {
        try
        {
            // 获取模型路径和设备
            std::string model_path = hv_Path.S().Text();
            std::string device = hv_Device.S().Text();
            //bool enable_fp16 = (hv_FP16ENABLE == HTuple("true") || hv_FP16ENABLE == HTuple("True"));
            // 修改3: 传递配置参数
            compiled_model = core.compile_model(model_path, device);

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

    // 修改4: 完整推理流程
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

            // 设置输入张量
            ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());
            infer_request.set_input_tensor(input_tensor);

            // 执行推理
            infer_request.infer();

            // 获取输出
            output_tensor = infer_request.get_output_tensor();

            return true;
        }
        catch (const std::exception &e)
        {
            // std::cerr << "推理失败: " << e.what() << std::endl;
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