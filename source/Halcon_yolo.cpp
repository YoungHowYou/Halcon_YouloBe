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
#include "Halcon_YouloBe.h"

using namespace std;
using namespace HalconCpp;

static ov::Core core[8];
static ov::CompiledModel compiled_model[8];
static ov::InferRequest infer_request[8];

Herror HInit_YoloSeg(Hproc_handle proc_handle)
{
	Hcpar *dict;
	INT4_8 num;
	HGetPPar(proc_handle, 1, &dict, &num);
	HTuple hv_Dict(dict, 1);
	HTuple hv_Path, hv_Device, hv_Index, FP16ENABLE;
	GetDictTuple(hv_Dict, u8"工程路径", &hv_Path);
	GetDictTuple(hv_Dict, u8"使用设备", &hv_Device);
	GetDictTuple(hv_Dict, u8"工程序号", &hv_Index);

	// ov::Core core;
	// ov::CompiledModel compiled_model;
	// ov::InferRequest infer_request;

	int idx = hv_Index.I();
	try
	{
		compiled_model[idx] = core[idx].compile_model(hv_Path.S().Text(), hv_Device.S().Text());
		infer_request[idx] = compiled_model[idx].create_infer_request();
		try
		{

			GetDictTuple(hv_Dict, u8"FP16加速", &FP16ENABLE);
			if (FP16ENABLE == 1)
			{
				ov::hint::inference_precision(ov::element::f16); // 关键
			}
		}
		catch (std::exception &e)
		{
			return H_MSG_FAIL;
		}
	}
	catch (std::exception &e)
	{
		return H_MSG_FAIL;
	}
	return H_MSG_TRUE;
}


float sigmoid_function(float a) 
{

	float b = 1. / (1. + exp(-a));
	return b;
}
Herror HProcess_YoloSeg(Hproc_handle proc_handle)
{
	Hcpar* dict;  INT4_8 num;
	HGetPPar(proc_handle, 1, &dict, &num);
	HTuple hv_Dict(dict, 1);
	HTuple hv_Index;
	HObject hv_Image;
	GetDictTuple(hv_Dict, u8"工程序号", &hv_Index);
	GetDictObject(&hv_Image, hv_Dict, "InputImage");
	HTuple hv_Num;
	HTuple hv_conf_thres;
	HTuple hv_nms_thres;

	GetDictTuple(hv_Dict, u8"置信度阈值", &hv_conf_thres);
	GetDictTuple(hv_Dict, u8"重叠度", &hv_nms_thres);


	int idx = hv_Index.I();
	// ----- HALCON → OpenCV -----
	HTuple ptrR, ptrG, ptrB, w, h, type;
	HTuple Channelsnum;
	CountChannels(hv_Image, &Channelsnum);
	cv::Mat img;

	if(Channelsnum==3)
	{ 
		GetImagePointer3(hv_Image, &ptrR, &ptrG, &ptrB, &type, &w, &h);

		cv::Mat planes[3] = 
		{
			cv::Mat(h, w, CV_8UC1, (uchar*)ptrB.L()),
			cv::Mat(h, w, CV_8UC1, (uchar*)ptrG.L()),
			cv::Mat(h, w, CV_8UC1, (uchar*)ptrR.L())
		};

		cv::merge(planes, 3, img);   // 这里仍有一次 memcpy，但比先 vector 再 merge 快

		//cv::Mat matR(h.I(), w.I(), CV_8UC1, (uchar*)ptrR.L());
		//cv::Mat matG(h.I(), w.I(), CV_8UC1, (uchar*)ptrG.L());
		//cv::Mat matB(h.I(), w.I(), CV_8UC1, (uchar*)ptrB.L());
		//std::vector<cv::Mat> ch = { matB,matG,matR };
		//cv::merge(ch, img);
	}
	else
	{
		GetImagePointer1(hv_Image, &ptrR,  &type, &w, &h);
		cv::Mat matR(h.I(), w.I(), CV_8UC1, (uchar*)ptrR.L());
		cv::cvtColor(matR, img, cv::COLOR_GRAY2BGR);
	}
	//img.convertTo(imgFloat, CV_32F, 1 / 255.0);

	cv::Mat imgFloat = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);

	int padw = 0, padh = 0;
	ov::Tensor input(ov::element::f32, { 1,3,640,640 }, imgFloat.data);
	infer_request[idx].set_input_tensor(input);
	infer_request[idx].infer();

	// ----- 取输出 -----
	auto output0 = infer_request[idx].get_output_tensor(0);
	auto output1 = infer_request[idx].get_output_tensor(1);

	auto out0_shape = output0.get_shape();               // [1, C+4+32, 8400]
	auto out1_shape = output1.get_shape();               // [1, 32, 25600]

	/*std::wostringstream woss;
	woss << L"out1_shape: [";
	for (size_t i = 0; i < out1_shape.size(); ++i) {
		woss << out1_shape[i];
		if (i + 1 < out1_shape.size()) woss << L", ";
	}
	woss << L"]";

	MessageBoxW(nullptr, woss.str().c_str(), L"Tensor Shape", MB_OK | MB_ICONINFORMATION);
	*/

	const int num_cls = out0_shape[1] - 4 - 32;          // 动态类别数
	const int num_pred = out0_shape[2];                 // 8400
	const int proto_ch = out1_shape[1];                 // 32
	const int proto_hw = out1_shape[2];                 // 25600
	// 6. 后处理
	//cv::Mat output_buffer(num_pred, 4 + num_cls + 32, CV_32F,output0.data<float>());
	cv::Mat output_buffer(out0_shape[1], out0_shape[2], CV_32F, output0.data<float>());
	cv::transpose(output_buffer, output_buffer);
	const float score_threshold = (float)hv_conf_thres.D();
	const float nms_threshold = (float)hv_nms_thres.D();
	std::vector<int>            class_ids;
	std::vector<float>          class_scores;
	std::vector<cv::Rect>       boxes;          // 这里暂存 cv::Rect（左上+宽高）
	std::vector<cv::Mat>        mask_confs;     // 每个目标 1×32 的 mask 系数

	//std::wstring txt = L"num_cls = " + std::to_wstring(output_buffer.rows);
	//MessageBoxW(nullptr, txt.c_str(), L"区块数", MB_OK | MB_ICONINFORMATION);

	float scale = 1;

	for (int i = 0; i < output_buffer.rows; ++i) 
	{
		// 6.1 找最大得分
		cv::Mat classes_scores = output_buffer.row(i).colRange(4, 4 + num_cls);
		
		cv::Point class_id;
		double maxClassScore;
		cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
		if (maxClassScore > score_threshold) 
		{
			class_scores.push_back(maxClassScore);
			class_ids.push_back(class_id.x);

			float cx = output_buffer.at<float>(i, 0);
			float cy = output_buffer.at<float>(i, 1);
			float w = output_buffer.at<float>(i, 2);
			float h = output_buffer.at<float>(i, 3);

			int left = int((cx - 0.5f * w) * scale);
			int top = int((cy - 0.5f * h) * scale);
			int width = int(w) * scale;
			int height = int(h) * scale;
			
			cv::Mat mask_conf = output_buffer.row(i).colRange(4+ num_cls, 32+4+ num_cls);
			mask_confs.push_back(mask_conf);
			boxes.push_back(cv::Rect(left, top, width, height));


		}
	}
		// 6.4 NMS
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

		cv::Mat proto(32, 25600, CV_32F, output1.data<float>()); //[32,25600]

		cv::Mat clsMap(img.rows, img.cols, CV_8UC1, cv::Scalar(0));   // 全 0 背景
		cv::Mat scoreMap(img.rows, img.cols, CV_32FC1, cv::Scalar(0)); // 记录最高
		HTuple H_x1;
		H_x1 = HTuple();
		HTuple H_y1;
		H_y1 = HTuple();
		HTuple H_x2;
		H_x2 = HTuple();
		HTuple H_y2;
		H_y2 = HTuple();
		HTuple H_class_scores;
		H_class_scores = HTuple();
		HTuple H_class_ids;
		H_class_ids = HTuple();
		for (int i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			int x1 = std::max(0, boxes[indices[i]].x);
 			int y1 = std::max(0, boxes[indices[i]].y);
			int x2 = std::max(0, boxes[indices[i]].br().x);
			int y2 = std::max(0, boxes[indices[i]].br().y);
			int mx1 = int(x1 / scale *0.25);
			int my1 = int(y1/ scale * 0.25);
			int mx2 =  int(x2/ scale * 0.25);
			int my2 = int(y2 / scale * 0.25);

			H_x1 = H_x1.TupleConcat(x1);
			H_y1 = H_y1.TupleConcat(y1);
			H_x2 = H_x2.TupleConcat(x2);
			H_y2 = H_y2.TupleConcat(y2);
			float score = class_scores[indices[i]];
			H_class_scores = H_class_scores.TupleConcat(score);
			int clsId = class_ids[indices[i]];
			H_class_ids = H_class_ids.TupleConcat(clsId);

			cv::Mat m = mask_confs[idx] * proto;
			for (int c = 0; c < m.cols; ++c) m.at<float>(0, c) = sigmoid_function(m.at<float>(0, c));
			cv::Mat m1 = m.reshape(1, 160);        // 160×160
		
			cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
			cv::Mat rm;


			cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1), 0, 0, cv::INTER_LINEAR);

			cv::Mat binMask(rm.size(), CV_8UC1);
			for (int r = 0; r < rm.rows; ++r)
				for (int c = 0; c < rm.cols; ++c)
					binMask.at<uchar>(r, c) = rm.at<float>(r, c) > 0.5f ? 255 : 0;

			/* 2. 逐像素“NMS”：分数高的覆盖低的 */
			for (int y = 0; y < binMask.rows; ++y) {
				for (int x = 0; x < binMask.cols; ++x) {

					


					if (binMask.at<uchar>(y, x) == 0) continue;
					int ux = x1 + x;
					int uy = y1 + y;

					/*std::wstring txt = L"num_cls = " + std::to_wstring(uy);
					MessageBoxW(nullptr, txt.c_str(), L"y", MB_OK | MB_ICONINFORMATION);
					txt = L"num_cls = " + std::to_wstring(ux);
					MessageBoxW(nullptr, txt.c_str(), L"x", MB_OK | MB_ICONINFORMATION);*/
					if (score > scoreMap.at<float>(uy, ux)) {   // 更高分才更新
						scoreMap.at<float>(uy, ux) = score;
						clsMap.at<uchar>(uy, ux) = static_cast<uchar>(clsId); // 1~255
					}
				}
			}
		}
	
	HObject Hrectss;
	GenRectangle1(&Hrectss, H_y1, H_x1, H_y2, H_x2);
	SetDictObject(Hrectss, hv_Dict, u8"目标region");
	SetDictTuple(hv_Dict, u8"置信度", H_class_scores);
	SetDictTuple(hv_Dict, u8"类别", H_class_ids);
	HObject ho_Image;
	GenImage1(&ho_Image, "byte", 640, 640, (int64)clsMap.data);
	SetDictObject(ho_Image, hv_Dict, "OutputImage");	
	return H_MSG_TRUE;
}
