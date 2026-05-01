#include "Halcon_Def.h"
#include <cmath>

#define H_Openvino_TAG 0xC0FFEE70
#define H_Openvino_SEM_TYPE "Openvino"
extern "C"
{
    typedef struct
    {
        OpenVINOModel OpenVINOModels;
    } OpenvinoHUserHandleData;

    static Herror OpenvinoHUserHandleDestructor(Hproc_handle ph, OpenvinoHUserHandleData *data)
    {
        data->OpenVINOModels.Release();
        return HFree(ph, data);
    }
    // 句柄类型描述符
    const HHandleInfo OpenvinoHandleTypeUser = HANDLE_INFO_INITIALIZER_NOSER(H_Openvino_TAG, H_Openvino_SEM_TYPE, OpenvinoHUserHandleDestructor, NULL, NULL);
}
#define Def_INOpenvinoObject(pos, pUserData) \
    OpenvinoHUserHandleData *(pUserData);    \
    HGetCElemH1(proc_handle, (pos), &OpenvinoHandleTypeUser, &(pUserData))

#define Def_OUTOpenvinoObject(pos, pUserData)                                        \
    OpenvinoHUserHandleData **(pUserData);                                           \
    HCkP(HAllocOutputHandle(proc_handle, 1, &(pUserData), &OpenvinoHandleTypeUser)); \
    HCkP(HAlloc(proc_handle, sizeof(OpenvinoHUserHandleData), (void **)(pUserData)))
#define OUTOpenvinoObject(pUserData) (*(pUserData))

Herror Openvino加载模型(Hproc_handle proc_handle)
{

    const Hcpar *容器;
    INT4_8 参数个数; // 参数个数
    HGetPPar(proc_handle, 1, &容器, &参数个数);
    HTuple hv_Dict(const_cast<Hcpar*>(容器), 1);
    HTuple hv_Path, hv_Device, hv_Index, FP16ENABLE;
    GetDictTuple(hv_Dict, u8"工程路径", &hv_Path);
    GetDictTuple(hv_Dict, u8"使用设备", &hv_Device);
    OpenVINOModel detect_model;
    if (!detect_model.load_model(hv_Path, hv_Device))
    {
        // std::cerr << "检测模型加载失败！" << std::endl;
        return 10000 * H__LINE__;
    }
    Def_OUTOpenvinoObject(1, handle_data);
    (*handle_data)->OpenVINOModels = detect_model;
    SetDictTuple(hv_Dict, u8"高",detect_model.input_height);
    SetDictTuple(hv_Dict, u8"宽",detect_model.input_width);

    return H_MSG_TRUE;
}

Herror Openvino推理模型(Hproc_handle proc_handle)
{

    const Hcpar *容器;
    INT4_8 参数个数; // 参数个数
    Def_INOpenvinoObject(1, handle_data);
    HGetPPar(proc_handle, 2, &容器, &参数个数);
    HTuple hv_Dict(const_cast<Hcpar*>(容器), 1);

    HObject hv_Image;
    GetDictObject(&hv_Image, hv_Dict, "InputImage");

    HTuple ptrR, ptrG, ptrB, w, h, type;
    HTuple Channelsnum;
    CountChannels(hv_Image, &Channelsnum);
    cv::Mat img;
    if (Channelsnum == 3)
    {
        GetImagePointer3(hv_Image, &ptrR, &ptrG, &ptrB, &type, &w, &h);

        cv::Mat planes[3] =
            {
                cv::Mat(h, w, CV_8UC1, (uchar *)ptrB.L()),
                cv::Mat(h, w, CV_8UC1, (uchar *)ptrG.L()),
                cv::Mat(h, w, CV_8UC1, (uchar *)ptrR.L())};

        cv::merge(planes, 3, img); // 这里仍有一次 memcpy，但比先 vector 再 merge 快
    }
    else
    {
        GetImagePointer1(hv_Image, &ptrR, &type, &w, &h);
        cv::Mat matR(h.I(), w.I(), CV_8UC1, (uchar *)ptrR.L());
        cv::cvtColor(matR, img, cv::COLOR_GRAY2BGR);
    }
    ov::Tensor detect_result;
    handle_data->OpenVINOModels.infer(img, detect_result);
    HObject ho_Image;
    const INT64 *data = detect_result.data<INT64>();
    GenImage1(&ho_Image, "int8",  handle_data->OpenVINOModels.input_width,  handle_data->OpenVINOModels.input_height, (int64)data);
    SetDictObject(ho_Image, hv_Dict, "OutputImage");

    return H_MSG_TRUE;
}

/*=============================================================================
 * 辅助结构体：用于存储单次检测结果，方便后续做 NMS
 *===========================================================================*/
struct DetectResult {
    int class_id;
    float conf;
    cv::Rect box;       // 在原图中的绝对坐标
    cv::Mat mask_roi;   // 160x160 尺度下截取的 mask (未上采样)
    cv::Rect mask_box;  // mask 对应的原图绝对坐标
};

/*=============================================================================
 * 全局 NMS 函数
 *===========================================================================*/
static void ApplyNMS(std::vector<DetectResult>& results, float nms_thresh) {
    if (results.empty()) return;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (const auto& r : results) {
        class_ids.push_back(r.class_id);
        confidences.push_back(r.conf);
        boxes.push_back(r.box);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0f, nms_thresh, indices);
    
    std::vector<DetectResult> nms_results;
    nms_results.reserve(indices.size());
    for (int idx : indices) {
        nms_results.push_back(results[idx]);
    }
    results = std::move(nms_results);
}

/*=============================================================================
 * 算子: Openvino_YOLO_Seg_Detect
 *
 * 模型输出格式 (best.xml):
 * - 检测结果 [1, 300, 38]: 
 *   - [0~3]: boxes (x1, y1, x2, y2) 相对于640x640
 *   - [4]: confidence
 *   - [5]: class_id
 *   - [6~37]: 32个mask系数
 * - 分割原型 [1, 32, 160, 160]
 *
 * 输出:
 *   - BoundingBoxes: Region 数组 (每个检测一个矩形Region)
 *   - ClassLabelImage: 灰度图像 (与原图同大小, 像素值=类别ID)
 *   - NumDetections: 检测数量
 *   - Confidences: 置信度数组
 *   - ClassIDs: 类别ID数组
 * 
 * 新增功能:
 *   - 支持滑窗检测大图 (通过字典参数 "步长X"/"步长Y" 或 "步长" 设置)
 *   - 支持小图自动缩放到模型尺寸
 *   - 滑窗模式下自动应用 NMS 去重
 *===========================================================================*/
Herror Openvino_YOLO_Seg_Detect(Hproc_handle proc_handle)
{
    const Hcpar *容器;
    INT4_8 参数个数;
    Def_INOpenvinoObject(1, handle_data);
    HGetPPar(proc_handle, 2, &容器, &参数个数);
    HTuple hv_Dict(const_cast<Hcpar*>(容器), 1);

    // 1. 获取输入图像与基本参数
    HObject hv_Image;
    GetDictObject(&hv_Image, hv_Dict, "InputImage");

    double conf_thresh = 0.25, mask_thresh = 0.5, nms_thresh = 0.45;
    try { HTuple hv_T; GetDictTuple(hv_Dict, "ConfThreshold", &hv_T); conf_thresh = hv_T.D(); } catch (...) {}
    try { HTuple hv_T; GetDictTuple(hv_Dict, "MaskThreshold", &hv_T); mask_thresh = hv_T.D(); } catch (...) {}
    try { HTuple hv_T; GetDictTuple(hv_Dict, "NMSThreshold", &hv_T); nms_thresh = hv_T.D(); } catch (...) {}

    // 2. 获取模型宽高与滑窗步长
    int model_w = handle_data->OpenVINOModels.input_width;
    int model_h = handle_data->OpenVINOModels.input_height;
    int step_x = model_w, step_y = model_h;
    try { HTuple hv_T; GetDictTuple(hv_Dict, u8"宽", &hv_T); model_w = hv_T.I(); } catch (...) {}
    try { HTuple hv_T; GetDictTuple(hv_Dict, u8"高", &hv_T); model_h = hv_T.I(); } catch (...) {}
    try { HTuple hv_T; GetDictTuple(hv_Dict, u8"步长", &hv_T); step_x = hv_T.I(); step_y = hv_T.I(); } catch (...) {}
    try { HTuple hv_T; GetDictTuple(hv_Dict, u8"步长X", &hv_T); step_x = hv_T.I(); } catch (...) {}
    try { HTuple hv_T; GetDictTuple(hv_Dict, u8"步长Y", &hv_T); step_y = hv_T.I(); } catch (...) {}

    // 3. 图像转换 (Halcon -> OpenCV)
    HTuple ptrR, ptrG, ptrB, w, h, type, Channelsnum;
    CountChannels(hv_Image, &Channelsnum);
    cv::Mat orig_img;
    int orig_w, orig_h;
    if (Channelsnum == 3) {
        GetImagePointer3(hv_Image, &ptrR, &ptrG, &ptrB, &type, &w, &h);
        orig_w = w.I(); orig_h = h.I();
        cv::Mat planes[3] = {
            cv::Mat(h, w, CV_8UC1, (uchar*)ptrB.L()), 
            cv::Mat(h, w, CV_8UC1, (uchar*)ptrG.L()), 
            cv::Mat(h, w, CV_8UC1, (uchar*)ptrR.L())
        };
        cv::merge(planes, 3, orig_img);
    } else {
        GetImagePointer1(hv_Image, &ptrR, &type, &w, &h);
        orig_w = w.I(); orig_h = h.I();
        cv::Mat matR(h.I(), w.I(), CV_8UC1, (uchar*)ptrR.L());
        cv::cvtColor(matR, orig_img, cv::COLOR_GRAY2BGR);
    }

    std::vector<DetectResult> all_results;
    all_results.reserve(64); // 预分配，避免 push_back 触发多次重分配

    // 内部 Lambda：执行推理并解析当前窗口结果
    auto DoInference = [&](const cv::Mat& input_img, int offset_x, int offset_y, float ratio_x, float ratio_y) {
        ov::Tensor det_output, proto_output;
        float scale_x, scale_y;
        if (!handle_data->OpenVINOModels.infer_yolo_seg(input_img, det_output, proto_output, scale_x, scale_y)) return;

        float* det_data = det_output.data<float>();
        auto det_shape = det_output.get_shape();
        int num_dets = (int)det_shape[1];  // 300
        int det_dims = (int)det_shape[2];  // 38

        float* proto_data = proto_output.data<float>();
        auto proto_shape = proto_output.get_shape();
        int proto_c = (int)proto_shape[1]; // 32
        int proto_h = (int)proto_shape[2]; // 160
        int proto_w = (int)proto_shape[3]; // 160

        // 将 proto_data 包装为 OpenCV Mat [32, 25600]（零拷贝）
        cv::Mat proto_mat(proto_c, proto_h * proto_w, CV_32FC1, proto_data);

        // 预计算合并后的缩放系数，减少循环内乘法次数
        const float fx = scale_x * ratio_x;
        const float fy = scale_y * ratio_y;

        for (int i = 0; i < num_dets; i++) {
            float* det = det_data + i * det_dims;
            float conf = det[4];
            if (conf < conf_thresh) continue;

            int cls = (int)det[5];
            float px1 = det[0], py1 = det[1], px2 = det[2], py2 = det[3];

            // 一步映射到原图绝对坐标（合并 scale 和 ratio 两次乘法）
            int x1 = (int)(px1 * fx) + offset_x;
            int y1 = (int)(py1 * fy) + offset_y;
            int x2 = (int)(px2 * fx) + offset_x;
            int y2 = (int)(py2 * fy) + offset_y;

            x1 = std::max(0, std::min(x1, orig_w - 1));
            y1 = std::max(0, std::min(y1, orig_h - 1));
            x2 = std::max(0, std::min(x2, orig_w - 1));
            y2 = std::max(0, std::min(y2, orig_h - 1));

            if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;

            // Mask 计算
            int mask_x1 = std::max(0, (int)(px1 / 4.0f));
            int mask_y1 = std::max(0, (int)(py1 / 4.0f));
            int mask_x2 = std::min(proto_w - 1, (int)(px2 / 4.0f));
            int mask_y2 = std::min(proto_h - 1, (int)(py2 / 4.0f));
            
            if (mask_x2 - mask_x1 <= 0 || mask_y2 - mask_y1 <= 0) continue;

            cv::Mat coeffs_mat(1, proto_c, CV_32FC1, det + 6);
            cv::Mat mask_160_flat = coeffs_mat * proto_mat;
            cv::Mat mask_160 = mask_160_flat.reshape(1, proto_h);
            // 必须 clone，因为底层内存会被下一次推理覆盖
            cv::Mat mask_roi = mask_160(cv::Rect(mask_x1, mask_y1, mask_x2 - mask_x1, mask_y2 - mask_y1)).clone();

            DetectResult res;
            res.class_id = cls;
            res.conf = conf;
            res.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            res.mask_roi = mask_roi;
            res.mask_box = res.box;
            all_results.push_back(res);
        }
    };

    // 4. 核心逻辑：尺寸判断与滑窗
    if (orig_w == model_w && orig_h == model_h) {
        // 分支 1：尺寸相等，直接检测
        DoInference(orig_img, 0, 0, 1.0f, 1.0f);
    } 
    else if (orig_w < model_w || orig_h < model_h) {
        // 分支 2：尺寸偏小，缩放到模型尺寸
        cv::Mat resized_img;
        cv::resize(orig_img, resized_img, cv::Size(model_w, model_h));
        float ratio_x = (float)orig_w / model_w;
        float ratio_y = (float)orig_h / model_h;
        DoInference(resized_img, 0, 0, ratio_x, ratio_y);
    } 
    else {
        // 分支 3：尺寸偏大，滑窗检测
        int start_y = 0; // 声明在外层循环作用域
        for (int y = 0; y < orig_h; y += step_y) {
            start_y = y;
            int start_x = 0; // 声明在外层循环作用域
            for (int x = 0; x < orig_w; x += step_x) {
                start_x = x;
                // 边界处理：如果剩余尺寸不足模型宽高，则向左/上平移保证窗口大小
                if (start_x + model_w > orig_w) start_x = std::max(0, orig_w - model_w);
                if (start_y + model_h > orig_h) start_y = std::max(0, orig_h - model_h);

                cv::Rect roi(start_x, start_y, model_w, model_h);
                cv::Mat crop_img = orig_img(roi);
                
                DoInference(crop_img, start_x, start_y, 1.0f, 1.0f);
                
                if (start_x + model_w >= orig_w) break; // 避免死循环
            }
            if (start_y + model_h >= orig_h) break; // 避免死循环
        }
        // 滑窗会导致重叠，必须进行 NMS
        ApplyNMS(all_results, nms_thresh);
    }

    // 5. 整合结果并生成 Halcon 对象
    int num_valid = (int)all_results.size();
    SetDictTuple(hv_Dict, "NumDetections", (Hlong)num_valid);

    if (num_valid > 0) {
        HTuple hv_Confs, hv_Classes;
        HTuple hv_Row1, hv_Col1, hv_Row2, hv_Col2; // 用于一次性生成所有矩形

        // 准备标签图像
        HObject ho_ClassLabelImage;
        GenImageConst(&ho_ClassLabelImage, "uint2", orig_w, orig_h);
        HTuple hv_LabelPtr, hv_LabelType, hv_LabelW, hv_LabelH;
        GetImagePointer1(ho_ClassLabelImage, &hv_LabelPtr, &hv_LabelType, &hv_LabelW, &hv_LabelH);
        ushort* label_data = (ushort*)hv_LabelPtr.L();
        memset(label_data, 0, orig_w * orig_h * sizeof(ushort));

        for (int i = 0; i < num_valid; i++) {
            const auto& res = all_results[i];
            hv_Confs[i] = (double)res.conf;
            hv_Classes[i] = (Hlong)res.class_id;

            hv_Row1[i] = (double)res.box.y;
            hv_Col1[i] = (double)res.box.x;
            hv_Row2[i] = (double)(res.box.y + res.box.height);
            hv_Col2[i] = (double)(res.box.x + res.box.width);

            // 恢复并填充 Mask：用 OpenCV ROI + copyTo 替代逐像素循环
            if (!res.mask_roi.empty() && res.box.width > 0 && res.box.height > 0) {
                cv::Mat mask_up;
                cv::resize(res.mask_roi, mask_up, cv::Size(res.box.width, res.box.height), 0, 0, cv::INTER_LINEAR);
                cv::Mat mask_bin = mask_up > 0.0f; // CV_8UC1

                // 将 label_data 包装为 Mat（零拷贝），取出对应 ROI
                cv::Mat label_mat(orig_h, orig_w, CV_16UC1, label_data);
                cv::Mat dst_roi = label_mat(res.box);

                // 用 fill_val 填充 mask 为 true 的像素，其余保持不变
                cv::Mat fill_mat(res.box.height, res.box.width, CV_16UC1,
                                 cv::Scalar((ushort)(res.class_id + 1)));
                fill_mat.copyTo(dst_roi, mask_bin);
            }
        }

        // 高能优化：使用 HTuple 数组一次性生成所有矩形，彻底告别 ConcatObj 的性能灾难！
        HObject ho_Rects;
        GenRectangle1(&ho_Rects, hv_Row1, hv_Col1, hv_Row2, hv_Col2);

        SetDictTuple(hv_Dict, "Confidences", hv_Confs);
        SetDictTuple(hv_Dict, "ClassIDs", hv_Classes);
        SetDictObject(ho_Rects, hv_Dict, "BoundingBoxes");
        SetDictObject(ho_ClassLabelImage, hv_Dict, "ClassLabelImage");
    }

    return H_MSG_TRUE;
}
/*=============================================================================
 * 算子: Openvino_YOLO_Seg_Detect
 *
 * 模型输出格式 (best.xml):
 * - 检测结果 [1, 300, 38]: 
 *   - [0~3]: boxes (x_center, y_center, width, height) 相对于640x640
 *   - [4]: confidence
 *   - [5]: class_id
 *   - [6~37]: 32个mask系数
 * - 分割原型 [1, 32, 160, 160]
 *
 * 输出:
 *   - BoundingBoxes: Region 数组 (每个检测一个矩形Region)
 *   - ClassLabelImage: 灰度图像 (与原图同大小, 像素值=类别ID)
 *===========================================================================*/

int roi_error(Himage small_image, Himage big_image, int x, int y, int w, int h)
{
	if (small_image.kind != UINT2_IMAGE)
	{
		return 1;
	}
	if (big_image.kind != UINT2_IMAGE)
	{
		return 2;
	}
	if ((x < 0) || (y < 0) || (w < 0) || (h < 0))
	{
		return 3;
	}
	if (x + w > big_image.width)
	{
		return 4;
	}
	if (y + h > big_image.height)
	{
		return 5;
	}
	if (small_image.width != w)
	{
		return 6;
	}
	if (small_image.height != h)
	{
		return 7;
	}
	return 0;
}
//A+(B∩Roi)
int add_roi(Himage small_image, Himage big_image, int x, int y, int w, int h)
{
	int error = roi_error(small_image, big_image, x, y, w, h);
	if (error != 0)
	{
		return error;
	}
	cv::Mat small_imagein((int)small_image.height, (int)small_image.width, CV_16UC1, small_image.pixel.u.p);
	cv::Mat big_imagein((int)big_image.height, (int)big_image.width, CV_16UC1, big_image.pixel.u.p);
	cv::add(small_imagein, big_imagein(cv::Rect(x, y, w, h)), big_imagein(cv::Rect(x, y, w, h)));
	return 0;
}
//A.*(B∩Roi)
int mul_roi(Himage small_image, Himage big_image, int x, int y, int w, int h)
{
	int error = roi_error(small_image, big_image, x, y, w, h);
	if (error != 0)
	{
		return error;
	}
	cv::Mat small_imagein((int)small_image.height, (int)small_image.width, CV_16UC1, small_image.pixel.u.p);
	cv::Mat big_imagein((int)big_image.height, (int)big_image.width, CV_16UC1, big_image.pixel.u.p);
	cv::multiply(small_imagein, big_imagein(cv::Rect(x, y, w, h)), big_imagein(cv::Rect(x, y, w, h)));
	return 0;
}
//A-(B∩Roi)
int sub_B_roi(Himage small_image, Himage big_image, int x, int y, int w, int h)
{
	int error = roi_error(small_image, big_image, x, y, w, h);
	if (error != 0)
	{
		return error;
	}
	cv::Mat small_imagein((int)small_image.height, (int)small_image.width, CV_16UC1, small_image.pixel.u.p);
	cv::Mat big_imagein((int)big_image.height, (int)big_image.width, CV_16UC1, big_image.pixel.u.p);
	cv::subtract(small_imagein, big_imagein(cv::Rect(x, y, w, h)), big_imagein(cv::Rect(x, y, w, h)));
	return 0;
}
//A/(B∩Roi)
int div_B_roi(Himage small_image, Himage big_image, int x, int y, int w, int h)
{
	int error = roi_error(small_image, big_image, x, y, w, h);
	if (error != 0)
	{
		return error;
	}
	cv::Mat small_imagein((int)small_image.height, (int)small_image.width, CV_16UC1, small_image.pixel.u.p);
	cv::Mat big_imagein((int)big_image.height, (int)big_image.width, CV_16UC1, big_image.pixel.u.p);
	cv::divide(small_imagein, big_imagein(cv::Rect(x, y, w, h)), big_imagein(cv::Rect(x, y, w, h)));
	return 0;
}
//(A∩Roi)/B
int div_A_roi(Himage big_image, Himage small_image, int x, int y, int w, int h)
{
	int error = roi_error(small_image, big_image, x, y, w, h);
	if (error != 0)
	{
		return error;
	}
	cv::Mat small_imagein((int)small_image.height, (int)small_image.width, CV_16UC1, small_image.pixel.u.p);
	cv::Mat big_imagein((int)big_image.height, (int)big_image.width, CV_16UC1, big_image.pixel.u.p);
	cv::divide(big_imagein(cv::Rect(x, y, w, h)), small_imagein, big_imagein(cv::Rect(x, y, w, h)));
	return 0;
}
//(A∩Roi)-B
int sub_A_roi(Himage big_image, Himage small_image, int x, int y, int w, int h)
{
	int error = roi_error(small_image, big_image, x, y, w, h);
	if (error != 0)
	{
		return error;
	}
	cv::Mat small_imagein((int)small_image.height, (int)small_image.width, CV_16UC1, small_image.pixel.u.p);
	cv::Mat big_imagein((int)big_image.height, (int)big_image.width, CV_16UC1, big_image.pixel.u.p);
	cv::subtract(big_imagein(cv::Rect(x, y, w, h)), small_imagein, big_imagein(cv::Rect(x, y, w, h)));
	return 0;
}



Herror HCadd_roi(Hproc_handle proc_handle)
{
	Hkey in_smallobj_key, in_bigobj_key, out_image_key;
	Himage    insmallimage;
	Himage    inbig_image;
	Hcpar sy, sx, ew, eh;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &sy, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &sx, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &ew, 1);
	HGetSPar(proc_handle, 4, LONG_PAR, &eh, 1);

	HGetObj(proc_handle, 1, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);
	HGetObj(proc_handle, 2, 1, &in_bigobj_key);
	HGetDImage(proc_handle, in_bigobj_key, 1, &inbig_image);

	iRes = add_roi(insmallimage, inbig_image, sx.par.l, sy.par.l, ew.par.l, eh.par.l);
	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;
}

Herror HCmul_roi(Hproc_handle proc_handle)
{
	Hkey in_smallobj_key, in_bigobj_key, out_image_key;
	Himage    insmallimage;
	Himage    inbig_image;
	Hcpar sy, sx, ew, eh;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &sy, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &sx, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &ew, 1);
	HGetSPar(proc_handle, 4, LONG_PAR, &eh, 1);

	HGetObj(proc_handle, 1, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);
	HGetObj(proc_handle, 2, 1, &in_bigobj_key);
	HGetDImage(proc_handle, in_bigobj_key, 1, &inbig_image);

	iRes = mul_roi(insmallimage, inbig_image, sx.par.l, sy.par.l, ew.par.l, eh.par.l);
	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;
}

Herror HCsub_B_roi(Hproc_handle proc_handle)
{
	Hkey in_smallobj_key, in_bigobj_key, out_image_key;
	Himage    insmallimage;
	Himage    inbig_image;
	Hcpar sy, sx, ew, eh;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &sy, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &sx, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &ew, 1);
	HGetSPar(proc_handle, 4, LONG_PAR, &eh, 1);

	HGetObj(proc_handle, 1, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);
	HGetObj(proc_handle, 2, 1, &in_bigobj_key);
	HGetDImage(proc_handle, in_bigobj_key, 1, &inbig_image);
	iRes = sub_B_roi(insmallimage, inbig_image, sx.par.l, sy.par.l, ew.par.l, eh.par.l);
	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;
}

Herror HCdiv_B_roi(Hproc_handle proc_handle)
{
	Hkey in_smallobj_key, in_bigobj_key, out_image_key;
	Himage    insmallimage;
	Himage    inbig_image;
	Hcpar sy, sx, ew, eh;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &sy, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &sx, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &ew, 1);
	HGetSPar(proc_handle, 4, LONG_PAR, &eh, 1);

	HGetObj(proc_handle, 1, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);
	HGetObj(proc_handle, 2, 1, &in_bigobj_key);
	HGetDImage(proc_handle, in_bigobj_key, 1, &inbig_image);

	iRes = div_B_roi(insmallimage, inbig_image, sx.par.l, sy.par.l, ew.par.l, eh.par.l);
	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;
}

Herror HCdiv_A_roi(Hproc_handle proc_handle)
{
	Hkey in_smallobj_key, in_bigobj_key, out_image_key;
	Himage    insmallimage;
	Himage    inbig_image;
	Hcpar sy, sx, ew, eh;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &sy, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &sx, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &ew, 1);
	HGetSPar(proc_handle, 4, LONG_PAR, &eh, 1);

	HGetObj(proc_handle, 2, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);
	HGetObj(proc_handle, 1, 1, &in_bigobj_key);
	HGetDImage(proc_handle, in_bigobj_key, 1, &inbig_image);

	iRes = div_A_roi(inbig_image, insmallimage, sx.par.l, sy.par.l, ew.par.l, eh.par.l);

	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;

}

Herror HCsub_A_roi(Hproc_handle proc_handle)
{
	Hkey in_smallobj_key, in_bigobj_key, out_image_key;
	Himage    insmallimage;
	Himage    inbig_image;
	Hcpar sy, sx, ew, eh;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &sy, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &sx, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &ew, 1);
	HGetSPar(proc_handle, 4, LONG_PAR, &eh, 1);

	HGetObj(proc_handle, 2, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);
	HGetObj(proc_handle, 1, 1, &in_bigobj_key);
	HGetDImage(proc_handle, in_bigobj_key, 1, &inbig_image);

	iRes = sub_A_roi(inbig_image, insmallimage, sx.par.l, sy.par.l, ew.par.l, eh.par.l);

	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;

}
// int CLAHE_image(Himage input_image, Himage output_image, int k_width, int k_height, int clipLimit)
// {
// 	cv::Mat imagein((int)input_image.height, (int)input_image.width, CV_8UC1, input_image.pixel.f);
// 	cv::Mat imageout((int)input_image.height, (int)input_image.width, CV_8UC1, input_image.pixel.f);
// 	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(k_width, k_height));
// 	clahe->apply(imagein, imageout);
// 	return 0;
// }
Herror HCremap(Hproc_handle proc_handle)
{

	const Hcpar* dict;
	INT4_8 num;
	HAllocStringMem(proc_handle, 1024);
	HGetPPar(proc_handle, 1, &dict, &num);

	HTuple hv_DictHandle(const_cast<Hcpar*>(dict), 1);
	HTuple HandleIndex;

	HObject iMAGE;
	HObject dstiMAGE;
	HObject MapXiMAGE;
	HObject MapYiMAGE;

	GetDictObject(&iMAGE, hv_DictHandle, u8"输入图");
	GetDictObject(&dstiMAGE, hv_DictHandle, u8"输出图");
	GetDictObject(&MapXiMAGE, hv_DictHandle, u8"MapX");
	GetDictObject(&MapYiMAGE, hv_DictHandle, u8"MapY");

	HTuple  hv_Pointer, hv_Type, hv_Width, hv_Height;
	GetImagePointer1(iMAGE, &hv_Pointer, &hv_Type, &hv_Width, &hv_Height);

	HTuple  dsthv_Pointer, dsthv_Type, dsthv_Width, dsthv_Height;
	GetImagePointer1(dstiMAGE, &dsthv_Pointer, &dsthv_Type, &dsthv_Width, &dsthv_Height);

	HTuple  Mapxhv_Pointer, Mapxhv_Type, Mapxhv_Width, Mapxhv_Height;
	GetImagePointer1(MapXiMAGE, &Mapxhv_Pointer, &Mapxhv_Type, &Mapxhv_Width, &Mapxhv_Height);

	HTuple  Mapyhv_Pointer, Mapyhv_Type, Mapyhv_Width, Mapyhv_Height;
	GetImagePointer1(MapYiMAGE, &Mapyhv_Pointer, &Mapyhv_Type, &Mapyhv_Width, &Mapyhv_Height);

	cv::Mat srcImage((int)hv_Height.L(), (int)hv_Width.L(), CV_16UC1, (char*)hv_Pointer.L());
	cv::Mat dstImage((int)dsthv_Height.L(), (int)dsthv_Width.L(), CV_16UC1, (char*)dsthv_Pointer.L());

	cv::Mat xMapArra((int)Mapxhv_Height.L(), (int)Mapxhv_Width.L(), CV_32FC1, (char*)Mapxhv_Pointer.L());
	cv::Mat yMapArra((int)Mapyhv_Height.L(), (int)Mapyhv_Width.L(), CV_32FC1, (char*)Mapyhv_Pointer.L());

	cv::remap(srcImage, dstImage, xMapArra, yMapArra, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

	return H_MSG_TRUE;
}


Herror HPNGIn(Hproc_handle proc_handle)
{
	Hcpar  acceleration;

	Hkey      in_smallobj_key, out_obj_key, out_image_key;
	Himage    insmallimage, outimage;
	HGetSPar(proc_handle, 1, LONG_PAR, &acceleration, 1);



	//HAllocStringMem(proc_handle, 1024);
	HGetObj(proc_handle, 1, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);

	//UCHAR BitNum;
	//UINT64 pointAddress;
	cv::Mat img;
	switch (insmallimage.kind)
	{
	case UINT2_IMAGE:
		//BitNum = 2;
		//pointAddress = (INT64)insmallimage.pixel.u.p;
		img=cv::Mat (insmallimage.height, insmallimage.width, CV_MAKETYPE(CV_16U, 1), insmallimage.pixel.u.p);

		break;
	case BYTE_IMAGE:
		//BitNum = 1;
		//pointAddress = (INT64)insmallimage.pixel.b;
		img = cv::Mat(insmallimage.height, insmallimage.width, CV_MAKETYPE(CV_8U, 1), insmallimage.pixel.b);

		break;
	default:
		return 10000;

	}

	std::vector<uchar> png_buf;          // 输出缓冲区
	std::vector<int> params = 
	{
		cv::IMWRITE_PNG_COMPRESSION,(int) acceleration.par.l   // 0-9 可选
	};

	cv::imencode(".png", img, png_buf, params);

	int dstHeight = (png_buf.size()+8 + insmallimage.width - 1) / insmallimage.width;
	size_t cmpBytes = png_buf.size();
	HCkP(HNewImage(proc_handle, &outimage, BYTE_IMAGE, insmallimage.width, dstHeight));	
	memcpy(outimage.pixel.b, &cmpBytes, 8);
	memcpy(outimage.pixel.b + 8, reinterpret_cast<char*>(png_buf.data()), cmpBytes);

	HCrObj(proc_handle, 1, &out_obj_key);
	HPutDImage(proc_handle, out_obj_key, 1, &outimage, FALSE, &out_image_key);//图像输出
	HPutRect(proc_handle, out_obj_key, outimage.width, outimage.height);

	return H_MSG_TRUE;
}
Herror HPNGOut(Hproc_handle proc_handle)
{
	//Hcpar  StrKey;

	Hkey      in_smallobj_key, out_obj_key, out_image_key;
	Himage    insmallimage, outimage;
	//HAllocStringMem(proc_handle, 1024);
	HGetObj(proc_handle, 1, 1, &in_smallobj_key);
	HGetDImage(proc_handle, in_smallobj_key, 1, &insmallimage);

	size_t cmpBytes;
	memcpy(&cmpBytes, insmallimage.pixel.b , 8);
	std::vector<uchar> png_buf;          // 目标容器
	
	png_buf.assign(insmallimage.pixel.b +8, insmallimage.pixel.b +8+ cmpBytes);  // 深拷贝

	cv::Mat decoded = cv::imdecode(png_buf, cv::IMREAD_UNCHANGED);

	INT64 PointAddress;
	/*std::wstring txt = L"Height = " + std::to_wstring(Height);
	MessageBoxW(nullptr, txt.c_str(), L"查看", MB_OK | MB_ICONINFORMATION);*/
	switch (decoded.type())
	{
		case CV_16UC1:
			HCkP(HNewImage(proc_handle, &outimage, UINT2_IMAGE, insmallimage.width, decoded.rows));
			memcpy(outimage.pixel.u.p, decoded.data, insmallimage.width*decoded.rows*2);

			break;
		case CV_8UC1:
			HCkP(HNewImage(proc_handle, &outimage, BYTE_IMAGE, insmallimage.width, decoded.rows));
			memcpy(outimage.pixel.b, decoded.data, insmallimage.width*decoded.rows);

			break;
		default:
			return 10000;
	}

	HCrObj(proc_handle, 1, &out_obj_key);
	HPutDImage(proc_handle, out_obj_key, 1, &outimage, FALSE, &out_image_key);//图像输出
	HPutRect(proc_handle, out_obj_key, outimage.width, outimage.height);
	return H_MSG_TRUE;
}


Herror HCCLAHE_image(Hproc_handle proc_handle)
{
    Hkey      in_obj_key, out_obj_key, out_image_key;
    Himage    inimage;
    Himage    outimage;
    Hcpar     k_width, k_height, clipLimit;
    
    // 获取控制参数 (CLAHE 网格大小和对比度限制)
    HGetSPar(proc_handle, 1, LONG_PAR, &k_width, 1);
    HGetSPar(proc_handle, 2, LONG_PAR, &k_height, 1);
    HGetSPar(proc_handle, 3, LONG_PAR, &clipLimit, 1);
    
    // 获取输入图像
    HGetObj(proc_handle, 1, 1, &in_obj_key);
    HGetDImage(proc_handle, in_obj_key, 1, &inimage);
    
    // 获取输出图像 (预先分配的缓冲区)
    //HGetObj(proc_handle, 2, 1, &out_obj_key);
    //HGetDImage(proc_handle, out_obj_key, 1, &outimage);
    
    // 参数有效性检查
    if (k_width.par.l <= 0 || k_height.par.l <= 0 || clipLimit.par.l < 0)
    {
        return 30001;  // 无效参数错误码
    }
    
    // 检查图像类型 - 只支持 BYTE_IMAGE (8位灰度)
    if (inimage.kind != BYTE_IMAGE)
    {
        return 30002;  // 不支持的图像类型
    }
    

    
    // 创建 OpenCV Mat 头 (零拷贝，直接指向 HALCON 图像数据)
    cv::Mat imageIn(inimage.height, inimage.width, CV_8UC1, inimage.pixel.b);


	HCkP(HNewImage(proc_handle, &outimage, BYTE_IMAGE, inimage.width, inimage.height));
	cv::Mat imageOut(outimage.height, outimage.width, CV_8UC1, outimage.pixel.b);

	
    
    // 创建 CLAHE 对象并应用
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        static_cast<double>(clipLimit.par.l), 
        cv::Size(static_cast<int>(k_width.par.l), static_cast<int>(k_height.par.l))
    );
    
    try
    {
        clahe->apply(imageIn, imageOut);
    }
    catch (const cv::Exception& e)
    {
        // OpenCV 处理异常
        return 30004;  // CLAHE 处理失败
    }
    HCrObj(proc_handle, 1, &out_obj_key);
	HPutDImage(proc_handle, out_obj_key, 1, &outimage, FALSE, &out_image_key);//图像输出
	HPutRect(proc_handle, out_obj_key, outimage.width, outimage.height);


    return H_MSG_TRUE;
}


//using namespace std;

// Windows UTF-8 路径转换：仅 Windows 需要在 wstring 与 string 之间穿梭，
// Linux 文件系统本身就是 UTF-8 字节序，path 在源码中始终保持 std::string。
#if defined(_WIN32) || defined(_WIN64)
string utf8Path(const wstring& wpath)
{
    if (wpath.empty()) return "";
    int size = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
    string result(size - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, &result[0], size, nullptr, nullptr);
    return result;
}

wstring widePath(const string& path)
{
    if (path.empty()) return L"";
    int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    wstring result(size - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &result[0], size);
    return result;
}
#endif

// 将十进制坐标转为 "度/1 分/1 秒/100" 字符串（EXIF Rational 数组格式）
string decimalToExifString(double decimal) 
{
    double abs_val = fabs(decimal);
    int32_t deg = static_cast<int32_t>(abs_val);
    double min_float = (abs_val - deg) * 60.0;
    int32_t min = static_cast<int32_t>(min_float);
    double sec = (min_float - min) * 60.0;
    
    // 秒数保留2位小数精度
    int32_t sec_num = static_cast<int32_t>(sec * 100);
    int32_t sec_den = 100;
    
    // 约分
    auto gcd = [](int32_t a, int32_t b) {
        while (b != 0) { int32_t t = b; b = a % b; a = t; }
        return a;
    };
    int32_t g = gcd(sec_num, sec_den);
    
    return to_string(deg) + "/1 " + to_string(min) + "/1 " + 
           to_string(sec_num/g) + "/" + to_string(sec_den/g);
}

// 辅助：删除已存在的 EXIF 键
void eraseExifKey(Exiv2::ExifData& exif, const char* key) 
{
    auto it = exif.findKey(Exiv2::ExifKey(key));
    if (it != exif.end()) exif.erase(it);
}

// 写入 EXIF 元数据（完整版）
bool writeImageExif(const string& imagePath,
                    double latitude, double longitude, double altitude,
                    double aperture, double shutterSpeed, int iso,
                    double focalLength, const string& dateTime,
                    const string& make, const string& model) 
{
    try {
        // 打开图像
        ExivImagePtr image = Exiv2::ImageFactory::open(imagePath);
        if (!image.get()) {
            cerr << "错误：无法打开图像 " << imagePath << endl;
            return false;
        }

        image->readMetadata();
        Exiv2::ExifData& exif = image->exifData();

        // ========== GPS 信息 ==========
        // 纬度
        eraseExifKey(exif, "Exif.GPSInfo.GPSLatitude");
        exif["Exif.GPSInfo.GPSLatitudeRef"] = (latitude >= 0) ? "N" : "S";
        ExivValuePtr latVal = Exiv2::Value::create(Exiv2::unsignedRational);
        latVal->read(decimalToExifString(latitude));
        exif.add(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude"), latVal.get());

        // 经度
        eraseExifKey(exif, "Exif.GPSInfo.GPSLongitude");
        exif["Exif.GPSInfo.GPSLongitudeRef"] = (longitude >= 0) ? "E" : "W";
        ExivValuePtr lonVal = Exiv2::Value::create(Exiv2::unsignedRational);
        lonVal->read(decimalToExifString(longitude));
        exif.add(Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude"), lonVal.get());

        // 海拔（米）
        eraseExifKey(exif, "Exif.GPSInfo.GPSAltitude");
        exif["Exif.GPSInfo.GPSAltitudeRef"] = static_cast<uint16_t>(0); // 0=海平面以上
        int32_t alt_num = static_cast<int32_t>(fabs(altitude) * 100);
        exif["Exif.GPSInfo.GPSAltitude"] = Exiv2::Rational(alt_num, 100);

        // GPS 版本 2.2.0.0
        exif["Exif.GPSInfo.GPSVersionID"] = static_cast<uint16_t>(2);

        // ========== 相机参数 ==========
        // 光圈 F值 (如 2.8 -> 28/10)
        if (aperture > 0) {
            int32_t fnum = static_cast<int32_t>(aperture * 10);
            exif["Exif.Photo.FNumber"] = Exiv2::Rational(fnum, 10);
            exif["Exif.Photo.ApertureValue"] = Exiv2::Rational(fnum, 10);
        }

        // 曝光时间 (如 1/1000秒)
        if (shutterSpeed > 0) {
            if (shutterSpeed >= 1.0) {
                exif["Exif.Photo.ExposureTime"] = Exiv2::Rational(static_cast<int32_t>(shutterSpeed), 1);
            } else {
                int32_t denom = static_cast<int32_t>(1.0 / shutterSpeed);
                exif["Exif.Photo.ExposureTime"] = Exiv2::Rational(1, denom);
            }
        }

        // ISO
        if (iso > 0) {
            exif["Exif.Photo.ISOSpeedRatings"] = static_cast<uint16_t>(iso);
        }

        // 焦距 (mm)
        if (focalLength > 0) {
            int32_t focal = static_cast<int32_t>(focalLength * 10);
            exif["Exif.Photo.FocalLength"] = Exiv2::Rational(focal, 10);
        }

        // 拍摄时间 (格式: 2026:03:09 14:30:00)
        if (!dateTime.empty()) {
            exif["Exif.Photo.DateTimeOriginal"] = dateTime;
            exif["Exif.Photo.DateTimeDigitized"] = dateTime;
            exif["Exif.Image.DateTime"] = dateTime;
        }

        // 相机厂商和型号
        if (!make.empty()) exif["Exif.Image.Make"] = make;
        if (!model.empty()) exif["Exif.Image.Model"] = model;

        // 写入文件
        image->setExifData(exif);
        image->writeMetadata();

        cout << "成功写入 EXIF: " << imagePath << endl;
        return true;

    } catch (Exiv2::Error& e) {
        cerr << "EXIF 错误: " << e.what() << endl;
        return false;
    } catch (exception& e) {
        cerr << "标准错误: " << e.what() << endl;
        return false;
    }
}



Herror HCWriteImageExif(Hproc_handle proc_handle)
{
    // 分配字符串内存
    HAllocStringMem(proc_handle, 1024);
    
    // 获取输入参数
    char const* const* imagePath;
    double const* latitude;
    double const* longitude;
    double const* altitude;
    double const* aperture;
    double const* shutterSpeed;
    INT4_8 const* iso;
    double const* focalLength;
    char const* const* dateTime;
    char const* const* make;
    char const* const* model;
    INT4_8 num;
    
    // 读取字符串参数
    HGetPElemS(proc_handle, 1, CONV_NONE, &imagePath, &num);
    // 读取 double 参数
    HGetPElemD(proc_handle, 2, CONV_NONE, &latitude, &num);
    HGetPElemD(proc_handle, 3, CONV_NONE, &longitude, &num);
    HGetPElemD(proc_handle, 4, CONV_NONE, &altitude, &num);
    HGetPElemD(proc_handle, 5, CONV_NONE, &aperture, &num);
    HGetPElemD(proc_handle, 6, CONV_NONE, &shutterSpeed, &num);
    // 读取整数参数
    HGetPElemL(proc_handle, 7, CONV_NONE, &iso, &num);
    HGetPElemD(proc_handle, 8, CONV_NONE, &focalLength, &num);
    // 读取字符串参数
    HGetPElemS(proc_handle, 9, CONV_NONE, &dateTime, &num);
    HGetPElemS(proc_handle, 10, CONV_NONE, &make, &num);
    HGetPElemS(proc_handle, 11, CONV_NONE, &model, &num);
    
    // 调用实际的 EXIF 写入函数
    bool result = writeImageExif(
        string(imagePath[0]),
        latitude[0],
        longitude[0],
        altitude[0],
        aperture[0],
        shutterSpeed[0],
        static_cast<int>(iso[0]),
        focalLength[0],
        string(dateTime[0]),
        string(make[0]),
        string(model[0])
    );
    
    if (!result)
    {
        return 30001;  // EXIF 写入失败错误码
    }
    
    return H_MSG_TRUE;
}






/*=============================================================================
 * 算子1: cv_orb_detect
 *
 * 功能: 对输入的 8 位灰度图执行 ORB 特征检测，输出关键点坐标和描述子矩阵。
 *
 * 字典输入:
 *   "InputImage"   — HObject, 8位灰度图 (byte)
 *   "NFeatures"    — HTuple(int), ORB 最大特征点数, 默认 3000
 *
 * 字典输出:
 *   "KeypointsRow" — HTuple(real[]), 关键点行坐标 (y)
 *   "KeypointsCol" — HTuple(real[]), 关键点列坐标 (x)
 *   "Descriptors"  — HObject, 描述子矩阵存为图像 (byte, W=32, H=N)
 *   "NumKeypoints" — HTuple(int), 检测到的关键点数量
 *===========================================================================*/
Herror HCcv_orb_detect(Hproc_handle proc_handle)
{
    const Hcpar *dict;
    INT4_8 num;
    HAllocStringMem(proc_handle, 1024);
    HGetPPar(proc_handle, 1, &dict, &num);
    HTuple hv_DictHandle(const_cast<Hcpar*>(dict), 1);

    // 获取输入图像
    HObject ho_InputImage;
    GetDictObject(&ho_InputImage, hv_DictHandle, "InputImage");

    HTuple hv_Pointer, hv_Type, hv_Width, hv_Height;
    GetImagePointer1(ho_InputImage, &hv_Pointer, &hv_Type, &hv_Width, &hv_Height);

    // 构造 cv::Mat (零拷贝)
    cv::Mat img((int)hv_Height.L(), (int)hv_Width.L(), CV_8UC1,
                (uchar *)hv_Pointer.L());

    // 获取参数 NFeatures
    HTuple hv_NFeatures;
    try {
        GetDictTuple(hv_DictHandle, "NFeatures", &hv_NFeatures);
    } catch (...) {
        hv_NFeatures = 3000;
    }
    int nFeatures = (int)hv_NFeatures.L();

    // 创建 ORB 检测器
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);

    // 检测关键点并计算描述子
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    int nKP = (int)keypoints.size();

    // 输出关键点坐标
    HTuple hv_Rows, hv_Cols;
    for (int i = 0; i < nKP; i++)
    {
        hv_Rows[i] = (double)keypoints[i].pt.y;
        hv_Cols[i] = (double)keypoints[i].pt.x;
    }
    SetDictTuple(hv_DictHandle, "KeypointsRow", hv_Rows);
    SetDictTuple(hv_DictHandle, "KeypointsCol", hv_Cols);
    SetDictTuple(hv_DictHandle, "NumKeypoints", (Hlong)nKP);

    // 输出描述子矩阵为 byte 图像 (H=nKP, W=32)
    if (nKP > 0 && !descriptors.empty())
    {
        // ORB 描述子: CV_8UC1, 每行 32 字节
        HObject ho_Desc;
        GenImage1(&ho_Desc, "byte", 32, nKP, (Hlong)descriptors.data);

        // 由于 GenImage1 会拷贝数据，这里是安全的
        SetDictObject(ho_Desc, hv_DictHandle, "Descriptors");
    }

    return H_MSG_TRUE;
}

/*=============================================================================
 * 算子2: cv_akaze_detect
 *
 * 功能: 对输入的 8 位灰度图执行 AKAZE 特征检测，输出关键点坐标和描述子矩阵。
 *
 * 字典输入:
 *   "InputImage"   — HObject, 8位灰度图 (byte)
 *
 * 字典输出:
 *   "KeypointsRow" — HTuple(real[]), 关键点行坐标 (y)
 *   "KeypointsCol" — HTuple(real[]), 关键点列坐标 (x)
 *   "Descriptors"  — HObject, 描述子矩阵存为图像 (byte, W=61, H=N)
 *   "NumKeypoints" — HTuple(int), 检测到的关键点数量
 *   "DescWidth"    — HTuple(int), 描述子宽度 (AKAZE 可能不固定)
 *===========================================================================*/
Herror HCcv_akaze_detect(Hproc_handle proc_handle)
{
    const Hcpar *dict;
    INT4_8 num;
    HAllocStringMem(proc_handle, 1024);
    HGetPPar(proc_handle, 1, &dict, &num);
    HTuple hv_DictHandle(const_cast<Hcpar*>(dict), 1);

    // 获取输入图像
    HObject ho_InputImage;
    GetDictObject(&ho_InputImage, hv_DictHandle, "InputImage");

    HTuple hv_Pointer, hv_Type, hv_Width, hv_Height;
    GetImagePointer1(ho_InputImage, &hv_Pointer, &hv_Type, &hv_Width, &hv_Height);

    cv::Mat img((int)hv_Height.L(), (int)hv_Width.L(), CV_8UC1,
                (uchar *)hv_Pointer.L());

    // 创建 AKAZE 检测器
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    akaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    int nKP = (int)keypoints.size();

    // 输出关键点坐标
    HTuple hv_Rows, hv_Cols;
    for (int i = 0; i < nKP; i++)
    {
        hv_Rows[i] = (double)keypoints[i].pt.y;
        hv_Cols[i] = (double)keypoints[i].pt.x;
    }
    SetDictTuple(hv_DictHandle, "KeypointsRow", hv_Rows);
    SetDictTuple(hv_DictHandle, "KeypointsCol", hv_Cols);
    SetDictTuple(hv_DictHandle, "NumKeypoints", (Hlong)nKP);

    // AKAZE 描述子: 默认 MLDB 二进制, CV_8UC1
    // 描述子宽度可能是 61 字节 (AKAZE 默认)
    if (nKP > 0 && !descriptors.empty())
    {
        int descWidth = descriptors.cols;

        // 确保是 CV_8U 类型 (二进制描述子)
        cv::Mat descU8;
        if (descriptors.type() != CV_8UC1)
            descriptors.convertTo(descU8, CV_8UC1);
        else
            descU8 = descriptors;

        HObject ho_Desc;
        GenImage1(&ho_Desc, "byte", descWidth, nKP, (Hlong)descU8.data);
        SetDictObject(ho_Desc, hv_DictHandle, "Descriptors");
        SetDictTuple(hv_DictHandle, "DescWidth", (Hlong)descWidth);
    }

    return H_MSG_TRUE;
}

/*=============================================================================
 * 算子3: cv_bf_knn_match
 *
 * 功能: 对两组二进制描述子执行 BF 暴力匹配 + Lowe's Ratio Test 筛选。
 *
 * 字典输入:
 *   "DescriptorsRef"    — HObject, 参考图描述子 (byte 图像, W=descW, H=N1)
 *   "DescriptorsTarget" — HObject, 目标图描述子 (byte 图像, W=descW, H=N2)
 *   "RatioThresh"       — HTuple(real), Lowe's ratio 阈值, 默认 0.75
 *   "DescWidth"         — HTuple(int), 描述子宽度 (ORB=32, AKAZE 可变)
 *
 * 字典输出:
 *   "MatchIdxRef"       — HTuple(int[]), 匹配的参考图关键点索引
 *   "MatchIdxTarget"    — HTuple(int[]), 匹配的目标图关键点索引
 *   "NumGoodMatches"    — HTuple(int), 优质匹配数量
 *===========================================================================*/
Herror HCcv_bf_knn_match(Hproc_handle proc_handle)
{
    const Hcpar *dict;
    INT4_8 num;
    HAllocStringMem(proc_handle, 1024);
    HGetPPar(proc_handle, 1, &dict, &num);
    HTuple hv_DictHandle(const_cast<Hcpar*>(dict), 1);

    // 获取参考图描述子
    HObject ho_DescRef;
    GetDictObject(&ho_DescRef, hv_DictHandle, "DescriptorsRef");
    HTuple ptrRef, typeRef, wRef, hRef;
    GetImagePointer1(ho_DescRef, &ptrRef, &typeRef, &wRef, &hRef);

    // 获取目标图描述子
    HObject ho_DescTarget;
    GetDictObject(&ho_DescTarget, hv_DictHandle, "DescriptorsTarget");
    HTuple ptrTarget, typeTarget, wTarget, hTarget;
    GetImagePointer1(ho_DescTarget, &ptrTarget, &typeTarget, &wTarget, &hTarget);

    // 获取描述子宽度
    HTuple hv_DescWidth;
    try {
        GetDictTuple(hv_DictHandle, "DescWidth", &hv_DescWidth);
    } catch (...) {
        hv_DescWidth = 32; // 默认 ORB
    }
    int descWidth = (int)hv_DescWidth.L();

    // 获取 ratio 阈值
    HTuple hv_RatioThresh;
    try {
        GetDictTuple(hv_DictHandle, "RatioThresh", &hv_RatioThresh);
    } catch (...) {
        hv_RatioThresh = 0.75;
    }
    double ratioThresh = hv_RatioThresh.D();

    int nRef = (int)hRef.L();
    int nTarget = (int)hTarget.L();

    // 构造 cv::Mat
    cv::Mat matRef(nRef, descWidth, CV_8UC1, (uchar *)ptrRef.L());
    cv::Mat matTarget(nTarget, descWidth, CV_8UC1, (uchar *)ptrTarget.L());

    // BF 匹配
    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knnMatches;

    if (nRef < 2 || nTarget < 2)
    {
        SetDictTuple(hv_DictHandle, "NumGoodMatches", (Hlong)0);
        return H_MSG_TRUE;
    }

    try {
        bf.knnMatch(matRef, matTarget, knnMatches, 2);
    } catch (const cv::Exception &) {
        SetDictTuple(hv_DictHandle, "NumGoodMatches", (Hlong)0);
        return H_MSG_TRUE;
    }

    // Lowe's Ratio Test
    std::vector<int> goodIdxRef, goodIdxTarget;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        if (knnMatches[i].size() == 2)
        {
            const cv::DMatch &m = knnMatches[i][0];
            const cv::DMatch &n = knnMatches[i][1];
            if (m.distance < ratioThresh * n.distance)
            {
                goodIdxRef.push_back(m.queryIdx);
                goodIdxTarget.push_back(m.trainIdx);
            }
        }
    }

    int nGood = (int)goodIdxRef.size();

    // 输出匹配索引
    HTuple hv_IdxRef, hv_IdxTarget;
    for (int i = 0; i < nGood; i++)
    {
        hv_IdxRef[i] = (Hlong)goodIdxRef[i];
        hv_IdxTarget[i] = (Hlong)goodIdxTarget[i];
    }
    SetDictTuple(hv_DictHandle, "MatchIdxRef", hv_IdxRef);
    SetDictTuple(hv_DictHandle, "MatchIdxTarget", hv_IdxTarget);
    SetDictTuple(hv_DictHandle, "NumGoodMatches", (Hlong)nGood);

    return H_MSG_TRUE;
}

/*=============================================================================
 * 算子4: cv_estimate_affine_partial2d
 *
 * 功能: 使用 RANSAC 估计部分仿射变换矩阵 (平移+旋转+均匀缩放)。
 *       输入为两组对应点坐标，输出 2x3 仿射矩阵的 6 个元素。
 *
 * 字典输入:
 *   "SrcRow"          — HTuple(real[]), 源点行坐标 (目标图关键点 y)
 *   "SrcCol"          — HTuple(real[]), 源点列坐标 (目标图关键点 x)
 *   "DstRow"          — HTuple(real[]), 目标点行坐标 (参考图关键点 y)
 *   "DstCol"          — HTuple(real[]), 目标点列坐标 (参考图关键点 x)
 *   "RansacThreshold" — HTuple(real), RANSAC 重投影阈值, 默认 3.0
 *
 * 字典输出:
 *   "HomMat2D"        — HTuple(real[6]), 仿射矩阵 [a00,a01,a02, a10,a11,a12]
 *                        可直接用于 Halcon 的 affine_trans_image
 *   "InlierCount"     — HTuple(int), 内点数量
 *   "Success"         — HTuple(int), 1=成功, 0=失败
 *   "TranslateRow"    — HTuple(real), 平移量 ty
 *   "TranslateCol"    — HTuple(real), 平移量 tx
 *   "Angle"           — HTuple(real), 旋转角度 (弧度)
 *   "Scale"           — HTuple(real), 缩放因子
 *===========================================================================*/
Herror HCcv_estimate_affine_partial2d(Hproc_handle proc_handle)
{
    const Hcpar *dict;
    INT4_8 num;
    HAllocStringMem(proc_handle, 1024);
    HGetPPar(proc_handle, 1, &dict, &num);
    HTuple hv_DictHandle(const_cast<Hcpar*>(dict), 1);

    // 获取对应点坐标
    HTuple hv_SrcRow, hv_SrcCol, hv_DstRow, hv_DstCol;
    GetDictTuple(hv_DictHandle, "SrcRow", &hv_SrcRow);
    GetDictTuple(hv_DictHandle, "SrcCol", &hv_SrcCol);
    GetDictTuple(hv_DictHandle, "DstRow", &hv_DstRow);
    GetDictTuple(hv_DictHandle, "DstCol", &hv_DstCol);

    int nPts = (int)hv_SrcRow.Length();
    if (nPts < 4)
    {
        SetDictTuple(hv_DictHandle, "Success", (Hlong)0);
        SetDictTuple(hv_DictHandle, "InlierCount", (Hlong)0);
        return H_MSG_TRUE;
    }

    // 获取 RANSAC 阈值
    HTuple hv_RansacThresh;
    try {
        GetDictTuple(hv_DictHandle, "RansacThreshold", &hv_RansacThresh);
    } catch (...) {
        hv_RansacThresh = 3.0;
    }
    double ransacThresh = hv_RansacThresh.D();

    // 构造 OpenCV 点数组
    // 注意: OpenCV 的 estimateAffinePartial2D 参数是 (from, to)
    // Python 代码中: estimateAffinePartial2D(dst_pts, src_pts) 即从目标到参考
    // 这里 Src = 目标图点, Dst = 参考图点
    std::vector<cv::Point2f> srcPts(nPts), dstPts(nPts);
    for (int i = 0; i < nPts; i++)
    {
        //srcPts[i] = cv::Point2f((float)hv_SrcCol.D()[i], (float)hv_SrcRow.D()[i]);
        //dstPts[i] = cv::Point2f((float)hv_DstCol.D()[i], (float)hv_DstRow.D()[i]);
        srcPts[i] = cv::Point2f((float)hv_SrcCol[i].D(), (float)hv_SrcRow[i].D());
        dstPts[i] = cv::Point2f((float)hv_DstCol[i].D(), (float)hv_DstRow[i].D());

    }

    // RANSAC 估计部分仿射变换
    cv::Mat inlierMask;
    cv::Mat M = cv::estimateAffinePartial2D(
        srcPts, dstPts,
        inlierMask,
        cv::RANSAC,
        ransacThresh
    );

    if (M.empty())
    {
        SetDictTuple(hv_DictHandle, "Success", (Hlong)0);
        SetDictTuple(hv_DictHandle, "InlierCount", (Hlong)0);
        return H_MSG_TRUE;
    }

    // 计算内点数
    int inlierCount = 0;
    if (!inlierMask.empty())
    {
        for (int i = 0; i < inlierMask.rows; i++)
        {
            if (inlierMask.at<uchar>(i, 0) != 0)
                inlierCount++;
        }
    }

    // 提取矩阵元素
    // M = [a00, a01, a02]   即 [cos*s, -sin*s, tx]
    //     [a10, a11, a12]      [sin*s,  cos*s, ty]
    double a00 = M.at<double>(0, 0);
    double a01 = M.at<double>(0, 1);
    double a02 = M.at<double>(0, 2); // tx
    double a10 = M.at<double>(1, 0);
    double a11 = M.at<double>(1, 1);
    double a12 = M.at<double>(1, 2); // ty

    double angle = std::atan2(a10, a00);
    double scale = std::sqrt(a00 * a00 + a10 * a10);

    // 输出 HomMat2D: Halcon 的 hom_mat2d 格式是行优先的 [a00,a01,a02,a10,a11,a12]
    // 但 Halcon affine_trans_image 需要的 HomMat2D 是一个 6 元素 tuple
    // 格式: [ScaleR*cos, -ScaleC*sin, Tx, ScaleR*sin, ScaleC*cos, Ty]
    // 即与 OpenCV 的 2x3 矩阵行优先展开一致
    HTuple hv_HomMat2D;
    //hv_HomMat2D[0] = a00;
    //hv_HomMat2D[1] = a01;
    //hv_HomMat2D[2] = a02;
    //hv_HomMat2D[3] = a10;
    //hv_HomMat2D[4] = a11;
    //hv_HomMat2D[5] = a12;
    // 修改后（正确：坐标系转换）
    hv_HomMat2D[0] = a11;  // h00: row→row
    hv_HomMat2D[1] = a10;  // h01: col→row
    hv_HomMat2D[2] = a12;  // h02: row平移 (ty)
    hv_HomMat2D[3] = a01;  // h10: row→col
    hv_HomMat2D[4] = a00;  // h11: col→col
    hv_HomMat2D[5] = a02;  // h12: col平移 (tx)
    
    SetDictTuple(hv_DictHandle, "HomMat2D", hv_HomMat2D);
    SetDictTuple(hv_DictHandle, "Success", (Hlong)1);
    SetDictTuple(hv_DictHandle, "InlierCount", (Hlong)inlierCount);
    SetDictTuple(hv_DictHandle, "TranslateRow", a12);
    SetDictTuple(hv_DictHandle, "TranslateCol", a02);
    SetDictTuple(hv_DictHandle, "Angle", angle);
    SetDictTuple(hv_DictHandle, "Scale", scale);

    return H_MSG_TRUE;
}
