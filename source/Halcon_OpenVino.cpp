#include "Halcon_Def.h"
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

    Hcpar *容器;
    INT4_8 参数个数; // 参数个数
    HGetPPar(proc_handle, 1, &容器, &参数个数);
    HTuple hv_Dict(容器, 1);
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

    Hcpar *容器;
    INT4_8 参数个数; // 参数个数
    Def_INOpenvinoObject(1, handle_data);
    HGetPPar(proc_handle, 2, &容器, &参数个数);
    HTuple hv_Dict(容器, 1);

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

	Hcpar* dict;
	INT4_8 num;
	HAllocStringMem(proc_handle, 1024);
	HGetPPar(proc_handle, 1, &dict, &num);

	HTuple hv_DictHandle(dict, 1);
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

// Windows UTF-8 路径转换
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
        Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(imagePath);
        if (!image) {
            cerr << "错误：无法打开图像 " << imagePath << endl;
            return false;
        }

        image->readMetadata();
        Exiv2::ExifData& exif = image->exifData();

        // ========== GPS 信息 ==========
        // 纬度
        eraseExifKey(exif, "Exif.GPSInfo.GPSLatitude");
        exif["Exif.GPSInfo.GPSLatitudeRef"] = (latitude >= 0) ? "N" : "S";
        Exiv2::Value::UniquePtr latVal = Exiv2::Value::create(Exiv2::unsignedRational);
        latVal->read(decimalToExifString(latitude));
        exif.add(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude"), latVal.get());

        // 经度
        eraseExifKey(exif, "Exif.GPSInfo.GPSLongitude");
        exif["Exif.GPSInfo.GPSLongitudeRef"] = (longitude >= 0) ? "E" : "W";
        Exiv2::Value::UniquePtr lonVal = Exiv2::Value::create(Exiv2::unsignedRational);
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
