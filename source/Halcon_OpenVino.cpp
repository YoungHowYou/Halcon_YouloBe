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



	HAllocStringMem(proc_handle, 1024);
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

int CLAHE_image(Himage input_image, Himage output_image, int k_width, int k_height, int clipLimit)
{
	cv::Mat imagein((int)input_image.height, (int)input_image.width, CV_8UC1, input_image.pixel.f);
	cv::Mat imageout((int)input_image.height, (int)input_image.width, CV_8UC1, input_image.pixel.f);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(k_width, k_height));
	clahe->apply(imagein, imageout);
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

Herror HCCLAHE_image(Hproc_handle proc_handle)
{
	Hkey in_inobj_key, in_outobj_key;
	Himage    inimage;
	Himage    outimage;
	Hcpar k_width, k_height, clipLimit;
	INT4_8 iRes;
	HGetSPar(proc_handle, 1, LONG_PAR, &k_width, 1);
	HGetSPar(proc_handle, 2, LONG_PAR, &k_height, 1);
	HGetSPar(proc_handle, 3, LONG_PAR, &clipLimit, 1);
	HGetObj(proc_handle, 1, 1, &in_inobj_key);
	HGetDImage(proc_handle, in_inobj_key, 1, &inimage);
	HGetObj(proc_handle, 2, 1, &in_outobj_key);
	HGetDImage(proc_handle, in_outobj_key, 1, &outimage);

	iRes = CLAHE_image(inimage, outimage, k_width.par.l, k_height.par.l, clipLimit.par.l);

	if (0 != iRes)
	{
		return 30000 + iRes;
	}
	return H_MSG_TRUE;
}