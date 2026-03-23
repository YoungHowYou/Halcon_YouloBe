#pragma once
#ifndef __APPLE__
#include "Halcon.h"
#else
#ifndef HC_LARGE_IMAGES
#include <HALCON/Halcon.h>
#else
#include <HALCONxl/Halcon.h>
#endif
#endif
#define Test_EXPORTS_API __declspec(dllexport)

#ifdef __cplusplus
extern "C"
{
#endif

    Test_EXPORTS_API Herror Openvino加载模型(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror Openvino推理模型(Hproc_handle proc_handle);
    
    Test_EXPORTS_API Herror HCremap(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HPNGOut(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HPNGIn(Hproc_handle proc_handle);

    Test_EXPORTS_API Herror HCadd_roi(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCmul_roi(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCsub_B_roi(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCdiv_B_roi(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCdiv_A_roi(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCsub_A_roi(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCCLAHE_image(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCWriteImageExif(Hproc_handle proc_handle);
/*******************************************************************************
 * 以下声明需要追加到 include/Halcon_YouloBe.h 中的 extern "C" { } 块内
 ******************************************************************************/

    Test_EXPORTS_API Herror HCcv_orb_detect(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCcv_akaze_detect(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCcv_bf_knn_match(Hproc_handle proc_handle);
    Test_EXPORTS_API Herror HCcv_estimate_affine_partial2d(Hproc_handle proc_handle);

#ifdef __cplusplus
}
#endif
