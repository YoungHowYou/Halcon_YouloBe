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

#ifdef __cplusplus
}
#endif
