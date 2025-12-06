#pragma once
#ifndef __APPLE__
#  include "Halcon.h"
#else
#  ifndef HC_LARGE_IMAGES
#    include <HALCON/Halcon.h>
#  else
#    include <HALCONxl/Halcon.h>
#  endif
#endif
#define Test_EXPORTS_API __declspec(dllexport)

#ifdef __cplusplus
extern "C" {
#endif

	extern Test_EXPORTS_API Herror HInit_YoloSeg(Hproc_handle proc_handle);
	extern Test_EXPORTS_API Herror HProcess_YoloSeg(Hproc_handle proc_handle);




#ifdef __cplusplus
}
#endif 
