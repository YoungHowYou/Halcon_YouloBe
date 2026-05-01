#include "Halcon_YouloBe.h"
Herror OpenvinoLoadModel(Hproc_handle proc_handle)
{

    return Openvino加载模型( proc_handle);

}
Herror OpenvinoInfer(Hproc_handle proc_handle)
{
    return Openvino推理模型( proc_handle);

}


 Herror CHCremap(Hproc_handle proc_handle)
 {
	 return HCremap(proc_handle);

 }

Herror CHPNGOut(Hproc_handle proc_handle)
{
	 return HPNGOut(proc_handle);

 }
 
Herror CHPNGIn(Hproc_handle proc_handle)
{
	 return HPNGIn(proc_handle);

 }

Herror Cadd_roi(Hproc_handle proc_handle)
{
	 return HCadd_roi(proc_handle);
}

Herror Cmul_roi(Hproc_handle proc_handle)
{
	 return HCmul_roi(proc_handle);
}

Herror Csub_B_roi(Hproc_handle proc_handle)
{
	 return HCsub_B_roi(proc_handle);
}

Herror Cdiv_B_roi(Hproc_handle proc_handle)
{
	 return HCdiv_B_roi(proc_handle);
}

Herror Cdiv_A_roi(Hproc_handle proc_handle)
{
	 return HCdiv_A_roi(proc_handle);
}

Herror Csub_A_roi(Hproc_handle proc_handle)
{
	 return HCsub_A_roi(proc_handle);
}

Herror CCLAHE_image(Hproc_handle proc_handle)
{
	 return HCCLAHE_image(proc_handle);
}

Herror CWriteImageExif(Hproc_handle proc_handle)
{
	 return HCWriteImageExif(proc_handle);
}
/*******************************************************************************
 * 以下代码需要追加到 source/Halcon_YouloBe.c 文件末尾
 * 这些是 C 转发函数，将 Halcon 调用转发到 C++ 实现
 ******************************************************************************/

Herror Ccv_orb_detect(Hproc_handle proc_handle)
{
    return HCcv_orb_detect(proc_handle);
}

Herror Ccv_akaze_detect(Hproc_handle proc_handle)
{
    return HCcv_akaze_detect(proc_handle);
}

Herror Ccv_bf_knn_match(Hproc_handle proc_handle)
{
    return HCcv_bf_knn_match(proc_handle);
}

Herror Ccv_estimate_affine_partial2d(Hproc_handle proc_handle)
{
    return HCcv_estimate_affine_partial2d(proc_handle);
}
