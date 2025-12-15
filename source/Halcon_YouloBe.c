#include "Halcon_YouloBe.h"
Herror OpenvinoLoadModel(Hproc_handle proc_handle)
{

    return Openvino加载模型( proc_handle);

}
Herror OpenvinoInfer(Hproc_handle proc_handle)
{
    return Openvino推理模型( proc_handle);

}
