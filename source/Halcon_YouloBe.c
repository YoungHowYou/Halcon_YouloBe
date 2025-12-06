#include "Halcon_YouloBe.h"

Herror CHInit_YoloSeg(Hproc_handle proc_handle)
{

    return HInit_YoloSeg(proc_handle);
}

Herror CHClass_YoloSeg(Hproc_handle proc_handle)
{

    return HProcess_YoloSeg(proc_handle);
}
