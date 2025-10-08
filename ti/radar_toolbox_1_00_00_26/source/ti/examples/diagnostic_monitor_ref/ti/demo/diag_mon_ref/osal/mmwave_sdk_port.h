/* This file is to make mmwave SDK ported with mmWave SDL interfaces */

#ifndef __MMWAVE_SDK_PORT_H__
#define __MMWAVE_SDK_PORT_H__

#ifdef OS_TI_RTOS
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/family/arm/exc/Exception.h>
#include <ti/sysbios/hal/hwi.h>
#include <ti/sysbios/knl/swi.h>
#include <ti/sysbios/knl/Task.h>
#include <xdc/runtime/Memory.h>
#include <xdc/runtime/Error.h>
#include <xdc/runtime/Types.h>
#include <xdc/runtime/System.h>
#include <ti/sysbios/heaps/HeapMem.h>
//typedef void (*Exception_ExceptionHookFuncPtr)(Exception_ExcContext*);

/* 1. Cycle Profiler from mmWave SDK to SDL */
#if defined (SUBSYS_DSS)
#include <c6x.h>
Cycleprofiler_init(void)
{
    TSCL = 0;
}
#define Cycleprofiler_getTimeStamp() TSCL
#endif

#ifdef SUBSYS_MSS
/* mmwave SDK driver includes */
#include "osal/osal.h"
#include <ti/drivers/esm/esm.h>
#include "osal/HwiP.h"
#define Cycleprofiler_init()         OSAL_R4F_CycleProfiler_init(0)
#define Cycleprofiler_getTimeStamp() OSAL_R4F_CycleProfiler_getCount()



#endif

#endif
#endif //end __MMWAVE_SDK_PORT_H__

