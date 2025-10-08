/*
 *   @file  diag_sbl.c
 *
 *   @brief
 *    Top Level wrapper function for SBL Diagnostics test function
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2020 Texas Instruments, Inc.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* BIOS/XDC Include Files. */
#include <xdc/std.h>
#include <xdc/cfg/global.h>
#include <xdc/runtime/IHeap.h>
#include <xdc/runtime/Error.h>
#include <xdc/runtime/Memory.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Task.h>
#include <ti/sysbios/knl/Event.h>
#include <ti/sysbios/knl/Clock.h>
#include <ti/sysbios/heaps/HeapBuf.h>
#include <ti/sysbios/heaps/HeapMem.h>
#include <ti/sysbios/knl/Event.h>
#include <ti/sysbios/family/arm/v7a/Pmu.h>

/* MMWSDK include files */
#include <ti/common/sys_common.h>
#include <ti/drivers/dma/dma.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/osal/SemaphoreP.h>
#include <ti/drivers/esm/esm.h>
#include <ti/drivers/osal/CycleprofilerP.h>

/* SBL internal include file. */
#include "ti/utils/diag_sbl/include/sbl_internal.h"
#include "ti/utils/diag_sbl/include/diag_sbl.h"

/* mmWave SDL Include Files. */
#include <ti/csl/csl.h>
#include <ti/diag/diag.h>

#include <ti/sysbios/family/arm/v7a/Pmu.h>
#include <ti/demo/diag_mon_ref/osal/osal.h>


volatile Diag_STC_TestType gSTCTestType = Diag_STC_Test_Normal;

const char * STCTestTypeName[4U] =
{
    "Invalid",
    "Normal",
    "Insert Failure",
    "Force Timeout"
};

/**************************************************************************
 *************************** Local Declarations ***************************
 **************************************************************************/
#define NUM_MSS_PBIST_MEM_GROUP (Diag_PBISTMemGroup_DSS_STC_ROM + 1U)
const char * MSSPBISTMemGroupName[NUM_MSS_PBIST_MEM_GROUP] =
{
    "Invalid",
    "MSS PBIST ROM",
    "MSS STC ROM",
    "SW BUFFER",
#ifndef SOC_XWR68XX
    "DCAN Memory SRAM",
    "DCAN Memory FRAM",
#endif /* SOC_XWR68XX */
    "DMA RAM",
    "MIBSPIA RAM",
    "MCAN Memory",
    "DTHE RAM",
    "Secure RAM",
    "TraceBuffer RAM",
    "MIBSPIB RAM",
    "Mailbox",
    "L3 Bank 0&1",
    "L3 Bank 2&3",
    "L3 Bank 4",
    "L3 Bank 5",
    "L3 Bank 6",
    "L3 Bank 7",
    "DSS ADCBUF and CQ",
#ifndef SOC_XWR16XX
    "DSS FFT",
#endif /* SOC_XWR16XX */
    "DSS TPCC",
    "DSS DATA TXFR RAM & HSRAM",
    "DSS CBUFF FIFO ECC",
    "DSS CBUFF FIFO",
    "DSS L2",
    "DSS PROGFILT",
    "DSS BPM",
    "DSS GEM PBIST ROM",
#ifdef SOC_XWR68XX
    "MSS TCMA ROM",
    "MSS TCMA ECC ROM",
#endif /* SOC_XWR68XX */
    "DSS STC ROM"
};


/**
 *  @b Description
 *  @n
 *      This is the implementation of the R4F LBIST STC diagnostic.
 *  @Input : None
 *  @retval : none
 *
 */
int32_t SBL_diag_stc(void)
{
    int32_t             retVal;
    retVal = Diag_R4F_STC_execute (gSTCTestType);
    if(retVal == DIAG_SUCCESS)
    {
        SBL_printf("Diag R4F STC [Test type: %s] Passed\r\n", STCTestTypeName[gSTCTestType]);
    }
    else
	{
        SBL_printf("Diag R4F STC [Test type: %s] Failed\r\n", STCTestTypeName[gSTCTestType]);			
	}
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the R4F PBIST Self test diagnostic.
 *  @Input : None
 *  @retval : none
 *
 */
int32_t SBL_diag_pbist(void)
{
    Diag_R4F_PBIST_Cfg      cfg;
    int32_t                 retVal;
    uint32_t                cycleCount;
    Diag_PBISTMemGroup      memGroup;
    CSL_TopRCMRegs*     	ptrTopRCMRegs;	
    uint8_t                 testStatus = 1U;
	
    /* Get the base address of the TopRCM Module */
    ptrTopRCMRegs = CSL_TopRCM_getBaseAddress();

    /* Ungate the PBIST pre clock divider */
    CSL_TopRCM_setPBISTGateStatus(ptrTopRCMRegs, 0U);	

    /* Loop through each MSS PBIST Memory group */
    for (memGroup = Diag_PBISTMemGroup_MSS_PBIST_ROM; memGroup <= Diag_PBISTMemGroup_DSS_STC_ROM; memGroup++)
    {
        /***************************************************************************
         * Diagnostic: PBIST memory group self test
         ***************************************************************************/
        (void)memset((void *)&cfg, 0, sizeof(Diag_R4F_PBIST_Cfg));

        /* Populate the configuration. */
        cfg.injectFault     = 0U;
        cfg.memoryGroup     = memGroup;
        cfg.maxCycleCount   = PBIST_MAX_CYCLE_COUNT;

        /* Execute the diagnostic. */
        retVal = Diag_R4F_PBIST_execute(&cfg, &cycleCount);

		if(retVal == DIAG_SUCCESS)
		{
			SBL_printf("Diag R4F PBIST [%s] Passed\r\n", MSSPBISTMemGroupName[memGroup]);
		}
		else
		{
			SBL_printf("Diag R4F PBIST [%s] Failed\r\n", MSSPBISTMemGroupName[memGroup]);
			testStatus = 0U; 	
		}								  

    }

    /***************************************************************************
     * Diagnostic: PBIST Fault Inject
     ***************************************************************************/
    (void)memset((void *)&cfg, 0, sizeof(Diag_R4F_PBIST_Cfg));

    /* Populate the configuration. */
    cfg.injectFault     = 1U;
    cfg.maxCycleCount   = PBIST_MAX_CYCLE_COUNT;

    /* Execute the diagnostic. */
    retVal = Diag_R4F_PBIST_execute(&cfg, &cycleCount);

    /* STF Test Result Reporting. */
	if(retVal == DIAG_SUCCESS)
	{
		SBL_printf("Diag R4F PBIST [Fault injection] Passed\r\n");
	}
	else
	{
		SBL_printf("Diag R4F PBIST [Fault injection] Failed\r\n");							  
	}


    if (retVal != DIAG_SUCCESS)
    {
        SBL_printf("Error: Diag R4F PBIST [Fault injection] Error Code[%d]\r\n", retVal);

        /* Test failed. */
        testStatus = 0U;
    }

    /* STF Test Reporting. */
	if (testStatus != 0)
	{
		SBL_printf("Diag R4F PBIST [All tests] Passed\r\n");
	}
	else{
		SBL_printf("Diag R4F PBIST [All tests] Failed\r\n");							  
	}
    return retVal;
}

uint32_t OSAL_R4F_CycleProfiler_getCount (void)
{
    return(0); //TEMP

}
#if 0
#define MAX_HWI_HANDLE     4
typedef struct hwiHandleMap_t
{
    HwiP_Handle hwiHandle;
    uint8_t     irq;
}hwiHandleMap;


hwiHandleMap gHwiHandleMap[MAX_HWI_HANDLE] = {0};

void OSAL_R4F_Interrupt_addHook (uint8_t irq, uint8_t fiqType, OSAL_HookFxn hookFxn)
{
    HwiP_Params     hwiParams;
    uint8_t idx;

    /* find the empty Hwi Handle map */
    for(idx=0; idx<MAX_HWI_HANDLE; idx++)
    {
        if((gHwiHandleMap[idx].hwiHandle == NULL) && (gHwiHandleMap[idx].irq == 0))
            break;
    }

    HwiP_Params_init(&hwiParams);
    hwiParams.name = "SDL_IRQ_HOOK";
    /* HwiP_Type_IRQ or HwiP_Type_FIQ; */
    hwiParams.type = (HwiP_Type)fiqType;
    gHwiHandleMap[idx].irq = irq;
    gHwiHandleMap[idx].hwiHandle = HwiP_create(irq, (HwiP_Fxn)hookFxn, &hwiParams);

}

void OSAL_R4F_Interrupt_delHook (uint8_t irq)
{
    uint8_t idx;

    /* find the empty Hwi Handle map */
    for(idx=0; idx < MAX_HWI_HANDLE; idx++)
    {
        if(gHwiHandleMap[idx].irq == irq)
            break;
    }

    HwiP_delete(((HwiP_Handle)gHwiHandleMap[idx].hwiHandle));

    gHwiHandleMap[idx].hwiHandle = NULL;
    gHwiHandleMap[idx].irq       = 0;
}
#endif
