/**
 *   @file  dss_main.c
 *
 *   @brief
 *      This is the main file which implements the millimeter wave Demo
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
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/esm/esm.h>
#include <ti/utils/cycleprofiler/cycle_profiler.h>

/* Demo Include Files */
#include "common/diag_mon_output.h"
#include "osal/osal.h"
#include "dss/diag_mon_dss.h"
#include "osal/DebugP.h"
#include "diag/diag_test_api.h"
#include "diag/diag_error_code.h"

/**
 * @brief Task Priority settings:
 */
#define MMWDEMO_DPC_OBJDET_DPM_TASK_PRIORITY      5

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
MmwDemo_DSS_MCB    gMmwDssMCB = {0};

/**
 *  @b Description
 *  @n
 *     Function to sleep the DSP using IDLE instruction. When DSP has no work left to do,
 *     the BIOS will be in Idle thread and will call this function. The DSP will
 *     wake-up on any interrupt (e.g chirp interrupt).
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_sleep(void)
{
    /* issue IDLE instruction */
    asm(" IDLE ");
}



void MmwDemo_nonOsLoop(void)
{
    while(1)
    {
        MmwDemo_sleep();
    }
}

/**
 *  @b Description
 *  @n
 *     Disable/Enable L1P, L1D, L2 Cache. It requires to disable these cache
 *     before DIAG Tests, which can be later enabled while main app execution.
 *
 *  @Input: enDis  1: enable, 0:disable cache
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_disEnCache(char enDis)
{
    /* L1P/L1D Cache Size */
    enum Dss_Cache_L1Cache_size {
        Dss_Cache_L1Size_0K = 0,
        Dss_Cache_L1Size_4K = 1,
        Dss_Cache_L1Size_8K = 2,
        Dss_Cache_L1Size_16K = 3,
        Dss_Cache_L1Size_32K = 4
    };
    /* L2Size */
    enum Dss_Cache_L2Size {
        Dss_Cache_L2Size_0K = 0,
        Dss_Cache_L2Size_32K = 1,
        Dss_Cache_L2Size_64K = 2,
        Dss_Cache_L2Size_128K = 3
    };

    CSL_DSPICFGRegs*    ptrDSPICFGRegs;
    /* Get the base address of DSP_ICFG module*/
    ptrDSPICFGRegs = CSL_DSPICFG_getBaseAddress();

    if(enDis)
    {
        /* set the cache as required for the application */

        /* Eanble L1P cache */
        CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, Dss_Cache_L1Size_32K);

        /* Enable L1D cache */
        CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, Dss_Cache_L1Size_32K);

        /* Enable L2 cache */
        CSL_DSPICFG_L2Cfg_setMode(ptrDSPICFGRegs, Dss_Cache_L2Size_32K);
    }
    else
    {
        /* Disable L1P cache */
        CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, Dss_Cache_L1Size_0K);

        /* Disable L1D cache */
        CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, Dss_Cache_L1Size_0K);

        /* Disable L2 cache */
        CSL_DSPICFG_L2Cfg_setMode(ptrDSPICFGRegs, Dss_Cache_L2Size_0K);
    }
}

/**
 *  @b Description
 *  @n
 *      System Initialization Task which initializes the various
 *      components in the system.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_dssInitTask(unsigned int arg0, unsigned int arg1)
{
    int32_t             errCode, retVal;
    extern uint32_t gDssDiagTestStatus;
    extern uint32_t gDssDiagTestExec;
#ifdef SOC_XWR68XX
    MmwDemo_dssDiagTestMsg *dssDiagTestStat = (MmwDemo_dssDiagTestMsg*)SOC_XWR68XX_DSS_HSRAM_BASE_ADDRESS;
#else
    MmwDemo_dssDiagTestMsg *dssDiagTestStat = (MmwDemo_dssDiagTestMsg*)SOC_XWR18XX_DSS_HSRAM_BASE_ADDRESS;
#endif
    /* Run all the Diagnostic Test on DSS */
    retVal = DssDiag_InjectTest();

    /* Enable the cache after diagnostic Test execution */
    MmwDemo_disEnCache(1);

    /* Store the Diagnostic Test status on HSRAM which is shared with MSS over SW Interrupt */
    dssDiagTestStat->diagTestBitStat = gDssDiagTestStatus;
    dssDiagTestStat->diagTestExecBits = gDssDiagTestExec;
    dssDiagTestStat->errVal = retVal;

    /* Set the operational status for the DSS
     * This will notify the MSS core that DSS application is up. */
    SOC_setMMWaveDSSLinkState(gMmwDssMCB.socHandle, 1, &errCode);

    /******************************************************************************
     * Synchronization
     * - The synchronization API always needs to be invoked.
     ******************************************************************************/
    while (1)
    {
        int32_t syncStatus;
        /* Get the synchronization status from MSS */
        syncStatus = SOC_isMMWaveMSSOperational (gMmwDssMCB.socHandle, &errCode);
        if (syncStatus < 0)
        {
            /* Error: Unable to synchronize the mmWave control module */
            printf ("Error: MMWDemoDSS mmWave Control Synchronization failed [Error code %d]\n", errCode);
            return;
        }
        if (syncStatus == 1)
        {
            /* Synchronization achieved: */
            break;
        }
        /* Sleep and poll again: */
        //MmwDemo_sleep();
    }

    /* Generate SW interrupt to MSS, so it can read the HSRAM for DIAG Test Status */
    SOC_triggerDSStoMSSsoftwareInterrupt(gMmwDssMCB.socHandle, 1, &errCode);

    MmwDemo_nonOsLoop();

    /*************************************************************************************
     * Here onward DSS application can setup other logic to do the signal processing part.
     * ***********************************************************************************/

    return;
}

/**
 *  @b Description
 *  @n
 *      Entry point into the Millimeter Wave Demo
 *
 *  @retval
 *      Not Applicable.
 */
int main (void)
{
    int32_t         errCode;
    SOC_Handle      socHandle;
    SOC_Cfg         socCfg;

    /* Disable the cache before diagnostic Test execution */
    MmwDemo_disEnCache(0);

    /* Initialize the SOC configuration: */
    memset ((void *)&socCfg, 0, sizeof(SOC_Cfg));

    /* Initialize the OSAL Cycle Profiling Module: */
    OSAL_C674_CycleProfiler_init ();

    /* Populate the SOC configuration: */
    socCfg.clockCfg = SOC_SysClock_BYPASS_INIT;

    /* Initialize the SOC Module: This is done as soon as the application is started
     * to ensure that the MPU is correctly configured. */
    socHandle = SOC_init (&socCfg, &errCode);
    if (socHandle == NULL)
    {
        return -1;
    }

    gMmwDssMCB.socHandle = socHandle;

    /* Initialize ESM driver, which is being used mainly for DIAG tests */
    OSAL_C674_ESMDrv_init();

    MmwDemo_dssInitTask(0, 0);

    return 0;
}

