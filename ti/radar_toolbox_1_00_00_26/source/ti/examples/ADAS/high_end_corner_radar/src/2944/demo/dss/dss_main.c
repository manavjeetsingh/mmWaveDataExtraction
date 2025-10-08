/**
 *   @file  dss_main.c
 *
 *   @brief
 *      This is the main file which implements the millimeter wave Demo
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2021 Texas Instruments, Inc.
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

/* MCU PLUS SDK Include Files */
#include "dssgenerated/ti_drivers_config.h"
#include "dssgenerated/ti_board_config.h"
#include "dssgenerated/ti_drivers_open_close.h"
#include "dssgenerated/ti_board_open_close.h"
#include <kernel/dpl/AddrTranslateP.h>
#include <kernel/dpl/DebugP.h>
#include "FreeRTOS.h"
#include "task.h"

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
#include <ti/control/dpm/dpm.h>
#include "../../datapath/dpc/objectdetection/objdethwaDDMA/objectdetection.h"
#include <ti/utils/mathutils/mathutils.h>

/* Demo Include Files */
#include "mmw_dss.h"
#include <kernel/dpl/CycleCounterP.h>
#include <kernel/dpl/TaskP.h>

/*! L3 RAM buffer for object detection DPC */
// uint8_t gMmwL3[(CSL_DSS_L3_U_SIZE - SYS_COMMON_HSRAM_SIZE)] __attribute__((aligned(4096)));
uint8_t gMmwL3[(0x240000)- SYS_COMMON_HSRAM_SIZE] __attribute__((aligned(4096)));

/* EDMA 4K silicon bug related : Align heap to 4K address boundary so that
 * non heap related changes (such as program code) does not alter the 4K related
 * behavior */
#pragma DATA_ALIGN(gMmwL3, 4096U);
#pragma DATA_SECTION(gMmwL3, ".l3ram");

 /*! L2 RAM buffer for object detection DPC */
#define MMWDEMO_OBJDET_L2RAM_SIZE (95 * 1024U)
uint8_t gDPC_ObjDetL2Heap[MMWDEMO_OBJDET_L2RAM_SIZE];

/* EDMA 4K silicon bug related : Align heap to 4K address boundary so that
 * non heap related changes (such as program code) does not alter the 4K related
 * behavior */
#pragma DATA_ALIGN(gDPC_ObjDetL2Heap, 4096U);
#pragma DATA_SECTION(gDPC_ObjDetL2Heap, ".dpc_l2Heap");

 /*! HSRAM for processing results */
#pragma DATA_SECTION(gHSRAM, ".demoSharedMem");
#pragma DATA_ALIGN(gHSRAM, 4);

/* Task declarations */
#define MMWDEMO_DSS_INIT_TASK_PRI         (1U)
#define MMWDEMO_DSS_INIT_TASK_STACK_SIZE  (1 * 1024u)
#define MMWDEMO_DPM_TASK_PRI              (5U)
#define MMWDEMO_DPM_TASK_STACK_SIZE       (8 * 1024u)

/* Task Stack variables.*/
StackType_t gMmwDemo_dssInitTaskStack[MMWDEMO_DSS_INIT_TASK_STACK_SIZE] __attribute__((aligned(32)));
StackType_t gMmwDemo_dpmTaskStack[MMWDEMO_DPM_TASK_STACK_SIZE] __attribute__((aligned(32)));

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
MmwDemo_DSS_MCB    gMmwDssMCB;

/**
 * @brief
 *  Global Variable for DPM result buffer
 */
DPM_Buffer  resultBuffer;

/**
 * @brief
 *  Global Variable for HSRAM buffer used to share results to remote
 */
MmwDemo_HSRAM gHSRAM;

/**************************************************************************
 ******************* Millimeter Wave Demo Functions Prototype *******************
 **************************************************************************/
static void MmwDemo_dssInitTask(void* args);
static void MmwDemo_DPC_ObjectDetection_reportFxn
(
    DPM_Report  reportType,
    uint32_t    instanceId,
    int32_t     errCode,
    uint32_t    arg0,
    uint32_t    arg1
);
static void MmwDemo_DPC_ObjectDetection_processFrameBeginCallBackFxn(uint8_t subFrameIndx);
static void MmwDemo_DPC_ObjectDetection_processInterFrameBeginCallBackFxn(uint8_t subFrameIndx);
static void MmwDemo_updateObjectDetStats
(
    DPC_ObjectDetection_Stats       *currDpcStats,
    MmwDemo_output_message_stats    *outputMsgStats
);

static int32_t MmwDemo_copyResultToHSRAM
(
    MmwDemo_HSRAM           *ptrHsramBuffer,
    DPC_ObjectDetection_ExecuteResult *result,
    MmwDemo_output_message_stats *outStats
);
static void MmwDemo_DPC_ObjectDetection_dpmTask(void* args);
static void MmwDemo_sensorStopEpilog(void);

/**************************************************************************
 ************************* Millimeter Wave Demo Functions **********************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      Epilog processing after sensor has stopped
 *
 *  @retval None
 */
static void MmwDemo_sensorStopEpilog(void)
{
    
    // Hwi_StackInfo   stackInfo;
    // Task_Stat       stat;
    // bool            hwiStackOverflow;

    test_print("Data Path Stopped (last frame processing done)\n");

}

/**
 *  @b Description
 *  @n
 *      DPM Registered Report Handler. The DPM Module uses this registered function to notify
 *      the application about DPM reports.
 *
 *  @param[in]  reportType
 *      Report Type
 *  @param[in]  instanceId
 *      Instance Identifier which generated the report
 *  @param[in]  errCode
 *      Error code if any.
 *  @param[in] arg0
 *      Argument 0 interpreted with the report type
 *  @param[in] arg1
 *      Argument 1 interpreted with the report type
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_DPC_ObjectDetection_reportFxn
(
    DPM_Report  reportType,
    uint32_t    instanceId,
    int32_t     errCode,
    uint32_t    arg0,
    uint32_t    arg1
)
{

    /* Only errors are logged on the console: */
    if (errCode != 0)
    {
        /* Error: Detected log on the console and die all errors are FATAL currently. */
        test_print ("Error: DPM Report %d received with error:%d arg0:0x%x arg1:0x%x\n",
                        reportType, errCode, arg0, arg1);
        DebugP_assert (0);
    }

    /* Processing further is based on the reports received: This is the control of the profile
     * state machine: */
    switch (reportType)
    {
        case DPM_Report_IOCTL:
        {
            /*****************************************************************
             * DPC has been configured without an error:
             * - This is an indication that the profile configuration commands
             *   went through without any issues.
             *****************************************************************/
            DebugP_logInfo("DSSApp: DPM Report IOCTL, command = %d\n", arg0);
            break;
        }
        case DPM_Report_DPC_STARTED:
        {
            /*****************************************************************
             * DPC has been started without an error:
             * - notify sensor management task that DPC is started.
             *****************************************************************/
            DebugP_logInfo("DSSApp: DPM Report start\n");
            break;
        }
        case DPM_Report_NOTIFY_DPC_RESULT:
        {
            /*****************************************************************
             * DPC Results have been passed:
             * - This implies that we have valid profile results which have
             *   been received from the profile.
             *****************************************************************/

            break;
        }
        case DPM_Report_NOTIFY_DPC_RESULT_ACKED:
        {
            /*****************************************************************
             * DPC Results have been acked:
             * - This implies that MSS received the results.
             *****************************************************************/

            break;
        }
        case DPM_Report_DPC_ASSERT:
        {
            DPM_DPCAssert*  ptrAssert;

            /*****************************************************************
             * DPC Fault has been detected:
             * - This implies that the DPC has crashed.
             * - The argument0 points to the DPC assertion information
             *****************************************************************/
            ptrAssert = (DPM_DPCAssert*)arg0;
            test_print ("DSS Exception: %s, line %d.\n", ptrAssert->fileName,
                       ptrAssert->lineNum);
            break;
        }
        case DPM_Report_DPC_STOPPED:
        {
            /*****************************************************************
             * DPC has been stopped without an error:
             * - This implies that the DPC can either be reconfigured or
             *   restarted.
             *****************************************************************/
            DebugP_logInfo("DSSApp: DPM Report stop\n");

            MmwDemo_sensorStopEpilog();
            break;
        }
        case DPM_Report_DPC_INFO:
        {
            /* Currently objDetHwa does not use this feature. */
            break;
        }
        default:
        {
            DebugP_assert (0);
            break;
        }
    }
    return;
}

/**
 *  @b Description
 *  @n
 *      Call back function that was registered during config time and is going
 *      to be called in DPC processing at the beginning of frame/sub-frame processing,
 *      we use this to issue BIOS calls for computing CPU load during inter-frame
 *
 *  @param[in] subFrameIndx Sub-frame index of the sub-frame during which processing
 *             this function was called.
 *
 *  @retval None
 */
static void MmwDemo_DPC_ObjectDetection_processFrameBeginCallBackFxn(uint8_t subFrameIndx)
{
    TaskP_loadUpdateAll();
    gMmwDssMCB.dataPathObj.subFrameStats[subFrameIndx].interFrameCPULoad = TaskP_loadGetTotalCpuLoad();
}

/**
 *  @b Description
 *  @n
 *      Call back function that was registered during config time and is going
 *      to be called in DPC processing at the beginning of
 *      inter-frame/inter-sub-frame processing,
 *      we use this to issue BIOS calls for computing CPU load during active frame
 *      (chirping)
 *
 *  @param[in] subFrameIndx Sub-frame index of the sub-frame during which processing
 *             this function was called.
 *
 *  @retval None
 */
static void MmwDemo_DPC_ObjectDetection_processInterFrameBeginCallBackFxn(uint8_t subFrameIndx)
{
    TaskP_loadUpdateAll();
    gMmwDssMCB.dataPathObj.subFrameStats[subFrameIndx].activeFrameCPULoad = TaskP_loadGetTotalCpuLoad();
}


/**
 *  @b Description
 *  @n
 *      Update stats based on the stats from DPC
 *
 *  @param[in]  currDpcStats        Pointer to DPC status
 *  @param[in]  outputMsgStats      Pointer to Output message stats 
 *
 *  @retval
 *      Not Applicable.
 */
 void MmwDemo_updateObjectDetStats
(
    DPC_ObjectDetection_Stats       *currDpcStats,
    MmwDemo_output_message_stats    *outputMsgStats
)
{
    static uint32_t prevInterFrameEndTimeStamp = 0U;

    /* Calculate interframe proc time */
    outputMsgStats->interFrameProcessingTime =
            (currDpcStats->interFrameEndTimeStamp - currDpcStats->interFrameStartTimeStamp)/DSP_CLOCK_MHZ; /* In micro seconds */

    outputMsgStats->interChirpProcessingMargin = currDpcStats->interChirpProcessingMargin/DSP_CLOCK_MHZ;

    /* Calculate interFrame processing Margin for previous frame, but saved to current frame */
    outputMsgStats->interFrameProcessingMargin =
        (currDpcStats->frameStartTimeStamp - prevInterFrameEndTimeStamp - currDpcStats->subFramePreparationCycles)/DSP_CLOCK_MHZ;

    prevInterFrameEndTimeStamp = currDpcStats->interFrameEndTimeStamp;
}

/**
 *  @b Description
 *  @n
 *      Copy DPC results and output stats to HSRAM to share with MSS
 *
 *  @param[in]  ptrHsramBuffer      Pointer to HSRAM buffer memory
 *  @param[in]  result              Pointer to DPC results
 *  @param[in]  outStats            Pointer to Output message stats
 *
 *  @retval
 *      Not Applicable.
 */
static int32_t MmwDemo_copyResultToHSRAM
(
    MmwDemo_HSRAM           *ptrHsramBuffer,
    DPC_ObjectDetection_ExecuteResult *result,
    MmwDemo_output_message_stats *outStats
)
{
    uint8_t             *ptrCurrBuffer;
    uint32_t            totalHsramSize;
    uint32_t            itemPayloadLen;

    /* Save result in HSRAM */
    if(ptrHsramBuffer == NULL)
    {
        return -1;
    }

    /* Save result in HSRAM */
    if(result != NULL)
    {
        itemPayloadLen = sizeof(DPC_ObjectDetection_ExecuteResult);
        memcpy((void *)&ptrHsramBuffer->result, (void *)result, itemPayloadLen);
    }
    else
    {
        return -1;
    }

    /* Save output Stats in HSRAM */
    if(outStats != NULL)
    {
        itemPayloadLen = sizeof(MmwDemo_output_message_stats);
        memcpy((void *)&ptrHsramBuffer->outStats, (void *)outStats, itemPayloadLen);
    }

    /* Set payload pointer to HSM buffer */
    ptrCurrBuffer = &ptrHsramBuffer->payload[0];
    totalHsramSize = MMWDEMO_HSRAM_PAYLOAD_SIZE;

    /* Save ObjOut in HSRAM */
    if(result->objOut != NULL)
    {
        itemPayloadLen = sizeof(DPIF_PointCloudCartesian) * result->numObjOut; //TODO
        if((totalHsramSize- itemPayloadLen) > 0)
        {
            memcpy(ptrCurrBuffer, (void *)result->objOut, itemPayloadLen);

            ptrHsramBuffer->result.objOut = (DPIF_PointCloudCartesian *)ptrCurrBuffer; //TODO
            ptrCurrBuffer+= itemPayloadLen;
            totalHsramSize -=itemPayloadLen;
        }
        else
        {
            return -1;
        }
    }

#if 0
    /* Save ObjOutSideInfo in HSRAM */
    if(result->objOutSideInfo != NULL)
    {
        itemPayloadLen = sizeof(DPIF_PointCloudSideInfo) * result->numObjOut;
        if((totalHsramSize- itemPayloadLen) > 0)
        {
            memcpy(ptrCurrBuffer, (void *)result->objOutSideInfo, itemPayloadLen);
            ptrHsramBuffer->result.objOutSideInfo = (DPIF_PointCloudSideInfo *)ptrCurrBuffer;
            ptrCurrBuffer+= itemPayloadLen;
            totalHsramSize -=itemPayloadLen;
        }
        else
        {
            return -1;
        }
    }
#endif

    /* Save DPC_ObjectDetection_Stats in HSRAM */
    if(result->stats != NULL)
    {
        itemPayloadLen = sizeof(DPC_ObjectDetection_Stats);
        if((totalHsramSize- itemPayloadLen) > 0)
        {
            memcpy(ptrCurrBuffer, (void *)result->stats, itemPayloadLen);
            ptrHsramBuffer->result.stats = (DPC_ObjectDetection_Stats *)ptrCurrBuffer;
            ptrCurrBuffer+= itemPayloadLen;
            totalHsramSize -=itemPayloadLen;
        }
        else
        {
            return -1;
        }
    }

    return totalHsramSize;
}

/**
 *  @b Description
 *  @n
 *      DPM Execution Task. DPM execute results are processed here:
 *      a) Update states based on timestamp from DPC.
 *      b) Copy results to shared memory to be shared with MSS.
 *      c) Send Results to MSS by calling DPM_sendResult()
 *
 *  @retval
 *      Not Applicable.
 */

#define GMAX 500
volatile uint32_t gDPMExecute[512] = {0};
volatile uint32_t gTotalExecuteOuter[512] = {0};
volatile uint32_t gTCtr1=0, gTCtr2=0;

static void MmwDemo_DPC_ObjectDetection_dpmTask(void* args)
{
    int32_t     retVal;
    DPC_ObjectDetection_ExecuteResult *result;
    volatile uint32_t              startTime;

    while (1)
    {
        /* Execute the DPM module: */
        gTotalExecuteOuter[gTCtr2%GMAX] = ClockP_getTicks();
        gDPMExecute[gTCtr1%GMAX] = ClockP_getTicks();
        retVal = DPM_execute (gMmwDssMCB.dataPathObj.objDetDpmHandle, &resultBuffer);
        gDPMExecute[gTCtr1%GMAX] = ClockP_getTicks()-gDPMExecute[gTCtr1%GMAX];
        if (retVal < 0) {
            test_print ("Error: DPM execution failed [Error code %d]\n", retVal);
            MmwDemo_debugAssert (0);
        }
        else
        {
            if ((resultBuffer.size[0] == sizeof(DPC_ObjectDetection_ExecuteResult)))
            {
                result = (DPC_ObjectDetection_ExecuteResult *)resultBuffer.ptrBuffer[0];

                /* Get the time stamp before copy data to HSRAM */
                startTime = CycleCounterP_getCount32();

                /* Update processing stats and added it to buffer 1*/
                MmwDemo_updateObjectDetStats(result->stats,
                                                &gMmwDssMCB.dataPathObj.subFrameStats[result->subFrameIdx]);

                /* Copy result data to HSRAM */
                if (MmwDemo_copyResultToHSRAM(&gHSRAM, result, &gMmwDssMCB.dataPathObj.subFrameStats[result->subFrameIdx]) >= 0)
                {
                    /* Update interframe margin with HSRAM copy time */
                    gHSRAM.outStats.interFrameProcessingMargin -= ((CycleCounterP_getCount32() - startTime)/DSP_CLOCK_MHZ);

                    /* Update DPM buffer */
                    resultBuffer.ptrBuffer[0] = (uint8_t *)&gHSRAM.result;
                    resultBuffer.ptrBuffer[1] = (uint8_t *)&gHSRAM.outStats;
                    resultBuffer.size[1] = sizeof(MmwDemo_output_message_stats);


                    /* YES: Results are available send them. */
                    retVal = DPM_sendResult (gMmwDssMCB.dataPathObj.objDetDpmHandle, true, &resultBuffer);
                    if (retVal < 0)
                    {
                        test_print ("Error: Failed to send results [Error: %d] to remote\n", retVal);
                    }
                }
                else
                {
                    test_print ("Error: Failed to copy processing results to HSRAM, error=%d\n", retVal);
                    MmwDemo_debugAssert (0);
                }
            }
        }
        gTotalExecuteOuter[gTCtr2%GMAX] = ClockP_getTicks()-gTotalExecuteOuter[gTCtr2%GMAX];

        gTCtr2++;
        gTCtr1++;
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
volatile uint32_t gDebug = 1;
static void MmwDemo_dssInitTask(void* args)
{
    int32_t             errCode;
    DPM_InitCfg         dpmInitCfg;
    DPC_ObjectDetection_InitParams      objDetInitParams;
    uint32_t            edmaCCIdx;

    CycleCounterP_reset();

    /*****************************************************************************
     * Driver Open/Configuraiton:
     *****************************************************************************/

    /* Initialize the Data Path: */
    // MmwDemo_dataPathInit(&gMmwDssMCB.dataPathObj); 
    MmwDemo_dataPathOpen(&gMmwDssMCB.dataPathObj); 

    /*****************************************************************************
     * Initialization of the DPM Module:
     *****************************************************************************/
    memset ((void *)&objDetInitParams, 0, sizeof(DPC_ObjectDetection_InitParams));

    /* Note this must be after MmwDemo_dataPathOpen() above which opens the hwa */
    objDetInitParams.hwaHandle = gMmwDssMCB.dataPathObj.hwaHandle;
    objDetInitParams.L3ramCfg.addr = (void *)&gMmwL3[0];
    objDetInitParams.L3ramCfg.size = sizeof(gMmwL3);
    objDetInitParams.CoreLocalRamCfg.addr = &gDPC_ObjDetL2Heap[0];
    objDetInitParams.CoreLocalRamCfg.size = sizeof(gDPC_ObjDetL2Heap);
    for (edmaCCIdx = 0; edmaCCIdx < EDMA_NUM_CC; edmaCCIdx++)
    {
        objDetInitParams.edmaHandle[edmaCCIdx] = gMmwDssMCB.dataPathObj.edmaHandle[edmaCCIdx];
    }

    /* DPC Call-back config */
    objDetInitParams.processCallBackCfg.processFrameBeginCallBackFxn =
        MmwDemo_DPC_ObjectDetection_processFrameBeginCallBackFxn;
    objDetInitParams.processCallBackCfg.processInterFrameBeginCallBackFxn =
        MmwDemo_DPC_ObjectDetection_processInterFrameBeginCallBackFxn;

    memset ((void *)&dpmInitCfg, 0, sizeof(DPM_InitCfg));

    /* Setup the configuration: */
    dpmInitCfg.ptrProcChainCfg  = &gDPC_ObjectDetectionCfg;
    dpmInitCfg.instanceId       = 0xFEEDFEED;
    dpmInitCfg.domain           = DPM_Domain_REMOTE;
    dpmInitCfg.reportFxn        = MmwDemo_DPC_ObjectDetection_reportFxn;
    dpmInitCfg.arg              = &objDetInitParams;
    dpmInitCfg.argSize          = sizeof(DPC_ObjectDetection_InitParams);

    /* Initialize the DPM Module: */
    gMmwDssMCB.dataPathObj.objDetDpmHandle = DPM_init (&dpmInitCfg, &errCode);
    if (gMmwDssMCB.dataPathObj.objDetDpmHandle == NULL)
    {
        test_print ("Error: Unable to initialize the DPM Module [Error: %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return;
    }

    /* Synchronization: This will synchronize the execution of the control module
     * between the domains. This is a prerequiste and always needs to be invoked. */
    while (1)
    {
        int32_t syncStatus;

        /* Get the synchronization status: */
        syncStatus = DPM_synch (gMmwDssMCB.dataPathObj.objDetDpmHandle, &errCode);
        if (syncStatus < 0)
        {
            /* Error: Unable to synchronize the framework */
            test_print ("Error: DPM Synchronization failed [Error code %d]\n", errCode);
            MmwDemo_debugAssert (0);
            return;
        }
        if (syncStatus == 1)
        {
            /* Synchronization acheived: */
            break;
        }
        /* Sleep and poll again: */
        ClockP_usleep(1 * 1000);
    }
    test_print ("Debug: DPM Module Sync is done\n");

    /* Launch the DPM Task */
    gMmwDssMCB.objDetDpmTaskHandle = xTaskCreateStatic( MmwDemo_DPC_ObjectDetection_dpmTask,   /* Pointer to the function that implements the task. */
                                  "MmwDemo_DPC_ObjectDetection_dpmTask", /* Text name for the task.  This is to facilitate debugging only. */
                                  MMWDEMO_DPM_TASK_STACK_SIZE,  /* Stack depth in units of StackType_t typically uint32_t on 32b CPUs */
                                  NULL,              /* We are not using the task parameter. */
                                  MMWDEMO_DPM_TASK_PRI,      /* task priority, 0 is lowest priority, configMAX_PRIORITIES-1 is highest */
                                  gMmwDemo_dpmTaskStack,  /* pointer to stack base */
                                  &gMmwDssMCB.objDetDpmTaskObj );    /* pointer to statically allocated task object memory */
    configASSERT(gMmwDssMCB.objDetDpmTaskHandle != NULL);


    while(gDebug){
        ClockP_usleep(1*1000);
    }
}

/**
 *  @b Description
 *  @n
 *     Function to sleep the DSP using IDLE instruction.
 *     When DSP has no work left to do,
 *     the BIOS will be in Idle thread and will call this function.
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_sleep(void)
{
    /* issue WFI (Wait For Interrupt) instruction */
    asm(" IDLE ");
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
    
    /* init SOC specific modules */
    System_init();
    Board_init();

    /* Initialize and populate the demo MCB */
    memset ((void*)&gMmwDssMCB, 0, sizeof(gMmwDssMCB));

    /* This task is created at highest priority, it should create more tasks and then delete itself */
    gMmwDssMCB.initTaskHandle = xTaskCreateStatic( MmwDemo_dssInitTask,   /* Pointer to the function that implements the task. */
                                  "MmwDemo_dssInitTask", /* Text name for the task.  This is to facilitate debugging only. */
                                  MMWDEMO_DSS_INIT_TASK_STACK_SIZE,  /* Stack depth in units of StackType_t typically uint32_t on 32b CPUs */
                                  NULL,              /* We are not using the task parameter. */
                                  MMWDEMO_DSS_INIT_TASK_PRI,      /* task priority, 0 is lowest priority, configMAX_PRIORITIES-1 is highest */
                                  gMmwDemo_dssInitTaskStack,  /* pointer to stack base */
                                  &gMmwDssMCB.initTaskObj );    /* pointer to statically allocated task object memory */
    configASSERT(gMmwDssMCB.initTaskHandle != NULL);

    /* Start the scheduler to start the tasks executing. */
    vTaskStartScheduler();

    /* The following line should never be reached because vTaskStartScheduler()
    will only return if there was not enough FreeRTOS heap memory available to
    create the Idle and (if configured) Timer tasks.  Heap management, and
    techniques for trapping heap exhaustion, are described in the book text. */
    DebugP_assertNoLog(0);

    return 0;
}
