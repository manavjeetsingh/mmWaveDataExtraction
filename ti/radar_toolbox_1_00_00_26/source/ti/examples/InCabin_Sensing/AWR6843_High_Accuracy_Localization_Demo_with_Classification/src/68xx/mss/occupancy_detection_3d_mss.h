/**
 *   @file  mmwLab_mss.h
 *
 *   @brief
 *      This is the main header file for the 3D people counting demo
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2016 Texas Instruments, Inc.
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
#ifndef MMW_MSS_H
#define MMW_MSS_H

#include <ti/sysbios/knl/Semaphore.h>
#include <ti/sysbios/knl/Task.h>

#include <ti/common/mmwave_error.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/uart/UART.h>
#include <ti/drivers/gpio/gpio.h>
#include <ti/drivers/mailbox/mailbox.h>

#include <ti/demo/utils/mmwdemo_adcconfig.h>
//#include <ti/demo/utils/mmwdemo_monitor.h>

#include <common/src/dpc/objdetrangehwa_overhead/objdetrangehwa.h>
#include "common/mmwLab_output.h"
#include "common/mmwLab_config.h"


#ifdef __cplusplus
extern "C" {
#endif

/*! @brief For advanced frame config, below define means the configuration given is
 * global at frame level and therefore it is broadcast to all sub-frames.
 */
#define MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG (-1)

/**
 * @defgroup configStoreOffsets     Offsets for storing CLI configuration
 * @brief    Offsets of config fields within the parent structures, note these offsets will be
 *           unique and hence can be used to differentiate the commands for processing purposes.
 * @{
 */
#define MMWLAB_ADCBUFCFG_OFFSET                 (offsetof(OccupancyDetection3D_SubFrameCfg, adcBufCfg))

#define MMWLAB_SUBFRAME_DSPDYNCFG_OFFSET        (offsetof(OccupancyDetection3D_SubFrameCfg, objDetDynCfg) + \
                                                  offsetof(OccupancyDetection3D_DPC_ObjDet_DynCfg, dspDynCfg))

#define MMWLAB_SUBFRAME_R4FDYNCFG_OFFSET        (offsetof(OccupancyDetection3D_SubFrameCfg, objDetDynCfg) + \
                                                  offsetof(OccupancyDetection3D_DPC_ObjDet_DynCfg, r4fDynCfg))

#define MMWLAB_CALIBDCRANGESIG_OFFSET           (MMWLAB_SUBFRAME_R4FDYNCFG_OFFSET + \
                                                  offsetof(DPC_ObjectDetectionRangeHWA_DynCfg, calibDcRangeSigCfg))

#define MMWLAB_CAPONCHAINCFG_OFFSET              (MMWLAB_SUBFRAME_DSPDYNCFG_OFFSET + \
                                                  offsetof(DPC_ObjectDetection_DynCfg, caponChainCfg))

#define MMWLAB_DYNRACFARCFG_OFFSET            (MMWLAB_CAPONCHAINCFG_OFFSET + \
                                                  offsetof(caponChainCfg, dynamicCfarConfig))

#define MMWLAB_STATICRACFARCFG_OFFSET              (MMWLAB_CAPONCHAINCFG_OFFSET + \
                                                  offsetof(caponChainCfg, staticCfarConfig))

#define MMWLAB_DOACAPONCFG_OFFSET                (MMWLAB_CAPONCHAINCFG_OFFSET + \
                                                  offsetof(caponChainCfg, doaConfig))

#define MMWLAB_DOACAPONRACFG_OFFSET                    (MMWLAB_DOACAPONCFG_OFFSET + \
                                                  offsetof(doaConfig, rangeAngleCfg))

#define MMWLAB_DOA2DESTCFG_OFFSET                 (MMWLAB_DOACAPONCFG_OFFSET + \
                                                  offsetof(doaConfig, angle2DEst))

#define MMWLAB_DOAFOVCFG_OFFSET                 (MMWLAB_DOACAPONCFG_OFFSET + \
                                                  offsetof(doaConfig, fovCfg))

#define MMWLAB_STATICANGESTCFG_OFFSET       (MMWLAB_DOACAPONCFG_OFFSET + \
                                                  offsetof(doaConfig, staticEstCfg))

#define MMWLAB_DOPCFARCFG_OFFSET          (MMWLAB_DOACAPONCFG_OFFSET + \
                                                  offsetof(doaConfig, dopCfarCfg))



/** @}*/ /* configStoreOffsets */

/**
 * @brief
 *  3D people counting Demo Sensor State
 *
 * @details
 *  The enumeration is used to define the sensor states used in 3D people counting Demo
 */
typedef enum OccupancyDetection3D_SensorState_e
{
    /*!  @brief Inital state after sensor is initialized.
     */
    OccupancyDetection3D_SensorState_INIT = 0,

    /*!  @brief Indicates sensor is started */
    OccupancyDetection3D_SensorState_STARTED,

    /*!  @brief  State after sensor has completely stopped */
    OccupancyDetection3D_SensorState_STOPPED
}OccupancyDetection3D_SensorState;

/**
 * @brief
 *  3D people counting Demo statistics
 *
 * @details
 *  The structure is used to hold the statistics information for the
 *  3D people counting Demo
 */
typedef struct OccupancyDetection3D_MSS_Stats_t
{
    /*! @brief   Counter which tracks the number of frame trigger events from BSS */
    uint64_t     frameTriggerReady;
    
    /*! @brief   Counter which tracks the number of failed calibration reports
     *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     failedTimingReports;

    /*! @brief   Counter which tracks the number of calibration reports received
     *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     calibrationReports;

     /*! @brief   Counter which tracks the number of sensor stop events received
      *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     sensorStopped;
}OccupancyDetection3D_MSS_Stats;

/**
 * @brief
 *  3D people counting Demo Data Path Information.
 *
 * @details
 *  The structure is used to hold all the relevant information for
 *  the data path.
 */
typedef struct OccupancyDetection3D_SubFrameCfg_t
{
    /*! @brief ADC buffer configuration storage */
    MmwDemo_ADCBufCfg adcBufCfg;

    /*! @brief Flag indicating if @ref adcBufCfg is pending processing. */
    uint8_t isAdcBufCfgPending : 1;

    /*! @brief Dynamic configuration storage for object detection DPC */
    mmwLab_DPC_ObjDet_DynCfg objDetDynCfg;

    /*! @brief  ADCBUF will generate chirp interrupt event every this many chirps - chirpthreshold */
    uint8_t     numChirpsPerChirpEvent;

    /*! @brief  Number of bytes per RX channel, it is aligned to 16 bytes as required by ADCBuf driver  */
    uint32_t    adcBufChanDataSize;

    /*! @brief  Number of ADC samples */
    uint16_t    numAdcSamples;

    /*! @brief  Number of chirps per sub-frame */
    uint16_t    numChirpsPerSubFrame;
    
    /*! @brief  Number of virtual antennas */
    uint8_t     numVirtualAntennas; 
} OccupancyDetection3D_SubFrameCfg;

/*!
 * @brief
 * Structure holds message stats information from data path.
 *
 * @details
 *  The structure holds stats information. This is a payload of the TLV message item
 *  that holds stats information.
 */
typedef struct OccupancyDetection3D_SubFrameStats_t
{
    /*! @brief   Frame processing stats */
    mmwLab_output_message_stats    outputStats;

    /*! @brief   Dynamic CLI configuration time in usec */
    uint32_t                        pendingConfigProcTime;

    /*! @brief   SubFrame Preparation time on MSS in usec */
    uint32_t                        subFramePreparationTime;
} OccupancyDetection3D_SubFrameStats;

/**
 * @brief Task handles storage structure
 */
typedef struct OccupancyDetection3D_TaskHandles_t
{
    /*! @brief   MMWAVE Control Task Handle */
    Task_Handle mmwaveCtrl;

    /*! @brief   ObjectDetection DPC related dpmTask */
    Task_Handle objDetDpmTask;

    /*! @brief   Demo init task */
    Task_Handle initTask;
} OccupancyDetection3D_taskHandles;

typedef struct OccupancyDetection3D_DataPathObj_t
{
    /*! @brief Handle to hardware accelerator driver. */
    HWA_Handle          hwaHandle;

    /*! @brief   Handle of the EDMA driver. */
    EDMA_Handle         edmaHandle;

    /*! @brief   Radar cube memory information from range DPC */
    DPC_ObjectDetectionRangeHWA_preStartCfg_radarCubeMem radarCubeMem;

    /*! @brief   Memory usage after the preStartCfg range DPC is applied */
    DPC_ObjectDetectionRangeHWA_preStartCfg_memUsage memUsage;

    /*! @brief   EDMA error Information when there are errors like missing events */
    EDMA_errorInfo_t    EDMA_errorInfo;

    /*! @brief EDMA transfer controller error information. */
    EDMA_transferControllerErrorInfo_t EDMA_transferControllerErrorInfo;

} OccupancyDetection3D_DataPathObj;

/**
 * @brief
 *  3D people counting Demo  MCB
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  3D people counting Demo
 */
typedef struct OccupancyDetection3D_MSS_MCB_t
{
    /*! @brief      Configuration which is used to execute the demo */
    mmwLab_Cfg                 cfg;

    /*! * @brief    Handle to the SOC Module */
    SOC_Handle                  socHandle;

    /*! @brief      UART Logging Handle */
    UART_Handle                 loggingUartHandle;

    /*! @brief      UART Command Rx/Tx Handle */
    UART_Handle                 commandUartHandle;

    /*! @brief      This is the mmWave control handle which is used
     * to configure the BSS. */
    MMWave_Handle               ctrlHandle;

    /*! @brief      ADCBuf driver handle */
    ADCBuf_Handle               adcBufHandle;

    /*! @brief      DSP chain DPM Handle */
    DPM_Handle                  objDetDpmHandle;

    /*! @brief      Object Detection DPC common configuration */
    mmwLab_DPC_ObjDet_CommonCfg objDetCommonCfg;

    /*! @brief      Data path object */
    OccupancyDetection3D_DataPathObj         dataPathObj;

    /*! @brief      Object Detection DPC subFrame configuration */
    OccupancyDetection3D_SubFrameCfg         subFrameCfg[RL_MAX_SUBFRAMES];

    /*! @brief      sub-frame stats */
    OccupancyDetection3D_SubFrameStats       subFrameStats[RL_MAX_SUBFRAMES];

    /*! @brief      Demo Stats */
    OccupancyDetection3D_MSS_Stats           stats;

    mmwLab_output_message_UARTpointCloud   pointCloudToUart;
    DPIF_DetMatrix                      heatMapOutFromDSP;
    DPIF_PointCloudSpherical            *pointCloudFromDSP;
    DPIF_PointCloudSideInfo             *pointCloudSideInfoFromDSP;
    DPC_ObjectDetection_Stats           *frameStatsFromDSP;
    uint32_t                            currSubFrameIdx;

    uint8_t                             numTargets;
    uint16_t                            numIndices;
    bool                                presenceDetEnabled;
    uint32_t                            presenceInd;
    uint16_t                            numDetectedPoints;
    uint32_t                            trackerProcessingTimeInUsec;
    uint32_t                            uartProcessingTimeInUsec;


    /*! @brief      Task handle storage */
    OccupancyDetection3D_taskHandles         taskHandles;

    /*! @brief   RF frequency scale factor, = 2.7 for 60GHz device, = 3.6 for 76GHz device */
    double                      rfFreqScaleFactor;

    /*! @brief   Semaphore handle to signal DPM started from DPM report function */
    Semaphore_Handle            DPMstartSemHandle;

    /*! @brief   Semaphore handle to signal DPM stopped from DPM report function. */
    Semaphore_Handle            DPMstopSemHandle;

    /*! @brief   Semaphore handle to signal DPM ioctl from DPM report function. */
    Semaphore_Handle            DPMioctlSemHandle;

    /*! @brief   Semaphore handle to run UART DMA task. */
    Semaphore_Handle            uartTxSemHandle;

    /*! @brief   Semaphore handle to trigger tracker DPU. */
    Semaphore_Handle            trackerDPUSemHandle;

    /*! @brief    Sensor state */
    OccupancyDetection3D_SensorState         sensorState;

    /*! @brief   Tracks the number of sensor start */
    uint32_t                    sensorStartCount;

    /*! @brief   Tracks the number of sensor sop */
    uint32_t                    sensorStopCount;

} OccupancyDetection3D_MSS_MCB;

/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/

/* Functions to handle the actions need to move the sensor state */
extern int32_t OccupancyDetection3D_openSensor(bool isFirstTimeOpen);
extern int32_t OccupancyDetection3D_configSensor(void);
extern int32_t OccupancyDetection3D_startSensor(void);
extern void OccupancyDetection3D_stopSensor(void);

/* functions to manage the dynamic configuration */
extern uint8_t OccupancyDetection3D_isAllCfgInPendingState(void);
extern uint8_t OccupancyDetection3D_isAllCfgInNonPendingState(void);
extern void OccupancyDetection3D_resetStaticCfgPendingState(void);
extern void OccupancyDetection3D_CfgUpdate(void *srcPtr, uint32_t offset, uint32_t size, int8_t subFrameNum);

extern void OccupancyDetection3D_CLIInit (uint8_t taskPriority);

/* Debug Functions */
extern void _OccupancyDetection3D_debugAssert(int32_t expression, const char *file, int32_t line);
#define OccupancyDetection3D_debugAssert(expression) {                                      \
                                         _OccupancyDetection3D_debugAssert(expression,      \
                                                  __FILE__, __LINE__);         \
                                         DebugP_assert(expression);             \
                                        }

#ifdef __cplusplus
}
#endif

#endif /* MMW_MSS_H */

