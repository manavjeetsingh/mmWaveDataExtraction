/**
 *   @file  diag_mon_config.h
 *
 *   @brief
 *      This is the header file that describes configurations for Diagnostic 
 *      and Monitoring Demo.
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
#ifndef DIAG_MON_CONFIG_H
#define DIAG_MON_CONFIG_H

/* MMWAVE library Include Files */
#include <ti/control/mmwave/mmwave.h>
#include <ti/common/sys_common.h>
#include <ti/datapath/dpc/objectdetection/objdethwa/objectdetection.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CRC_TYPE_16BIT        0
#define CRC_TYPE_32BIT        1
#define CRC_TYPE_64BIT        2



/* Change these configuration as required */
#if defined(SOC_XWR18XX)
#define MMW_FREQUENCY_LIMIT_LOW            77U
#define MMW_FREQUENCY_LIMIT_HIGH           81U

#elif defined(SOC_XWR68XX)
#define MMW_FREQUENCY_LIMIT_LOW            60U
#define MMW_FREQUENCY_LIMIT_HIGH           64U
#endif

/* Set the CRC Type for mmWaveLink communication with RadarSS */
#define MMWAVELINK_CRC_TYPE                 CRC_TYPE_32BIT

/* Monitoring Related configurations */
#define MMW_CALIB_MON_TIME_UNIT               1U

/* disable frame start async-event */
#define MMW_FRAME_START_ASYNC_EVENT_DISABLE   1U
/* disable frame Stop async-event */
#define MMW_FRAME_STOP_ASYNC_EVENT_DISABLE    1U


/**
 * @brief
 *  Millimeter Wave Demo Gui Monitor Selection
 *
 * @details
 *  The structure contains the selection for what information is placed to
 *  the output packet, and sent out to GUI. Unless otherwise specified,
 *  if the flag is set to 1, information
 *  is sent out. If the flag is set to 0, information is not sent out.
 *
 */
typedef struct MmwDemo_GuiMonSel_t
{
    /*! @brief   if 1: Send list of detected objects (see @ref DPIF_PointCloudCartesian) and
     *                 side info (@ref DPIF_PointCloudSideInfo).\n
     *           if 2: Send list of detected objects only (no side info)\n
     *           if 0: Don't send anything */
    uint8_t        detectedObjects;

    /*! @brief   Send log magnitude range array  */
    uint8_t        logMagRange;

    /*! @brief   Send noise floor profile */
    uint8_t        noiseProfile;

    /*! @brief   Send complex range bins at zero doppler, all antenna symbols for range-azimuth heat map */
    uint8_t        rangeAzimuthHeatMap;

    /*! @brief   Send complex range bins at zero doppler, (all antenna symbols), for range-azimuth heat map */
    uint8_t        rangeDopplerHeatMap;

    /*! @brief   Send stats */
    uint8_t        statsInfo;
} MmwDemo_GuiMonSel;

/**
 * @brief
 *  Millimeter Wave Demo Data Path Information.
 *
 * @details
 *  The structure is used to hold all the relevant information for
 *  the data path.
 */
typedef struct MmwDemo_platformCfg_t
{
    /*! @brief   GPIO index for sensor status */
    uint32_t            SensorStatusGPIO;
    
    /*! @brief   CPU Clock Frequency. */
    uint32_t            sysClockFrequency;

    /*! @brief   UART Command Baud Rate. */
    uint32_t            commandBaudRate;
} MmwDemo_platformCfg;

/**
 * @brief
 *  Millimeter Wave Demo configuration
 *
 * @details
 *  The structure is used to hold all the relevant configuration
 *  which is used to execute the Millimeter Wave Demo.
 */
typedef struct MmwDemo_Cfg_t
{
    /*! @brief   Platform specific configuration. */
    MmwDemo_platformCfg platformCfg;

    /*! @brief   Datapath output loggerSetting
                 0 (default): MSS UART logger
                 1: DSS UART logger
     */
    uint8_t              dataLogger;
} MmwDemo_Cfg;


extern rlRfLdoBypassCfg_t gRfLdoBypassCfg;
extern rlChanCfg_t gRfChannelCfg;
extern rlAdcOutCfg_t gAdcOutCfg;
extern rlLowPowerModeCfg_t gLowPowerModeCfg;
extern rlProfileCfg_t gProfileCfg;
extern rlChirpCfg_t gChirpCfg;
extern rlFrameCfg_t gFrameCfg;
extern rlRfCalMonTimeUntConf_t gCalMonTimeUnitConf;
extern rlRunTimeCalibConf_t gRunTimeCalibCfg;
extern rlMonAnaEnables_t  gRfAnaMonitorEn;
extern rlTempMonConf_t gTempMonCfg ;
extern rlRxGainPhaseMonConf_t gRxGainPhaseMonCfg;
extern rlRxNoiseMonConf_t gRxNoiseMonCfg;
extern rlRxIfStageMonConf_t gRxIfStageMonCfg;
extern rlTxPowMonConf_t gTx0PowMonCfg;
extern rlTxPowMonConf_t gTx1PowMonCfg;
extern rlTxPowMonConf_t gTx2PowMonCfg;
extern rlTxBallbreakMonConf_t  gTx0BallBreakMonCfg;
extern rlTxBallbreakMonConf_t  gTx1BallBreakMonCfg;
extern rlTxBallbreakMonConf_t  gTx2BallBreakMonCfg;
extern rlTxGainPhaseMismatchMonConf_t gTxGainPhseMonCfg;
extern rlSynthFreqMonConf_t gSynthFreqMonCfg;
extern rlExtAnaSignalsMonConf_t gExtAnaSigMonCfg;
extern rlTxIntAnaSignalsMonConf_t gTx0IntAnaSigMonCfg;
extern rlTxIntAnaSignalsMonConf_t gTx1IntAnaSigMonCfg;
extern rlTxIntAnaSignalsMonConf_t gTx2IntAnaSigMonCfg;
extern rlRxIntAnaSignalsMonConf_t gRxIntAnaSigMonCfg;
extern rlPmClkLoIntAnaSignalsMonConf_t gPmClkLoIntAnaSigMonCfg;
extern rlGpadcIntAnaSignalsMonConf_t gGpadcIntAnaSigMonCfg;
extern rlPllContrVoltMonConf_t gPllConVoltMonCfg;
extern rlDualClkCompMonConf_t gDualClkCompMonCfg;
extern rlRxSatMonConf_t gRxIfSatMonCfg;
extern rlSigImgMonConf_t gSigImgMonCfg;
extern rlRxMixInPwrMonConf_t gRxMixInpwrMonCfg;
extern rlMonDigEnables_t gDigMonitorEn;
extern rlDigMonPeriodicConf_t gDigMonitorPeriod;




#ifdef __cplusplus
}
#endif

#endif /* DIAG_MON_CONFIG_H */
