/*
 *   @file  mmw_cli.c
 *
 *   @brief
 *      Mmw (Milli-meter wave) DEMO CLI Implementation
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2018 Texas Instruments, Inc.
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

/* BIOS/XDC Include Files. */
#include <xdc/runtime/System.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
#include <ti/drivers/uart/UART.h>
#include <ti/control/mmwavelink/mmwavelink.h>
#include <ti/utils/mathutils/mathutils.h>

/* Demo Include Files */
#include "mss/mmw_mss.h"
#include "mss/mmw_cli.h"
#include "common/mmw_rfparser.h"
#include "common/rfmonitor.h"
#include "common/rfmonitor_internal.h"
#include "common/mmw_adcconfig.h"

/**************************************************************************
*************************** Extern Definitions *******************************
**************************************************************************/
extern MmwDemo_MSS_MCB    gMmwMssMCB;//TODO JIT : may remove this global struct or reduce element within it.
extern RFMonitorCfg     gRFMonitorCfg;

/**************************************************************************
 *************************** Local function prototype****************************
 **************************************************************************/

/* CLI Extended Command Functions */

extern int32_t MmwDemo_dataPathConfig (void);
static int32_t MmwDemo_CLISensorStart (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLISensorStop (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIGuiMonSel (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIADCBufCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIChirpQualityRxSatMonCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIChirpQualitySigImgMonCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLILvdsStreamCfg (int32_t argc, char* argv[]);

/*This is used in several formulas that translate CLI input to mmwavelink units.
  It must be defined as double to achieve the correct precision
  on the formulas (if defined as float there will be small precision errors
  that may result in the computed value being out of mmwavelink range if the 
  CLI input is a borderline value).

  The vaiable is initialized in @ref CLI_MMWaveExtensionInit() */
double gCLI_mmwave_freq_scale_factor;


/**
 * @brief   Global variable which tracks the CLI MCB
 */
CLI_MCB     gCliMcb;
MMW_CLI_AllCfg  gCLICmdCfg;

volatile uint8_t sensorStartCmdCnt = 0;
/**
 * @brief
 *  Global mmwave device part number and string table
 */
CLI_partInfoString partInfoStringArray[] =
{
    /* Maximum part number string size is CLI_MAX_PARTNO_STRING_LEN */
#ifdef SOC_XWR14XX
    {SOC_AWR14XX_nonSecure_PartNumber,  "AWR14xx non-secure"},
    {SOC_IWR14XX_nonSecure_PartNumber,  "IWR14xx non-secure"},
#endif

#ifdef SOC_XWR16XX
    {SOC_AWR16XX_nonSecure_PartNumber,  "AWR16xx non-secure"},
    {SOC_IWR16XX_nonSecure_PartNumber,  "IWR16xx non-secure"},
    {SOC_AWR16XX_Secure_PartNumber,     "AWR16xx secure"},
    {SOC_IWR16XX_Secure_PartNumber,     "IWR16xx secure"},
#endif

#ifdef SOC_XWR18XX
    {SOC_AWR18XX_nonSecure_PartNumber,  "AWR18xx non-secure"},
    {SOC_IWR18XX_nonSecure_PartNumber,  "IWR18xx non-secure"},
    {SOC_AWR18XX_Secure_PartNumber,     "AWR18xx secure"},
#endif

#ifdef SOC_XWR68XX
    {SOC_IWR64XX_nonSecure_PartNumber,  "IWR64xx non-secure"},    
    {SOC_IWR68XX_nonSecure_PartNumber,  "IWR68xx non-secure"},    
    {SOC_IWR68XX_Secure_PartNumber,     "IWR68xx secure"},
#endif

    /* last entry of the table */
    {0U,                                "Not Supported"}
};


/**************************************************************************
 *************************** CLI  Function Definitions **************************
 **************************************************************************/
/**
 *  @b Description
 *  @n
 *     Helper function used by RF Monitor module to get rxGain from a
 *   given profile id.
 *
 *  @retval
 *      Success         RX Gain programmed in profile
 *      Fail            -1
 */
uint16_t MmwDemo_getProfileRxGain(uint16_t profileIdx)
{
    return ( gCLICmdCfg.profChirpCfg.profileCfg[profileIdx].rxGain);
}

/**  @b Description
 *  @n
 *      This is the CLI Handler for the sensor start command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLISensorStart (int32_t argc, char* argv[])
{
    bool        doReconfig = true;
    int32_t     retVal = 0;

    /*  Only following command syntax will be supported 
        sensorStart
        sensorStart 0
    */
    if (argc == 2)
    {
        doReconfig = (bool) atoi (argv[1]);

        if (doReconfig == true)
        {
            /* Error: Reconfig is not supported, only argument of 0 is
             * do not reconfig, just re-start the sensor) valid */
            return MMW_CLI_ERROR_CODE_SENSOR_START_RECONFIG;
        }
    }
    else
    {
        /* In case there is no argument for sensorStart, always do reconfig */
        doReconfig = true;
    }

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignore this command */
        return MMW_CLI_ERROR_CMD_SENS_ALREADY_START_STOP;
    }
    /***********************************************************************************
     * Do sensor state management to influence the sensor actions
     ***********************************************************************************/

    retVal = MmwDemo_openSensor(!sensorStartCmdCnt);
    if(retVal != 0)
    {
        return MMW_CLI_ERROR_DATA_PATH_CONFIG;
    }

    if(MmwDemo_dataPathConfig() < 0)
        return MMW_CLI_ERROR_DATA_PATH_CONFIG;

    retVal = MmwDemo_startSensor(!sensorStartCmdCnt);
    sensorStartCmdCnt++;
    if(retVal != 0)
    {
        return -1;
    }

    /***********************************************************************************
     * Set the state
     ***********************************************************************************/
    gMmwMssMCB.sensorState = MmwDemo_SensorState_STARTED;

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the sensor stop command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLISensorStop (int32_t argc, char* argv[])
{
    if ((gMmwMssMCB.sensorState == MmwDemo_SensorState_STOPPED) ||
        (gMmwMssMCB.sensorState == MmwDemo_SensorState_INIT) ||
        (gMmwMssMCB.sensorState == MmwDemo_SensorState_OPENED))
    {
        /* Ignored: Sensor is already stopped */
        return MMW_CLI_ERROR_CMD_SENS_ALREADY_START_STOP;
    }

    MmwDemo_stopSensor();

    gMmwMssMCB.sensorState = MmwDemo_SensorState_STOPPED;
    return 0;
}

/**
 *  @b Description
 *  @n
 *      Utility function to get sub-frame number
 *
 *  @param[in] argc  Number of arguments
 *  @param[in] argv  Arguments
 *  @param[in] expectedArgc Expected number of arguments
 *  @param[out] subFrameNum Sub-frame Number (0 based)
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLIGetSubframe (int32_t argc, char* argv[], int32_t expectedArgc,
                                       int8_t* subFrameNum)
{
    int8_t subframe;
    
    /* Sanity Check: Minimum argument check */
    if (argc != expectedArgc)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /*Subframe info is always in position 1*/
    subframe = (int8_t) atoi(argv[1]);

    if(subframe >= (int8_t)RL_MAX_SUBFRAMES)
    {
        /* Error: Subframe number is invalid */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    *subFrameNum = (int8_t)subframe;

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for gui monitoring configuration
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLIGuiMonSel (int32_t argc, char* argv[])
{
    MmwDemo_GuiMonSel   guiMonSel;
    int8_t              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 8, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize the guiMonSel configuration: */
    memset ((void *)&guiMonSel, 0, sizeof(MmwDemo_GuiMonSel));

    /* Populate configuration: */
    guiMonSel.detectedObjects           = atoi (argv[2]);
    guiMonSel.logMagRange               = atoi (argv[3]);
    guiMonSel.noiseProfile              = atoi (argv[4]);
    guiMonSel.rangeAzimuthHeatMap       = atoi (argv[5]);
    guiMonSel.rangeDopplerHeatMap       = atoi (argv[6]);
    guiMonSel.statsInfo                 = atoi (argv[7]);

    MmwDemo_CfgUpdate((void *)&guiMonSel, MMWDEMO_GUIMONSEL_OFFSET,
        sizeof(MmwDemo_GuiMonSel), subFrameNum);

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for data logger set command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLIADCBufCfg (int32_t argc, char* argv[])
{
    Mmw_ADCBufCfg      adcBufCfg;
    int8_t              subFrameNum;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
       /* Ignored: This command is not allowed after sensor has started */
        return 0;
    }

    if(MmwDemo_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Initialize the ADC Output configuration: */
    memset ((void *)&adcBufCfg, 0, sizeof(adcBufCfg));

    /* Populate configuration: */
    adcBufCfg.adcFmt          = (uint8_t) atoi (argv[2]);
    adcBufCfg.iqSwapSel       = (uint8_t) atoi (argv[3]);
    adcBufCfg.chInterleave    = (uint8_t) atoi (argv[4]);
    adcBufCfg.chirpThreshold  = (uint8_t) atoi (argv[5]);

    /* This demo is using HWA for 1D processing which does not allow multi-chirp
     * processing */
    if (adcBufCfg.chirpThreshold != 1)
    {
        /* Error: chirpThreshold must be 1, multi-chirp is not allowed */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }
    /* store to Global structure */
    gCLICmdCfg.adcBufCfg = adcBufCfg;

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&adcBufCfg,
                      MMWDEMO_ADCBUFCFG_OFFSET,
                      sizeof(Mmw_ADCBufCfg), subFrameNum);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for configuring CQ RX Saturation monitor
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLIChirpQualityRxSatMonCfg (int32_t argc, char* argv[])
{
    int32_t                 retVal = -1;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignored: This command is not allowed after sensor has started */
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 6)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Populate configuration: */
    gRFMonitorCfg.rxSatMonCfg->profileIndx                 = (uint8_t) atoi (argv[1]);

    if( gRFMonitorCfg.rxSatMonCfg->profileIndx < RL_MAX_PROFILES_CNT)
    {
        /* Save Configuration to use later */
        gRFMonitorCfg.rxSatMonCfg->satMonSel                   = (uint8_t) atoi (argv[2]);
        gRFMonitorCfg.rxSatMonCfg->primarySliceDuration        = (uint16_t) atoi (argv[3]);
        gRFMonitorCfg.rxSatMonCfg->numSlices                   = (uint16_t) atoi (argv[4]);
        gRFMonitorCfg.rxSatMonCfg->rxChannelMask               = (uint8_t) atoi (argv[5]);
        
        /* Enable this Monitor on CLI CMD */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
                                                    RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
                                                    1U);
        retVal = rlRfRxIfSatMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlRxSatMonConf_t*)gRFMonitorCfg.rxSatMonCfg);
        
        return retVal;
    }
    else
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for configuring CQ Singal & Image band monitor
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLIChirpQualitySigImgMonCfg (int32_t argc, char* argv[])
{
    int32_t                 retVal = -1;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignored: This command is not allowed after sensor has started */
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Populate configuration: */
    gRFMonitorCfg.sigImageBandMonCfg->profileIndx         = (uint8_t) atoi (argv[1]);

    if(gRFMonitorCfg.sigImageBandMonCfg->profileIndx < RL_MAX_PROFILES_CNT)
    {
        gRFMonitorCfg.sigImageBandMonCfg->numSlices            = (uint8_t) atoi (argv[2]);
        gRFMonitorCfg.sigImageBandMonCfg->timeSliceNumSamples  = (uint16_t) atoi (argv[3]);
        /* Enable this Monitor on CLI CMD */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
                                                    RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
                                                    1U);
        retVal = rlRfRxSigImgMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlSigImgMonConf_t*)gRFMonitorCfg.sigImageBandMonCfg);
        
        return retVal;
    }
    else
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the High Speed Interface
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLILvdsStreamCfg (int32_t argc, char* argv[])
{
    MmwDemo_LvdsStreamCfg   cfg;
    int8_t                  subFrameNum;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignored: This command is not allowed after sensor has started */
        return 0;
    }

    if(MmwDemo_CLIGetSubframe(argc, argv, 5, &subFrameNum) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Initialize configuration for DC range signature calibration */
    memset ((void *)&cfg, 0, sizeof(MmwDemo_LvdsStreamCfg));

    /* Populate configuration: */
    cfg.isHeaderEnabled = (bool)    atoi(argv[2]);
    cfg.dataFmt         = (uint8_t) atoi(argv[3]);
    cfg.isSwEnabled     = (bool)    atoi(argv[4]);

    /* in this application software LVDS streaming is not supported */
    if((cfg.isSwEnabled == true) || (cfg.isHeaderEnabled == true))
    {
        return MMW_CLI_ERROR_LVDS_SW_HEADER_NOT_SUPPORTED;
    }

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&cfg,
                      MMWDEMO_LVDSSTREAMCFG_OFFSET,
                      sizeof(MmwDemo_LvdsStreamCfg), subFrameNum);

    return 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to set Init time calibration.
 *      using the mmWave link API.
 *
 *  @param  Enable/Disable Calibrations. 
 *          1 - Enable all calibrations. 0 - Disable all calibrations
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MmwDemo_RfInitCalibConfig (int32_t argc, char* argv[])
{
    int32_t         retVal;
    rlRfInitCalConf_t rfInitCalCfg = { 0 };
    uint16_t          enCustomCalib;

    /* Sanity check on number of parameters */
    if(argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    
    enCustomCalib = (uint16_t)atoi(argv[1]);

    if (enCustomCalib == 1)
    {
        rfInitCalCfg.calibEnMask = (uint32_t)atoi(argv[2]);
    }
    else
    {
        rfInitCalCfg.calibEnMask = 0x1FF0;
    }

    /* store to Global structure */
    gCLICmdCfg.openCfg.customCalibrationEnableMask = rfInitCalCfg.calibEnMask;
    gCLICmdCfg.openCfg.useCustomCalibration = 1;


    /* Enable/Disable calibrations */
    retVal = rlRfInitCalibConfig(RL_DEVICE_MAP_INTERNAL_BSS, &rfInitCalCfg);

    //TODO JIT: how and where to integrate rf_init API.??

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to test RF Init/Calibration API.
 *      using the mmWave link API.
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MmwaveLink_rfCalibration (void)
{
    int32_t     retVal;
    extern uint32_t gInitTimeCalibStatus;

    /* RF Initialization */
    retVal = rlRfInit(RL_DEVICE_MAP_INTERNAL_BSS);
    if (retVal != 0)
    {
        System_printf ("Error: Unable to start RF [Error %d]\n", retVal);
        return retVal;
    }

    /* wait for Rf Init Async-Event message from BSS */
    while(gInitTimeCalibStatus == 0U)
    {
        // Sleep and poll again:
        Task_sleep(1);
    }
    gInitTimeCalibStatus = 0U;

    return 0;
}

int32_t MmwaveLink_devCfg(void)
{
    rlRfDevCfg_t devCfg = {0};
    return rlRfSetDeviceCfg(RL_DEVICE_MAP_INTERNAL_BSS, &devCfg);
}

/**
 *  @b Description
 *  @n
 *
 *      This is the CLI Handler for Calibration and monitor general configuraiton
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_CLICalibMonCfg (int32_t argc, char* argv[])
{
    uint32_t oneTimeCalibEnMask = 0, periodicCalibEnMask = 0;
    int32_t retVal = -1;

    rlRfCalMonTimeUntConf_t calMonTimeUnitConf =
    {
        .calibMonTimeUnit = 1,
        .numOfCascadeDev = 1,
        .devId = 1,
        .reserved = 0
    };

    rlRfCalMonFreqLimitConf_t freqLimit =
    {
#ifndef SOC_XWR68XX
        .freqLimitLow = 760,
        .freqLimitHigh = 810,
#else
        .freqLimitLow = 570,
        .freqLimitHigh = 640,
#endif
        .reserved0 = 0,
        .reserved1 = 0
    };

    rlRunTimeCalibConf_t runTimeCalibCfg =
    {
        .oneTimeCalibEnMask = 0x710,  /* Enable All Run time Calibration */
        .periodicCalibEnMask = 0x710, /* Enable All Run time Calibration */
        .calibPeriodicity = 5,
        .reportEn = 1,
        .txPowerCalMode = 0,
        .reserved1 = 0
    };

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignored: This command is not allowed after sensor has started */
        return 0;
    }

    /* Sanity check on number of parameters */
    if((argc != 3) && (argc != 4))
    {
        return -1;
    }

    /* Save in mmw demo , they are passed to mmwave control module during
      sensor open/start process
     */
    calMonTimeUnitConf.calibMonTimeUnit = (uint16_t)atoi(argv[1]);
    runTimeCalibCfg.calibPeriodicity = (uint32_t)atoi(argv[2]);

    if(argc == 4)
    {
        periodicCalibEnMask = (uint32_t)atoi(argv[3]);
        runTimeCalibCfg.periodicCalibEnMask = periodicCalibEnMask;
    }
    
    if(argc == 5)
    {
        oneTimeCalibEnMask = (uint32_t)atoi(argv[4]);
        runTimeCalibCfg.oneTimeCalibEnMask = oneTimeCalibEnMask;
    }

    /* store to Global structure */
    gCLICmdCfg.openCfg.calibMonTimeUnit = calMonTimeUnitConf.calibMonTimeUnit;
    gCLICmdCfg.runTimeCalibCfg = runTimeCalibCfg;

    if(calMonTimeUnitConf.calibMonTimeUnit > 0)
    {
        retVal = rlRfSetCalMonTimeUnitConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlRfCalMonTimeUntConf_t*)&calMonTimeUnitConf);
    }
    
    if(runTimeCalibCfg.calibPeriodicity > 0)
    {
        retVal = rlRfRunTimeCalibConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlRunTimeCalibConf_t*)&runTimeCalibCfg);
    }
    
    
    /* Set calib/monitoring freq limits configuration */
    retVal = rlRfSetCalMonFreqLimitConfig(RL_DEVICE_MAP_INTERNAL_BSS,
                                     (rlRfCalMonFreqLimitConf_t*)&freqLimit);
                                     
    return retVal;
}

/**************************************************************************
 **************************** CLI Functions *******************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is the HELP generated by the CLI Module
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not Applicable.
 */
static int32_t CLI_help (int32_t argc, char* argv[])
{
    uint32_t    index;

    /* Display the banner: */
    MmwDemo_CLI_write ("Help: This will display the usage of the CLI commands\n");
    MmwDemo_CLI_write ("Command: Help Description\n");

    /* Cycle through all the registered CLI commands: */
    for (index = 0; index < gCliMcb.numCLICommands; index++)
    {
        /* Display the help string*/
        MmwDemo_CLI_write ("%s: %s\n",
                    gCliMcb.cfg.tableEntry[index].cmd,
                   (gCliMcb.cfg.tableEntry[index].helpString == NULL) ?
                    "No help available" :
                    gCliMcb.cfg.tableEntry[index].helpString);
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the version command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveVersion (int32_t argc, char* argv[])
{
    rlVersion_t    verArgs;
    int32_t        retVal;
    uint8_t        rfPatchBuildVer, rfPatchDebugVer;
    rlRfDieIdCfg_t  dieId = { 0 };
    SOC_PartNumber  devicePartNumber;
    uint8_t         index;
    int32_t         errCode;

    if(gCliMcb.cfg.overridePlatform == false)
    {
        /* print the platform */
#ifdef SOC_XWR14XX
        MmwDemo_CLI_write ("Platform                : xWR14xx\n");
#elif defined(SOC_XWR16XX)              
        MmwDemo_CLI_write ("Platform                : xWR16xx\n");
#elif defined(SOC_XWR18XX)              
        MmwDemo_CLI_write ("Platform                : xWR18xx\n");
#elif defined(SOC_XWR68XX)
        MmwDemo_CLI_write ("Platform                : xWR68xx\n");
#else                                   
        MmwDemo_CLI_write ("Platform                : unknown\n");
#endif
    }
    else
    {
        MmwDemo_CLI_write ("Platform                : %s\n", gCliMcb.cfg.overridePlatformString);
    }

    /* Get the version string: */
    retVal = SOC_getDevicePartNumber(gCliMcb.cfg.socHandle, &devicePartNumber, &errCode);
    if (retVal < 0)
    {
        MmwDemo_CLI_write ("Error: Unable to get the device version from SOC[Error %d]\n", errCode);
        return -1;
    }

    /* Get the version string: */
    retVal = rlDeviceGetVersion(RL_DEVICE_MAP_INTERNAL_BSS, &verArgs);
    if (retVal < 0)
    {
        MmwDemo_CLI_write ("Error: Unable to get the device version from mmWave link [Error %d]\n", retVal);
        return -1;
    }

    /* Display the version information on the CLI Console: */
    MmwDemo_CLI_write ("mmWave SDK Version      : %02d.%02d.%02d.%02d\n",
                            MMWAVE_SDK_VERSION_MAJOR,
                            MMWAVE_SDK_VERSION_MINOR,
                            MMWAVE_SDK_VERSION_BUGFIX,
                            MMWAVE_SDK_VERSION_BUILD);
    index = 0;
    while(partInfoStringArray[index].partNumber !=0U)
    {
        if(devicePartNumber == partInfoStringArray[index].partNumber)
        {
            /* found device in supported list */
            break;
        }
        index++;
    }
    if(partInfoStringArray[index].partNumber ==0U)
    {
        /* Device is not found in the list: */
        /* Display the device information on the CLI Console: */
        MmwDemo_CLI_write ("Device Info             : Device#%x ES %02d.%02d\n",
                   devicePartNumber,verArgs.rf.hwMajor, verArgs.rf.hwMinor);
    }
    else
    {
        /* Display the device information on the CLI Console: */
        MmwDemo_CLI_write ("Device Info             : %s ES %02d.%02d\n",
                   partInfoStringArray[index].partNumString,verArgs.rf.hwMajor, verArgs.rf.hwMinor);
    }

    MmwDemo_CLI_write ("RF F/W Version          : %02d.%02d.%02d.%02d.%02d.%02d.%02d\n",
                verArgs.rf.fwMajor, verArgs.rf.fwMinor, verArgs.rf.fwBuild, verArgs.rf.fwDebug,
                verArgs.rf.fwYear, verArgs.rf.fwMonth, verArgs.rf.fwDay);
                                
    rfPatchDebugVer = ((verArgs.rf.patchBuildDebug) & 0x0F);
    rfPatchBuildVer = (((verArgs.rf.patchBuildDebug) & 0xF0) >> 4);
    
    MmwDemo_CLI_write ("RF F/W Patch            : %02d.%02d.%02d.%02d.%02d.%02d.%02d\n",
                verArgs.rf.patchMajor, verArgs.rf.patchMinor, rfPatchBuildVer, rfPatchDebugVer,
                verArgs.rf.patchYear, verArgs.rf.patchMonth, verArgs.rf.patchDay);            
    MmwDemo_CLI_write ("mmWaveLink Version      : %02d.%02d.%02d.%02d\n",
                verArgs.mmWaveLink.major, verArgs.mmWaveLink.minor,
                verArgs.mmWaveLink.build, verArgs.mmWaveLink.debug);
                
    /* Get the die ID: */
    retVal = rlGetRfDieId(RL_DEVICE_MAP_INTERNAL_BSS, &dieId);
    if (retVal < 0)
    {
        MmwDemo_CLI_write ("Error: Unable to get the device die ID from mmWave link [Error %d]\n", retVal);
        return 0;
    }

    MmwDemo_CLI_write ("Lot number              : %d\n", dieId.lotNo);
    MmwDemo_CLI_write ("Wafer number            : %d\n", dieId.waferNo);
    MmwDemo_CLI_write ("Die coordinates in wafer: X = %d, Y = %d\n",
               dieId.devX, dieId.devY);

    /* Version string has been formatted successfully. */
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the flush configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveFlushCfg (int32_t argc, char* argv[])
{
    /* Reset the open configuration: */
    memset ((void*)&gCLICmdCfg, 0, sizeof(MMWave_OpenCfg));

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the DFE Data Output mode.
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveDataOutputMode (int32_t argc, char* argv[])
{
    uint32_t cfgMode;

    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Get the configuration mode: */
    cfgMode = atoi (argv[1]);
    switch (cfgMode)
    {
        case 1U:
        {
            /* store to Global structure */
            gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_FRAME;
            break;
        }
        case 2U:
        {
            gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_CONTINUOUS;
            break;
        }
        case 3U:
        {
            gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode = MMWave_DFEDataOutputMode_ADVANCED_FRAME;
            break;
        }
        default:
        {
            /* Error: Invalid argument. */
            /* Error: Invalid mode. */
            return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
        }
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the channel configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveChannelCfg (int32_t argc, char* argv[])
{
    rlChanCfg_t     chCfg = {0};
    int32_t         retVal = -1;
    
    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Populate the channel configuration: */
    chCfg.rxChannelEn = atoi (argv[1]);
    chCfg.txChannelEn = atoi (argv[2]);
    chCfg.cascading   = atoi (argv[3]);

    /* Save Configuration to use later */
    memcpy((void *)&gCLICmdCfg.openCfg.chCfg, (void *)&chCfg, sizeof(rlChanCfg_t));
    
    /* Set channel configuration */
    retVal = rlSetChannelConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlChanCfg_t*)&chCfg);
    
    if(retVal == 0)
    {
        extern volatile char isRfInitCmd;
        extern Semaphore_Handle  gMonReportStreamSem;
        isRfInitCmd = 1;
        //Trigger rfinit
        Semaphore_post(gMonReportStreamSem);
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the ADC configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveADCCfg (int32_t argc, char* argv[])
{
    rlAdcOutCfg_t   adcOutCfg = {0};
    int32_t         retVal = -1;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Populate the ADC Output configuration: */
    adcOutCfg.fmt.b2AdcBits   = atoi (argv[1]);
    adcOutCfg.fmt.b2AdcOutFmt = atoi (argv[2]);

    /* Save Configuration to use later */
    memcpy((void *)&gCLICmdCfg.openCfg.adcOutCfg, (void *)&adcOutCfg, sizeof(rlAdcOutCfg_t));
    
    /* Set ADC out configuration */
    retVal = rlSetAdcOutConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlAdcOutCfg_t*)&adcOutCfg);
    
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the profile configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveProfileCfg (int32_t argc, char* argv[])
{
    rlProfileCfg_t          profileCfg = {0};
    uint8_t                 index;
    int32_t                 retVal;

    /* Sanity Check: Minimum argument check */
    if (argc != 15)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Sanity Check: Profile configuration is valid only for the Frame or 
                     Advanced Frame Mode: */
    if ((gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME) &&
        (gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME))
    {
        /* Error: Configuration is valid only if the DFE Output Mode is Frame or Advanced Frame */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }
    
    index = atoi (argv[1]);
    /* profile ID out of max limit */
    if(index >= (RL_MAX_PROFILES_CNT))
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }
    /* Populate the profile configuration: */
    profileCfg.profileId             = index;

    /* Translate from GHz to [1 LSB = gCLI_mmwave_freq_scale_factor * 1e9 / 2^26 Hz] units
     * of mmwavelink format */
    profileCfg.startFreqConst        = (uint32_t) (atof(argv[2]) * (1U << 26) /
                                                   gCLI_mmwave_freq_scale_factor);

    /* Translate below times from us to [1 LSB = 10 ns] units of mmwavelink format */
    profileCfg.idleTimeConst         = (uint32_t)((float)atof(argv[3]) * 1000 / 10);
    profileCfg.adcStartTimeConst     = (uint32_t)((float)atof(argv[4]) * 1000 / 10);
    profileCfg.rampEndTime           = (uint32_t)((float)atof(argv[5]) * 1000 / 10);

    profileCfg.txOutPowerBackoffCode = atoi (argv[6]);
    profileCfg.txPhaseShifter        = atoi (argv[7]);

    /* Translate from MHz/us to [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e6 * 900) / 2^26 kHz/uS]
     * units of mmwavelink format */
    profileCfg.freqSlopeConst        = (int16_t)(atof(argv[8]) * (1U << 26) /
                                                 ((gCLI_mmwave_freq_scale_factor * 1e3) * 900.0));

    /* Translate from us to [1 LSB = 10 ns] units of mmwavelink format */
    profileCfg.txStartTime           = (int32_t)((float)atof(argv[9]) * 1000 / 10);

    profileCfg.numAdcSamples         = atoi (argv[10]);
    profileCfg.digOutSampleRate      = atoi (argv[11]);
    profileCfg.hpfCornerFreq1        = atoi (argv[12]);
    profileCfg.hpfCornerFreq2        = atoi (argv[13]);
    profileCfg.rxGain                = atoi (argv[14]);


    if (index == MMWAVE_MAX_PROFILE)
    {
        /* Error: All the profiles have been exhausted */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* store to Global structure */
    gCLICmdCfg.profChirpCfg.profileCfg[index] = profileCfg;

    gCLICmdCfg.profChirpCfg.rcvProfileCfgCnt++;

    /* Set Profile configuration */
    retVal = rlSetProfileConfig(RL_DEVICE_MAP_INTERNAL_BSS, 1U, (rlProfileCfg_t*)&profileCfg);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the chirp configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveChirpCfg (int32_t argc, char* argv[])
{
    rlChirpCfg_t            chirpCfg = {0};
    int32_t                 retVal;
    /* Sanity Check: Minimum argument check */
    if (argc != 9)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Sanity Check: Chirp configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if ((gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME) &&
        (gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME))
    {
        /* Error: Configuration is valid only if the DFE Output Mode is Chirp */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Populate the chirp configuration: */
    chirpCfg.chirpStartIdx   = atoi (argv[1]);
    chirpCfg.chirpEndIdx     = atoi (argv[2]);
    chirpCfg.profileId       = atoi (argv[3]);

    /* Translate from Hz to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e9) / 2^26 Hz]
     * units of mmwavelink format */
    chirpCfg.startFreqVar    = (uint32_t) ((float)atof(argv[4]) * (1U << 26) /
                                            (gCLI_mmwave_freq_scale_factor * 1e9));

    /* Translate from KHz/us to number of [1 LSB = (gCLI_mmwave_freq_scale_factor * 1e6) * 900 /2^26 KHz/us]
     * units of mmwavelink format */
    chirpCfg.freqSlopeVar    = (uint16_t) ((float)atof(argv[5]) * (1U << 26) /
                                           ((gCLI_mmwave_freq_scale_factor * 1e6) * 900.0));

    /* Translate from us to [1 LSB = 10ns] units of mmwavelink format */
    chirpCfg.idleTimeVar     = (uint32_t)((float)atof (argv[6]) * 1000.0 / 10.0);

    /* Translate from us to [1 LSB = 10ns] units of mmwavelink format */
    chirpCfg.adcStartTimeVar = (uint32_t)((float)atof (argv[7]) * 1000.0 / 10.0);

    chirpCfg.txEnable        = atoi (argv[8]);

    if(gCLICmdCfg.profChirpCfg.rcvdChirpCfgCnt < MAX_CHIRP_CFG_STORED)
    {
        gCLICmdCfg.profChirpCfg.chirpCfg[gCLICmdCfg.profChirpCfg.rcvdChirpCfgCnt] = chirpCfg;
    }
    gCLICmdCfg.profChirpCfg.rcvdChirpCfgCnt++;
    /* Set chirp configuration */
    retVal = rlSetChirpConfig(RL_DEVICE_MAP_INTERNAL_BSS, 1U, (rlChirpCfg_t*)&chirpCfg);
    
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the frame configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveFrameCfg (int32_t argc, char* argv[])
{
    rlFrameCfg_t    frameCfg = {0};
    int32_t         retVal = -1;
    
    /* Sanity Check: Minimum argument check */
    if (argc != 8)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Sanity Check: Frame configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if (gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_FRAME)
    {
        /* Error: Configuration is valid only if the DFE Output Mode is Chirp */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Populate the frame configuration: */
    frameCfg.chirpStartIdx      = atoi (argv[1]);
    frameCfg.chirpEndIdx        = atoi (argv[2]);
    frameCfg.numLoops           = atoi (argv[3]);
    frameCfg.numFrames          = atoi (argv[4]);
    frameCfg.framePeriodicity   = (uint32_t)((float)atof(argv[5]) * 1000000 / 5);
    frameCfg.triggerSelect      = atoi (argv[6]);
    frameCfg.frameTriggerDelay  = (uint32_t)((float)atof(argv[7]) * 1000000 / 5);

    /* Save Configuration to use later */
    memcpy((void *)&gCLICmdCfg.cliCtrlCfg.u.frameCfg.frameCfg, (void *)&frameCfg, sizeof(rlFrameCfg_t)); //TODO JIT: may remove many global struct
    
    retVal = retVal = rlSetFrameConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlFrameCfg_t*)&frameCfg);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the advanced frame configuration command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveAdvFrameCfg (int32_t argc, char* argv[])
{
    rlAdvFrameCfg_t  advFrameCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 6)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    
    /* Sanity Check: Frame configuration is valid only for the Frame or
                     Advanced Frame Mode: */
    if (gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME)
    {
        /* Error: Configuration is valid only if the DFE Output Mode is Advanced Frame */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Initialize the frame configuration: */
    memset ((void *)&advFrameCfg, 0, sizeof(rlAdvFrameCfg_t));

    /* Populate the frame configuration: */
    advFrameCfg.frameSeq.numOfSubFrames      = atoi (argv[1]);
    advFrameCfg.frameSeq.forceProfile        = atoi (argv[2]);
    advFrameCfg.frameSeq.numFrames           = atoi (argv[3]);
    advFrameCfg.frameSeq.triggerSelect       = atoi (argv[4]);
    advFrameCfg.frameSeq.frameTrigDelay      = (uint32_t)((float)atof(argv[5]) * 1000000 / 5);

    /* Save Configuration to use later */
    memcpy ((void *)&gCLICmdCfg.cliCtrlCfg.u.advancedFrameCfg.frameCfg,
            (void *)&advFrameCfg, sizeof(rlAdvFrameCfg_t));
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the subframe configuration command.
 *      Only valid when used in conjunction with the advanced frame configuration.
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveSubFrameCfg (int32_t argc, char* argv[])
{
    rlSubFrameCfg_t  subFrameCfg = {0};
    uint8_t          subFrameNum;
    int32_t          retVal = -1;

    /* Sanity Check: Minimum argument check */
    if (argc != 11)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    
    /* Sanity Check: Sub Frame configuration is valid only for the Advanced Frame Mode: */
    if (gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_ADVANCED_FRAME)
    {
        /* Error: Configuration is valid only if the DFE Output Mode is Advanced Frame */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Initialize the frame configuration: */
    memset ((void *)&subFrameCfg, 0, sizeof(rlSubFrameCfg_t));

    /* Populate the frame configuration: */
    subFrameNum                                  = (uint8_t)atoi (argv[1]);
    if (subFrameNum > (gCLICmdCfg.cliCtrlCfg.u.advancedFrameCfg.frameCfg.frameSeq.numOfSubFrames-1))
    {
        /* Error: Invalid subframe number. */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }
    subFrameCfg.forceProfileIdx     = atoi (argv[2]);
    subFrameCfg.chirpStartIdx       = atoi (argv[3]);
    subFrameCfg.numOfChirps         = atoi (argv[4]);
    subFrameCfg.numLoops            = atoi (argv[5]);
    subFrameCfg.burstPeriodicity    = (uint32_t)((float)atof(argv[6]) * 1000000 / 5);
    subFrameCfg.chirpStartIdxOffset = atoi (argv[7]);
    subFrameCfg.numOfBurst          = atoi (argv[8]);
    subFrameCfg.numOfBurstLoops     = atoi (argv[9]);
    subFrameCfg.subFramePeriodicity = (uint32_t)((float)atof(argv[10]) * 1000000 / 5);
        /* Save Configuration to use later */
    memcpy((void *)&gCLICmdCfg.cliCtrlCfg.u.advancedFrameCfg.frameCfg.frameSeq.subFrameCfg[subFrameNum],
        (void *)&subFrameCfg, sizeof(rlSubFrameCfg_t));
        
    gCLICmdCfg.subFrameCfgRcvCnt++;
    
    if(gCLICmdCfg.subFrameCfgRcvCnt == gCLICmdCfg.cliCtrlCfg.u.advancedFrameCfg.frameCfg.frameSeq.numOfSubFrames)
    {
        /* invoke advFrameConfig API */
        /* Set advance frame configuration */
        retVal = rlSetAdvFrameConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlAdvFrameCfg_t*)&gCLICmdCfg.cliCtrlCfg.u.advancedFrameCfg.frameCfg);
    }
    else
    {
        retVal = 0;
    }    

    return retVal;
    
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the low power command
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveLowPowerCfg (int32_t argc, char* argv[])
{
    int32_t                 retVal = -1;
    rlLowPowerModeCfg_t     lowPowerCfg = {0};

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Populate the channel configuration: */
    lowPowerCfg.lpAdcMode     = atoi (argv[2]);

    /* Save Configuration to use later */
    memcpy((void *)&gCLICmdCfg.openCfg.lowPowerMode, (void *)&lowPowerCfg, sizeof(rlLowPowerModeCfg_t));

    retVal = rlSetLowPowerModeConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlLowPowerModeCfg_t*)&lowPowerCfg);
    
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for the continuous mode
 *
 *  @param[in] argc
 *      Number of arguments
 *  @param[in] argv
 *      Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MMWaveContModeCfg (int32_t argc, char* argv[])
{
    int32_t retVal = -1;
    rlContModeCfg_t contModeCfg = {0};
    
    /* Sanity Check: Minimum argument check */
    if (argc != 10)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Sanity Check: Continuous configuration is valid only for the Continuous Mode: */
    if (gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode != MMWave_DFEDataOutputMode_CONTINUOUS)
    {
        /* Error: Configuration is valid only if the DFE Output Mode is Continuous */
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Populate the configuration: */
    contModeCfg.startFreqConst        = (uint32_t) (atof(argv[1]) * (1U << 26) /
    gCLI_mmwave_freq_scale_factor);
    contModeCfg.txOutPowerBackoffCode = (uint32_t) atoi (argv[2]);
    contModeCfg.txPhaseShifter        = (uint32_t) atoi (argv[3]);
    contModeCfg.digOutSampleRate      = (uint16_t) atoi (argv[4]);
    contModeCfg.hpfCornerFreq1        = (uint8_t)  atoi (argv[5]);
    contModeCfg.hpfCornerFreq2        = (uint8_t)  atoi (argv[6]);
    contModeCfg.rxGain                = (uint16_t) atoi (argv[7]);
    /* reserved  (argv[8]); */
    gCLICmdCfg.cliCtrlCfg.u.continuousModeCfg.dataTransSize  = (uint16_t) atoi (argv[9]);
    
    gCLICmdCfg.cliCtrlCfg.u.continuousModeCfg.contModecfg = contModeCfg;

    /* Set continue mode configuration */
    retVal = rlSetContModeConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlContModeCfg_t*)&contModeCfg);
    
    
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      Function to Setup the HSI Clock. Required for LVDS streaming.
 *
 *  @retval
 *      0  - Success.
 *      <0 - Failed with errors
 */
static int32_t Mmwdemo_CLIHsiClkCfg(int32_t argc, char* argv[])
{
    int32_t  retVal;
    rlDevHsiClk_t       hsiClkgs;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignored: This command is not allowed after sensor has started */
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Initialize configuration for HSI clock configuration */
    memset ((void *)&hsiClkgs, 0, sizeof(rlDevHsiClk_t));

    hsiClkgs.hsiClk = (uint16_t)atoi(argv[1]);

	/* if gSysValRFCfg.hsiClkCfg.hsiClk is not configured from cfg file, default is  9*/
	if (hsiClkgs.hsiClk == 0)
		hsiClkgs.hsiClk = 9;
	
    gCLICmdCfg.hsiClkCfg = hsiClkgs;

    /* Setup the HSI in the radar link: */
    retVal = rlDeviceSetHsiClk(RL_DEVICE_MAP_CASCADED_1, &hsiClkgs);

    /*The delay below is needed only if the DCA1000EVM is being used to capture the data traces.
      This is needed because the DCA1000EVM FPGA needs the delay to lock to the
      bit clock before they can start capturing the data correctly. */
    Task_sleep(HSI_DCA_MIN_DELAY_MSEC);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      Function to configure LDO bypass.
 *
 *  @retval
 *      0  - Success.
 *      <0 - Failed with errors
 */
static int32_t MmwDemo_CLILdoBypassCfg(int32_t argc, char* argv[])
{
    int32_t            retVal;
    /**
     * @brief
     *  Global Variable for LDO BYPASS config   
     **********************************************************
      AS PER RECOMMENDATION FROM THE xWR1843BOOST EVM User guide
     **********************************************************
       set LDO bypass since the board has 1.0V RF supply 1 
       and 1.0V RF supply 2. Please update this API if this 
       assumption is not valid else it may DAMAGE your board!
     **********************************************************
     **********************************************************
     */
    rlRfLdoBypassCfg_t rfLdoBypassCfg =
    {
        .ldoBypassEnable   = 3, /* 1.0V RF supply 1 and 1.0V RF supply 2 */
        .supplyMonIrDrop   = 1, /* IR drop of 3% */
        .ioSupplyIndicator = 0, /* 3.3 V IO supply */
    };

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        /* Ignored: This command is not allowed after sensor has started */
        return MMW_CLI_ERROR_CMD_AFTER_SENSOR_START;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        /* Error: Invalid usage of the CLI command */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Populate configuration: */
    if((uint8_t)atoi(argv[1]) != 0)
    {
        rfLdoBypassCfg.ldoBypassEnable = CSL_FMKR (0U, 0U, 1U);
    }
    if((uint8_t)atoi(argv[2]) != 0)
    {
        rfLdoBypassCfg.ldoBypassEnable |= CSL_FMKR (1U, 1U, 1U);
    }
    if((uint8_t)atoi(argv[3]) > 3)
    {
        /* Supported supplyMonIrDrop value is 0-3 */
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    rfLdoBypassCfg.supplyMonIrDrop     = (uint8_t)atoi(argv[3]);
    rfLdoBypassCfg.ioSupplyIndicator   = (uint8_t) 0; /* Hard-code to 0, since only 3.3V is supported */

    gCLICmdCfg.ldoBypassCfg = rfLdoBypassCfg;

    /* Setup the HSI in the radar link: */
    retVal = rlRfSetLdoBypassConfig(RL_DEVICE_MAP_CASCADED_1, &rfLdoBypassCfg);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Execution Task
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not Applicable.
 */
uint8_t                 rcvCmd[256];
static void CLI_task(UArg arg0, UArg arg1)
{
    uint8_t                 cmdString[256];
    char*                   tokenizedArgs[MMW_CLI_MAX_ARGS];
    char*                   ptrCLICommand;
    char                    delimitter[] = " \r\n";
    uint32_t                argIndex;
    CLI_CmdTableEntry*      ptrCLICommandEntry;
    int32_t                 cliStatus;
    uint32_t                index;
    int32_t                 retVal;

    /* Do we have a banner to be displayed? */
    if (gCliMcb.cfg.cliBanner != NULL)
    {
        /* YES: Display the banner */
        MmwDemo_CLI_write (gCliMcb.cfg.cliBanner);
    }

    /* Loop around forever: */
    while (1)
    {
        /* Demo Prompt: */
        MmwDemo_CLI_write (gCliMcb.cfg.cliPrompt);

        /* Reset the command string: */
        memset ((void *)&cmdString[0], 0, sizeof(cmdString));

        /* Read the command message from the UART: */
        retVal = UART_read (gCliMcb.cfg.cliUartHandle, &cmdString[0], (sizeof(cmdString) - 1));

        memcpy(rcvCmd, cmdString, retVal);
#ifdef DEBUG_PC_TOOL        
        UART_writePolling (gMmwMssMCB.loggingUartHandle,
                           (uint8_t*)&cmdString[0],
                           (sizeof(cmdString) - 1));
        UART_writePolling (gMmwMssMCB.loggingUartHandle,
                           (uint8_t*)"\r",
                           1);
#endif
        /* Reset all the tokenized arguments: */
        memset ((void *)&tokenizedArgs, 0, sizeof(tokenizedArgs));
        argIndex      = 0;
        ptrCLICommand = (char*)&cmdString[0];

        /* comment lines found - ignore the whole line*/
        if (cmdString[0]=='%') {
            MmwDemo_CLI_write ("Skipped\n");
            continue;
        }

        /* Set the CLI status: */
        cliStatus = -1;

        /* The command has been entered we now tokenize the command message */
        while (1)
        {
            /* Tokenize the arguments: */
            tokenizedArgs[argIndex] = strtok(ptrCLICommand, delimitter);
            if (tokenizedArgs[argIndex] == NULL)
                break;

            /* Increment the argument index: */
            argIndex++;
            if (argIndex >= MMW_CLI_MAX_ARGS)
                break;

            /* Reset the command string */
            ptrCLICommand = NULL;
        }

        /* Were we able to tokenize the CLI command? */
        if (argIndex == 0)
            continue;

        /* Cycle through all the registered CLI commands: */
        for (index = 0; index < gCliMcb.numCLICommands; index++)
        {
            ptrCLICommandEntry = &gCliMcb.cfg.tableEntry[index];

            /* Do we have a match? */
            if (strcmp(ptrCLICommandEntry->cmd, tokenizedArgs[0]) == 0)
            {
                /* YES: Pass this to the CLI registered function */
                cliStatus = ptrCLICommandEntry->cmdHandlerFxn (argIndex, tokenizedArgs);
                if (cliStatus == 0)
                {
                    MmwDemo_CLI_write ("Done\n");
                }
                else
                {
                    MmwDemo_CLI_write ("Error %d\n", cliStatus);
                }
                break;
            }
        }

        /* Did we get a matching CLI command? */
        if (index == gCliMcb.numCLICommands)
        {
            /* NO matching command found. Is the mmWave extension enabled? */
            if (gCliMcb.cfg.enableMMWaveExtension == 1U)
            {
                /* Yes: Pass this to the mmWave extension handler */
                cliStatus = CLI_RFMonitorExtensionHandler (argIndex, tokenizedArgs);
                if (cliStatus == 0)
                {
                    MmwDemo_CLI_write ("Done\n");
                }
                else
                {
                    MmwDemo_CLI_write ("Error %d\n", cliStatus);
                }
            }

            /* Was the CLI command found? */
            if (cliStatus == -1)
            {
                /* No: The command was still not found */
                MmwDemo_CLI_write ("Error %d\n", MMW_CLI_ERROR_CODE_INVALID_COMMAND);
            }
        }
    }
}

/**
 *  @b Description
 *  @n
 *      Logging function which can log the messages to the CLI console
 *
 *  @param[in]  format
 *      Format string
 *
 *  \ingroup CLI_UTIL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_CLI_write (const char* format, ...)
{
    va_list     arg;
    char        logMessage[256];
    int32_t     sizeMessage;

    /* Format the message: */
    va_start (arg, format);
    sizeMessage = vsnprintf (&logMessage[0], sizeof(logMessage), format, arg);
    va_end (arg);

    /* If MmwDemo_CLI_write is called before CLI init has happened, return */
    if (gCliMcb.cfg.cliUartHandle == NULL)
    {
        return;        
    }
    
    /* Log the message on the UART CLI console: */
    if (gCliMcb.cfg.usePolledMode == true)
    {
        /* Polled mode: */
        UART_writePolling (gCliMcb.cfg.cliUartHandle, (uint8_t*)&logMessage[0], sizeMessage);
    }
    else
    {
        /* Blocking Mode: */
        UART_write (gCliMcb.cfg.cliUartHandle, (uint8_t*)&logMessage[0], sizeMessage);
    }
}

/**
 *  @b Description
 *  @n
 *      This is the function which is used to initialize and setup the CLI
 *
 *  @param[in]  ptrCLICfg
 *      Pointer to the CLI configuration
 *
 *  \ingroup CLI_UTIL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MmwDemo_CLI_open (CLI_Cfg* ptrCLICfg)
{
    Task_Params     taskParams;
    uint32_t        index;

    /* Sanity Check: Validate the arguments */
    if (ptrCLICfg == NULL)
        return -1;

    /* Cycle through and determine the number of supported CLI commands: */
    for (index = 0; index < CLI_MAX_CMD; index++)
    {
        /* Do we have a valid entry? */
        if (gCliMcb.cfg.tableEntry[index].cmd == NULL)
        {
            /* NO: This is the last entry */
            break;
        }
        else
        {
            /* YES: Increment the number of CLI commands */
            gCliMcb.numCLICommands = gCliMcb.numCLICommands + 1;
        }
    }

    /* Do we have a CLI Prompt specified?  */
    if (gCliMcb.cfg.cliPrompt == NULL)
        gCliMcb.cfg.cliPrompt = "CLI:/>";

    /* The CLI provides a help command by default:
     * - Since we are adding this at the end of the table; a user of this module can also
     *   override this to provide its own implementation. */
    gCliMcb.cfg.tableEntry[gCliMcb.numCLICommands].cmd           = "help";
    gCliMcb.cfg.tableEntry[gCliMcb.numCLICommands].helpString    = NULL;
    gCliMcb.cfg.tableEntry[gCliMcb.numCLICommands].cmdHandlerFxn = CLI_help;

    /* Increment the number of CLI commands: */
    gCliMcb.numCLICommands++;

    /* Initialize Monitor configurations with global settings */
    RFMon_initCfg();
    
    /* Initialize the task parameters and launch the CLI Task: */
    Task_Params_init(&taskParams);
    taskParams.priority  = gCliMcb.cfg.taskPriority;
    taskParams.stackSize = 4*1024;
    gCliMcb.cliTaskHandle = Task_create(CLI_task, &taskParams, NULL);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the function which is used to close the CLI module
 *
 *  \ingroup CLI_UTIL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MmwDemo_CLI_close (void)
{
    /* Shutdown the CLI Task */
    Task_delete(&gCliMcb.cliTaskHandle);

    /* Cleanup the memory */
    memset ((void*)&gCliMcb, 0, sizeof(CLI_MCB));
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Execution Task
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_CLIInit (uint8_t taskPriority)
{
    char        demoBanner[256];
    uint32_t    cnt;
    int32_t errCode;

    /* initialize */
    memset((void*)&gCLICmdCfg, 0, sizeof(MMW_CLI_OpenCfg));

    /* Create Demo Banner to be printed out by CLI */
    sprintf(&demoBanner[0], 
                       "******************************************\n" \
                       "xWR18xx MMW Demo %02d.%02d.%02d.%02d\n"  \
                       "******************************************\n", 
                        MMWAVE_SDK_VERSION_MAJOR,
                        MMWAVE_SDK_VERSION_MINOR,
                        MMWAVE_SDK_VERSION_BUGFIX,
                        MMWAVE_SDK_VERSION_BUILD
            );

    /* Initialize the CLI configuration: */
    memset ((void *)&gCliMcb.cfg, 0, sizeof(CLI_Cfg));

    gCLI_mmwave_freq_scale_factor = SOC_getDeviceRFFreqScaleFactor(gMmwMssMCB.socHandle, &errCode);
    
    /* Populate the CLI configuration: */
    gCliMcb.cfg.cliPrompt                    = "mmwDemo:/>";
    gCliMcb.cfg.cliBanner                    = demoBanner;
    gCliMcb.cfg.cliUartHandle                = gMmwMssMCB.commandUartHandle;
    gCliMcb.cfg.taskPriority                 = taskPriority;
    gCliMcb.cfg.socHandle                    = gMmwMssMCB.socHandle;
    gCliMcb.cfg.mmWaveHandle                 = gMmwMssMCB.ctrlHandle;
    gCliMcb.cfg.enableMMWaveExtension        = 1U;
    gCliMcb.cfg.usePolledMode                = true;
    gCliMcb.cfg.overridePlatform             = false;
    gCliMcb.cfg.overridePlatformString       = NULL;

    cnt=0;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "version";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "No arguments";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveVersion;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "flushCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "No arguments";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveFlushCfg;
    cnt++;
    
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "dfeDataOutputMode";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<modeType>   1-Chirp and 2-Continuous";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveDataOutputMode;
    cnt++;
    

    gCliMcb.cfg.tableEntry[cnt].cmd            = "channelCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<rxChannelEn> <txChannelEn> <cascading>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveChannelCfg;
    cnt++;
     
    gCliMcb.cfg.tableEntry[cnt].cmd            = "adcCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<numADCBits> <adcOutputFmt>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveADCCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "profileCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<profileId> <startFreq> <idleTime> <adcStartTime> <rampEndTime> <txOutPower> <txPhaseShifter> <freqSlopeConst> <txStartTime> <numAdcSamples> <digOutSampleRate> <hpfCornerFreq1> <hpfCornerFreq2> <rxGain>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveProfileCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "chirpCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<startIdx> <endIdx> <profileId> <startFreqVar> <freqSlopeVar> <idleTimeVar> <adcStartTimeVar> <txEnable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveChirpCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "frameCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<chirpStartIdx> <chirpEndIdx> <numLoops> <numFrames> <framePeriodicity> <triggerSelect> <frameTriggerDelay>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveFrameCfg;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "advFrameCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<numOfSubFrames> <forceProfile> <numFrames> <triggerSelect> <frameTrigDelay>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveAdvFrameCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "subFrameCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<subFrameNum> <forceProfileIdx> <chirpStartIdx> <numOfChirps> <numLoops> <burstPeriodicity> <chirpStartIdxOffset> <numOfBurst> <numOfBurstLoops> <subFramePeriodicity>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveSubFrameCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "lowPower";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<reserved> <lpAdcMode>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveLowPowerCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "contModeCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<startFreq> <txOutPower> <txPhaseShifter> <digOutSampleRate> <hpfCornerFreq1> <hpfCornerFreq2> <rxGain> <vcoSelect> <numSamples>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_MMWaveContModeCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "sensorStart";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "[doReconfig(optional, default:enabled)]";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLISensorStart;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "sensorStop";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "No arguments";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLISensorStop;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "guiMonitor";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<subFrameIdx> <detectedObjects> <logMagRange> <noiseProfile> <rangeAzimuthHeatMap> <rangeDopplerHeatMap> <statsInfo>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGuiMonSel;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "adcbufCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<subFrameIdx> <adcOutputFmt> <SampleSwap> <ChanInterleave> <ChirpThreshold>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIADCBufCfg;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "CQRxSatMonitor";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<profile> <satMonSel> <priSliceDuration> <numSlices> <rxChanMask>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIChirpQualityRxSatMonCfg;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "CQSigImgMonitor";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<profile> <numSlices> <numSamplePerSlice>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIChirpQualitySigImgMonCfg;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "lvdsStreamCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<subFrameIdx> <enableHeader> <dataFmt> <enableSW>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLILvdsStreamCfg;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "ldoBypass";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<RF LDO bypass> <PA LDO bypass> <supplyMonIrDrop>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLILdoBypassCfg;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "calibConfig";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<Enable Custom Mask> <calibEnMask>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_RfInitCalibConfig;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "hsiClkCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<Clock rate code>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = Mmwdemo_CLIHsiClkCfg;
    cnt++;

    gCliMcb.cfg.tableEntry[cnt].cmd            = "calibMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<calibMonTimeUnit> <calibPeriodicity> <periodicCalibEnMask>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICalibMonCfg;
    cnt++;
        
    gCliMcb.cfg.tableEntry[cnt].cmd            = "gpadcSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "tempMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <tempDiffThresh>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "extAnaSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "txPowerMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <txAnt> <profile index>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "txBallbreakMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <txAnt> ";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "rxGainPhaseMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <profile> <rxGainMismatch array> <rxGainPhaseMismatch array>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "synthFreqMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <profile index>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "pllConVoltMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "dualClkCompMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "rxIfStageMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <profile index>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "extAnaSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "pmClkSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <profile index>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "rxIntAnaSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <profile index>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "gpadcSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    
    gCliMcb.cfg.tableEntry[cnt].cmd            = "txIntAnaSigMonCfg";
    gCliMcb.cfg.tableEntry[cnt].helpString     = "<enable> <txAnt> <profile index>";
    gCliMcb.cfg.tableEntry[cnt].cmdHandlerFxn  = CLI_RFMonitorExtensionHandler;
    cnt++;
    
    /* Open the CLI: */
    if (MmwDemo_CLI_open (&gCliMcb.cfg) < 0)
    {
        System_printf ("Error: Unable to open the CLI\n");
        return;
    }
    System_printf ("Debug: CLI is operational\n");
    return;
}




