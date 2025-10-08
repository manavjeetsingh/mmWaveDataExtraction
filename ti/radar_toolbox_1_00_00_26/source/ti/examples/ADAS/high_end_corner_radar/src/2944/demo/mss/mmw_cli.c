/*
 *   @file  mmw_cli.c
 *
 *   @brief
 *      Mmw (Milli-meter wave) DEMO CLI Implementation
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
#include <stdbool.h>

/* MCU + SDK Include Files: */
#include <drivers/uart.h>
/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
// #include <ti/drv/uart/UART.h>
#include <ti/control/mmwavelink/mmwavelink.h>
#include <ti/utils/cli/cli.h>
#include <ti/utils/mathutils/mathutils.h>
#include <ti/control/dpm/dpm.h>

/* Demo Include Files */
#include "../include/mmw_config.h"
#include "mmw_mss.h"
#include "../utils/mmwdemo_adcconfig.h"
#include "../utils/mmwdemo_rfparser.h"
#define GTRACK_3D
#include "../../alg/gtrack/gtrack.h"

/**************************************************************************
 *************************** Local function prototype****************************
 **************************************************************************/

/* CLI Extended Command Functions */
static int32_t MmwDemo_CLICfarCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLICompressionCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLILocalMaxCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIIntfMitigCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIAntennaCalibParams (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLISensorStart (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLISensorStop (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIGuiMonSel (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIADCBufCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIAnalogMonitorCfg (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIConfigDataPort (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIRansac (int32_t argc, char* argv[]);
static int32_t MmwDemo_CLIGtrackCfg (int32_t argc, char* argv[]);

/**************************************************************************
 *************************** Extern Definitions *******************************
 **************************************************************************/

//redefinition of CLI_MCB
typedef struct CLI_MCB_t
{
    /**
     * @brief   Configuration which was used to configure the CLI module
     */
    CLI_Cfg         cfg;

    /**
     * @brief   This is the number of CLI commands which have been added to the module
     */
    uint32_t        numCLICommands;

    /**
     * @brief   CLI Task Handle:
     */
    TaskHandle_t     cliTaskHandle;
}CLI_MCB;

extern MmwDemo_MSS_MCB    gMmwMssMCB;
extern uint8_t gRangeCfarEnable;
extern CLI_MCB     gCLI;
extern int32_t CLI_MMWaveExtensionHandler(int32_t argc, char* argv[]);

int32_t MmwDemo_DPM_ioctl_blocking
(
    DPM_Handle handle,
    uint32_t cmd,
    void* arg,
    uint32_t argLen
);

/**************************************************************************
 *************************** Local Definitions ****************************
 **************************************************************************/

#define MMWDEMO_DATAUART_MAX_BAUDRATE_SUPPORTED 3125000

#define CLI_BYPASS

/**************************************************************************
 *************************** Global Definitions ****************************
 **************************************************************************/ 
// #define MAX_RADAR_CMD               27
// char *gHardcodedCliConfigs[MAX_RADAR_CMD] = {
// "sensorStop",
// "flushCfg",
// "dfeDataOutputMode 1", 
// "channelCfg 15 15 0",
// "adcCfg 2 0",
// "adcbufCfg -1 1 0 0 1",
// "lowPower 0 0",
// "profileCfg 0 77 5 5 18.81 0 0 8.883 0 384 30000 0 0 42",
// "chirpCfg 0 5 0 0 0 0 0 15",
// "frameCfg 0 5 128 0 384 250 1 0",
// "guiMonitor -1 3 0 0 0 0 0",
// "cfarCfg -1 1 3 16 0 0 1 12.0 0 24 0",
// "cfarCfg -1 0 3 16 0 0 1 15.0 0 7 0",
// "compressionCfg -1 1 0 0.5 8",
// "intfMitigCfg -1 15 18",
// "localMaxCfg -1 4 40",
// "antennaCalibParams 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1",
// "analogMonitor 0 0",
// "calibData 0 0 0",
// "appSceneryParams 0.0 0.0 1.0 0.0 0.0 -14.0 14.0 1.0 231.0 -5.0 5.0",
// "appGatingParams 8.0 10.0 8.0 4.0 40.0",
// "appStateParams 3 2 10 100 10 0",
// "appAllocParams 20.0 20.0 1.0 6 4.0 20.0",
// "gtrack 1 800 30 0.0 40.8781 0.1065 1.0 10.0 0.0 0.250 0",
// "ransac 1 200 1.0",
// "rangeCfar 0",
// "sensorStart"
// };


#define MAX_RADAR_CMD               27
char *gHardcodedCliConfigs[MAX_RADAR_CMD] = {
"sensorStop",
"flushCfg",
"dfeDataOutputMode 1",
"channelCfg 15 15 0",
"adcCfg 2 0",
"adcbufCfg -1 1 0 0 1",
"lowPower 0 0",
"profileCfg 0 77 5 5 18.81 0 0 8.883 0 384 30000 0 0 42",
"chirpCfg 0 5 0 0 0 0 0 15",
"frameCfg 0 5 128 0 384 250 1 0",
"guiMonitor -1 1 0 0 0 0 0",
"cfarCfg -1 1 3 16 0 0 1 12.0 0 24 0",
"cfarCfg -1 0 3 16 0 0 1 15.0 0 7 0",
"compressionCfg -1 1 0 0.5 8",
"intfMitigCfg -1 15 18",
"localMaxCfg -1 4 40",
"antennaCalibParams 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1",
"analogMonitor 0 0",
"calibData 0 0 0",
"appSceneryParams 0.0 0.0 1.0 -25.0 0.0 -14.0 14.0 1.0 231.0 -5.0 5.0",
"appGatingParams 8.0 10.0 8.0 4.0 40.0",
"appStateParams 3 2 10 100 10 0",
"appAllocParams 20.0 20.0 1.0 6 4.0 20.0",
"gtrack 1 800 30 0.0 40.8781 0.1065 1.0 10.0 0.0 0.250 0",
"ransac 1 200 1.0",
"rangeCfar 0",
"sensorStart",
};


bool gRansacEnabled = false;
uint16_t gRansacIterations = 0;
float gRansacThresh = 0.0;

//handle to gtrack
void *gHTrackModule;

//is gtrack created?
bool gGtrackInstanceCreated = false;
bool gGtrackEnabled = false;

// gtrack default configurations
GTRACK_sceneryParams gAppSceneryParams;// =
// {
//     /* sensor Position is (0,0,1) */
//     .sensorPosition.x = 0.f,
//     .sensorPosition.y = 0.f,
//     .sensorPosition.z = 1.0f,

//     /* Sensor orientation is (0 degrees down, 0 degrees azimuthal tilt */
//     .sensorOrientation.azimTilt = 0.0f,
//     .sensorOrientation.elevTilt = 0.f,

//     .numBoundaryBoxes = 1,

//     /* one boundary box {x1,x2,y1,y2,z1,z2} */
//     .boundaryBox[0].x1 = -14.0f,
//     .boundaryBox[0].x2 = 14.0f,
//     .boundaryBox[0].y1 = 1.f,
//     .boundaryBox[0].y2 = 231.f,
//     .boundaryBox[0].z1 = -5.f,
//     .boundaryBox[0].z2 = 5.0f,

//     .boundaryBox[1].x1 = 0.f,
//     .boundaryBox[1].x2 = 0.f,
//     .boundaryBox[1].y1 = 0.f,
//     .boundaryBox[1].y2 = 0.f,
//     .boundaryBox[1].z1 = 0.f,
//     .boundaryBox[1].z2 = 0.f,

//     .numStaticBoxes = 0,

//     /* one static box {x1,x2,y1,y2,z1,z2} */
//     .staticBox[0].x1 = 0.0f,
//     .staticBox[0].x2 = 0.0f,
//     .staticBox[0].y1 = 0.0f,
//     .staticBox[0].y2 = 0.0f,
//     .staticBox[0].z1 = 0.0f,
//     .staticBox[0].z2 = 0.0f,

//     .staticBox[1].x1 = 0.f,
//     .staticBox[1].x2 = 0.f,
//     .staticBox[1].y1 = 0.f,
//     .staticBox[1].y2 = 0.f,
//     .staticBox[1].z1 = 0.f,
//     .staticBox[1].z2 = 0.f,
// };

GTRACK_gatingParams gAppGatingParams;// =
// {
//     .gain = 8.f,

//     .limitsArray[0] = 10.f,
//     .limitsArray[1] = 8.f,
//     .limitsArray[2] = 4.f,
//     .limitsArray[3] = 40.f,
// };

GTRACK_stateParams gAppStateParams;// = {
//      3U, 2U, 10U, 100U, 10U, 0U              /* det2act, det2free, act2free, stat2free, exit2free sleep2freeThre */
// };

GTRACK_allocationParams gAppAllocationParams;// = {  //same number of elements as matlab
//      20.f, 20.f, 1.f, 6U, 4.f, 20.f           /* 60 in clear, 200 obscured SNRs, 0.1m/s minimal velocity, 5 points, 1.5m in distance, 2m/s in velocity */
// };

/**************************************************************************
 *************************** CLI  Function Definitions ********************
 **************************************************************************/
/**
 *  @b Description
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
            CLI_write ("Error: Reconfig is not supported, only argument of 0 is\n"
                       "(do not reconfig, just re-start the sensor) valid\n");
            return -1;
        }
    }
    else
    {
        /* In case there is no argument for sensorStart, always do reconfig */
        doReconfig = true;
    }

    /***********************************************************************************
     * Do sensor state management to influence the sensor actions
     ***********************************************************************************/

    /* Error checking initial state: no partial config is allowed 
       until the first sucessful sensor start state */
    if ((gMmwMssMCB.sensorState == MmwDemo_SensorState_INIT) || 
         (gMmwMssMCB.sensorState == MmwDemo_SensorState_OPENED))
    {
        MMWave_CtrlCfg ctrlCfg;

        /* need to get number of sub-frames so that next function to check
         * pending state can work */
        CLI_getMMWaveExtensionConfig (&ctrlCfg);
        gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames =
            MmwDemo_RFParser_getNumSubFrames(&ctrlCfg);

        //TODO
        // if (MmwDemo_isAllCfgInPendingState() == 0)
        // {
        //     CLI_write ("Error: Full configuration must be provided before sensor can be started "
        //                "the first time\n");

        //     /* Although not strictly needed, bring back to the initial value since we
        //      * are rejecting this first time configuration, prevents misleading debug. */
        //     gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames = 0;

        //     return -1;
        // }
    }

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: Sensor is already started\n");
        return 0;
    }

    if (doReconfig == false)
    {
        //TODO
        //  /* User intends to issue sensor start without config, check if no
        //     config was issued after stop and generate error if this is the case. */
        //  if (MmwDemo_isAllCfgInNonPendingState() == 0)
        //  {
        //      /* Message user differently if all config was issued or partial config was
        //         issued. */
        //      if (MmwDemo_isAllCfgInPendingState())
        //      {
        //          CLI_write ("Error: You have provided complete new configuration, "
        //                     "issue \"sensorStart\" (without argument) if you want it to "
        //                     "take effect\n");
        //      }
        //      else
        //      {
        //          CLI_write ("Error: You have provided partial configuration between stop and this "
        //                     "command and partial configuration cannot be undone."
        //                     "Issue the full configuration and do \"sensorStart\" \n");
        //      }
        //      return -1;
        //  }
    }
    else
    {
        /* User intends to issue sensor start with full config, check if all config
           was issued after stop and generate error if  is the case. */
        MMWave_CtrlCfg ctrlCfg;

        /* need to get number of sub-frames so that next function to check
         * pending state can work */
        CLI_getMMWaveExtensionConfig (&ctrlCfg);
        gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames =
            MmwDemo_RFParser_getNumSubFrames(&ctrlCfg);
        
        // if (MmwDemo_isAllCfgInPendingState() == 0)
        // {
        //     /* Message user differently if no config was issued or partial config was
        //        issued. */
        //     if (MmwDemo_isAllCfgInNonPendingState())
        //     {
        //         CLI_write ("Error: You have provided no configuration, "
        //                    "issue \"sensorStart 0\" OR provide "
        //                    "full configuration and issue \"sensorStart\"\n");
        //     }
        //     else
        //     {
        //         CLI_write ("Error: You have provided partial configuration between stop and this "
        //                    "command and partial configuration cannot be undone."
        //                    "Issue the full configuration and do \"sensorStart\" \n");
        //     }
        //     /* Although not strictly needed, bring back to the initial value since we
        //      * are rejecting this first time configuration, prevents misleading debug. */
        //     gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames = 0;
        //     return -1;
        // }
    }

    /***********************************************************************************
     * Retreive and check mmwave Open related config before calling openSensor
     ***********************************************************************************/

    /*  Fill demo's MCB mmWave openCfg structure from the CLI configs*/
    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_INIT)
    {
        /* Get the open configuration: */
        CLI_getMMWaveExtensionOpenConfig (&gMmwMssMCB.cfg.openCfg);
        /* call sensor open */
        retVal = MmwDemo_openSensor(true);
        if(retVal != 0)
        {
            return -1;
        }
        gMmwMssMCB.sensorState = MmwDemo_SensorState_OPENED;    
    }
    else
    {
        /* openCfg related configurations like chCfg, lowPowerMode, adcCfg
         * are only used on the first sensor start. If they are different
         * on a subsequent sensor start, then generate a fatal error
         * so the user does not think that the new (changed) configuration
         * takes effect, the board needs to be reboot for the new
         * configuration to be applied.
         */
        MMWave_OpenCfg openCfg;
        CLI_getMMWaveExtensionOpenConfig (&openCfg);
        /* Compare openCfg->chCfg*/
        if(memcmp((void *)&gMmwMssMCB.cfg.openCfg.chCfg, (void *)&openCfg.chCfg,
                          sizeof(rlChanCfg_t)) != 0)
        {
            MmwDemo_debugAssert(0);
        }
        
        /* Compare openCfg->lowPowerMode*/
        if(memcmp((void *)&gMmwMssMCB.cfg.openCfg.lowPowerMode, (void *)&openCfg.lowPowerMode,
                          sizeof(rlLowPowerModeCfg_t)) != 0)
        {
            MmwDemo_debugAssert(0);
        }
        /* Compare openCfg->adcOutCfg*/
        if(memcmp((void *)&gMmwMssMCB.cfg.openCfg.adcOutCfg, (void *)&openCfg.adcOutCfg,
                          sizeof(rlAdcOutCfg_t)) != 0)
        {
            MmwDemo_debugAssert(0);
        }
    }

    

    /***********************************************************************************
     * Retrieve mmwave Control related config before calling startSensor
     ***********************************************************************************/
    /* Get the mmWave ctrlCfg from the CLI mmWave Extension */
    if(doReconfig)
    {
        /* if MmwDemo_openSensor has non-first time related processing, call here again*/
        /* call sensor config */
        CLI_getMMWaveExtensionConfig (&gMmwMssMCB.cfg.ctrlCfg);
        retVal = MmwDemo_configSensor();
        if(retVal != 0)
        {
            return -1;
        }
    }
    retVal = MmwDemo_startSensor();
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
        CLI_write ("Ignored: Sensor is already stopped\n");
        return 0;
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
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /*Subframe info is always in position 1*/
    subframe = (int8_t) atoi(argv[1]);

    if(subframe >= (int8_t)RL_MAX_SUBFRAMES)
    {
        CLI_write ("Error: Subframe number is invalid\n");
        return -1;
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
 *      This is the CLI Handler for CFAR configuration
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
static int32_t MmwDemo_CLICfarCfg (int32_t argc, char* argv[])
{
    DPU_DopplerProc_CfarCfg   cfarCfg;
    uint32_t            procDirection;
    int8_t              subFrameNum;
    float               threshold;

    if(MmwDemo_CLIGetSubframe(argc, argv, 12, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cfarCfg, 0, sizeof(cfarCfg));

    /* Populate configuration: */
    procDirection             = (uint32_t) atoi (argv[2]);
    cfarCfg.averageMode       = (uint8_t) atoi (argv[3]);
    cfarCfg.winLen            = (uint8_t) atoi (argv[4]);
    cfarCfg.guardLen          = (uint8_t) atoi (argv[5]);
    cfarCfg.noiseDivShift     = (uint8_t) atoi (argv[6]);
    cfarCfg.cyclicMode        = (uint8_t) atoi (argv[7]);
    threshold                 = (float) atof (argv[8]);
    cfarCfg.peakGroupingEn    = (uint8_t) atoi (argv[9]);
    cfarCfg.osKvalue          = (uint8_t) atoi (argv[10]);
    cfarCfg.osEdgeKscaleEn    = (uint8_t) atoi (argv[11]);

    if (threshold > 100.0)
    {
        CLI_write("Error: Maximum value for CFAR thresholdScale is 100.0 dB.\n");
        return -1;
    }   
    
    /* threshold is a float value from 0-100dB. It needs to
       be later converted to linear scale (conversion can only be done
       when the number of virtual antennas is known) before passing it
       to CFAR DPU.
       For now, the threshold will be coded in a 16bit integer in the following
       way:
       suppose threshold is a float represented as XYZ.ABC
       it will be saved as a 16bit integer XYZAB       
       that is, 2 decimal cases are saved.*/
    threshold = threshold * MMWDEMO_CFAR_THRESHOLD_ENCODING_FACTOR;   
    cfarCfg.thresholdScale    = (uint16_t) threshold;
    
    /* Save Configuration to use later */     
    if (procDirection == 0)
    {
        MmwDemo_CfgUpdate((void *)&cfarCfg, MMWDEMO_CFARCFGRANGE_OFFSET,
                          sizeof(cfarCfg), subFrameNum);
    }
    else
    {
        MmwDemo_CfgUpdate((void *)&cfarCfg, MMWDEMO_CFARDOPPLERCFG_OFFSET,
                          sizeof(cfarCfg), subFrameNum);
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Compression configuration
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
static int32_t MmwDemo_CLICompressionCfg (int32_t argc, char* argv[])
{
    DPU_RangeProcHWA_CompressionCfg   compressionCfg;
    int8_t                            subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&compressionCfg, 0, sizeof(compressionCfg));

    /* Populate configuration: */
    compressionCfg.isEnabled              = (bool) atoi (argv[2]);
    compressionCfg.compressionMethod      = (uint16_t) atoi (argv[3]);
    compressionCfg.compressionRatio       = (float) atof (argv[4]);
    compressionCfg.rangeBinsPerBlock      = (uint16_t) atoi (argv[5]);
    /* rxAntennasPerBlock will be fixed to the number of Rx antennas */

    if (!((compressionCfg.rangeBinsPerBlock & (compressionCfg.rangeBinsPerBlock - 1)) == 0)) /* is it a power of 2? */
    {
        CLI_write("Error: rangeBinsPerBlock should be a power of 2 \n");
        return -1;
    }   
    
    MmwDemo_CfgUpdate((void *)&compressionCfg, MMWDEMO_COMPRESSIONCFG_OFFSET,
                          sizeof(compressionCfg), subFrameNum);
    
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Local Max configuration
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
static int32_t MmwDemo_CLILocalMaxCfg (int32_t argc, char* argv[])
{

    DPU_DopplerProc_LocalMaxCfg         localMaxCfg;
    int8_t                              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 4, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&localMaxCfg, 0, sizeof(localMaxCfg));

    /* Populate configuration: */
    localMaxCfg.azimThreshold                = (uint16_t) atoi (argv[2]);
    localMaxCfg.dopplerThreshold             = (uint16_t) atoi (argv[3]);
    
    MmwDemo_CfgUpdate((void *)&localMaxCfg, MMWDEMO_LOCALMAXCFG_OFFSET,
                          sizeof(localMaxCfg), subFrameNum);
    
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Interference Mitigation configuration
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
static int32_t MmwDemo_CLIIntfMitigCfg (int32_t argc, char* argv[])
{

    DPU_RangeProcHWADDMA_intfStatsdBCfg  intfStatsdBCfg;
    int8_t                               subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 4, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&intfStatsdBCfg, 0, sizeof(intfStatsdBCfg));

    /* Populate configuration: */
    intfStatsdBCfg.intfMitgMagSNRdB               = (uint32_t) atoi (argv[2]);
    intfStatsdBCfg.intfMitgMagDiffSNRdB           = (uint32_t) atoi (argv[3]);
    
    MmwDemo_CfgUpdate((void *)&intfStatsdBCfg, MMWDEMO_INTFMITIGCFG_OFFSET,
                          sizeof(intfStatsdBCfg), subFrameNum);
    
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Antenna Calibration configuration
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
static int32_t MmwDemo_CLIAntennaCalibParams (int32_t argc, char* argv[])
{

    /*! @brief      Antenna Calbration parameters in Im/Re format */
    float antennaCalibParams[SYS_COMMON_NUM_RX_CHANNEL * SYS_COMMON_NUM_TX_ANTENNAS * 2];
    int32_t argInd, i;

    /* Sanity Check: Minimum argument check */
    if (argc != (1 + SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL*2))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&antennaCalibParams, 0, sizeof(antennaCalibParams));

    argInd = 1;
    for (i = 0; i < SYS_COMMON_NUM_TX_ANTENNAS * SYS_COMMON_NUM_RX_CHANNEL * 2; i++)
    {
        antennaCalibParams[i] = (float) atof (argv[i+argInd]);
    }

    /* Save Configuration to use later */
    memcpy((void *) &gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.antennaCalibParams,
           &antennaCalibParams, sizeof(antennaCalibParams));

    gMmwMssMCB.objDetCommonCfg.isAntennaCalibParamCfgPending = 1;

    return 0;

}

#if 0
/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for CFAR FOV (Field Of View) configuration
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
static int32_t MmwDemo_CLICfarFovCfg (int32_t argc, char* argv[])
{
    
    DPU_CFARProc_FovCfg   fovCfg;
    uint32_t            procDirection;
    int8_t              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 5, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&fovCfg, 0, sizeof(fovCfg));

    /* Populate configuration: */
    procDirection             = (uint32_t) atoi (argv[2]);
    fovCfg.min                = (float) atof (argv[3]);
    fovCfg.max                = (float) atof (argv[4]);

    /* Save Configuration to use later */
    if (procDirection == 0)
    {
        MmwDemo_CfgUpdate((void *)&fovCfg, MMWDEMO_FOVRANGE_OFFSET,
                          sizeof(fovCfg), subFrameNum);
    }
    else
    {
        MmwDemo_CfgUpdate((void *)&fovCfg, MMWDEMO_FOVDOPPLER_OFFSET,
                          sizeof(fovCfg), subFrameNum);
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for AoA FOV (Field Of View) configuration
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
static int32_t MmwDemo_CLIAoAFovCfg (int32_t argc, char* argv[])
{
    DPU_AoAProc_FovAoaCfg   fovCfg;
    int8_t              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&fovCfg, 0, sizeof(fovCfg));

    /* Populate configuration: */
    fovCfg.minAzimuthDeg      = (float) atoi (argv[2]);
    fovCfg.maxAzimuthDeg      = (float) atoi (argv[3]);
    fovCfg.minElevationDeg    = (float) atoi (argv[4]);
    fovCfg.maxElevationDeg    = (float) atoi (argv[5]);

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&fovCfg, MMWDEMO_FOVAOA_OFFSET,
                      sizeof(fovCfg), subFrameNum);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for extended maximum velocity configuration
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
static int32_t MmwDemo_CLIExtendedMaxVelocity (int32_t argc, char* argv[])
{
    DPU_AoAProc_ExtendedMaxVelocityCfg   cfg;
    int8_t              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 3, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cfg, 0, sizeof(cfg));

    /* Populate configuration: */
    cfg.enabled      = (uint8_t) atoi (argv[2]);

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&cfg, MMWDEMO_EXTMAXVEL_OFFSET,
                      sizeof(cfg), subFrameNum);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for multi object beam forming configuration
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
static int32_t MmwDemo_CLIMultiObjBeamForming (int32_t argc, char* argv[])
{
    DPU_AoAProc_MultiObjBeamFormingCfg cfg;
    int8_t              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 4, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cfg, 0, sizeof(cfg));

    /* Populate configuration: */
    cfg.enabled                     = (uint8_t) atoi (argv[2]);
    cfg.multiPeakThrsScal           = (float) atof (argv[3]);

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&cfg, MMWDEMO_MULTIOBJBEAMFORMING_OFFSET,
                      sizeof(cfg), subFrameNum);

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for DC range calibration
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
static int32_t MmwDemo_CLICalibDcRangeSig (int32_t argc, char* argv[])
{
    DPU_RangeProc_CalibDcRangeSigCfg cfg;
    uint32_t                   log2NumAvgChirps;
    int8_t                     subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration for DC range signature calibration */
    memset ((void *)&cfg, 0, sizeof(cfg));

    /* Populate configuration: */
    cfg.enabled          = (uint16_t) atoi (argv[2]);
    cfg.negativeBinIdx   = (int16_t)  atoi (argv[3]);
    cfg.positiveBinIdx   = (int16_t)  atoi (argv[4]);
    cfg.numAvgChirps     = (uint16_t) atoi (argv[5]);

    if (cfg.negativeBinIdx > 0)
    {
        CLI_write ("Error: Invalid negative bin index\n");
        return -1;
    }
    if (cfg.positiveBinIdx < 0)
    {
        CLI_write ("Error: Invalid positive bin index\n");
        return -1;
    }	
    if ((cfg.positiveBinIdx - cfg.negativeBinIdx + 1) > DPU_RANGEPROC_SIGNATURE_COMP_MAX_BIN_SIZE)
    {
        CLI_write ("Error: Number of bins exceeds the limit\n");
        return -1;
    }
    log2NumAvgChirps = (uint32_t) mathUtils_ceilLog2(cfg.numAvgChirps);
    if (cfg.numAvgChirps != (1U << log2NumAvgChirps))
    {
        CLI_write ("Error: Number of averaged chirps is not power of two\n");
        return -1;
    }

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&cfg, MMWDEMO_CALIBDCRANGESIG_OFFSET,
                      sizeof(cfg), subFrameNum);

    return 0;
}

/**
 *  @b Description
 *  @n
 *      Clutter removal Configuration
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
static int32_t MmwDemo_CLIClutterRemoval (int32_t argc, char* argv[])
{
    DPC_ObjectDetection_StaticClutterRemovalCfg_Base cfg;
    int8_t              subFrameNum;

    if(MmwDemo_CLIGetSubframe(argc, argv, 3, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration for clutter removal */
    memset ((void *)&cfg, 0, sizeof(cfg));

    /* Populate configuration: */
    cfg.enabled          = (uint8_t) atoi (argv[2]);

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&cfg, MMWDEMO_STATICCLUTTERREMOFVAL_OFFSET,
                      sizeof(cfg), subFrameNum);

    return 0;
}
#endif

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
    MmwDemo_ADCBufCfg   adcBufCfg;
    int8_t              subFrameNum;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    if(MmwDemo_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
    {
        return -1;
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
        CLI_write("Error: chirpThreshold must be 1, multi-chirp is not allowed\n");
        return -1;
    }

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&adcBufCfg,
                      MMWDEMO_ADCBUFCFG_OFFSET,
                      sizeof(MmwDemo_ADCBufCfg), subFrameNum);
    return 0;
}

#if 0
/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for compensation of range bias and channel phase offsets
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
static int32_t MmwDemo_CLICompRangeBiasAndRxChanPhaseCfg (int32_t argc, char* argv[])
{
    DPU_AoAProc_compRxChannelBiasCfg   cfg;
    int32_t Re, Im;
    int32_t argInd;
    int32_t i;

    /* Sanity Check: Minimum argument check */
    if (argc != (1+1+SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL*2))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cfg, 0, sizeof(cfg));

    /* Populate configuration: */
    cfg.rangeBias          = (float) atof (argv[1]);

    argInd = 2;
    for (i=0; i < SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL; i++)
    {
        Re = (int32_t) (atof (argv[argInd++]) * 32768.);
        MATHUTILS_SATURATE16(Re);
        cfg.rxChPhaseComp[i].real = (int16_t) Re;

        Im = (int32_t) (atof (argv[argInd++]) * 32768.);
        MATHUTILS_SATURATE16(Im);
        cfg.rxChPhaseComp[i].imag = (int16_t) Im;

    }
    /* Save Configuration to use later */
    memcpy((void *) &gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.compRxChanCfg,
           &cfg, sizeof(cfg));

    gMmwMssMCB.objDetCommonCfg.isCompRxChannelBiasCfgPending = 1;

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for measurement configuration of range bias
 *      and channel phase offsets
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
static int32_t MmwDemo_CLIMeasureRangeBiasAndRxChanPhaseCfg (int32_t argc, char* argv[])
{
    DPC_ObjectDetection_MeasureRxChannelBiasCfg   cfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cfg, 0, sizeof(cfg));

    /* Populate configuration: */
    cfg.enabled          = (uint8_t) atoi (argv[1]);
    cfg.targetDistance   = (float) atof (argv[2]);
    cfg.searchWinSize   = (float) atof (argv[3]);

    /* Save Configuration to use later */
    memcpy((void *) &gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.measureRxChannelBiasCfg,
           &cfg, sizeof(cfg));

    gMmwMssMCB.objDetCommonCfg.isMeasureRxChannelBiasCfgPending = 1;

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
    rlRxSatMonConf_t        cqSatMonCfg;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 6)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cqSatMonCfg, 0, sizeof(rlRxSatMonConf_t));

    /* Populate configuration: */
    cqSatMonCfg.profileIndx                 = (uint8_t) atoi (argv[1]);

    if(cqSatMonCfg.profileIndx < RL_MAX_PROFILES_CNT)
    {

        cqSatMonCfg.satMonSel                   = (uint8_t) atoi (argv[2]);
        cqSatMonCfg.primarySliceDuration        = (uint16_t) atoi (argv[3]);
        cqSatMonCfg.numSlices                   = (uint16_t) atoi (argv[4]);
        cqSatMonCfg.rxChannelMask               = (uint8_t) atoi (argv[5]);

        /* Save Configuration to use later */
        gMmwMssMCB.cqSatMonCfg[cqSatMonCfg.profileIndx] = cqSatMonCfg;

        return 0;
    }
    else
    {
        return -1;
    }
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for configuring CQ Signal & Image band monitor
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
    rlSigImgMonConf_t       cqSigImgMonCfg;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Initialize configuration: */
    memset ((void *)&cqSigImgMonCfg, 0, sizeof(rlSigImgMonConf_t));

    /* Populate configuration: */
    cqSigImgMonCfg.profileIndx              = (uint8_t) atoi (argv[1]);

    if(cqSigImgMonCfg.profileIndx < RL_MAX_PROFILES_CNT)
    {
        cqSigImgMonCfg.numSlices            = (uint8_t) atoi (argv[2]);
        cqSigImgMonCfg.timeSliceNumSamples  = (uint16_t) atoi (argv[3]);

        /* Save Configuration to use later */
        gMmwMssMCB.cqSigImgMonCfg[cqSigImgMonCfg.profileIndx] = cqSigImgMonCfg;

        return 0;
    }
    else
    {
        return -1;
    }
}
#endif

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for enabling analog monitors
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
static int32_t MmwDemo_CLIAnalogMonitorCfg (int32_t argc, char* argv[])
{
    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }

    /* Save Configuration to use later */
    gMmwMssMCB.anaMonCfg.rxSatMonEn = atoi (argv[1]);
    gMmwMssMCB.anaMonCfg.sigImgMonEn = atoi (argv[2]);
    gMmwMssMCB.isAnaMonCfgPending = 1;

    return 0;
}

#if 0
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
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    if(MmwDemo_CLIGetSubframe(argc, argv, 5, &subFrameNum) < 0)
    {
        return -1;
    }

    /* Initialize configuration for DC range signature calibration */
    memset ((void *)&cfg, 0, sizeof(MmwDemo_LvdsStreamCfg));

    /* Populate configuration: */
    cfg.isHeaderEnabled = (bool)    atoi(argv[2]);
    cfg.dataFmt         = (uint8_t) atoi(argv[3]);
    cfg.isSwEnabled     = (bool)    atoi(argv[4]);

    /* If both h/w and s/w are enabled, HSI header must be enabled, because
     * we don't allow mixed h/w session without HSI header
     * simultaneously with s/w session with HSI header (s/w session always
     * streams HSI header) */
    if ((cfg.isSwEnabled == true) && (cfg.dataFmt != MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED))
    {
        if (cfg.isHeaderEnabled == false)
        {
            CLI_write("Error: header must be enabled when both h/w and s/w streaming are enabled\n");
            return -1;
        }
    }

    /* Save Configuration to use later */
    MmwDemo_CfgUpdate((void *)&cfg,
                      MMWDEMO_LVDSSTREAMCFG_OFFSET,
                      sizeof(MmwDemo_LvdsStreamCfg), subFrameNum);

    return 0;
}
#endif


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for configuring the data port
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
static int32_t MmwDemo_CLIConfigDataPort (int32_t argc, char* argv[])
{
    uint32_t baudrate;
    bool  ackPing;
    uint8_t ackData[16];
    UART_Transaction trans;

    trans.buf   = &ackData[0U];
    trans.count = sizeof(ackData);

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    /* Populate configuration: */
    baudrate = (uint32_t) atoi(argv[1]);
    ackPing = (bool) atoi(argv[2]);

    /* check if requested value is less than max supported value */
    if (baudrate > MMWDEMO_DATAUART_MAX_BAUDRATE_SUPPORTED)
    {
        CLI_write ("Ignored: Invalid baud rate (%d) specified\n",baudrate);
        return 0;
    }

    /* regardless of baud rate update, ack back to the host over this UART 
       port if handle is valid and user has requested the ack back */
    if ((gMmwMssMCB.loggingUartHandle != NULL) && (ackPing == true))
    {
        memset(ackData,0xFF,sizeof(ackData));
        // UART_writePolling (gMmwMssMCB.loggingUartHandle,
        //                    (uint8_t*)ackData,
        //                    sizeof(ackData));
        UART_write(gMmwMssMCB.loggingUartHandle, &trans);
    }

    return 0;
}





/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for querying Demo status
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
static int32_t MmwDemo_CLIQueryDemoStatus (int32_t argc, char* argv[])
{
    CLI_write ("Sensor State: %d\n",gMmwMssMCB.sensorState);
    CLI_write ("Data port baud rate: %d\n",gMmwMssMCB.cfg.platformCfg.loggingBaudRate);

    return 0;
}

#if 0
/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for querying Demo status
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
static int32_t MmwDemo_CLIQueryLocalIp (int32_t argc, char* argv[])
{
    if(gMmwMssMCB.enetCfg.status == 1){
        CLI_write ("Local IP is: %s\n", ip4addr_ntoa((const ip4_addr_t *)&gMmwMssMCB.enetCfg.localIp));
    }
    else{
        CLI_write ("Local IP is not up yet !!\n");
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for ethernet configuration
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
static int32_t MmwDemo_CLIEnetCfg(int32_t argc, char* argv[])
{

    volatile uint32_t remoteIp[4] = {0};
    uint8_t idx;

    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    /* Sanity Check: Minimum argument check */
    if (argc != 6)
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        return -1;
    }
    
    /* Populate configuration: */
    gMmwMssMCB.enetCfg.streamEnable = (bool) atoi(argv[1]);
    /* Get the IP Address */
    for(idx = 0; idx < 4; idx++){
        remoteIp[idx] = (uint32_t)atoi(argv[idx+2]);
    }
    /* Populate the IP Address */
    gMmwMssMCB.enetCfg.remoteIp = (ip_addr_t) IPADDR4_INIT_BYTES(remoteIp[0],remoteIp[1],remoteIp[2],remoteIp[3]);
    CLI_write("Remote IP Address is %s\n", ip4addr_ntoa(&gMmwMssMCB.enetCfg.remoteIp));

    if(gMmwMssMCB.enetCfg.streamEnable){
        MmwDemo_mssEnetCfgDone();
    }

    return 0;
}
#endif

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for save/restore calibration data to/from flash
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
static int32_t MmwDemo_CLICalibDataSaveRestore(int32_t argc, char* argv[])
{
    if (gMmwMssMCB.sensorState == MmwDemo_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    /* Validate inputs */
    if ( ((uint32_t) atoi(argv[1]) == 1) && ((uint32_t) atoi(argv[2] ) == 1))
    {
        CLI_write ("Error: Save and Restore can be enabled only one at a time\n");
        return -1;
    }

    /* Populate configuration: */
    gMmwMssMCB.calibCfg.saveEnable = (uint32_t) atoi(argv[1]);
    gMmwMssMCB.calibCfg.restoreEnable = (uint32_t) atoi(argv[2]);
    sscanf(argv[3], "0x%x", &gMmwMssMCB.calibCfg.flashOffset);

    gMmwMssMCB.isCalibCfgPending = 1;

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to send out the processing chain type
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
static int32_t MmwDemo_CLIProcChain(int32_t argc, char* argv[])
{
    
#ifdef DEMO_DDM
        CLI_write ("ProcChain: DDM\n");
#else
        CLI_write ("ProcChain: TDM\n");
#endif

    return 0;

}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to enable/disable ransac
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
static int32_t MmwDemo_CLIRansac(int32_t argc, char* argv[])
{
    //ransac <enable> <iterations> <thresh>
    if (argc != 4){
        return -1;
    }

    gRansacEnabled = atoi(argv[1]);
    gRansacIterations = atoi(argv[2]);
    gRansacThresh = atof(argv[3]);
    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to configure and create gtrack
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
static int32_t MmwDemo_CLIGtrackCfg(int32_t argc, char* argv[])
{
    // "gtrack <enable> <maxNumPoints> <maxNumTracks> <initialRadialVelocity> <maxRadialVelocity> <radialVelocityResolution> <maxAcceleration0> <maxAcceleration1> <maxAcceleration2> <deltaT> <boresightFilteringEnable>"
    if(argc != 12){
        return -1;
    }

    uint16_t enable;
    int32_t errCode;
    GTRACK_moduleConfig config;
    GTRACK_advancedParameters advParams;

    enable = atoi(argv[1]);

    //delete gtrack instance if it already exists
    if(gGtrackInstanceCreated){
        gtrack_delete(gHTrackModule);
        gHTrackModule = NULL;
        gGtrackInstanceCreated = false;
    }

    if(!enable){  
        gGtrackEnabled = false;
        return 0;
    }
    else{
        //create gtrack instance
        memset((void *)&config, 0, sizeof(GTRACK_moduleConfig));
        memset((void *)&advParams, 0, sizeof(GTRACK_advancedParameters));

        config.stateVectorType = GTRACK_STATE_VECTORS_3DA;
        config.verbose = GTRACK_VERBOSE_NONE;

        config.maxNumPoints = atoi(argv[2]);
        config.maxNumTracks = atoi(argv[3]);
        config.initialRadialVelocity = atof(argv[4]);
        config.maxRadialVelocity = atof(argv[5]);
        config.radialVelocityResolution = atof(argv[6]);
        config.maxAcceleration[0] = atof(argv[7]);
        config.maxAcceleration[1] = atof(argv[8]);
        config.maxAcceleration[2] = atof(argv[9]);
        config.deltaT = atof(argv[10]);
        config.boresightFilteringEnable = atoi(argv[11]);
        

        advParams.allocationParams = &gAppAllocationParams;
        advParams.gatingParams = &gAppGatingParams;
        advParams.stateParams = &gAppStateParams;
        advParams.sceneryParams = &gAppSceneryParams;

        config.advParams = &advParams;
        gHTrackModule = gtrack_create(&config, &errCode);
        if(gHTrackModule == NULL){
            test_print("gtrack creation error\n");
            return -1;
        }

        gGtrackInstanceCreated = true;
        gGtrackEnabled = true;
        return 0;
    }
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to configure app scenery parameters for gtrack
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
static int32_t MmwDemo_CLIGAppSceneryParams(int32_t argc, char* argv[])
{
    //"appSceneryParams <sensorPosX> <sensorPosY> <sensorPosZ> <orientationAzim> <orientationElev> <bBoxX1> <bBoxX2> <bBoxY1> <bBoxY2> <bBoxZ1> <bBoxZ2>";
    
    if(argc != 12){
        return -1;
    }
    
    memset(&gAppSceneryParams, 0, sizeof(gAppSceneryParams));

    gAppSceneryParams.sensorPosition.x = atof(argv[1]);
    gAppSceneryParams.sensorPosition.y = atof(argv[2]);
    gAppSceneryParams.sensorPosition.z = atof(argv[3]);

    gAppSceneryParams.sensorOrientation.azimTilt = atof(argv[4]);
    gAppSceneryParams.sensorOrientation.elevTilt = atof(argv[5]);

    gAppSceneryParams.numBoundaryBoxes = 1;
    gAppSceneryParams.boundaryBox[0].x1 = atof(argv[6]);
    gAppSceneryParams.boundaryBox[0].x2 = atof(argv[7]);
    gAppSceneryParams.boundaryBox[0].y1 = atof(argv[8]);
    gAppSceneryParams.boundaryBox[0].y2 = atof(argv[9]);
    gAppSceneryParams.boundaryBox[0].z1 = atof(argv[10]);
    gAppSceneryParams.boundaryBox[0].z2 = atof(argv[11]);

    gAppSceneryParams.numStaticBoxes = 0;

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to configure app gating parameters for gtrack
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
static int32_t MmwDemo_CLIGAppGatingParams(int32_t argc, char* argv[])
{
    //"gAppGatingParams <gain> <limitsArr0> <limitsArr1> <limitsArr2> <limitsArr3>"

    if(argc != 6){
        return -1;
    }

    memset(&gAppGatingParams, 0, sizeof(gAppGatingParams));

    gAppGatingParams.gain = atof(argv[1]);
    gAppGatingParams.limitsArray[0] = atof(argv[2]);
    gAppGatingParams.limitsArray[1] = atof(argv[3]);
    gAppGatingParams.limitsArray[2] = atof(argv[4]);
    gAppGatingParams.limitsArray[3] = atof(argv[5]);

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to configure app state parameters for gtrack
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
static int32_t MmwDemo_CLIGAppStateParams(int32_t argc, char* argv[])
{
    // "appStateParams <det2actThre> <det2freeThre> <active2freeThre> <static2freeThre> <exit2freeThre> <sleep2freeThre>"

    if(argc != 7){
        return -1;
    }

    memset(&gAppStateParams, 0, sizeof(gAppStateParams));

    gAppStateParams.det2actThre = atoi(argv[1]);
    gAppStateParams.det2freeThre = atoi(argv[2]);
    gAppStateParams.active2freeThre = atoi(argv[3]);
    gAppStateParams.static2freeThre = atoi(argv[4]);
    gAppStateParams.exit2freeThre = atoi(argv[5]);
    gAppStateParams.sleep2freeThre = atoi(argv[6]);

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to configure app alloc parameters for gtrack
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
static int32_t MmwDemo_CLIGAppAllocParams(int32_t argc, char* argv[])
{
    // "appAllocParams <snrThre> <snrThreObscured> <velocityThre> <pointsThre> <maxDistanceThre> <maxVelThre>"

    if(argc != 7){
        return -1;
    }

    memset(&gAppAllocationParams, 0, sizeof(gAppAllocationParams));

    gAppAllocationParams.snrThre = atof(argv[1]);
    gAppAllocationParams.snrThreObscured = atof(argv[2]);
    gAppAllocationParams.velocityThre = atof(argv[3]);
    gAppAllocationParams.pointsThre = atoi(argv[4]);
    gAppAllocationParams.maxDistanceThre = atof(argv[5]);
    gAppAllocationParams.maxVelThre = atof(argv[6]);

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to enable/disable rangeCfar
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
static int32_t MmwDemo_EnableRangeCfar(int32_t argc, char* argv[])
{
    // "rangeCfar <enable> "

    if(argc != 2){
        return -1;
    }

    uint8_t enable = atoi(argv[1]);
    gRangeCfarEnable = enable;

    return 0;
}


static int32_t R79_CLI_ByPassApi(CLI_Cfg* ptrCLICfg)
{
    //uint8_t                 cmdString[128];
    char*                   tokenizedArgs[CLI_MAX_ARGS];
    char*                   ptrCLICommand;
    char                    delimitter[] = " \r\n";
    uint32_t                argIndex;
    CLI_CmdTableEntry*      ptrCLICommandEntry;
    int32_t                 cliStatus;
    uint32_t                index, idx;
    uint16_t numCLICommands = 0U;
    
    /* Sanity Check: Validate the arguments */
    if (ptrCLICfg == NULL)
        return -1;

    /* Cycle through and determine the number of supported CLI commands: */
    for (index = 0; index < CLI_MAX_CMD; index++)
    {
        /* Do we have a valid entry? */
        if (ptrCLICfg->tableEntry[index].cmd == NULL)
        {
            /* NO: This is the last entry */
            break;
        }
        else
        {
            /* YES: Increment the number of CLI commands */
            numCLICommands = numCLICommands + 1;
        }
    }

    /* Execute All Radar Commands */
    for (idx = 0; idx < MAX_RADAR_CMD; idx++)
    {
        /* Reset all the tokenized arguments: */
        memset ((void *)&tokenizedArgs, 0, sizeof(tokenizedArgs));
        argIndex      = 0;
        ptrCLICommand = (char*)gHardcodedCliConfigs[idx];

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
            if (argIndex >= CLI_MAX_ARGS)
                break;

            /* Reset the command string */
            ptrCLICommand = NULL;
        }

        /* Were we able to tokenize the CLI command? */
        if (argIndex == 0)
            continue;

        /* Cycle through all the registered CLI commands: */
        for (index = 0; index < numCLICommands; index++)
        {
            ptrCLICommandEntry = &ptrCLICfg->tableEntry[index];

            /* Do we have a match? */
            if (strcmp(ptrCLICommandEntry->cmd, tokenizedArgs[0]) == 0)
            {
                CLI_write("%d\n", index);
                CLI_write("%s\n", gCLI.cfg.tableEntry[index].cmd);
                /* YES: Pass this to the CLI registered function */
                cliStatus = ptrCLICommandEntry->cmdHandlerFxn (argIndex, tokenizedArgs);
                if (cliStatus == 0)
                {
                    DebugP_log ("Done\r\n\n");
                }
                else
                {
                    DebugP_log ("Error %d\r\n", cliStatus);
                }
                break;
            }
        }

        /* Did we get a matching CLI command? */
        if (index == numCLICommands)
        {
            /* NO matching command found. Is the mmWave extension enabled? */
            if (ptrCLICfg->enableMMWaveExtension == 1U)
            {
                /* Yes: Pass this to the mmWave extension handler */
                cliStatus = CLI_MMWaveExtensionHandler (argIndex, tokenizedArgs);
            }

            /* Was the CLI command found? */
            if (cliStatus == -1)
            {
                /* No: The command was still not found */
                CLI_write ("'%s' is not recognized as a CLI command\n", tokenizedArgs[0]);
            }
        }
    }
    
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
    CLI_Cfg     cliCfg;
    char        demoBanner[256];
    uint32_t    cnt;

    /* Create Demo Banner to be printed out by CLI */
    sprintf(&demoBanner[0], 
                       "******************************************\n" \
                       "AWR294X MMW Demo %02d.%02d.%02d.%02d\n"  \
                       "******************************************\n", 
                        MMWAVE_SDK_VERSION_MAJOR,
                        MMWAVE_SDK_VERSION_MINOR,
                        MMWAVE_SDK_VERSION_BUGFIX,
                        MMWAVE_SDK_VERSION_BUILD
            );

    /* Initialize the CLI configuration: */
    memset ((void *)&cliCfg, 0, sizeof(CLI_Cfg));

    /* Populate the CLI configuration: */
    cliCfg.cliPrompt                    = "mmwDemo:/>";
    cliCfg.cliBanner                    = demoBanner;
    cliCfg.cliUartHandle                = gMmwMssMCB.commandUartHandle;
    cliCfg.taskPriority                 = taskPriority;
    cliCfg.mmWaveHandle                 = gMmwMssMCB.ctrlHandle;
    cliCfg.enableMMWaveExtension        = 1U;
    cliCfg.usePolledMode                = false;
    cliCfg.overridePlatform             = false;
    cliCfg.overridePlatformString       = "AWR294X";
        
    cnt=0;
    cliCfg.tableEntry[cnt].cmd            = "sensorStart";
    cliCfg.tableEntry[cnt].helpString     = "[doReconfig(optional, default:enabled)]";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLISensorStart;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "sensorStop";
    cliCfg.tableEntry[cnt].helpString     = "No arguments";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLISensorStop;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "guiMonitor";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <detectedObjects> <logMagRange> <noiseProfile> <rangeAzimuthHeatMap> <rangeDopplerHeatMap> <statsInfo>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGuiMonSel;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "cfarCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <procDirection> <averageMode> <winLen> <guardLen> <noiseDiv> <cyclicMode> <thresholdScale> <peakGroupingEn>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICfarCfg;
    cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "multiObjBeamForming";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <enabled> <threshold>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIMultiObjBeamForming;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "calibDcRangeSig";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <enabled> <negativeBinIdx> <positiveBinIdx> <numAvgFrames>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICalibDcRangeSig;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "clutterRemoval";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <enabled>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIClutterRemoval;
    // cnt++;

    cliCfg.tableEntry[cnt].cmd            = "adcbufCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <adcOutputFmt> <SampleSwap> <ChanInterleave> <ChirpThreshold>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIADCBufCfg;
    cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "compRangeBiasAndRxChanPhase";
    // cliCfg.tableEntry[cnt].helpString     = "<rangeBias> <Re00> <Im00> <Re01> <Im01> <Re02> <Im02> <Re03> <Im03> <Re10> <Im10> <Re11> <Im11> <Re12> <Im12> <Re13> <Im13> ";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICompRangeBiasAndRxChanPhaseCfg;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "measureRangeBiasAndRxChanPhase";
    // cliCfg.tableEntry[cnt].helpString     = "<enabled> <targetDistance> <searchWin>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIMeasureRangeBiasAndRxChanPhaseCfg;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "aoaFovCfg";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <minAzimuthDeg> <maxAzimuthDeg> <minElevationDeg> <maxElevationDeg>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIAoAFovCfg;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "cfarFovCfg";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <procDirection> <min (meters or m/s)> <max (meters or m/s)>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICfarFovCfg;
    // cnt++;
    // cliCfg.tableEntry[cnt].cmd            = "extendedMaxVelocity";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <enabled>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIExtendedMaxVelocity;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "CQRxSatMonitor";
    // cliCfg.tableEntry[cnt].helpString     = "<profile> <satMonSel> <priSliceDuration> <numSlices> <rxChanMask>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIChirpQualityRxSatMonCfg;
    // cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "CQSigImgMonitor";
    // cliCfg.tableEntry[cnt].helpString     = "<profile> <numSlices> <numSamplePerSlice>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIChirpQualitySigImgMonCfg;
    // cnt++;

    cliCfg.tableEntry[cnt].cmd            = "analogMonitor";
    cliCfg.tableEntry[cnt].helpString     = "<rxSaturation> <sigImgBand>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIAnalogMonitorCfg;
    cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "lvdsStreamCfg";
    // cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <enableHeader> <dataFmt> <enableSW>";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLILvdsStreamCfg;
    // cnt++;
    
    cliCfg.tableEntry[cnt].cmd            = "configDataPort";
    cliCfg.tableEntry[cnt].helpString     = "<baudrate> <ackPing>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIConfigDataPort;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "queryDemoStatus";
    cliCfg.tableEntry[cnt].helpString     = "";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIQueryDemoStatus;
    cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "queryLocalIp";
    // cliCfg.tableEntry[cnt].helpString     = "";
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIQueryLocalIp;
    // cnt++;

    cliCfg.tableEntry[cnt].cmd            = "calibData";
    cliCfg.tableEntry[cnt].helpString    = "<save enable> <restore enable> <Flash offset>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICalibDataSaveRestore;
    cnt++;

    // cliCfg.tableEntry[cnt].cmd            = "enetStreamCfg";
    // cliCfg.tableEntry[cnt].helpString     = "<isEnabled> <remoteIpD> <remoteIpC> <remoteIpB> <remoteIpA>"; /* Ip: D.C.B.A */
    // cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIEnetCfg;
    // cnt++;

    cliCfg.tableEntry[cnt].cmd            = "compressionCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <compressionRatio> <rangeBinsPerBlock> <compressionMethod>"; 
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICompressionCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "localMaxCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <azimThreshdB> <dopplerThreshdB>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLILocalMaxCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "intfMitigCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx>  <magSNRdB> <magDiffSNRdB>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIIntfMitigCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "antennaCalibParams";
    cliCfg.tableEntry[cnt].helpString     = "<I0> <Q0> .... <I11> <Q11>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIAntennaCalibParams;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "procChain";
    cliCfg.tableEntry[cnt].helpString     = "";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIProcChain;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "ransac";
    cliCfg.tableEntry[cnt].helpString     = "<enable> <iterations> <thresh>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIRansac;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "gtrack";
    cliCfg.tableEntry[cnt].helpString     = "<enable> <maxNumPoints> <maxNumTracks> <initialRadialVelocity> <maxRadialVelocity> <radialVelocityResolution> <maxAcceleration0> <maxAcceleration1> <maxAcceleration2> <deltaT> <boresightFilteringEnable>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGtrackCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "appSceneryParams";
    cliCfg.tableEntry[cnt].helpString     = "<sensorPosX> <sensorPosY> <sensorPosZ> <orientationAzim> <orientationElev> <bBoxX1> <bBoxX2> <bBoxY1> <bBoxY2> <bBoxZ1> <bBoxZ2>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGAppSceneryParams;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "appGatingParams";
    cliCfg.tableEntry[cnt].helpString     = "<gain> <limitsArr0> <limitsArr1> <limitsArr2> <limitsArr3>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGAppGatingParams;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "appStateParams";
    cliCfg.tableEntry[cnt].helpString     = "<det2actThre> <det2freeThre> <active2freeThre> <static2freeThre> <exit2freeThre> <sleep2freeThre>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGAppStateParams;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "appAllocParams";
    cliCfg.tableEntry[cnt].helpString     = "<snrThre> <snrThreObscured> <velocityThre> <pointsThre> <maxDistanceThre> <maxVelThre>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLIGAppAllocParams;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "rangeCfar";
    cliCfg.tableEntry[cnt].helpString     = "<enable>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_EnableRangeCfar;
    cnt++;
    
    
    /* Open the CLI: */
    if (CLI_open (&cliCfg) < 0)
    {
        test_print ("Error: Unable to open the CLI\n");
        return;
    }
    test_print ("Debug: CLI is operational\n");

#ifdef CLI_BYPASS
    R79_CLI_ByPassApi(&cliCfg);
#endif
    // test_print ("hardcoded configs sent\n");

    return;
}


