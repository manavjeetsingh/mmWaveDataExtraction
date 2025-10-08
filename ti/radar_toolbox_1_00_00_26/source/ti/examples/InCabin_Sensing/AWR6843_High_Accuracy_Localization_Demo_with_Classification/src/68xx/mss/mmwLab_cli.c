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
#include <occupancy_detection_3d_mss.h>
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
#include <ti/utils/cli/cli.h>
#include <ti/utils/mathutils/mathutils.h>

/* Demo Include Files */
#include "mmwdemo_rfparser.h"
#include "mmwdemo_adcconfig.h"

#include "common/mmwLab_config.h"

#define DEBUG(_x) //_x

/**************************************************************************
 *************************** Local function prototype****************************
 **************************************************************************/

/* CLI Extended Command Functions */
static int32_t mmwLab_CLIDynRACfarCfg (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIStaticRACfarCfg (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIDynRngAngleCfg (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIStaticRngAngleCfg (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIDynAngleEstCfg (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIDopplerCFARCfg (int32_t argc, char* argv[]);

static int32_t mmwLab_CLIBoardAntGeometry0 (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIBoardAntGeometry1 (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIBoardAntPhaseRot (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIAntAngleFoV (int32_t argc, char* argv[]);

static int32_t mmwLab_CLISensorStart (int32_t argc, char* argv[]);
static int32_t mmwLab_CLISensorStop (int32_t argc, char* argv[]);
static int32_t mmwLab_CLIADCBufCfg (int32_t argc, char* argv[]);
static int32_t mmwLab_CLICompRangeBiasAndRxChanPhaseCfg (int32_t argc, char* argv[]);

/**************************************************************************
 *************************** Extern Definitions *******************************
 **************************************************************************/

extern OccupancyDetection3D_MSS_MCB    gMmwMssMCB;

/**************************************************************************
 *************************** Local Definitions ****************************
 **************************************************************************/


/**************************************************************************
 *************************** CLI  Function Definitions **************************
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
static int32_t mmwLab_CLISensorStart (int32_t argc, char* argv[])
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

    /* Error checking initial state */
    if (gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_INIT)
    {
        MMWave_CtrlCfg ctrlCfg;

        /* need to get number of sub-frames so that next function to check
         * pending state can work */
        CLI_getMMWaveExtensionConfig (&ctrlCfg);
        gMmwMssMCB.objDetCommonCfg.numSubFrames =
            MmwDemo_RFParser_getNumSubFrames(&ctrlCfg);

        if (OccupancyDetection3D_isAllCfgInPendingState() == 0)
        {
            CLI_write ("Error: Full configuration must be provided before sensor can be started "
                       "the first time\n");

            /* Although not strictly needed, bring back to the initial value since we
             * are rejecting this first time configuration, prevents misleading debug. */
            gMmwMssMCB.objDetCommonCfg.numSubFrames = 0;

            return -1;
        }
    }

    if (gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_STARTED)
    {
        CLI_write ("Ignored: Sensor is already started\n");
        return 0;
    }

    if (doReconfig == false)
    {
         /* User intends to issue sensor start without config, check if no
            config was issued after stop and generate error if this is the case. */
         if (OccupancyDetection3D_isAllCfgInNonPendingState() == 0)
         {
             /* Message user differently if all config was issued or partial config was
                issued. */
             if (OccupancyDetection3D_isAllCfgInPendingState())
             {
                 CLI_write ("Error: You have provided complete new configuration, "
                            "issue \"sensorStart\" (without argument) if you want it to "
                            "take effect\n");
             }
             else
             {
                 CLI_write ("Error: You have provided partial configuration between stop and this "
                            "command and partial configuration cannot be undone."
                            "Issue the full configuration and do \"sensorStart\" \n");
             }
             return -1;
         }
    }
    else
    {
        /* User intends to issue sensor start with full config, check if all config
           was issued after stop and generate error if  is the case. */
        if (OccupancyDetection3D_isAllCfgInPendingState() == 0)
        {
            /* Message user differently if no config was issued or partial config was
               issued. */
            if (OccupancyDetection3D_isAllCfgInNonPendingState())
            {
                CLI_write ("Error: You have provided no configuration, "
                           "issue \"sensorStart 0\" OR provide "
                           "full configuration and issue \"sensorStart\"\n");
            }
            else
            {
                CLI_write ("Error: You have provided partial configuration between stop and this "
                           "command and partial configuration cannot be undone."
                           "Issue the full configuration and do \"sensorStart\" \n");
            }
            return -1;
        }
    }

    /***********************************************************************************
     * Retreive and check mmwave Open related config before calling openSensor
     ***********************************************************************************/

    /*  Fill demo's MCB mmWave openCfg structure from the CLI configs*/
    if (gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_INIT)
    {
        /* Get the open configuration: */
        CLI_getMMWaveExtensionOpenConfig (&gMmwMssMCB.cfg.openCfg);
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
            OccupancyDetection3D_debugAssert(0);
        }
        
        /* Compare openCfg->lowPowerMode*/
        if(memcmp((void *)&gMmwMssMCB.cfg.openCfg.lowPowerMode, (void *)&openCfg.lowPowerMode,
                          sizeof(rlLowPowerModeCfg_t)) != 0)
        {
            OccupancyDetection3D_debugAssert(0);
        }
        /* Compare openCfg->adcOutCfg*/
        if(memcmp((void *)&gMmwMssMCB.cfg.openCfg.adcOutCfg, (void *)&openCfg.adcOutCfg,
                          sizeof(rlAdcOutCfg_t)) != 0)
        {
            OccupancyDetection3D_debugAssert(0);
        }
    }

    retVal = OccupancyDetection3D_openSensor(gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_INIT);
    if(retVal != 0)
    {
        return -1;
    }

    /***********************************************************************************
     * Retrieve mmwave Control related config before calling startSensor
     ***********************************************************************************/
    /* Get the mmWave ctrlCfg from the CLI mmWave Extension */
    if(doReconfig)
    {
        CLI_getMMWaveExtensionConfig (&gMmwMssMCB.cfg.ctrlCfg);
        retVal = OccupancyDetection3D_configSensor();
        if(retVal != 0)
        {
            return -1;
        }
    }
    retVal = OccupancyDetection3D_startSensor();
    if(retVal != 0)
    {
        return -1;
    }

    /***********************************************************************************
     * Set the state
     ***********************************************************************************/
    gMmwMssMCB.sensorState = OccupancyDetection3D_SensorState_STARTED;
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
static int32_t mmwLab_CLISensorStop (int32_t argc, char* argv[])
{
    if ((gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_STOPPED) ||
        (gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_INIT))
    {
        CLI_write ("Ignored: Sensor is already stopped\n");
        return 0;
    }

    OccupancyDetection3D_stopSensor();

    OccupancyDetection3D_resetStaticCfgPendingState();

    gMmwMssMCB.sensorState = OccupancyDetection3D_SensorState_STOPPED;
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
static int32_t mmwLab_CLIGetSubframe (int32_t argc, char* argv[], int32_t expectedArgc,
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
 *      This is the CLI Handler for dynamic scene RA CFAR configuration
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
static int32_t mmwLab_CLIDynRACfarCfg (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    if(mmwLab_CLIGetSubframe(argc, argv, 17 , &subFrameNum) < 0)
    {
        DEBUG(System_printf ("Error: mmwLab_CLIDynRACfarCfg argc  = %d, argv2 = %d\n", argc, (uint8_t) atoi (argv[2]));)
        return -1;
    }

    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.leftSkipSize        =   (uint8_t) atoi (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.rightSkipSize       =   (uint8_t) atoi (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.leftSkipSizeAzimuth =   (uint8_t) atoi (argv[4]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.rightSkipSizeAzimuth=   (uint8_t) atoi (argv[5]);

            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.searchWinSizeRange  =   (uint8_t) atoi (argv[6]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.searchWinSizeDoppler=   (uint8_t) atoi (argv[7]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.searchWinSizeNear   =   (uint8_t) atoi (argv[8]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.guardSizeRange      =   (uint8_t) atoi (argv[9]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.guardSizeDoppler    =   (uint8_t) atoi (argv[10]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.guardSizeNear       =   (uint8_t) atoi (argv[11]);

            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.K0                  =   (float) atof (argv[12]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.dopplerSearchRelThr =   (float) atof (argv[13]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicSideLobeThr                    =   (float) atof (argv[14]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.enableSecondPassSearch =   (uint8_t) atoi (argv[15]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.rangeRefIndex       =   (uint8_t) atoi (argv[16]);
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.leftSkipSize        =   (uint8_t) atoi (argv[2]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.rightSkipSize       =   (uint8_t) atoi (argv[3]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.leftSkipSizeAzimuth =   (uint8_t) atoi (argv[4]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.rightSkipSizeAzimuth=   (uint8_t) atoi (argv[5]);

        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.searchWinSizeRange  =   (uint8_t) atoi (argv[6]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.searchWinSizeDoppler=   (uint8_t) atoi (argv[7]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.searchWinSizeNear   =   (uint8_t) atoi (argv[8]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.guardSizeRange      =   (uint8_t) atoi (argv[9]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.guardSizeDoppler    =   (uint8_t) atoi (argv[10]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.guardSizeNear       =   (uint8_t) atoi (argv[11]);

        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.K0                  =   (float) atof (argv[12]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.dopplerSearchRelThr =   (float) atof (argv[13]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicSideLobeThr                    =   (float) atof (argv[14]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.enableSecondPassSearch =   (uint8_t) atoi (argv[15]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.dynamicCfarConfig.rangeRefIndex       =   (uint8_t) atoi (argv[16]);
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for static scene RA CFAR configuration
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
static int32_t mmwLab_CLIStaticRACfarCfg (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    if(mmwLab_CLIGetSubframe(argc, argv, 14, &subFrameNum) < 0)
    {
        DEBUG(System_printf ("Error: mmwLab_CLIStaticRACfarCfg argc  = %d, argv2 = %d\n", argc, (uint8_t) atoi (argv[2]));)
        return -1;
    }

    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.leftSkipSize        =   (uint8_t) atoi (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.rightSkipSize       =   (uint8_t) atoi (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.leftSkipSizeAzimuth =   (uint8_t) atoi (argv[4]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.rightSkipSizeAzimuth=   (uint8_t) atoi (argv[5]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.searchWinSizeRange  =   (uint8_t) atoi (argv[6]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.searchWinSizeDoppler=   (uint8_t) atoi (argv[7]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.guardSizeRange      =   (uint8_t) atoi (argv[8]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.guardSizeDoppler    =   (uint8_t) atoi (argv[9]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.K0                  =   (float) atof (argv[10]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.dopplerSearchRelThr =   (float) atof (argv[11]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.enableSecondPassSearch =   (uint8_t) atoi (argv[13]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticSideLobeThr                    =   (float) atof (argv[12]);
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.leftSkipSize        =   (uint8_t) atoi (argv[2]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.rightSkipSize       =   (uint8_t) atoi (argv[3]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.leftSkipSizeAzimuth =   (uint8_t) atoi (argv[4]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.rightSkipSizeAzimuth=   (uint8_t) atoi (argv[5]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.searchWinSizeRange  =   (uint8_t) atoi (argv[6]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.searchWinSizeDoppler=   (uint8_t) atoi (argv[7]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.guardSizeRange      =   (uint8_t) atoi (argv[8]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.guardSizeDoppler    =   (uint8_t) atoi (argv[9]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.K0                  =   (float) atof (argv[10]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.dopplerSearchRelThr =   (float) atof (argv[11]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticCfarConfig.enableSecondPassSearch =   (uint8_t) atoi (argv[13]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.staticSideLobeThr                    =   (float) atof (argv[12]);
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for dynamic scene range-angle config
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
static int32_t mmwLab_CLIDynRngAngleCfg (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    if(mmwLab_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
    {
        DEBUG(System_printf ("Error: mmwLab_CLIDynRngAngleCfg argc  = %d, argv2 = %f\n", argc, (float) atof (argv[2]));)
        return -1;
    }


    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.searchStep        =   (float) atof (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.mvdr_alpha        =   (float) atof (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.detectionMethod   =   (uint8_t) atoi (argv[4]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.dopplerEstMethod  =   (uint8_t) atoi (argv[5]);
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.searchStep        =   (float) atof (argv[2]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.mvdr_alpha        =   (float) atof (argv[3]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.detectionMethod   =   (uint8_t) atoi (argv[4]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.dopplerEstMethod  =   (uint8_t) atoi (argv[5]);
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for dynamic scene 2D angle estimation config
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
static int32_t mmwLab_CLIDynAngleEstCfg (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    subFrameNum         =   (int8_t) atoi(argv[1]);


    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        DEBUG(System_printf ("Error: mmwLab_CLIDynAngleEstCfg all detectionMethod  = %d\n", gMmwMssMCB.subFrameCfg[0].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.detectionMethod);)
        if (gMmwMssMCB.subFrameCfg[0].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.detectionMethod <= 1)
        {
            if (argc != 10)
            {
                DEBUG(System_printf ("Error: mmwLab_CLIDynAngleEstCfg argc  = %d, argv2 = %f\n", argc, (float) atof (argv[2]));)
                return -1;
            }
            for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
            {
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.elevSearchStep   =   (float) atof (argv[2]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.mvdr_alpha       =   (float) atof (argv[3]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.maxNpeak2Search  =   (uint8_t) atoi (argv[4]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.peakExpSamples   =   (uint8_t) atoi (argv[5]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.elevOnly         =   (uint8_t) atoi (argv[6]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.sideLobThr       =   (float) atof (argv[7]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.peakExpRelThr    =   (float) atof (argv[8]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.peakExpSNRThr    =   (float) atof (argv[9]);
            }
        }
        else
        {
            if (argc != 8)
            {
                DEBUG(System_printf ("Error: mmwLab_CLIDynAngleEstCfg argc  = %d, argv2 = %f\n", argc, (float) atof (argv[2]));)
                return -1;
            }
            for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
            {
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.zoominFactor       =   (uint8_t) atoi (argv[2]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.zoominNn8bors      =   (uint8_t) atoi (argv[3]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.peakExpSamples     =   (uint8_t) atoi (argv[4]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.peakExpRelThr      =   (float) atof (argv[5]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.peakExpSNRThr      =   (float) atof (argv[6]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.localMaxCheckFlag  =   (uint8_t) atoi(argv[7]);
            }
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        if (gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.detectionMethod <= 1)
        {
            if (argc != 10)
            {
                return -1;
            }
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.elevSearchStep   =   (float) atof (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.mvdr_alpha       =   (float) atof (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.maxNpeak2Search  =   (uint8_t) atoi (argv[4]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.peakExpSamples   =   (uint8_t) atoi (argv[5]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.elevOnly         =   (uint8_t) atoi (argv[6]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.sideLobThr       =   (float) atof (argv[7]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.peakExpRelThr    =   (float) atof (argv[8]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevAngleEstCfg.peakExpSNRThr    =   (float) atof (argv[9]);
        }
        else
        {
            if (argc != 8)
            {
                return -1;
            }
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.zoominFactor       =   (uint8_t) atoi (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.zoominNn8bors      =   (uint8_t) atoi (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.peakExpSamples     =   (uint8_t) atoi (argv[4]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.peakExpRelThr      =   (float) atof (argv[5]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.peakExpSNRThr      =   (float) atof (argv[6]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.angle2DEst.azimElevZoominCfg.localMaxCheckFlag  =   (uint8_t) atoi(argv[7]);

        }
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Doppler Estimation configuration if the Doppler estimation method is CFAR
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
static int32_t mmwLab_CLIDopplerCFARCfg (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    if(mmwLab_CLIGetSubframe(argc, argv, 7, &subFrameNum) < 0)
    {
        DEBUG(System_printf ("Error: mmwLab_CLIDopplerCFARCfg argc  = %d, argv2\n", argc, (uint8_t) atoi (argv[2]));)
        return -1;
    }

    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        if (gMmwMssMCB.subFrameCfg[0].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.dopplerEstMethod == 1)
        {
            for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
            {
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.cfarDiscardLeft  =   (uint8_t) atoi (argv[2]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.cfarDiscardRight =   (uint8_t) atoi (argv[3]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.guardWinSize     =   (uint8_t) atoi (argv[4]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.refWinSize       =   (uint8_t) atoi (argv[5]);
                gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.thre             =   (float) atof (argv[6]);
            }
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        if (gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.rangeAngleCfg.dopplerEstMethod == 1)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.cfarDiscardLeft  =   (uint8_t) atoi (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.cfarDiscardRight =   (uint8_t) atoi (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.guardWinSize     =   (uint8_t) atoi (argv[4]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.refWinSize       =   (uint8_t) atoi (argv[5]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.dopCfarCfg.thre             =   (float) atof (argv[6]);
        }
    }
    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for static scene range-angle config
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
static int32_t mmwLab_CLIStaticRngAngleCfg (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    if(mmwLab_CLIGetSubframe(argc, argv, 5, &subFrameNum) < 0)
    {
        DEBUG(System_printf ("Error: mmwLab_CLIStaticRngAngleCfg argc  = %d, argv2 = %d\n", argc, (uint8_t) atoi (argv[2]));)
        return -1;
    }

    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.staticEstCfg.staticProcEnabled          =   (uint8_t) atoi (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.staticEstCfg.staticAzimStepDeciFactor   =   (uint8_t) atoi (argv[3]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.staticEstCfg.staticElevStepDeciFactor   =   (uint8_t) atoi (argv[4]);
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.staticEstCfg.staticProcEnabled          =   (uint8_t) atoi (argv[2]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.staticEstCfg.staticAzimStepDeciFactor   =   (uint8_t) atoi (argv[3]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.staticEstCfg.staticElevStepDeciFactor   =   (uint8_t) atoi (argv[4]);
    }
    return 0;
}



/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for ADC buffer command
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
static int32_t mmwLab_CLIADCBufCfg (int32_t argc, char* argv[])
{
    MmwDemo_ADCBufCfg   adcBufCfg;
    int8_t              subFrameNum;

    if (gMmwMssMCB.sensorState == OccupancyDetection3D_SensorState_STARTED)
    {
        CLI_write ("Ignored: This command is not allowed after sensor has started\n");
        return 0;
    }

    if(mmwLab_CLIGetSubframe(argc, argv, 6, &subFrameNum) < 0)
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
    OccupancyDetection3D_CfgUpdate((void *)&adcBufCfg,
                           MMWLAB_ADCBUFCFG_OFFSET,
                      sizeof(MmwDemo_ADCBufCfg), subFrameNum);
    return 0;
}


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
static int32_t mmwLab_CLICompRangeBiasAndRxChanPhaseCfg (int32_t argc, char* argv[])
{
    int32_t argInd;
    int32_t i;
    uint8_t  indx;

    /* Sanity Check: Minimum argument check */
    if (argc != (1+1+SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL*2))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        DEBUG(System_printf ("Error: mmwLab_CLICompRangeBiasAndRxChanPhaseCfg argc  = %d, argv2=%f\n", argc, (float)atof (argv[2]));)
        return -1;
    }

    for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
    {
        argInd = 2;
        for (i=0; i < SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL; i++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.phaseCompVect[i].real   =   (float)atof (argv[argInd++]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.phaseCompVect[i].imag   =   (float)atof (argv[argInd++]);
        }
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for board antenna geometry matrix row 0
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
static int32_t mmwLab_CLIBoardAntGeometry0 (int32_t argc, char* argv[])
{
    int32_t argInd;
    int32_t i;
    uint8_t  indx;

    /* Sanity Check: Minimum argument check */
    if (argc != (1+SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        DEBUG(System_printf ("Error: mmwLab_CLIBoardAntGeometry0 argc  = %d, argv2 = %d\n", argc, (int8_t)atoi (argv[2]));)
        return -1;
    }


    for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
    {
        argInd = 1;
        for (i=0; i < SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL; i++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.m_ind[i]   =   (int8_t)atoi (argv[argInd++]);
        }
    }

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for board antenna geometry matrix row 1
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
static int32_t mmwLab_CLIBoardAntGeometry1 (int32_t argc, char* argv[])
{
    int32_t argInd;
    int32_t i;
    uint8_t  indx;

    /* Sanity Check: Minimum argument check */
    if (argc != (1+SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        DEBUG(System_printf ("Error: mmwLab_CLIBoardAntGeometry1 argc  = %d, argv2 = %d\n", argc, (int8_t)atoi (argv[2]));)
        return -1;
    }

    for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
    {
        argInd = 1;
        for (i=0; i < SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL; i++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.n_ind[i]   =   (int8_t)atoi (argv[argInd++]);
        }
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for board antenna phase rotation
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
static int32_t mmwLab_CLIBoardAntPhaseRot (int32_t argc, char* argv[])
{
    int32_t argInd;
    int32_t i;
    uint8_t  indx;

    /* Sanity Check: Minimum argument check */
    if (argc != (1+SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL))
    {
        CLI_write ("Error: Invalid usage of the CLI command\n");
        DEBUG(System_printf ("Error: mmwLab_CLIBoardAntPhaseRot argc  = %d, argv2 = %d\n", argc, (int8_t)atoi (argv[2]));)
        return -1;
    }

    for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
    {
        argInd = 1;
        for (i=0; i < SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL; i++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.phaseRot[i]   =   (int8_t)atoi (argv[argInd++]);
        }
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for angle FOV config
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
static int32_t mmwLab_CLIAntAngleFoV (int32_t argc, char* argv[])
{
    int8_t              subFrameNum;

    if(mmwLab_CLIGetSubframe(argc, argv, 4, &subFrameNum) < 0)
    {
        DEBUG(System_printf ("Error: mmwLab_CLIAntAngleFoV argc  = %d, argv2 = %f\n", argc, (float) atof (argv[2]));)
        return -1;
    }

    if(subFrameNum == MMWLAB_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.fovCfg[0]   =   (float) atof (argv[2]);
            gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.fovCfg[1]   =   (float) atof (argv[3]);
        }
    }
    else
    {
        uint8_t  indx = subFrameNum;
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.fovCfg[0]   =   (float) atof (argv[2]);
        gMmwMssMCB.subFrameCfg[indx].objDetDynCfg.dspDynCfg.caponChainCfg.doaConfig.fovCfg[1]   =   (float) atof (argv[3]);
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
void mmwLab_CLIInit (uint8_t taskPriority)
{
    CLI_Cfg     cliCfg;
    char        demoBanner[110];
    uint32_t    cnt;

    /* Create Demo Banner to be printed out by CLI */
    sprintf(&demoBanner[0], 
                       "\n***********************************\n" \
                       "    xWR68xx Proprietary VOD Demo   \n" \
                       "***********************************\n"
            );

    /* Initialize the CLI configuration: */
    memset ((void *)&cliCfg, 0, sizeof(CLI_Cfg));

    /* Populate the CLI configuration: */
    cliCfg.cliPrompt                    = "mmwDemo:/>";
    cliCfg.cliBanner                    = demoBanner;
    cliCfg.cliUartHandle                = gMmwMssMCB.commandUartHandle;
    cliCfg.taskPriority                 = taskPriority;
    cliCfg.socHandle                    = gMmwMssMCB.socHandle;
    cliCfg.mmWaveHandle                 = gMmwMssMCB.ctrlHandle;
    cliCfg.enableMMWaveExtension        = 1U;
    cliCfg.usePolledMode                = true;
    cliCfg.overridePlatform             = false;
    cliCfg.overridePlatformString       = NULL;    
    
    cnt=0;
    cliCfg.tableEntry[cnt].cmd            = "sensorStart";
    cliCfg.tableEntry[cnt].helpString     = "[doReconfig(optional, default:enabled)]";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLISensorStart;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "sensorStop";
    cliCfg.tableEntry[cnt].helpString     = "No arguments";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLISensorStop;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "dynamicRACfarCfg";
    cliCfg.tableEntry[cnt].helpString     = "<leftSkipSize> <rightSkipSize> <leftSkipSizeAzimuth> <rightSkipSizeAngle> <searchWinSizeRange> <searchWinSizeAngle> <searchWinSizeNear> <guardSizeRange> <guardSizeAngle> <guardSizeNear> <threRange> <threAngle> <threSidelob> <enSecondPass> <rangeRefIndex>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIDynRACfarCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "staticRACfarCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <leftSkipSize> <rightSkipSize> <leftSkipSizeAzimuth> <rightSkipSizeAngle> <searchWinSizeRange> <searchWinSizeAngle> <guardSizeRange> <guardSizeAngle> <threRange> <threAngle> <threSidelob> <enSecondPass>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIStaticRACfarCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "dynamicRangeAngleCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <searchStep> <mvdr_alpha> <detectionMethod> <dopplerEstMethod>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIDynRngAngleCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "dynamic2DAngleCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <elevSearchStep> <mvdr_alpha> <maxNpeak2Search> <peakExpSamples> <elevOnly> <sideLobThr> <peakExpRelThr> <peakExpSNRThr>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIDynAngleEstCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "dopplerCfarCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <discardLeft> <discardRight> <guardWinSize> <refWinSize> <threshold> ";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIDopplerCFARCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "staticRangeAngleCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <staticProcEnabled> <staticAzimStepDeciFactor> <staticElevStepDeciFactor>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIStaticRngAngleCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "fovCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <azimFoV> <elevFoV> ";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIAntAngleFoV;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "antGeometry0";
    cliCfg.tableEntry[cnt].helpString     = "<elem0> <elem1> <elem2> <elem3> <elem4> <elem5> <elem6> <elem7> <elem8>  <elem9> <elem10> <elem11> <elem12> ";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIBoardAntGeometry0;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "antGeometry1";
    cliCfg.tableEntry[cnt].helpString     = "<elem0> <elem1> <elem2> <elem3> <elem4> <elem5> <elem6> <elem7> <elem8>  <elem9> <elem10> <elem11> <elem12> ";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIBoardAntGeometry1;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "antPhaseRot";
    cliCfg.tableEntry[cnt].helpString     = "<elem0> <elem1> <elem2> <elem3> <elem4> <elem5> <elem6> <elem7> <elem8>  <elem9> <elem10> <elem11> <elem12> ";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIBoardAntPhaseRot;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "adcbufCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <adcOutputFmt> <SampleSwap> <ChanInterleave> <ChirpThreshold>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLIADCBufCfg;
    cnt++;

    cliCfg.tableEntry[cnt].cmd            = "compRangeBiasAndRxChanPhase";
    cliCfg.tableEntry[cnt].helpString     = "<rangeBias> <Re00> <Im00> <Re01> <Im01> <Re02> <Im02> <Re03> <Im03> <Re10> <Im10> <Re11> <Im11> <Re12> <Im12> <Re13> <Im13>  <Re20> <Im20> <Re21> <Im21> <Re22> <Im22> <Re23> <Im23> ";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = mmwLab_CLICompRangeBiasAndRxChanPhaseCfg;
    cnt++;

    /* Open the CLI: */
    if (CLI_open (&cliCfg) < 0)
    {
        System_printf ("Error: Unable to open the CLI\n");
        return;
    }
    System_printf ("Debug: CLI is operational\n");
    return;
}


