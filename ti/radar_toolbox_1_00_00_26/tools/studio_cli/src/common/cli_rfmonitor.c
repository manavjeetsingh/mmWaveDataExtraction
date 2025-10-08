/*
 *   @file  cli_rfmonitor.c
 *
 *   @brief
 *      CLI Extension which handles the mmWave link monitoring
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2019 Texas Instruments, Inc.
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

/* Standard Include Files */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

/* mmWave SDK Include Files */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
#include <ti/drivers/uart/UART.h>
#include <ti/control/mmwavelink/mmwavelink.h>
#include <ti/control/mmwave/mmwave.h>

/* App Include Files */
#include "mss/mmw_cli.h"
#include "common/rfmonitor_internal.h"
#include "common/rfmonitor.h"

/**************************************************************************
 ******************* CLI mmWave Extension Local Definitions****************
 **************************************************************************/

extern RFMonitorCfg     gRFMonitorCfg;

/**************************************************************************
 ******************* CLI mmWave Extension Local Functions *****************
 **************************************************************************/

/* CLI Command Functions */
static int32_t CLI_RxGainPhaseMonCfg (int32_t argc, char* argv[]);
static int32_t CLI_TempMonCfg (int32_t argc, char* argv[]);
static int32_t CLI_TxPowerMonCfg (int32_t argc, char* argv[]);
static int32_t CLI_TxBallbreakMonCfg (int32_t argc, char* argv[]);
static int32_t CLI_SynthFreqMonCfg (int32_t argc, char* argv[]);
static int32_t CLI_PllConVoltMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_DualClkCompMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_RxIfStageMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_TxBpmMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_ExtAnaSigMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_TxIntAnaSigMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_RxIntAnaSigMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_PmClkSigMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_GpadcSigMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_RxMixInpwrMonCfg(int32_t argc, char* argv[]);
static int32_t CLI_GetMonStats(int32_t argc, char* argv[]);
static int32_t CLI_MonCalibReportCfg(int32_t argc, char* argv[]);
static int32_t CLI_RfDigitalMonCfg(int32_t argc, char* argv[]);

#undef CLI_MMWAVE_HELP_SUPPORT

/**************************************************************************
 ************************ CLI mmWave Extension Globals ********************
 **************************************************************************/

/**
 * @brief
 *  This is the RF monitor table added to the CLI.
 */
CLI_CmdTableEntry gCLIRFMonitorExtensionTable[] =
{
    {
        "rxGainPhaseMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <profile>",
#else
        NULL,
#endif
        CLI_RxGainPhaseMonCfg
    },
    {
        "tempMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <tempDiffThresh>",
#else
        NULL,
#endif
        CLI_TempMonCfg
    },
    {
        "txPowerMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <txAnt> <profile index>",
#else
        NULL,
#endif
        CLI_TxPowerMonCfg
    },
    {
        "txBallbreakMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <txAnt>",
#else
        NULL,
#endif
        CLI_TxBallbreakMonCfg
    },
    {
        "synthFreqMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <profile index>",
#else
        NULL,
#endif
        CLI_SynthFreqMonCfg
    },
    {
        "pllConVoltMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable>",
#else
        NULL,
#endif
        CLI_PllConVoltMonCfg
    },
    {
        "dualClkCompMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable>",
#else
        NULL,
#endif
        CLI_DualClkCompMonCfg
    },
    {
        "rxIfStageMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <profile idx>",
#else
        NULL,
#endif
        CLI_RxIfStageMonCfg
    },
    {
        "txBpmMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <txAnt> <profile idx>",
#else
        NULL,
#endif
        CLI_TxBpmMonCfg
    },
    {
        "extAnaSigMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable>",
#else
        NULL,
#endif
        CLI_ExtAnaSigMonCfg
    },
    {
        "txIntAnaSigMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <txAnt> <profile idx>",
#else
        NULL,
#endif
        CLI_TxIntAnaSigMonCfg
    },
    {
        "rxIntAnaSigMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <profile idx>",
#else
        NULL,
#endif
        CLI_RxIntAnaSigMonCfg
    },
    {
        "pmClkSigMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <profile idx>",
#else
        NULL,
#endif
        CLI_PmClkSigMonCfg
    },
    {
        "gpadcSigMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable>",
#else
        NULL,
#endif
        CLI_GpadcSigMonCfg
    },
    {
        "rxMixInpwrMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <profile idx>",
#else
        NULL,
#endif
        CLI_RxMixInpwrMonCfg
    },
    {
        "getMonStats",
        NULL,
        CLI_GetMonStats
    },
    {
        "monCalibReportCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable calib report> <enable monStats> <enable failure report>",
#else
        NULL,
#endif
        CLI_MonCalibReportCfg
    },
    {
         "digMonCfg",
#ifdef CLI_MMWAVE_HELP_SUPPORT
        "<enable> <enMask>",
#else
        NULL,
#endif
        CLI_RfDigitalMonCfg
    }
};


/**************************************************************************
 ********************** CLI mmWave Extension Functions ********************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      Helper function to get profile index from command arguments
 *
 *  @param[in] profileId       Profile index string
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   profile index in range [0, RL_MAX_PROFILES_CNT- 1]
 *  @retval
 *      Error   -   -1
 */
static int8_t CLI_getProfileIdx(char* profileId)
{
    int8_t    profileIdx = -1;

    profileIdx = (uint8_t)atoi(profileId);
    if ((profileIdx >= RL_MAX_PROFILES_CNT) ||
       (profileIdx < 0) )
    {
        profileIdx = MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    return profileIdx;
}

/**
 *  @b Description
 *  @n
 *      Helper function to get tx antenna index from command arguments
 *
 *  @param[in] txAntString       Tx antenna index string
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   profile index in range [0, RL_TX_CNT- 1]
 *  @retval
 *      Error   -   -1
 */
static int8_t CLI_geTxAntIdx(char* txAntString)
{
    int8_t    txIdx = -1;

    txIdx = (uint8_t)(atoi(txAntString));
    if((txIdx >= RL_TX_CNT) ||
       (txIdx < 0) )
    {
        txIdx = -1;
    }

    return txIdx;
}

/**
 *  @b Description
 *  @n
 *      Helper function to validate number of arguments
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] expectedArgc   Expected number of arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t CLI_validateArgc(int32_t argc, int32_t expecteArgc)
{
    if (argc != expecteArgc)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Rx Gain phase monitor
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_RxGainPhaseMonCfg(int32_t argc, char* argv[])
{
    int8_t profileIdx;
    int32_t retVal = 0;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    
    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    gRFMonitorCfg.rxGainPhaseMonCfg->profileIndx = profileIdx;

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT,
                                                    RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT,
                                                    RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT,
                                                    1U);
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for temperater monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_TempMonCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TEMP_EN_BIT,
                                                    RFMON_ANAMON_TEMP_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TEMP_EN_BIT,
                                                    RFMON_ANAMON_TEMP_EN_BIT,
                                                    1U);
    }

    gRFMonitorCfg.tempMonCfg->tempDiffThresh = (uint16_t)atoi(argv[2]);


    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Tx Power monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_TxPowerMonCfg(int32_t argc, char* argv[])
{
    int8_t         txIdx;
    int8_t          profileIdx;
    rlTxPowMonConf_t **ptrTxPowerMonCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[3])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Set tx antenna index */
    if((txIdx = CLI_geTxAntIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    switch(txIdx)
    {
        case 0:
            ptrTxPowerMonCfg = &gRFMonitorCfg.allTxPowerMonCfg.tx0PowrMonCfg;
            break;

        case 1:
            ptrTxPowerMonCfg = &gRFMonitorCfg.allTxPowerMonCfg.tx1PowrMonCfg;
            break;

        case 2:
            ptrTxPowerMonCfg = &gRFMonitorCfg.allTxPowerMonCfg.tx2PowrMonCfg;
            break;

        default:
            break;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TX0_POWER_EN_BIT + txIdx,
                                                    RFMON_ANAMON_TX0_POWER_EN_BIT + txIdx,
                                                    0U);

        *ptrTxPowerMonCfg = NULL;
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TX0_POWER_EN_BIT + txIdx,
                                                    RFMON_ANAMON_TX0_POWER_EN_BIT + txIdx,
                                                    1U);

        *ptrTxPowerMonCfg = &gTxPowMonCfg[txIdx];
        (*ptrTxPowerMonCfg)->profileIndx = (rlUInt8_t)profileIdx;
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Tx Ball break monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_TxBallbreakMonCfg(int32_t argc, char* argv[])
{
    int8_t         txIdx;
    rlTxBallbreakMonConf_t  **ptrTxBallbreakMonCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set tx antenna index */
    if((txIdx = CLI_geTxAntIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Set profile index */
    switch(txIdx)
    {
        case 0:
            ptrTxBallbreakMonCfg = &gRFMonitorCfg.allTxBallbreakMonCfg.tx0BallBrkMonCfg;
            break;

        case 1:
            ptrTxBallbreakMonCfg = &gRFMonitorCfg.allTxBallbreakMonCfg.tx1BallBrkMonCfg;
            break;

        case 2:
            ptrTxBallbreakMonCfg = &gRFMonitorCfg.allTxBallbreakMonCfg.tx2BallBrkMonCfg;
            break;

        default:
            break;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TX0_BALLBREAK_EN_BIT + txIdx,
                                                    RFMON_ANAMON_TX0_BALLBREAK_EN_BIT + txIdx,
                                                    0U);
        *ptrTxBallbreakMonCfg = NULL;

    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TX0_BALLBREAK_EN_BIT + txIdx,
                                                    RFMON_ANAMON_TX0_BALLBREAK_EN_BIT + txIdx,
                                                    1U);

        *ptrTxBallbreakMonCfg = &gTxBallbreakMonCfg[txIdx];
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Synthesizer frequency monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_SynthFreqMonCfg(int32_t argc, char* argv[])
{
    int8_t         profileIdx;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_SYNTH_FREQ_EN_BIT,
                                                    RFMON_ANAMON_SYNTH_FREQ_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_SYNTH_FREQ_EN_BIT,
                                                    RFMON_ANAMON_SYNTH_FREQ_EN_BIT,
                                                    1U);
    }

    gRFMonitorCfg.synthFreqMonCfg->profileIndx = (rlUInt8_t)profileIdx;

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for PLL Control voltage Monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_PllConVoltMonCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT,
                                                    RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT,
                                                    RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT,
                                                    1U);
    }

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for dual clock comp monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_DualClkCompMonCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT,
                                                    RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT,
                                                    RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT,
                                                    1U);
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for RX IF stage monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_RxIfStageMonCfg(int32_t argc, char* argv[])
{
    int8_t         profileIdx;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }
    
    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_IFSTAGE_EN_BIT,
                                                    RFMON_ANAMON_RX_IFSTAGE_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_IFSTAGE_EN_BIT,
                                                    RFMON_ANAMON_RX_IFSTAGE_EN_BIT,
                                                    1U);
    }

    gRFMonitorCfg.rxIfStageMonCfg->profileIndx = (rlUInt8_t)profileIdx;

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Tx BPM monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_TxBpmMonCfg(int32_t argc, char* argv[])
{
    int8_t         txIdx;
    int8_t         profileIdx;
    rlTxBpmMonConf_t **ptrTxBpmMonCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[3])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Set tx antenna index */
    if((txIdx = CLI_geTxAntIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    switch(txIdx)
    {
        case 0:
            ptrTxBpmMonCfg = &gRFMonitorCfg.allTxBpmMonCfg.tx0BpmMonCfg;
            break;

        case 1:
            ptrTxBpmMonCfg = &gRFMonitorCfg.allTxBpmMonCfg.tx1BpmMonCfg;
            break;

        case 2:
            ptrTxBpmMonCfg = &gRFMonitorCfg.allTxBpmMonCfg.tx2BpmMonCfg;
            break;

        default:
            break;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TX0_BPM_EN_BIT + txIdx,
                                                    RFMON_ANAMON_TX0_BPM_EN_BIT + txIdx,
                                                    0U);

        *ptrTxBpmMonCfg = NULL;
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_TX0_BPM_EN_BIT + txIdx,
                                                    RFMON_ANAMON_TX0_BPM_EN_BIT + txIdx,
                                                    1U);

        *ptrTxBpmMonCfg = &gTxBpmMonCfg[txIdx];
        (*ptrTxBpmMonCfg)->profileIndx = (rlUInt8_t)profileIdx;

    }

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for external Analog signal monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_ExtAnaSigMonCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT,
                                                    1U);
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Tx Internal analog signal monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_TxIntAnaSigMonCfg(int32_t argc, char* argv[])
{
    int8_t          txIdx;
    int8_t          profileIdx;
    rlTxIntAnaSignalsMonConf_t **ptrTxIntAnaSigMonCfg;

    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[3])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    /* Set tx antenna index */
    if((txIdx = CLI_geTxAntIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    switch(txIdx)
    {
        case 0:
            ptrTxIntAnaSigMonCfg = &gRFMonitorCfg.allTxIntAnaSigMonCfg.tx0IntAnaSgnlMonCfg;
            break;

        case 1:
            ptrTxIntAnaSigMonCfg = &gRFMonitorCfg.allTxIntAnaSigMonCfg.tx1IntAnaSgnlMonCfg;
            break;

        case 2:
            ptrTxIntAnaSigMonCfg = &gRFMonitorCfg.allTxIntAnaSigMonCfg.tx2IntAnaSgnlMonCfg;
            break;

        default:
            break;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_INT_TX0_SIGNALS_EN_BIT + txIdx,
                                                    RFMON_ANAMON_INT_TX0_SIGNALS_EN_BIT + txIdx,
                                                    0U);

        *ptrTxIntAnaSigMonCfg = NULL;
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_INT_TX0_SIGNALS_EN_BIT + txIdx,
                                                    RFMON_ANAMON_INT_TX0_SIGNALS_EN_BIT + txIdx,
                                                    1U);

        *ptrTxIntAnaSigMonCfg = &gTxIntAnaSigMonCfg[txIdx];
        (*ptrTxIntAnaSigMonCfg)->profileIndx = (rlUInt8_t)profileIdx;
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Internal signals in the RX path monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_RxIntAnaSigMonCfg(int32_t argc, char* argv[])
{
    int8_t         profileIdx;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT,
                                                    1U);
    }

    gRFMonitorCfg.rxIntAnaSigMonCfg->profileIndx = (rlUInt8_t)profileIdx;

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Internal signals for PM, CLK and LO monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_PmClkSigMonCfg(int32_t argc, char* argv[])
{
    int8_t         profileIdx;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT,
                                                    1U);
    }

    gRFMonitorCfg.pmClkLoIntAnaSigMonCfg->profileIndx = (rlUInt8_t)profileIdx;

    return 0;
}



/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for Internal signals for GPADC monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_GpadcSigMonCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 2)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_GPADC_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_GPADC_SIGNALS_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_GPADC_SIGNALS_EN_BIT,
                                                    RFMON_ANAMON_GPADC_SIGNALS_EN_BIT,
                                                    1U);
    }

    return 0;
}


/**
 *  @b Description
 *  @n
 *      This is the CLI Handler for internal RX mixer input power monitor configuration
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_RxMixInpwrMonCfg(int32_t argc, char* argv[])
{
    int8_t         profileIdx;

    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Set profile index */
    if((profileIdx = CLI_getProfileIdx(argv[2])) < 0)
    {
        return MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM;
    }

    if(atoi(argv[1]) == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT,
                                                    RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT,
                                                    RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT,
                                                    1U);
    }

    gRFMonitorCfg.rxMixInpwrMonCfg->profileIndx = (rlUInt8_t)profileIdx;

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to get monitor stats
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_GetMonStats(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 1)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    RFMon_reportStatsToHost(0);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the CLI Handler to configure monitor reports
 *
 *  @param[in] argc           Number of arguments
 *  @param[in] argv           Arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_MonCalibReportCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 4)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Save report configuration */
    gRFMonitorCfg.reportCfg.enCalibReport     = (uint8_t)atoi(argv[1]);

    /*
     *   0 - Quite mode, turn off failure report
     *   1 - Active mode, report is sent when Async event is received from RF
     *   N - Report is sent after N frame period
    */
    gRFMonitorCfg.reportCfg.enMonReportMode   = (uint8_t)atoi(argv[2]);
    /* This Parameter is NOT SUPPORTED in this version */
    gRFMonitorCfg.reportCfg.enMonStats        = (uint8_t)atoi(argv[3]);

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave extension initialization API
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   number of commands added
 *  @retval
 *      Error   -   <0
 */
static int32_t CLI_RfDigitalMonCfg(int32_t argc, char* argv[])
{
    /* Sanity Check: Minimum argument check */
    if (argc != 3)
    {
        return MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD;
    }

    /* Save enable Mask configuration */
    gRFMonitorCfg.rfDigMonitorEn->enMask = (uint32_t)atoi(argv[2]);

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave extension initialization API
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   number of commands added
 *  @retval
 *      Error   -   <0
 */
int32_t CLI_RFMonitorExtensionInit(int32_t argc, char* argv[])
{
    /* Configuration initialization */
    RFMon_initCfg();

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave extension handler which executes mmWave extension
 *      commands. This is invoked by the main CLI wrapper only if the extension
 *      was enabled.
 *
 *  @param[in]  argc
 *      Number of detected arguments
 *  @param[in] argv
 *      Detected arguments
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      0   -   Matching mmWave extension command found
 *  @retval
 *      -1  -   No Matching mmWave extension command
 */
int32_t CLI_RFMonitorExtensionHandler(int32_t argc, char* argv[])
{
    CLI_CmdTableEntry*  ptrCLICommandEntry;
    int32_t             retVal = 0;

    /* Get the pointer to the mmWave extension table */
    ptrCLICommandEntry = &gCLIRFMonitorExtensionTable[0];

    /* Cycle through all the registered externsion CLI commands: */
    while (ptrCLICommandEntry->cmdHandlerFxn != NULL)
    {
        /* Do we have a match? */
        if (strcmp(ptrCLICommandEntry->cmd, argv[0]) == 0)
        {
            /* YES: Pass this to the CLI registered function */
            retVal = ptrCLICommandEntry->cmdHandlerFxn (argc, argv);
#if 0 /* this print is getting printed in CLI_task */
            if (cliStatus == 0)
            {
                /* Successfully executed the CLI command: */
                MmwDemo_CLI_write ("Done\n");
            }
            else
            {
                /* Error: The CLI command failed to execute */
                MmwDemo_CLI_write ("Error %d\n", cliStatus);
            }
#endif
            break;
        }

        /* Get the next entry: */
        ptrCLICommandEntry++;
    }

    /* Was this a valid CLI command? */
    if (ptrCLICommandEntry->cmdHandlerFxn == NULL)
    {
        /* NO: The command was not a valid CLI mmWave extension command. Setup
         * the return value correctly. */
        retVal = MMW_CLI_ERROR_CODE_INVALID_COMMAND;
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the mmWave extension handler which is invoked by the
 *      CLI Help command handler only if the extension was enabled.
 *
 *  \ingroup CLI_UTIL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
void CLI_RFMonitorExtensionHelp(void)
{
    CLI_CmdTableEntry*  ptrCLICommandEntry;

    /* Get the pointer to the mmWave extension table */
    ptrCLICommandEntry = &gCLIRFMonitorExtensionTable[0];

    /* Display the banner: */
    MmwDemo_CLI_write ("****************************************************\n");
    MmwDemo_CLI_write ("mmWave RF Monitor Extension Help\n");
    MmwDemo_CLI_write ("****************************************************\n");

    /* Cycle through all the registered externsion CLI commands: */
    while (ptrCLICommandEntry->cmdHandlerFxn != NULL)
    {
        /* Display the help string*/
        MmwDemo_CLI_write ("%s: %s\n",
                    ptrCLICommandEntry->cmd,
                   (ptrCLICommandEntry->helpString == NULL) ?
                    "No help available" :
                    ptrCLICommandEntry->helpString);

        /* Get the next entry: */
        ptrCLICommandEntry++;
    }
    return;
}

