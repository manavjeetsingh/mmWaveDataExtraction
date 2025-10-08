/**
 *   @file  mmw_cli.h
 *
 *   @brief
 *      This is the main header file for the CLI
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
/** @defgroup CLI_UTIL      CLI Utility
 */
#ifndef MMW_CLI_H
#define MMW_CLI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ti/drivers/uart/uart.h>
#include <./common/mmw_adcconfig.h>
#include <ti/sysbios/knl/Task.h>
#include <./common/mmw_adcconfig.h>

/**************************************************************************
 ************************* CLI Module Definitions *************************
 **************************************************************************/

/** @addtogroup CLI_UTIL_EXTERNAL_DEFINITION
 @{ */

/**
 * @brief   This is the maximum number of CLI commands which are supported
 */
#ifdef CLI_MAX_CMD
#undef CLI_MAX_CMD
#endif
#define MMW_CLI_ERROR_CODE_INVALID_COMMAND                  -50
#define MMW_CLI_ERROR_CODE_INVALID_USAGE_OF_CMD             -51
#define MMW_CLI_ERROR_CODE_INVALID_INPUT_PARAM              -52
#define MMW_CLI_ERROR_CODE_SENSOR_START_RECONFIG            -53
#define MMW_CLI_ERROR_CMD_SENS_ALREADY_START_STOP           -54
#define MMW_CLI_ERROR_LVDS_SW_HEADER_NOT_SUPPORTED          -55
#define MMW_CLI_ERROR_CMD_EXECUTE_TIME_OUT                  -56
#define MMW_CLI_ERROR_DATA_PATH_CONFIG                      -57
#define MMW_CLI_ERROR_CMD_AFTER_SENSOR_START                -58


#define     CLI_MAX_CMD         60
/* max chirp config stored in global structure */
#define MAX_CHIRP_CFG_STORED    5
/**
 * @brief
 *  This is the maximum number of CLI arguments which can be passed to a
 *  command.
 */
#define     MMW_CLI_MAX_ARGS        40

#define CLI_MAX_PARTNO_STRING_LEN          31U

/**
@}
*/

/**************************************************************************
 ************************** CLI Data Structures ***************************
 **************************************************************************/


/**
 * @brief
 *  Frame mode configuration
 *
 * @details
 *  The structure specifies the configuration which is required to configure
 *  the mmWave link to operate in frame mode
 */
typedef struct MMW_CLI_FrameCfg_t
{
    /**
     * @brief   List of all the active profile handles which can be configured.
     * Setting to NULL indicates that the profile is skipped.
     */
    MMWave_ProfileHandle    profileHandle[MMWAVE_MAX_PROFILE];

    /**
     * @brief   Configuration which is used to setup Frame
     */
    rlFrameCfg_t            frameCfg;
}MMW_CLI_FrameCfg;

/**
 * @brief
 *  Advanced frame configuration
 *
 * @details
 *  The structure specifies the configuration which is required to configure
 *  the mmWave link to operate in advanced frame mode
 */
typedef struct MMW_CLI_AdvancedFrameCfg_t
{
    /**
     * @brief   List of all the active profile handles which can be configured.
     * Setting to NULL indicates that the profile is skipped.
     */
    MMWave_ProfileHandle    profileHandle[MMWAVE_MAX_PROFILE];

    /**
     * @brief   Advanced Frame configuration
     */
    rlAdvFrameCfg_t         frameCfg;
}MMW_CLI_AdvancedFrameCfg;

/**
 * @brief
 *  Continuous mode configuration
 *
 * @details
 *  The structure specifies the configuration which is required to configure
 *  the mmWave link to operate in continuous mode
 */
typedef struct MMW_CLI_ContModeCfg_t
{
    /**
     * @brief   Continuous mode configuration
     */
    rlContModeCfg_t         contModecfg;

    /**
     * @brief   Sample count: This refers to the number of samples per
     * channel.
     */
    uint16_t                dataTransSize;
}MMW_CLI_ContModeCfg;

/**
 * @brief
 *  Control configuration
 *
 * @details
 *  The structure specifies the configuration which is required to configure
 *  and setup the BSS.
 */
typedef struct MMW_CLI_CtrlCfg_t
{//TODO JIT: define all these below struct locally
    /**
     * @brief   DFE Data Output Mode:
     */
    MMWave_DFEDataOutputMode        dfeDataOutputMode;

    union
    {
        /**
         * @brief   Chirp configuration to be used: This is only applicable
         * if the data output mode is set to Chirp
         */
        MMW_CLI_FrameCfg         frameCfg;

        /**
         * @brief   Continuous configuration to be used: This is only applicable
         * if the data output mode is set to Continuous
         */
        MMW_CLI_ContModeCfg          continuousModeCfg;

        /**
         * @brief   Advanced Frame configuration: This is only applicable
         * if the data output mode is set to Advanced Frame
         */
        MMW_CLI_AdvancedFrameCfg     advancedFrameCfg;
    }u;
}MMW_CLI_CtrlCfg;



/**
 * @brief
 *  Open Configuration
 *
 * @details
 *  The structure specifies the configuration which is required to open the
 *  MMWave module. Once the MMWave module has been opened the mmWave link
 *  to the BSS is operational.
 */
typedef struct MMW_CLI_OpenCfg_t
{
    /**
     * @brief   Low Frequency Limit for calibrations:
     */
    uint16_t                    freqLimitLow;

    /**
     * @brief   High Frequency Limit for calibrations
     */
    uint16_t                    freqLimitHigh;

    /**
     * @brief   Configuration which is used to setup channel
     */
    rlChanCfg_t                 chCfg;

    /**
     * @brief   Low power mode configuration
     */
    rlLowPowerModeCfg_t         lowPowerMode;

    /**
     * @brief   Configuration which is used to setup ADC
     */
    rlAdcOutCfg_t               adcOutCfg;

#if (defined(SOC_XWR16XX) || defined(SOC_XWR18XX) || defined(SOC_XWR68XX))
    /**
     * @brief   Designate the default asynchronous event handler. By default
     * the BSS assumes that the default asynchronous event handler is the MSS. \n
     * Field Not valid for xwr14xx.
     */
    MMWave_DefaultAsyncEventHandler     defaultAsyncEventHandler;
#endif

    /**
     * @brief   Flag that determines if frame start async event is disabled.
     * For more information refer to the mmWave Link documentation for:-
     * - RL_RF_AE_FRAME_TRIGGER_RDY_SB
     * - rlSensorStart
     */
    bool                disableFrameStartAsyncEvent;

    /**
     * @brief   Flag that determines if frame stop async event is disabled.
     * For more information refer to the mmWave Link documentation for:-
     * - RL_RF_AE_FRAME_END_SB
     * - rlSensorStop
     */
    bool                disableFrameStopAsyncEvent;

    /**
     * @brief   Set the flag to enable the application to specify the calibration
     * mask which is to be used. If the flag is set to false the MMWave module will
     * default and enable all the calibrations.
     */
    bool                useCustomCalibration;

    /**
     * @brief   This is the custom calibration enable mask which is to be used and
     * is applicable only if the application has enabled "Custom Calibration"
     */
    uint32_t            customCalibrationEnableMask;

    /**
     * @brief   Calibration Monitor time unit configuration in units of frame.
     *          Value of 1 here means Calibration Monitor time unit = 1 frame duration.
     *          See rlRfCalMonTimeUntConf_t for details on this configuration.
     */
    rlUInt16_t                  calibMonTimeUnit;
}MMW_CLI_OpenCfg;

/* stores all the recieved chirp and profile config */
typedef struct MMW_CLI_ProfChirpCfg_t
{
    uint8_t                     rcvProfileCfgCnt;
    rlProfileCfg_t              profileCfg[RL_MAX_PROFILES_CNT];
    uint16_t                    rcvdChirpCfgCnt;
    rlChirpCfg_t                chirpCfg[MAX_CHIRP_CFG_STORED];

}MMW_CLI_ProfChirpCfg;

/**
 * @brief
 *  Open Configuration
 *
 * @details
 *  The structure specifies the configuration which is required to open the
 *  MMWave module. Once the MMWave module has been opened the mmWave link
 *  to the BSS is operational.
 */
typedef struct MMW_CLI_AllCfg_t
{

    MMW_CLI_OpenCfg             openCfg;

    /* this stores frame/advFrame/contiModeCfg along with profileHandle */
    MMW_CLI_CtrlCfg             cliCtrlCfg;
    uint8_t                     subFrameCfgRcvCnt;

    Mmw_ADCBufCfg               adcBufCfg;

    rlRfLdoBypassCfg_t          ldoBypassCfg;
    
    rlDevHsiClk_t               hsiClkCfg;

    /**
     *
     * @brief ADC Bit and ADC Output format Configuration
     */
    rlAdcBitFormat_t            adcBitFormatCfg;

    uint8_t                     bpmCfgRcvCnt;
    rlBpmChirpCfg_t             bpmChirpCfg[MAX_CHIRP_CFG_STORED];

    MMW_CLI_ProfChirpCfg        profChirpCfg;

    rlTestSource_t              testSrcCfg;
    
    rlDynPwrSave_t              dynPwrSaveCfg;
    
    rlRfDevCfg_t                rfDevCfg;

    rlRfLdoBypassCfg_t          ldoByPassCfg;
    uint16_t                    rcvdPhShiftCfgCnt;
    rlRfPhaseShiftCfg_t         rfPhaseShiftCfg[MAX_CHIRP_CFG_STORED];

    rlRunTimeCalibConf_t        runTimeCalibCfg;

    rlRfBootStatusCfg_t         rfBootStatCfg;

    rlRfDieIdCfg_t              rfDieIdCfg;
 
#ifdef STORE_ADV_API
    rlRfPALoopbackCfg_t         rfPaLoopBckCfg;
    rlRfPSLoopbackCfg_t         rfPsLoopBckCfg;
    rlRfIFLoopbackCfg_t         rfIfLoopBckCfg;
    rlRfProgFiltCoeff_t         rfProgFiltCoeff;
    rlRfProgFiltConf_t          rfProgFiltConf;
    rlRfMiscConf_t              rfMiscConf;
    rlRfCalMonTimeUntConf_t     calMonTimeUntCfg;
#endif
    
    
}MMW_CLI_AllCfg;


/** @addtogroup CLI_UTIL_EXTERNAL_DATA_STRUCTURE
 @{ */

/**
 * @brief   Handle to the CLI module:
 */
typedef void*   MMW_CLI_Handle;

/**
 * @brief   CLI command handler:
 *
 *  @param[in]  argc
 *      Number of arguments
 *  @param[in]  argv
 *      Pointer to the arguments
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
typedef int32_t (*CLI_CmdHandler)(int32_t argc, char* argv[]);


/**
 * @brief
 *  CLI command table entry
 *
 * @details
 *  This is command entry which holds information which maps a
 *  command string to the corresponding command handler.
 */
typedef struct CLI_CmdTableEntry_t
{
    /**
     * @brief   Command string
     */
    char*               cmd;

    /**
     * @brief   CLI Command Help string
     */
    char*               helpString;

    /**
     * @brief   Command Handler to be executed
     */
    CLI_CmdHandler      cmdHandlerFxn;
}CLI_CmdTableEntry;

/**
 * @brief
 *  CLI configuration
 *
 * @details
 *  This is the configuration structure which is used to initialize and open
 *  the CLI module.
 */
typedef struct CLI_Cfg_t
{
    /**
     * @brief   CLI Prompt string (if any to be displayed)
     */
    char*               cliPrompt;

    /**
     * @brief   Optional banner string if any to be displayed on startup of the CLI
     */
    char*               cliBanner;

    /**
     * @brief   UART Command Handle used by the CLI
     */
    UART_Handle         cliUartHandle;

    /**
     * @brief   The CLI has an mmWave extension which can be enabled by this
     * field. The extension supports the well define mmWave link CLI command(s)
     * In order to use the extension the application should have initialized
     * and setup the mmWave.
     */
    uint8_t             enableMMWaveExtension;

    /**
     * @brief   The SOC driver handle is used to acquire device part number
     */
    SOC_Handle          socHandle;

    /**
     * @brief   The mmWave control handle which needs to be specified if
     * the mmWave extensions are being used. The CLI Utility works only
     * in the FULL configuration mode. If the handle is opened in
     * MINIMAL configuration mode the CLI mmWave extension will fail
     */
    MMWave_Handle       mmWaveHandle;

    /**
     * @brief   Task Priority: The CLI executes in the context of a task
     * which executes with this priority
     */
    uint8_t             taskPriority;

    /**
     * @brief   Flag which determines if the CLI Write should use the UART
     * in polled or blocking mode.
     */
    bool                usePolledMode;

    /**
     * @brief   Flag which determines if the CLI should override the platform
     * string reported in @ref CLI_MMWaveVersion.
     */
    bool                overridePlatform;

    /**
     * @brief   Optional platform string to be used in @ref CLI_MMWaveVersion
     */
    char*               overridePlatformString;

    /**
     * @brief   This is the table which specifies the supported CLI commands
     */
    CLI_CmdTableEntry   tableEntry[CLI_MAX_CMD];
}CLI_Cfg;

/**
@}
*/

/**
 * @brief
 *  CLI Master control block
 *
 * @details
 *  This is the MCB which tracks the CLI module
 */
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
    Task_Handle     cliTaskHandle;
}CLI_MCB;

/**
 * @brief
 *  CLI device part number information
 *
 * @details
 *  This is the struct to define part number and its corresponding string to be used in CLI
 */
typedef struct CLI_partInfoString_t
{
    uint8_t     partNumber;
    uint8_t     partNumString[CLI_MAX_PARTNO_STRING_LEN];
}CLI_partInfoString;



/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/
extern void MmwDemo_CLIInit (uint8_t taskPriority);
extern int32_t MmwDemo_CLI_open (CLI_Cfg* ptrCLICfg);
extern void    MmwDemo_CLI_write (const char* format, ...);
extern int32_t MmwDemo_CLI_close (void);

#ifdef __cplusplus
}
#endif

#endif /* MMW_CLI_H */

