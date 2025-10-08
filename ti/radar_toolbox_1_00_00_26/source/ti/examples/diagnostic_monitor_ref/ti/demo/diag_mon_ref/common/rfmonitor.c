/*
 *   @file  rfmonitor.c
 *
 *   @brief
 *      Mmwave link RF monitoring functions
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

/* BIOS/XDC Include Files. */
#include <xdc/runtime/System.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/control/mmwavelink/mmwavelink.h>
#include <ti/drivers/uart/UART.h>
#include <ti/utils/cli/cli.h>
#include <ti/drivers/osal/DebugP.h>

#include "common/diag_mon_config.h"
#include "common/rfmonitor_internal.h"
#include "common/rfmonitor.h"

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/
 /* Monitor configuration */
RFMonitorCfg     gRFMonitorCfg;
/* Monitor reports */
rfMonReport gRFMonReport;
rfMonFailureReport gRFMonFailureReport;

/* Global to track FTTI */
uint32_t    gLastFtti = 0;

/**Global variable to enable uart_writepolling */
volatile uint8_t temp_mon_report_enb = 0U;
volatile uint8_t tx0pow_mon_report_enb = 0U;

extern void MmwDemo_CLI_write (const char* format, ...);

/**************************************************************************
 *************************** Monitoring functions ******************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is used to update CQ monitor enable bits
 *
 *  @param[in]  rxSatMonEn          Flag indicates if rxSat monitor is enabled/disabled
 *  @param[in]  sigImgMonEn         Flag indicates if sigImg monitor is enabled/disabled
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void RFMon_updateCQEnMask(uint8_t rxSatMonEn, uint8_t sigImgMonEn)
{
    if(rxSatMonEn == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
                                                    RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
                                                    RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
                                                    1U);
    }

    if(sigImgMonEn == 0)
    {
        /* Monitor is disabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
                                                    RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
                                                    0U);
    }
    else
    {
        /* Monitor is enabled */
        gRFMonitorCfg.rfAnaMonitorEn->enMask = CSL_FINSR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                                                    RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
                                                    RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
                                                    1U);
    }
}

/**
 *  @b Description
 *  @n
 *      The function configures Analog monitors based on CLI inputs
 *  and default configurations.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t RFMon_configAnaMonitors(void)
{
    int32_t     retVal = 0;
    
    /*********************************
     * temperature monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_TEMP_EN_BIT, RFMON_ANAMON_TEMP_EN_BIT))
    {
        retVal = rlRfTempMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlTempMonConf_t*)gRFMonitorCfg.tempMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfTempMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * RX Gain Phase monitor
     * rxGainPhase monitor depends on
     * temperature report to calculate
     * rxGain and rxPhase
     *********************************/
	if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
	              RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT, RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT) )
	{
        retVal = rlRfRxGainPhMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                       (rlRxGainPhaseMonConf_t*)gRFMonitorCfg.rxGainPhaseMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfRxGainPhMonConfig retVal=%d\n", retVal);
        }
	}

	/*********************************
     * RX Noise monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_RX_NOISE_EN_BIT, RFMON_ANAMON_RX_NOISE_EN_BIT) )
    {
        retVal = rlRfRxNoiseMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                      (rlRxNoiseMonConf_t*)gRFMonitorCfg.rxNoiseFigMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfRxNoiseMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * RX IF Stage monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_RX_IFSTAGE_EN_BIT, RFMON_ANAMON_RX_IFSTAGE_EN_BIT) )
    {
        retVal = rlRfRxIfStageMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                        (rlRxIfStageMonConf_t*)gRFMonitorCfg.rxIfStageMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfRxIfStageMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * TX Power monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_TX2_POWER_EN_BIT, RFMON_ANAMON_TX0_POWER_EN_BIT) )
    {
        retVal = rlRfTxPowrMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                     (rlAllTxPowMonConf_t*)&gRFMonitorCfg.allTxPowerMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfTxPowrMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * Tx Ball break monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_TX2_BALLBREAK_EN_BIT, RFMON_ANAMON_TX0_BALLBREAK_EN_BIT) )
    {
        retVal = rlRfTxBallbreakMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                          (rlAllTxBallBreakMonCfg_t*)&gRFMonitorCfg.allTxBallbreakMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfTxBallbreakMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * Tx Gain Phase monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_TX_GAIN_PHASE_EN_BIT, RFMON_ANAMON_TX_GAIN_PHASE_EN_BIT) )
    {
        retVal = rlRfTxGainPhaseMismatchMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                                  (rlTxGainPhaseMismatchMonConf_t*)gRFMonitorCfg.txGainPhMisMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfTxGainPhaseMismatchMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * TX BPM monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_TX2_BPM_EN_BIT, RFMON_ANAMON_TX0_BPM_EN_BIT) )
    {
        retVal = rlRfTxBpmMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                    (rlAllTxBpmMonConf_t*)&gRFMonitorCfg.allTxBpmMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfTxBpmMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * Synthesizer frequency monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_SYNTH_FREQ_EN_BIT, RFMON_ANAMON_SYNTH_FREQ_EN_BIT) )
    {
        retVal = rlRfSynthFreqMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                        (rlSynthFreqMonConf_t*)gRFMonitorCfg.synthFreqMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfSynthFreqMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * Ext analog signal monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT, RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT) )
    {
        retVal = rlRfExtAnaSignalsMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                            (rlExtAnaSignalsMonConf_t*)gRFMonitorCfg.extAnaSigMonCfg);

        if(retVal != 0)
        {
            System_printf("Error: rlRfExtAnaSignalsMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * TX Internal Analog Signal monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_INT_TX2_SIGNALS_EN_BIT, RFMON_ANAMON_INT_TX0_SIGNALS_EN_BIT) )
    {
        retVal = rlRfTxIntAnaSignalsMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                              (rlAllTxIntAnaSignalsMonConf_t*)&gRFMonitorCfg.allTxIntAnaSigMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfTxIntAnaSignalsMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * RX Internal Analog Signal monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT, RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT) )
    {
        retVal = rlRfRxIntAnaSignalsMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                              (rlRxIntAnaSignalsMonConf_t*)gRFMonitorCfg.rxIntAnaSigMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfRxIntAnaSignalsMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * PM clock internal analog signal monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT, RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT) )
    {
        retVal = rlRfPmClkLoIntAnaSignalsMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                                   (rlPmClkLoIntAnaSignalsMonConf_t*)gRFMonitorCfg.pmClkLoIntAnaSigMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfPmClkLoIntAnaSignalsMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * GPADC signal monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_GPADC_SIGNALS_EN_BIT, RFMON_ANAMON_GPADC_SIGNALS_EN_BIT) )
    {
        retVal = rlRfGpadcIntAnaSignalsMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                                 (rlGpadcIntAnaSignalsMonConf_t*)gRFMonitorCfg.gpadcIntAnaSigMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfGpadcIntAnaSignalsMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * PLL control voltage monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT, RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT) )
    {
        retVal = rlRfPllContrlVoltMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                            (rlPllContrVoltMonConf_t*)gRFMonitorCfg.pllConVoltMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfPllContrlVoltMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * Dual clock comp monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT, RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT) )
    {
        retVal = rlRfDualClkCompMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                          (rlDualClkCompMonConf_t*)gRFMonitorCfg.dualClkCompMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfDualClkCompMonConfig retVal=%d\n", retVal);
        }
    }

    /*********************************
     * RX IF Saturation monitor
     *********************************/
    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                  RFMON_ANAMON_RX_IF_SATURATION_EN_BIT, RFMON_ANAMON_RX_IF_SATURATION_EN_BIT) )
    {
        retVal = rlRfRxIfSatMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlRxSatMonConf_t*)gRFMonitorCfg.rxSatMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfRxIfSatMonConfig retVal=%d\n", retVal);
        }
    }

     /*********************************
      * RX Signal Image band monitor
      *********************************/
     if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                   RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT, RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT) )
     {
         retVal = rlRfRxSigImgMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlSigImgMonConf_t*)gRFMonitorCfg.sigImageBandMonCfg);
         if(retVal != 0)
         {
             System_printf("Error: rlRfRxSigImgMonConfig retVal=%d\n", retVal);
         }
     }

    /*********************************
     * RX MIX input power signal monitor
     *********************************/
     if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
                   RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT, RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT) )
     {
        retVal = rlRfRxMixerInPwrConfig(RL_DEVICE_MAP_INTERNAL_BSS, \
                                        (rlRxMixInPwrMonConf_t*)gRFMonitorCfg.rxMixInpwrMonCfg);
        if(retVal != 0)
        {
            System_printf("Error: rlRfRxMixerInPwrConfig retVal=%d\n", retVal);
        }
     }

    /*********************************
     * LastStep: enable analog monitor
     *********************************/
    retVal = rlRfAnaMonConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlMonAnaEnables_t*)gRFMonitorCfg.rfAnaMonitorEn);
    if(retVal != 0)
    {
        System_printf("Error: rlRfAnaMonConfig retVal=%d\n", retVal);
    }

    System_printf("Debug: Finished All analog monitor configurations to BSS\n");

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function configures digital monitors based on CLI inputs
 *  and default configurations.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t RFMon_configDigMonitors(void)
{
    int32_t         retVal;

    /*********************************
     * Enable digital monitors
     *********************************/
    /* Digital monitoring configuration */
    retVal = rlRfDigMonEnableConfig(RL_DEVICE_MAP_INTERNAL_BSS,  gRFMonitorCfg.rfDigMonitorEn);
    if(retVal != 0)
    {
        System_printf("Error: rlRfDigMonEnableConfig retVal=%d\n", retVal);
        return retVal;
    }
    System_printf("Debug: Finished rlRfDigMonEnableConfig configurations to BSS\n");

    /*********************************
     * Config digital monitors
     *********************************/
    /* Set Digital periodic monitor configuration */
    retVal = rlRfDigMonPeriodicConfig(RL_DEVICE_MAP_INTERNAL_BSS, gRFMonitorCfg.rfDigMonPeriodicCfg);
    if(retVal != 0)
    {
        System_printf("Error: rlRfDigMonPeriodicConfig retVal=%d\n", retVal);
        return retVal;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function initializes the RF monitor configuraiton and reports
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void RFMon_initCfg(  void)
{
    /* Reset configuration */
    memset((void *)&gRFMonitorCfg, 0, sizeof(RFMonitorCfg));

    /* Take the defaults, some parameter maybe modified by CLI command */
    gRFMonitorCfg.rfDigMonitorEn           = &gDigMonitorEn;
    gRFMonitorCfg.rfDigMonPeriodicCfg      = &gDigMonitorPeriod;
    gRFMonitorCfg.rfAnaMonitorEn           = &gRfAnaMonitorEn;
    gRFMonitorCfg.tempMonCfg               = &gTempMonCfg;
    gRFMonitorCfg.rxGainPhaseMonCfg        = &gRxGainPhaseMonCfg;
    gRFMonitorCfg.rxNoiseFigMonCfg         = &gRxNoiseMonCfg;
    gRFMonitorCfg.allTxPowerMonCfg.tx0PowrMonCfg = &gTx0PowMonCfg;
    gRFMonitorCfg.allTxPowerMonCfg.tx1PowrMonCfg = &gTx1PowMonCfg;
    gRFMonitorCfg.allTxPowerMonCfg.tx2PowrMonCfg = &gTx2PowMonCfg;
    gRFMonitorCfg.allTxBallbreakMonCfg.tx0BallBrkMonCfg = &gTx0BallBreakMonCfg;
    gRFMonitorCfg.allTxBallbreakMonCfg.tx1BallBrkMonCfg = &gTx1BallBreakMonCfg;
    gRFMonitorCfg.allTxBallbreakMonCfg.tx2BallBrkMonCfg = &gTx2BallBreakMonCfg;
    gRFMonitorCfg.synthFreqMonCfg = &gSynthFreqMonCfg;
    gRFMonitorCfg.pllConVoltMonCfg = &gPllConVoltMonCfg;
    gRFMonitorCfg.dualClkCompMonCfg = &gDualClkCompMonCfg;
    gRFMonitorCfg.rxIfStageMonCfg = &gRxIfStageMonCfg;
    gRFMonitorCfg.extAnaSigMonCfg = &gExtAnaSigMonCfg;
    gRFMonitorCfg.allTxIntAnaSigMonCfg.tx0IntAnaSgnlMonCfg = &gTx0IntAnaSigMonCfg;
    gRFMonitorCfg.allTxIntAnaSigMonCfg.tx1IntAnaSgnlMonCfg = &gTx1IntAnaSigMonCfg;
    gRFMonitorCfg.allTxIntAnaSigMonCfg.tx2IntAnaSgnlMonCfg = &gTx2IntAnaSigMonCfg;
    gRFMonitorCfg.rxIntAnaSigMonCfg = &gRxIntAnaSigMonCfg;
    gRFMonitorCfg.pmClkLoIntAnaSigMonCfg = &gPmClkLoIntAnaSigMonCfg;
    gRFMonitorCfg.gpadcIntAnaSigMonCfg = &gGpadcIntAnaSigMonCfg;
    gRFMonitorCfg.rxMixInpwrMonCfg = &gRxMixInpwrMonCfg;
    /* CQ is not available in this application
     gRFMonitorCfg.rxSatMonCfg      = &gRxIfSatMonCfg;
     gRFMonitorCfg.sigImageBandMonCfg = &gSigImgMonCfg;
    */
    /* Reset reports */
    memset((void *)&gRFMonReport, 0, sizeof(rfMonReport));
    memset((void *)&gRFMonFailureReport, 0, sizeof(rfMonFailureReport));
}


/**
 *  @b Description
 *  @n
 *      This function resets all monitor reports
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void RFMon_resetFailureReports(  void)
{
    /* Reset reports */
    memset((void *)&gRFMonReport, 0, sizeof(rfMonReport));
    memset((void *)&gRFMonFailureReport, 0, sizeof(rfMonFailureReport));
}

/**
 *  @b Description
 *  @n
 *      The function configures Digital and Analog monitors based on CLI inputs
 *  and default configurations.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t RFMon_config(int32_t *errCode)
{
    int32_t     retVal = 0;
#ifdef DIGITAL_MONITOR_FEATURE_EN
    retVal = RFMon_configDigMonitors();
    if(retVal !=0)
    {
        goto EXIT;
    }
#endif
    retVal = RFMon_configAnaMonitors();
    if(retVal !=0)
    {
        goto EXIT;
    }

    /* only one profile supported in this application */
    gRFMonitorCfg.rxGain = gProfileCfg.rxGain;

EXIT:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function reports monitoring failures stats to CLI UART port..
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void RFMon_reportStatsToHost(uint8_t checkFTTI)
{
    /* Monitor reports */
    if((checkFTTI == 0) || (gLastFtti != gRFMonReport.rfMonHdrReport.fttiCount))
    {
        /* Monitor report:
          FTTI::temp:digFailure:tempMon:rxGainPhase:rxIfStage:tx0Power:tx1Power:tx2Power:tx0Ballbreak:tx1Ballbreak:tx2Ballbreak:
                    synthFreq:pllconVolt:tx0IntSig:tx2IntSig:tx2IntSig:pmClkLoIntAnaSig:gpadc:dualClkComp:extAnaSig
        */
            MmwDemo_CLI_write("RFMon: %d::%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d\n",
                      gRFMonReport.rfMonHdrReport.fttiCount,
                      gRFMonReport.rfMonHdrReport.avgTemp,
                      gRFMonFailureReport.rfDigMonFailureCnt,
                      gRFMonFailureReport.rfDigPeriodFailureCnt,
                      gRFMonFailureReport.tempMonFailureCnt,
                      gRFMonFailureReport.rxGainPhaseFailureCnt,
                      gRFMonFailureReport.rxIfStageFailureCnt,
                      gRFMonFailureReport.rxIntAnaSigFailureCnt,
                      gRFMonFailureReport.tx0PowerFailureCnt,
                      gRFMonFailureReport.tx1PowerFailureCnt,
                      gRFMonFailureReport.tx2PowerFailureCnt,
                      gRFMonFailureReport.tx0BallbreakFailureCnt,
                      gRFMonFailureReport.tx1BallbreakFailureCnt,
                      gRFMonFailureReport.tx2BallbreakFailureCnt,
                      gRFMonFailureReport.synthFreqFailureCnt,
                      gRFMonFailureReport.pllConVoltFailureCnt,
                      gRFMonFailureReport.tx0IntAnaSigFailureCnt,
                      gRFMonFailureReport.tx1IntAnaSigFailureCnt,
                      gRFMonFailureReport.tx2IntAnaSigFailureCnt,
                      gRFMonFailureReport.pmClkLoIntAnaSigFailureCnt,
                      gRFMonFailureReport.gpadcIntAnaSigFailureCnt,
                      gRFMonFailureReport.dualClkCompFailureCnt,
                      gRFMonFailureReport.extAnaSigFailureCnt
                      //gRFMonFailureReport.rxMixInpwrFailureCnt defeatured
                      );

        /* Save last ftti */
        gLastFtti = gRFMonReport.rfMonHdrReport.fttiCount;
    }
}

/**
 *  @b Description
 *  @n
 *      The function reports monitoring failures through CLI UART port.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static void RFMon_reportFailureToHost(uint8_t monitor)
{
    switch(monitor)
    {
        case RF_DIG_PERIODIC_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Digital Monitor :0x%x\n",gRFMonReport.monDigPeriodReport.digMonPeriodicStatus);
            break;
        }
        case RF_DIG_LATENT_FAULT_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Digital Latent Fault :0x%x\n",gRFMonReport.monDigLatentFaultReport.digMonLatentFault);
            break;
        }
        case RF_TEMPERATURE_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Temperature Monitor, Status :0x%x ErrorCode :0x%x\n", gRFMonReport.monTempReport.statusFlags,\
                      gRFMonReport.monTempReport.errorCode);
            break;
        }
        case RF_RXGAIN_PHASE_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Rx Gain Phase Monitor, Status :0x%x ErrorCode :0x%x\n", gRFMonReport.monRxGainPhReport.statusFlags,\
                      gRFMonReport.monRxGainPhReport.errorCode);
            break;
        }
        case RF_RXIFA_STAGE_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Rx If Stage Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monRxIfStageReport.statusFlags,\
                      gRFMonReport.monRxIfStageReport.errorCode);
            break;
        }
        case RF_RX_INT_ANA_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Rx IntAna Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monRxIntAnaSigReport.statusFlags,\
                      gRFMonReport.monRxIntAnaSigReport.errorCode);
            break;
        }
        case RF_TX0_POWER_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx0 Power Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx0powReport.statusFlags,\
                      gRFMonReport.monTx0powReport.errorCode);
            break;
        }
        case RF_TX1_POWER_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx1 Power Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx1powReport.statusFlags,\
                      gRFMonReport.monTx1powReport.errorCode);
            break;
        }
        case RF_TX2_POWER_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx2 Power Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx2powReport.statusFlags,\
                      gRFMonReport.monTx2powReport.errorCode);
            break;
        }
        case RF_TX0_BALLBREAK_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx0 Ballbreak Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx0BallbreakReport.statusFlags,\
                      gRFMonReport.monTx0BallbreakReport.errorCode);
            break;
        }
        case RF_TX1_BALLBREAK_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx1 Ballbreak Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx1BallbreakReport.statusFlags,\
                      gRFMonReport.monTx1BallbreakReport.errorCode);
            break;
        }
        case RF_TX2_BALLBREAK_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx2 Ballbreak Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx2BallbreakReport.statusFlags,\
                      gRFMonReport.monTx2BallbreakReport.errorCode);
            break;
        }
        case RF_SYNTH_FREQ_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Synth Freq Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monSynthFreqReport.statusFlags,\
                      gRFMonReport.monSynthFreqReport.errorCode);
            break;
        }
        case RF_PLL_CTRL_VOL_MON:
        {
            MmwDemo_CLI_write("[FAILURE] PLL Ctrl Vol Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monPllConvVoltReport.statusFlags,\
                      gRFMonReport.monPllConvVoltReport.errorCode);
            break;
        }
        case RF_TX0_INT_ANA_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx0 IntAna Sig Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx0IntAnaSigReport.statusFlags,\
                      gRFMonReport.monTx0IntAnaSigReport.errorCode);
            break;
        }
        case RF_TX1_INT_ANA_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx1 IntAna Sig Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx1IntAnaSigReport.statusFlags,\
                      gRFMonReport.monTx1IntAnaSigReport.errorCode);
            break;
        }
        case RF_TX2_INT_ANA_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx2 IntAna Sig Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx2IntAnaSigReport.statusFlags,\
                      gRFMonReport.monTx2IntAnaSigReport.errorCode);
            break;
        }
        case RF_PMCLK_LO_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] PM CLK LO Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monPmClkIntAnaSigReport.statusFlags,\
                      gRFMonReport.monPmClkIntAnaSigReport.errorCode);
            break;
        }
        case RF_GPADC_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] PM CLK LO Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monGpadcIntAnaSigReport.statusFlags,\
                      gRFMonReport.monGpadcIntAnaSigReport.errorCode);
            break;
        }
        case RF_DCC_CLK_FREQ_MON:
        {
            MmwDemo_CLI_write("[FAILURE] DCC CLK Freq Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monDccClkFreqReport.statusFlags,\
                      gRFMonReport.monDccClkFreqReport.errorCode);
            break;
        }
        case RF_EXT_ANA_SIG_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Ext Ana Sig Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monExtAnaSigReport.statusFlags,\
                      gRFMonReport.monExtAnaSigReport.errorCode);
            break;
        }
        case RF_TX0_BPM_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx0 BPM Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx0BpmReport.statusFlags,\
                      gRFMonReport.monTx0BpmReport.errorCode);
            break;
        }
        case RF_TX1_BPM_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx1 BPM Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx1BpmReport.statusFlags,\
                      gRFMonReport.monTx1BpmReport.errorCode);
            break;
        }
        case RF_TX2_BPM_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Tx2 BPM Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monTx2BpmReport.statusFlags,\
                      gRFMonReport.monTx2BpmReport.errorCode);
            break;
        }
        case RF_RX_MIXER_IN_PWR_MON:
        {
            MmwDemo_CLI_write("[FAILURE] Rx Mixer Pwr Monitor, Status :0x%x ErrorCode :0x%x\n",gRFMonReport.monRxMixrInPwrReport.statusFlags,\
                      gRFMonReport.monRxMixrInPwrReport.errorCode);
            break;
        }
        default:
        {
            break;
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is monitoring report handler. Based on the report type, it either saves
 *  the reports or increment failure counter if failure is detected.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t RFMon_reportHandler(uint16_t msgId, uint16_t asyncSB, uint8_t *payload)
{
    int16_t monitorFailure = -1;
    int32_t retVal = 0;

    switch (msgId)
    {
        case RL_RF_ASYNC_EVENT_MSG:
        {
            /* Received Asynchronous Message: */
            switch (asyncSB)
            {
                case RL_RF_AE_MON_REPORT_HEADER_SB:
                {
                    gRFMonReport.gNumFtti++;
                    memcpy(&gRFMonReport.rfMonHdrReport, payload, sizeof(rlMonReportHdrData_t));
                    break;
                }
                case RL_RF_AE_MON_TEMPERATURE_REPORT_SB:
                {
                    /** Enable Temperature Monitor Report  **/
                    gmonReportData |= (1<< RF_TEMPERATURE_MON);
                    /* Save report */
                    memcpy(&gRFMonReport.monTempReport, payload, sizeof(rlMonTempReportData_t));
                    
                    if(gRFMonitorCfg.tempMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTempReport.errorCode != 0U)
                        {
                            gRFMonFailureReport.tempMonFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.tempMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tempMonFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TEMPERATURE_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_RX_GAIN_PHASE_REPORT:
                {
                    uint8_t rxAntIdx;
                    uint8_t rfFreqBandIdx;
                    uint8_t *rfTemp;
                    uint8_t validateFailed = 0;
                    uint32_t rxGainMax, rxGainMin;
                    uint32_t rxPhaseMax, rxPhaseMin;

                    gmonReportData |= (1<< RF_RXGAIN_PHASE_MON);
                    
                    memcpy((void*)&gRFMonReport.monRxGainPhReport, payload, sizeof(rlMonRxGainPhRep_t));
                    
                    /* The pointer will be treated as array with 3 elements */
                    rfTemp = &gRFMonReport.monRxGainPhReport.loopbackPowerRF1;

                    /* Noise Power shows a decreasing trend with
                    temperature with average noise power being -41.48 dBm
                    at 25 C and decreasing with slope = -7.74 dBm/100C .
                    Noise Power dBm (Temp[C]) =-0.0774 * Temp [C] -39.55 */

                    /* TODO: which temperature should be used here? avg or per RxAnt?
                       temperature monitor is modatory to be on? */
                    gRFMonReport.monRxGainPhRawVal.rxNoise = 
                        - ( gRFMonReport.rfMonHdrReport.avgTemp * 0.0774 + 39.55);

                    for(rfFreqBandIdx = 0; rfFreqBandIdx < RL_MON_RF_FREQ_CNT; rfFreqBandIdx++)
                    {
                        float loopbackPower;

                        rxGainMax = 0;
                        rxGainMin = 0xFFFFFFFF;
                        rxPhaseMax = 0;
                        rxPhaseMin = 0xFFFFFFFF;

                        gRFMonReport.monRxGainPhRawVal.loopBackPower[rfFreqBandIdx] = (uint32_t)(- 2 * rfTemp[rfFreqBandIdx]);

                        for(rxAntIdx = 0; rxAntIdx < RL_RX_CNT; rxAntIdx++)
                        {
                            /* LOOPBACK POWER TEMP dBm (Temp[C]) =-0.0665*Temp[C] – 34.53 */
                            loopbackPower = -(gRFMonReport.monTempReport.tempValues[rxAntIdx] * 0.0665 + 34.53);

                            /* The actual Rx gain of the device in dB = RX GAIN VALUE(Rf freq, Rx) + (-38dBm) – LOOPBACK POWER TEMP dBm(Temp[C]) */
                            gRFMonReport.monRxGainPhRawVal.rxGainVal[rfFreqBandIdx][rxAntIdx]= 
                                gRFMonReport.monRxGainPhReport.rxGainVal[rfFreqBandIdx * RL_RX_CNT + rxAntIdx] - (int16_t)(loopbackPower * 10) -380;

                            /* 1 LSB = 360/2^16*/
                            gRFMonReport.monRxGainPhRawVal.rxPhaseVal[rfFreqBandIdx][rxAntIdx]= 
                                (gRFMonReport.monRxGainPhReport.rxPhaseVal[rfFreqBandIdx * RL_RX_CNT + rxAntIdx] * 360) >>16;

                            /* Report raw values */
                            if(gRFMonitorCfg.reportCfg.enRxGainPhaseReport == 1U)
                            {
                                MmwDemo_CLI_write("RxGainPh(%d:%d) : \t%d \t%d \t%d : \t%d \t%d \t%d\n",
                                    rfFreqBandIdx, rxAntIdx,
                                    gRFMonReport.monRxGainPhReport.rxGainVal[rfFreqBandIdx * RL_RX_CNT + rxAntIdx],
                                    gRFMonReport.monRxGainPhReport.rxPhaseVal[rfFreqBandIdx * RL_RX_CNT + rxAntIdx],
                                    gRFMonReport.monRxGainPhRawVal.loopBackPower[rfFreqBandIdx],
                                    gRFMonReport.monRxGainPhRawVal.rxGainVal[rfFreqBandIdx][rxAntIdx],
                                    gRFMonReport.monRxGainPhRawVal.rxPhaseVal[rfFreqBandIdx][rxAntIdx],
                                    (uint32_t)gRFMonReport.monRxGainPhRawVal.rxNoise);
                            }

                            /* Validate rxGainAbsThresh */
                            if(abs(gRFMonReport.monRxGainPhRawVal.rxGainVal[rfFreqBandIdx][rxAntIdx] - gRFMonitorCfg.rxGain * 10) > 
                                 gRFMonitorCfg.rxGainPhaseMonCfg->rxGainAbsThresh)
                            {
                                validateFailed ++;
                            }

                            /* For now, no need to validate rxGainFlatnessErrThresh */

                            /* Find maxmin rxGain and rxPhase across all RF and RX antenna */
                            rxGainMax = MAX(rxGainMax, gRFMonReport.monRxGainPhRawVal.rxGainVal[rfFreqBandIdx][rxAntIdx]);
                            rxPhaseMax = MAX(rxPhaseMax, gRFMonReport.monRxGainPhRawVal.rxPhaseVal[rfFreqBandIdx][rxAntIdx]);
                            rxGainMin = MIN(rxGainMin, gRFMonReport.monRxGainPhRawVal.rxGainVal[rfFreqBandIdx][rxAntIdx]);
                            rxPhaseMin = MIN(rxPhaseMin, gRFMonReport.monRxGainPhRawVal.rxPhaseVal[rfFreqBandIdx][rxAntIdx]);
                        }

                        /* Validate rxGainMismatchErrThresh*/
                        if((rxGainMax - rxGainMin) > gRFMonitorCfg.rxGainPhaseMonCfg->rxGainMismatchErrThresh)
                        {
                            /* rxGainMismatchErrThresh failure */
                            validateFailed ++;
                        }

                        /* Validate rxGainPhaseMismatchErrThresh */
                        if((rxPhaseMax - rxPhaseMin) > gRFMonitorCfg.rxGainPhaseMonCfg->rxGainPhaseMismatchErrThresh)
                        {
                            /* rxGainMismatchErrThresh failure */
                            validateFailed ++;
                        }
                    }

                    if((gRFMonitorCfg.rxGainPhaseMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY) ||\
                            (validateFailed))
                    {
                        gRFMonFailureReport.rxGainPhaseFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_RXGAIN_PHASE_MON);
                    }

                    break;
                }
                case RL_RF_AE_MON_RX_IF_STAGE_REPORT:
                {
                    gmonReportData |= (1<< RF_RXIFA_STAGE_MON);
                    /* Save report */
                    memcpy((void*)&gRFMonReport.monRxIfStageReport, payload, sizeof(rlMonRxIfStageRep_t));
                    
                    if(gRFMonitorCfg.rxIfStageMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monRxIfStageReport.statusFlags != 0x7)
                        {
                            gRFMonFailureReport.rxIfStageFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.rxIfStageMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.rxIfStageFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_RXIFA_STAGE_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX0_POWER_REPORT:
                {
                    /* Save report */
                    memcpy((void*)&gRFMonReport.monTx0powReport, payload, sizeof(rlMonTxPowRep_t));
                    gmonReportData |= (1U<< RF_TX0_POWER_MON);
                    
                    if(gRFMonitorCfg.allTxPowerMonCfg.tx0PowrMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx0powReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx0PowerFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxPowerMonCfg.tx0PowrMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx0PowerFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX0_POWER_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX1_POWER_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX1_POWER_MON);
                    
                    memcpy((void*)&gRFMonReport.monTx1powReport, payload, sizeof(rlMonTxPowRep_t));

                    if(gRFMonitorCfg.allTxPowerMonCfg.tx1PowrMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx1powReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx1PowerFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxPowerMonCfg.tx1PowrMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx1PowerFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX1_POWER_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX2_POWER_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX2_POWER_MON);
                    
                    memcpy((void*)&gRFMonReport.monTx2powReport, payload, sizeof(rlMonTxPowRep_t));

                    if(gRFMonitorCfg.allTxPowerMonCfg.tx2PowrMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx2powReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx2PowerFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxPowerMonCfg.tx2PowrMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx2PowerFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX2_POWER_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX0_BALLBREAK_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX0_BALLBREAK_MON);
                    
                    memcpy((void*)&gRFMonReport.monTx0BallbreakReport, payload, sizeof(rlMonTxBallBreakRep_t));

                    if(gRFMonitorCfg.allTxBallbreakMonCfg.tx0BallBrkMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx0BallbreakReport.statusFlags != 0x1)
                        {
                            gRFMonFailureReport.tx0BallbreakFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxBallbreakMonCfg.tx0BallBrkMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx0BallbreakFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX0_BALLBREAK_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX1_BALLBREAK_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX1_BALLBREAK_MON);
                    memcpy((void*)&gRFMonReport.monTx1BallbreakReport, payload, sizeof(rlMonTxBallBreakRep_t));

                    if(gRFMonitorCfg.allTxBallbreakMonCfg.tx1BallBrkMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx1BallbreakReport.statusFlags != 0x1)
                        {
                            gRFMonFailureReport.tx1BallbreakFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxBallbreakMonCfg.tx1BallBrkMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx1BallbreakFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX1_BALLBREAK_MON);
                    }
                    break;
                }
                case RL_RF_AE_DIG_LATENTFAULT_REPORT_SB:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_DIG_LATENT_FAULT_MON);

                    memcpy((void*)&gRFMonReport.monDigLatentFaultReport, payload, sizeof(rlDigLatentFaultReportData_t));

                    if(gRFMonReport.monDigLatentFaultReport.digMonLatentFault != gRFMonitorCfg.rfDigMonitorEn->enMask)
                    {
                        gRFMonFailureReport.rfDigMonFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_DIG_LATENT_FAULT_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_DIG_PERIODIC_REPORT_SB:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_DIG_PERIODIC_MON);

                    memcpy((void*)&gRFMonReport.monDigPeriodReport, payload, sizeof(rlDigPeriodicReportData_t));

                    if(gRFMonitorCfg.rfDigMonPeriodicCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monDigPeriodReport.digMonPeriodicStatus != gRFMonitorCfg.rfDigMonPeriodicCfg->periodicEnableMask)
                        {
                            gRFMonFailureReport.rfDigPeriodFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.rfDigMonPeriodicCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.rfDigPeriodFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_DIG_PERIODIC_MON);
                    }
                    break;
                }
                case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_RUN_TIME_CALIB);
                    /* This is not monitoring report but calibration report, put here for easy merge in the future */
                    memcpy((void*)&gRFMonReport.runtimeCalibReport, payload, sizeof(rlRfRunTimeCalibReport_t));

                    if(gRFMonitorCfg.reportCfg.enCalibReport)
                    {
                        /* Output of calibration status to CLI */
                        MmwDemo_CLI_write("%d:CalibStatus:0x%x:0x%x, temp=%d\n",
                            gRFMonReport.rfMonHdrReport.fttiCount,
                            gRFMonReport.runtimeCalibReport.calibErrorFlag,
                            gRFMonReport.runtimeCalibReport.calibUpdateStatus,
                            gRFMonReport.runtimeCalibReport.temperature);
                    }

                    /* Updated calibration all passes */
                    if((gRFMonReport.runtimeCalibReport.calibErrorFlag & gRFMonReport.runtimeCalibReport.calibUpdateStatus) != 
                        gRFMonReport.runtimeCalibReport.calibUpdateStatus)
                    {
                        gRFMonFailureReport.runtimeCalibFailureCnt++;
                        MmwDemo_CLI_write("Runtime Calib failure\n");
                    }
                    break;
                }
                default:
                {
                    System_printf ("RFMonitor Error: Asynchronous Event SB Id %d not handled with msg ID [0x%x] \n", asyncSB,msgId);
                    retVal = -1;
                    break;
                }
            }
            break;
        }
        case RL_RF_ASYNC_EVENT_1_MSG:
        {
            switch (asyncSB)
            {
                case RL_RF_AE_MON_TX2_BALLBREAK_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX2_BALLBREAK_MON);
                    
                    memcpy((void*)&gRFMonReport.monTx2BallbreakReport, payload, sizeof(rlMonTxBallBreakRep_t));

                    if(gRFMonitorCfg.allTxBallbreakMonCfg.tx2BallBrkMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx2BallbreakReport.statusFlags != 0x1)
                        {
                            gRFMonFailureReport.tx2BallbreakFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxBallbreakMonCfg.tx2BallBrkMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx2BallbreakFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX2_BALLBREAK_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX_GAIN_PAHSE_MON);

                    memcpy((void*)&gRFMonReport.monTxPhaseMisReport, payload, sizeof(rlMonTxGainPhaMisRep_t));


                    if(gRFMonitorCfg.txGainPhMisMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.txGainPhaseFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX_GAIN_PAHSE_MON);
                    }

                    break;
                }
                case RL_RF_AE_MON_TX0_BPM_REPORT:
                {
                    gmonReportData |= (1U<< RF_TX0_BPM_MON);

                    /* Save report */
                    memcpy((void*)&gRFMonReport.monTx0BpmReport, payload, sizeof(rlMonTxBpmRep_t));

                    if(gRFMonitorCfg.allTxBpmMonCfg.tx0BpmMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx0BpmReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx0BpmFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxBpmMonCfg.tx0BpmMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx0BpmFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX0_BPM_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX1_BPM_REPORT:
                {
                    /* Save report */
                    memcpy((void*)&gRFMonReport.monTx1BpmReport, payload, sizeof(rlMonTxBpmRep_t));

                    if(gRFMonitorCfg.allTxBpmMonCfg.tx1BpmMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx1BpmReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx1BpmFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxBpmMonCfg.tx1BpmMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx1BpmFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX1_BPM_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX2_BPM_REPORT:
                {
                    /* Save report */
                    memcpy((void*)&gRFMonReport.monTx2BpmReport, payload, sizeof(rlMonTxBpmRep_t));

                    if(gRFMonitorCfg.allTxBpmMonCfg.tx2BpmMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx2BpmReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx2BpmFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxBpmMonCfg.tx2BpmMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx2BpmFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX2_BPM_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_SYNTH_FREQ_MON);
                    memcpy((void*)&gRFMonReport.monSynthFreqReport, payload, sizeof(rlMonSynthFreqRep_t));

                    if(gRFMonitorCfg.synthFreqMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monSynthFreqReport.statusFlags != 0x1)
                        {
                            gRFMonFailureReport.synthFreqFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.synthFreqMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.synthFreqFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_SYNTH_FREQ_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_EXT_ANA_SIG_MON);
                    memcpy((void*)&gRFMonReport.monExtAnaSigReport, payload, sizeof(rlMonExtAnaSigRep_t));

                    if(gRFMonitorCfg.extAnaSigMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {

                        /* TODO : there 2 enable variable signalInpEnables & signalBuffEnables, however there is only one report */
                        if(gRFMonReport.monExtAnaSigReport.statusFlags != gRFMonitorCfg.extAnaSigMonCfg->signalInpEnables)
                        {
                            gRFMonFailureReport.extAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.extAnaSigMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.extAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_EXT_ANA_SIG_MON);
                    }
                    break;
                }

                case RL_RF_AE_MON_TX0_INT_ANA_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX0_INT_ANA_SIG_MON);
                    memcpy((void*)&gRFMonReport.monTx0IntAnaSigReport, payload, sizeof(rlMonTxIntAnaSigRep_t));

                    if(gRFMonitorCfg.allTxIntAnaSigMonCfg.tx0IntAnaSgnlMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx0IntAnaSigReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx0IntAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxIntAnaSigMonCfg.tx0IntAnaSgnlMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx0IntAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX0_INT_ANA_SIG_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX1_INT_ANA_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX1_INT_ANA_SIG_MON);
                    memcpy((void*)&gRFMonReport.monTx1IntAnaSigReport, payload, sizeof(rlMonTxIntAnaSigRep_t));

                    if(gRFMonitorCfg.allTxIntAnaSigMonCfg.tx1IntAnaSgnlMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx1IntAnaSigReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx1IntAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxIntAnaSigMonCfg.tx1IntAnaSgnlMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx1IntAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX1_INT_ANA_SIG_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX2_INT_ANA_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX2_INT_ANA_SIG_MON);
                    
                    memcpy((void*)&gRFMonReport.monTx2IntAnaSigReport, payload, sizeof(rlMonTxIntAnaSigRep_t));

                    if(gRFMonitorCfg.allTxIntAnaSigMonCfg.tx2IntAnaSgnlMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monTx2IntAnaSigReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.tx2IntAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.allTxIntAnaSigMonCfg.tx2IntAnaSgnlMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.tx2IntAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_TX2_INT_ANA_SIG_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_RX_INT_ANALOG_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_RX_INT_ANA_SIG_MON);
                    
                    memcpy((void*)&gRFMonReport.monRxIntAnaSigReport, payload, sizeof(rlMonRxIntAnaSigRep_t));

                    if(gRFMonitorCfg.rxIntAnaSigMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monRxIntAnaSigReport.statusFlags != 0xFFF)
                        {
                            gRFMonFailureReport.rxIntAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.rxIntAnaSigMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.rxIntAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_RX_INT_ANA_SIG_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_PMCLK_LO_SIG_MON);
                    
                    memcpy((void*)&gRFMonReport.monPmClkIntAnaSigReport, payload, sizeof(rlMonPmclkloIntAnaSigRep_t));

                    if(gRFMonitorCfg.pmClkLoIntAnaSigMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if ((gRFMonitorCfg.pmClkLoIntAnaSigMonCfg->sync20GSigSel != 0) &&
                          (gRFMonReport.monPmClkIntAnaSigReport.statusFlags != 0xF))
                        {
                            gRFMonFailureReport.pmClkLoIntAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.pmClkLoIntAnaSigMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.pmClkLoIntAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_PMCLK_LO_SIG_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_GPADC_SIG_MON);

                    memcpy((void*)&gRFMonReport.monGpadcIntAnaSigReport, payload, sizeof(rlMonGpadcIntAnaSigRep_t));

                    if(gRFMonitorCfg.gpadcIntAnaSigMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monGpadcIntAnaSigReport.statusFlags != 0x3)
                        {
                            gRFMonFailureReport.gpadcIntAnaSigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.gpadcIntAnaSigMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.gpadcIntAnaSigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_GPADC_SIG_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_PLL_CTRL_VOL_MON);
                    
                    memcpy((void*)&gRFMonReport.monPllConvVoltReport, payload, sizeof(rlMonPllConVoltRep_t));
                    
                    if(gRFMonitorCfg.pllConVoltMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(((gRFMonitorCfg.pllConVoltMonCfg->signalEnables & 0x1) &&
                          (gRFMonReport.monPllConvVoltReport.statusFlags != 0x1) ) ||
                            ((gRFMonitorCfg.pllConVoltMonCfg->signalEnables & 0x2) &&
                          (gRFMonReport.monPllConvVoltReport.statusFlags != 0xE) ) ||
                            ((gRFMonitorCfg.pllConVoltMonCfg->signalEnables & 0x4) &&
                          (gRFMonReport.monPllConvVoltReport.statusFlags != 0x70) ) )
                        {
                            gRFMonFailureReport.pllConVoltFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.pllConVoltMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.pllConVoltFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_PLL_CTRL_VOL_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_DCC_CLK_FREQ_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_DCC_CLK_FREQ_MON);
                    memcpy((void*)&gRFMonReport.monDccClkFreqReport, payload, sizeof(rlMonDccClkFreqRep_t));

                    if(gRFMonitorCfg.dualClkCompMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monDccClkFreqReport.statusFlags != gRFMonitorCfg.dualClkCompMonCfg->dccPairEnables)
                        {
                            gRFMonFailureReport.dualClkCompFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.dualClkCompMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.dualClkCompFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = (RF_DCC_CLK_FREQ_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_RX_MIXER_IN_PWR_REPORT:
                {
                    gmonReportData |= RF_RX_MIXER_IN_PWR_MON;
                    /* Save report */
                    memcpy((void*)&gRFMonReport.monRxMixrInPwrReport, payload, sizeof(rlMonRxMixrInPwrRep_t));

                    if(gRFMonitorCfg.rxMixInpwrMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monRxMixrInPwrReport.statusFlags != 0xF)
                        {
                            gRFMonFailureReport.rxMixInpwrFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.rxMixInpwrMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.rxMixInpwrFailureCnt++;

                        /* Set failure bit */
                        monitorFailure = RF_RX_MIXER_IN_PWR_MON;
                    }
                    break;
                }
                default:
                {
                    System_printf ("RF monitor Error: Asynchronous Event SB Id %d not handled with msg ID 0x%x\n", asyncSB,msgId);
                    retVal = -1;
                    break;
                }
            }
            break;
        }
        default:
        {
            System_printf ("RFMonitor Error: Asynchronous message %d is NOT handled\n", msgId);
            retVal = -1;
            break;
        }
    }

    /* Check if there is a failure */
    if(monitorFailure >= 0)
    {
        /* Report failure during active frame if enabled */
        if(gRFMonitorCfg.reportCfg.enMonFailureReport == MON_REPORT_MODE_AT_FAILURE_ONLY)
        {
            RFMon_reportFailureToHost((uint8_t)monitorFailure);
        }
    }
    return retVal;
}

