/*
 *   @file  rfmonitor.c
 *
 *   @brief
 *      Mmwave link RF monitoring functions
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

#include "common/mmw_config.h"
#include "common/rfmonitor_internal.h"
#include "common/rfmonitor.h"

/* External definiton */
extern uint16_t MmwDemo_getProfileRxGain(uint16_t profileIdx);

#ifdef CLI_DBG_MON_REPORTING
#define DBG_MON_REPORT_WRITE     MmwDemo_CLI_write
#else
#define DBG_MON_REPORT_WRITE    System_printf
#endif
/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/
 /* Monitor configuration */
RFMonitorCfg     gRFMonitorCfg;

/* Monitor reports */
rfMonReport gRFMonReport;
rfMonFailureReport gRFMonFailureReport;

/* Globle to track FTTI */
uint32_t    gLastFtti = 0;

/**Global variable to enable uart_writepolling */
volatile uint8_t temp_mon_report_enb = 0U;
volatile uint8_t tx0pow_mon_report_enb = 0U;

/**************************************************************************
 *************************** Monitoring functions ******************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is used to update CQ monitor enable bits
 *
 *  @param[in]  rxSatMonEn          Flag indicates if rxSat monitor is enalbed/disabed
 *  @param[in]  sigImgMonEn         Flag indicates if sigImg monitor is enalbed/disabed
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

    gRFMonitorCfg.rfDigMonitorEn->testMode = 0; /* Production mode */
#if 0
    gRFMonitorCfg.rfDigMonitorEn->enMask = CSL_FMKR (1U, 1U,  1U) |    /* CR4 and VIM lockstep test of diagnostic */
                                      CSL_FMKR (3U,  3U, 1U) |     /* VIM test of diagnostic */
                                      CSL_FMKR (6U,  6U, 1U) |     /* CRC test of diagnostic */
                                      CSL_FMKR (8U, 8U, 1U) |       /* DFE Parity */
                                      CSL_FMKR (9U, 9U, 1U) |       /* DFE memory ECC  */
                                      CSL_FMKR (10U, 10U, 1U) |     /* RAMPGEN lockstep test */
                                      CSL_FMKR (11U, 11U, 1U) |     /* FRC lockstep test */
                                      CSL_FMKR (16U, 16U, 1U) |     /* ESM test */
                                      CSL_FMKR (17U, 17U, 1U) |     /* DFE STC */
                                      CSL_FMKR (19U, 19U, 1U) |     /* ATCM, BTCM ECC */
                                      CSL_FMKR (20U, 20U, 1U) |     /* ATCM, BTCM parity */
                                      CSL_FMKR (24U, 24U, 1U) |     /* FFT test */
                                      CSL_FMKR (25U, 25U, 1U) ;     /* RTI test */
#endif
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
    gRFMonitorCfg.allTxPowerMonCfg.tx0PowrMonCfg = &gTxPowMonCfg[0];
    gRFMonitorCfg.allTxPowerMonCfg.tx1PowrMonCfg = &gTxPowMonCfg[1];
    gRFMonitorCfg.allTxPowerMonCfg.tx2PowrMonCfg = &gTxPowMonCfg[2];
    gRFMonitorCfg.allTxBallbreakMonCfg.tx0BallBrkMonCfg = &gTxBallbreakMonCfg[0];
    gRFMonitorCfg.allTxBallbreakMonCfg.tx1BallBrkMonCfg = &gTxBallbreakMonCfg[1];
    gRFMonitorCfg.allTxBallbreakMonCfg.tx2BallBrkMonCfg = &gTxBallbreakMonCfg[2];
    gRFMonitorCfg.synthFreqMonCfg = &gSynthFreqMonCfg;
    gRFMonitorCfg.pllConVoltMonCfg = &gPllConVoltMonCfg;
    gRFMonitorCfg.dualClkCompMonCfg = &gDualClkCompMonCfg;
    gRFMonitorCfg.rxIfStageMonCfg = &gRxIfStageMonCfg;
    gRFMonitorCfg.extAnaSigMonCfg = &gExtAnaSigMonCfg;
    gRFMonitorCfg.allTxIntAnaSigMonCfg.tx0IntAnaSgnlMonCfg = &gTxIntAnaSigMonCfg[0];
    gRFMonitorCfg.allTxIntAnaSigMonCfg.tx1IntAnaSgnlMonCfg = &gTxIntAnaSigMonCfg[1];
    gRFMonitorCfg.allTxIntAnaSigMonCfg.tx2IntAnaSgnlMonCfg = &gTxIntAnaSigMonCfg[2];
    gRFMonitorCfg.rxIntAnaSigMonCfg = &gRxIntAnaSigMonCfg;
    gRFMonitorCfg.pmClkLoIntAnaSigMonCfg = &gPmClkLoIntAnaSigMonCfg;
    gRFMonitorCfg.gpadcIntAnaSigMonCfg = &gGpadcIntAnaSigMonCfg;
    gRFMonitorCfg.rxMixInpwrMonCfg = &gRxMixInpwrMonCfg;
    gRFMonitorCfg.rxSatMonCfg      = &gRxIfSatMonCfg;
    gRFMonitorCfg.sigImageBandMonCfg = &gSigImgMonCfg;

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

    /* Enable Digital monitor if requested over CLI CMD */
    if(gRFMonitorCfg.rfDigMonitorEn->enMask)
    {
        retVal = RFMon_configDigMonitors();
        if(retVal !=0)
        {
            goto EXIT;
        }
    }

    retVal = RFMon_configAnaMonitors();
    if(retVal !=0)
    {
        goto EXIT;
    }

    /* Assumes that the parsing of RF configuration is already done */
    retVal = MmwDemo_getProfileRxGain(gRFMonitorCfg.rxGainPhaseMonCfg->profileIndx);
    if(retVal < 0)
    {
        /* TODO: define errCode */
        *errCode = -1;
        return -1;
    }
    else
    {
        gRFMonitorCfg.rxGain = (uint16_t)retVal;
    }
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
        /* Monitor stats report*/
        DBG_MON_REPORT_WRITE("RFMon: %d::%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d\n",
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
#if 0
    switch(monitor)
    {
        case RFMON_DIGITAL_BIT:
        {
            DBG_MON_REPORT_WRITE("DigMon failure:0x%x\n",gRFMonReport.monDigLatentFaultReport.digMonLatentFault);
            break;
        }
        case RFMON_DIGPERIOD_BIT:
        {
            DBG_MON_REPORT_WRITE("DigPeriodic failure:0x%x\n",gRFMonReport.monDigLatentFaultReport.digMonLatentFault);
            break;
        }
        case RFMON_TEMP_BIT:
        {
            DBG_MON_REPORT_WRITE("tempMon failure:%x\n", gRFMonReport.monTempReport.errorCode);
            break;
        }
        case RFMON_RXGAINPHASE_BIT:
        {
            DBG_MON_REPORT_WRITE("rxGainPhase failure\n");
            break;
        }
        case RFMON_RXIFSTAGE_BIT:
        {
            DBG_MON_REPORT_WRITE("rxIfStage failure:0x%x\n",gRFMonReport.monRxIfStageReport.statusFlags);
            break;
        }
        case RFMON_RXINTANA_BIT:
        {
            DBG_MON_REPORT_WRITE("rxIntAna failure:0x%x\n", gRFMonReport.monRxIntAnaSigReport.statusFlags);
            break;
        }
        case RFMON_TX0POWER_BIT:
        {
            DBG_MON_REPORT_WRITE("tx0Power failure:0x%x\n", gRFMonReport.monTx0powReport.statusFlags);
            break;
        }
        case RFMON_TX1POWER_BIT:
        {
            DBG_MON_REPORT_WRITE("tx1Power failure:0x%x\n", gRFMonReport.monTx1powReport.statusFlags);
            break;
        }
        case RFMON_TX2POWER_BIT:
        {
            DBG_MON_REPORT_WRITE("tx2Power failure:0x%x\n", gRFMonReport.monTx2powReport.statusFlags);
            break;
        }
        case RFMON_TX0BALLBREAK_BIT:
        {
            DBG_MON_REPORT_WRITE("tx0Ballbreak failure:0x%x\n", gRFMonReport.monTx0BallbreakReport.statusFlags);
            break;
        }
        case RFMON_TX1BALLBREAK_BIT:
        {
            DBG_MON_REPORT_WRITE("tx1Ballbreak failure:0x%x\n", gRFMonReport.monTx1BallbreakReport.statusFlags);
            break;
        }
        case RFMON_TX2BALLBREAK_BIT:
        {
            DBG_MON_REPORT_WRITE("tx2Ballbreak failure:0x%x\n", gRFMonReport.monTx2BallbreakReport.statusFlags);
            break;
        }
        case RFMON_SYNTHFREQ_BIT:
        {
            DBG_MON_REPORT_WRITE("synthFreq failure:0x%x\n", gRFMonReport.monSynthFreqReport.statusFlags);
            break;
        }
        case RFMON_PLLCONVOLT_BIT:
        {
            DBG_MON_REPORT_WRITE("pllConVolt failure:0x%x\n", gRFMonReport.monPllConvVoltReport.statusFlags );
            break;
        }
        case RFMON_TX0INTANA_BIT:
        {
            DBG_MON_REPORT_WRITE("tx0IntAna failure:0x%x\n", gRFMonReport.monTx0IntAnaSigReport.statusFlags );
            break;
        }
        case RFMON_TX1INTANA_BIT:
        {
            DBG_MON_REPORT_WRITE("tx1IntAna failure:0x%x\n", gRFMonReport.monTx1IntAnaSigReport.statusFlags );
            break;
        }
        case RFMON_TX2INTANA_BIT:
        {
            DBG_MON_REPORT_WRITE("tx2IntAna failure:0x%x\n", gRFMonReport.monTx2IntAnaSigReport.statusFlags );
            break;
        }
        case RFMON_PMCLKLO_BIT:
        {
            DBG_MON_REPORT_WRITE("pmclklo failure:0x%x\n", gRFMonReport.monPmClkIntAnaSigReport.statusFlags );
            break;
        }
        case RFMON_GPADCINTANA_BIT:
        {
            DBG_MON_REPORT_WRITE("gpadcIntAna failure:0x%x\n", gRFMonReport.monGpadcIntAnaSigReport.statusFlags );
            break;
        }
        case RFMON_DUALCLKCOMP_BIT:
        {
            DBG_MON_REPORT_WRITE("dualClk failure:0x%x\n", gRFMonReport.monDccClkFreqReport.statusFlags );
            break;
        }
        case RFMON_EXTANA_BIT:
        {
            DBG_MON_REPORT_WRITE("extAna failure:0x%x\n", gRFMonReport.monExtAnaSigReport.statusFlags );
            break;
        }
        case RFMON_TX0BPM_BIT:
        {
            DBG_MON_REPORT_WRITE("tx0Bpm failure:0x%x\n", gRFMonReport.monTx0BpmReport.statusFlags );
            break;
        }
        case RFMON_TX1BPM_BIT:
        {
            DBG_MON_REPORT_WRITE("tx1Bpm failure:0x%x\n", gRFMonReport.monTx1BpmReport.statusFlags );
            break;
        }
        case RFMON_TX2BPM_BIT:
        {
            DBG_MON_REPORT_WRITE("tx2Bpm failure:0x%x\n", gRFMonReport.monTx2BpmReport.statusFlags );
            break;
        }
        case RFMON_RXMIXER_BIT:
        {
            DBG_MON_REPORT_WRITE("rxMixer failure:0x%x\n", gRFMonReport.monRxMixrInPwrReport.statusFlags);
            break;
        }
        default:
        {
            break;
        }
    }
#endif
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
void RFMon_monReportToHost(void)
{
    uint8_t failureBit = 0;

    /* Monitor failures: only for MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK mode
      since monFailureMask is only modified in that mode.
     */
    if ((gRFMonReport.monFailureMask != 0) && 
       (gRFMonitorCfg.reportCfg.enMonReportMode== 2U) )
    {
        while(gRFMonReport.monFailureMask !=0)
        {
            if((gRFMonReport.monFailureMask & 0x1) != 0U)
            {
                RFMon_reportFailureToHost(failureBit);
            }
            failureBit++;
            gRFMonReport.monFailureMask = gRFMonReport.monFailureMask >>1;
        }
    }

    /* Reset monFailureMask */
    gRFMonReport.monFailureMask = 0;

    /* Report rxGainphase raw values */
    if(gRFMonitorCfg.reportCfg.enRxGainPhaseReport == 2U)
    {
        uint8_t rfFreqBandIdx;
        uint8_t rxAntIdx;

        for(rfFreqBandIdx=0; rfFreqBandIdx< RL_MON_RF_FREQ_CNT; rfFreqBandIdx++)
        {
            for(rxAntIdx=0; rxAntIdx< RL_RX_CNT; rxAntIdx++)
            {
                DBG_MON_REPORT_WRITE("RxGainPh(%d:%d) : \t%d \t%d \t%d : \t%d \t%d \t%d\n",
                    rfFreqBandIdx, rxAntIdx,
                    gRFMonReport.monRxGainPhReport.rxGainVal[rfFreqBandIdx * RL_RX_CNT + rxAntIdx],
                    gRFMonReport.monRxGainPhReport.rxPhaseVal[rfFreqBandIdx * RL_RX_CNT + rxAntIdx],
                    gRFMonReport.monRxGainPhRawVal.loopBackPower[rfFreqBandIdx],
                    gRFMonReport.monRxGainPhRawVal.rxGainVal[rfFreqBandIdx][rxAntIdx],
                    gRFMonReport.monRxGainPhRawVal.rxPhaseVal[rfFreqBandIdx][rxAntIdx],
                    (uint32_t)gRFMonReport.monRxGainPhRawVal.rxNoise);
            }
        }
    }

    /* Report monitor Stats if enabled */
    if(gRFMonitorCfg.reportCfg.enMonStats)
    {
        RFMon_reportStatsToHost(1);
    }
}

/**
 *  @b Description
 *  @n
 *      The function is minitoring report handler. Based on the report type, it either saves
 *  the reports or increment failure counter if failure is detected.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void RFMon_reportHandler(uint16_t msgId, uint16_t asyncSB, uint8_t *payload)
{
    int32_t monitorFailure = 0;

    switch (msgId)
    {
        case RL_RF_ASYNC_EVENT_MSG:
        {
            /* Received Asychronous Message: */
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
                    /** Enable Temperaure Monitor Report  **/
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
                        monitorFailure |= (1<< RF_TEMPERATURE_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_RX_GAIN_PHASE_REPORT:
                {

                    gmonReportData |= (1<< RF_RXGAIN_PHASE_MON);
                    
                    memcpy((void*)&gRFMonReport.monRxGainPhReport, payload, sizeof(rlMonRxGainPhRep_t));
                    
                    if(gRFMonitorCfg.rxGainPhaseMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monRxGainPhReport.errorCode != 0U)
                        {
                            gRFMonFailureReport.rxGainPhaseFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.rxGainPhaseMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.rxGainPhaseFailureCnt++;

                        /* Set failure bit */
                        monitorFailure |= (1<< RF_RXGAIN_PHASE_MON);
                    }

                    break;
                }
                case RL_RF_AE_MON_RX_NOISE_FIG_REPORT:
                {

                    gmonReportData |= (1<< RF_RX_NOISE_MON);

                    memcpy((void*)&gRFMonReport.monRxNoiseFigReport, payload, sizeof(rlMonRxNoiseFigRep_t));

                    if(gRFMonitorCfg.rxNoiseFigMonCfg->reportMode == MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK)
                    {
                        if(gRFMonReport.monRxGainPhReport.errorCode != 0U)
                        {
                            gRFMonFailureReport.rxNoiseFigFailureCnt++;
                        }
                    }

                    if(gRFMonitorCfg.rxNoiseFigMonCfg->reportMode == MON_REPORT_MODE_AT_FAILURE_ONLY)
                    {
                        gRFMonFailureReport.rxNoiseFigFailureCnt++;

                        /* Set failure bit */
                        monitorFailure |= (1<< RF_RX_NOISE_MON);
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
                        monitorFailure |= (1<< RF_RXIFA_STAGE_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX0_POWER_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX0_POWER_MON);
                    memcpy((void*)&gRFMonReport.monTx0powReport, payload, sizeof(rlMonTxPowRep_t));

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
                        monitorFailure |= (1U<< RF_TX0_POWER_MON);
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
                        monitorFailure |= (1U<< RF_TX1_POWER_MON);
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
                        monitorFailure |= (1U<< RF_TX2_POWER_MON);
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
                        monitorFailure |= (1U<< RF_TX0_BALLBREAK_MON);
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
                        monitorFailure |= (1U<< RF_TX1_BALLBREAK_MON);
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
                        monitorFailure |= (1U<< RF_DIG_LATENT_FAULT_MON);
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
                        monitorFailure |= (1U<< RF_DIG_PERIODIC_MON);
                    }
                    break;
                }
                case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
                {

                    if(gRFMonitorCfg.reportCfg.enCalibReport)
                    {
                        /* Save report */
                        gmonReportData |= (1U<< RF_RUN_TIME_CALIB);
                        /* This is not monitoring report but calibration report, put here for easy merge in the future */
                        memcpy((void*)&gRFMonReport.runtimeCalibReport, payload, sizeof(rlRfRunTimeCalibReport_t));
                    }

                    /* Updated calibration all passes */
                    if((gRFMonReport.runtimeCalibReport.calibErrorFlag & gRFMonReport.runtimeCalibReport.calibUpdateStatus) != 
                        gRFMonReport.runtimeCalibReport.calibUpdateStatus)
                    {
                        gRFMonFailureReport.runtimeCalibFailureCnt++;
                        DBG_MON_REPORT_WRITE("Runtime Calib failure\n");
                    }
                    break;
                }
                default:
                {
                    System_printf ("RFMonitor Error: Asynchronous Event SB Id %d not handled with msg ID [0x%x] \n", asyncSB,msgId);
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
                        monitorFailure |= (1U<< RF_TX2_BALLBREAK_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX0_BPM_REPORT:
                {

                    /* Save report */
                    gmonReportData |= (1U<< RF_TX0_BPM_MON);

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
                        monitorFailure |= (1U<< RF_TX0_BPM_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX1_BPM_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX1_BPM_MON);

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
                        monitorFailure |= (1U<< RF_TX1_BPM_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_TX2_BPM_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U<< RF_TX2_BPM_MON);
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
                        monitorFailure |= (1U<< RF_TX2_BPM_MON);
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
                        monitorFailure |= (1U<< RF_SYNTH_FREQ_MON);
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
                        monitorFailure |= (1U<< RF_EXT_ANA_SIG_MON);
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
                        monitorFailure |= (1U<< RF_TX0_INT_ANA_SIG_MON);
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
                        monitorFailure |= (1U<< RF_TX1_INT_ANA_SIG_MON);
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
                        monitorFailure |= (1U<< RF_TX2_INT_ANA_SIG_MON);
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
                        monitorFailure |= (1U<< RF_RX_INT_ANA_SIG_MON);
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
                        monitorFailure |= (1U<< RF_PMCLK_LO_SIG_MON);
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
                        monitorFailure |= (1U<< RF_GPADC_SIG_MON);
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
                        monitorFailure |= (1U<< RF_PLL_CTRL_VOL_MON);
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
                        monitorFailure |= (1U<< RF_DCC_CLK_FREQ_MON);
                    }
                    break;
                }
                case RL_RF_AE_MON_RX_MIXER_IN_PWR_REPORT:
                {
                    /* Save report */
                    gmonReportData |= (1U << RF_RX_MIXER_IN_PWR_MON);
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
                        monitorFailure |= (1U << RF_RX_MIXER_IN_PWR_MON);
                    }
                    break;
                }
                default:
                {
                    System_printf ("RF monitor Error: Asynchronous Event SB Id %d not handled with msg ID 0x%x\n", asyncSB,msgId);
                    break;
                }
            }
            break;
        }
        default:
        {
            System_printf ("RFMonitor Error: Asynchronous message %d is NOT handled\n", msgId);
            break;
        }
    }

    /* Check if there is a failure */
    if((monitorFailure >= 0) && (monitorFailure < 32))
    {
        /* Update failure Monitor bit mask */
        gRFMonReport.monFailureMask |= monitorFailure;

        /* Report failure during active frame if enabled */
        if(gRFMonitorCfg.reportCfg.enMonReportMode == 1U)
        {
            RFMon_reportFailureToHost((uint8_t)monitorFailure);
        }
    }
}

