/*
 *   @file  rfmonitor_defaultCfg.c
 *
 *   @brief
 *      Contains all the RF and Monitor default configuration values
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

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/
 
/* Standard Include Files */
#include <stdio.h>
#include <stdbool.h>

/* mmWave SK Include File */
#include <ti/control/mmwavelink/mmwavelink.h>
#include <./common/rfmonitor_internal.h>
#include <./common/rfmonitor.h>

#define LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW                          (0U)
#define CENTER_RF_FRQ_IN_PROFILES_SWEEP_BW                          (1U)
#define HIGHEST_RF_FRQ_IN_PROFILES_SWEEP_BW                         (2U)
#define HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW           (7U)

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

/* Temperature Monitor */
rlTempMonConf_t gTempMonCfg =
{
    //.reportMode = MON_REPORT_MODE_AT_FAILURE_ONLY,
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 0,
    .anaTempThreshMin = -40,
    .anaTempThreshMax = 140,
    .digTempThreshMin = -40,
    .digTempThreshMax = 140,
    .tempDiffThresh = 50,
    .reserved1 = 0,
    .reserved2 = 0,
};

rlRxGainPhaseMonConf_t gRxGainPhaseMonCfg =
{
    .profileIndx = 0,
    .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
    .reserved0 = 0,
    .txSel = 0,                         /* TX0 */
    .rxGainAbsThresh = 360,             /* 4dB = 53dB - 48dB, 9dB = 48-39 , programmed value = 48*/
    .rxGainMismatchErrThresh = 45,      /* Rel gain : setting 3.0dB */
    .rxGainFlatnessErrThresh = 65535,   /* set to max value */
    .rxGainPhaseMismatchErrThresh = (30 * (1U << 16)) / 360U, /* 15 degree: setting */
    .rxGainMismatchOffsetVal[0][0] = 0x1A,
    .rxGainMismatchOffsetVal[0][1] = 0x1A,
    .rxGainMismatchOffsetVal[0][2] = 0x1A,
    .rxGainMismatchOffsetVal[1][0] = 0x1A,
    .rxGainMismatchOffsetVal[1][1] = 0x1A,
    .rxGainMismatchOffsetVal[1][2] = 0x1A,
    .rxGainMismatchOffsetVal[2][0] = 0x1A,
    .rxGainMismatchOffsetVal[2][1] = 0x1A,
    .rxGainMismatchOffsetVal[2][2] = 0x1A,
    .rxGainMismatchOffsetVal[3][0] = 0x1A,
    .rxGainMismatchOffsetVal[3][1] = 0x1A,
    .rxGainMismatchOffsetVal[3][2] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[0][0] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[0][1] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[0][2] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[1][0] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[1][1] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[1][2] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[2][0] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[2][1] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[2][2] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[3][0] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[3][1] = 0x1A,
    .rxGainPhaseMismatchOffsetVal[3][2] = 0x1A,
    .reserved1 = 0,
};

rlTxPowMonConf_t gTxPowMonCfg[3] =
{
    {
        .profileIndx = 0x0,
        .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
        .reserved0 = 0x0,
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .reserved1 = 0x0,
        .txPowAbsErrThresh = 35,
        .txPowFlatnessErrThresh = 50,
        .reserved2 = 0x0,
        .reserved3 = 0x0,
    },
    {
        .profileIndx = 0x0,
        .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
        .reserved0 = 0x0,
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .reserved1 = 0x0,
        .txPowAbsErrThresh = 35,
        .txPowFlatnessErrThresh = 45,
        .reserved2 = 0x0,
        .reserved3 = 0x0,
    },
    {
        .profileIndx = 0x0,
        .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
        .reserved0 = 0x0,
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .reserved1 = 0x0,
        .txPowAbsErrThresh = 35,
        .txPowFlatnessErrThresh = 60,
        .reserved2 = 0x0,
        .reserved3 = 0x0
    }
};

rlTxBallbreakMonConf_t gTxBallbreakMonCfg[3] =
{
    {
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .reserved0 = 0x0,
        .txReflCoeffMagThresh = -15,
    #ifdef SOC_XWR68XX
        .monStartFreqConst = 0x58E3A1CD, /* 60GHz */
    #else
        .monStartFreqConst = 0x0,
    #endif
        .reserved1 = 0x0,
    },
    {
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .reserved0 = 0x0,
        .txReflCoeffMagThresh = -15,
    #ifdef SOC_XWR68XX
        .monStartFreqConst = 0x58E3A1CD,
    #else
        .monStartFreqConst = 0x0,
    #endif
        .reserved1 = 0x0
        },
    {
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .reserved0 = 0x0,
        .txReflCoeffMagThresh = -15,
    #ifdef SOC_XWR68XX
        .monStartFreqConst = 0x58E3A1CD,
    #else
        .monStartFreqConst = 0x0,
    #endif
        .reserved1 = 0x0
    }
};

rlSynthFreqMonConf_t gSynthFreqMonCfg =
{
    .profileIndx = 0,
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .freqErrThresh = 1000,  /*  setting 20MHz */
    .monStartTime = 35,     /* >=6us into ramp*/
    .monitorMode = 0,
    .vcoMonEn = 0,
    .reserved1 = 0,
};

rlPllContrVoltMonConf_t gPllConVoltMonCfg =
{
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 0,
    .signalEnables = 7, /* It monitors APLL control voltage and Synth VCO1 control voltage. Not Synth VCO2 control voltage */
    .reserved1 = 0,
};

rlDualClkCompMonConf_t gDualClkCompMonCfg =
{
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 10,
    .dccPairEnables = 0x3F,
    .reserved1 = 0,
};

rlRxIfStageMonConf_t gRxIfStageMonCfg =
{
    .profileIndx = 0,
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 0,
    .reserved1 = 0,
    .hpfCutoffErrThresh = 20,
    .lpfCutoffErrThresh = 100, /* defeatured, set to max */
    .ifaGainErrThresh = 30,
    .reserved2 = 0,
};

/* De-featured */
rlTxBpmMonConf_t gTxBpmMonCfg[3] =
{
    {
        .profileIndx = 0x0,
        .phaseShifterMonCnfg = 0xC2,
        .phaseShifterMon1 = 0x06,
        .phaseShifterMon2 = 0x06,
        .reportMode = MON_REPORT_MODE_AT_FAILURE_ONLY,
        .rxEn = 0x1,
        .txBpmPhaseErrThresh = 0x1555,
        .txBpmAmplErrThresh = 30,
        .phaseShifterThreshMax = 40,
        .phaseShifterThreshMin = 0,
        .reserved = 0x0,
    },
    {
        .profileIndx = 0x0,
        .phaseShifterMonCnfg = 0xC2,
        .phaseShifterMon1 = 0x06,
        .phaseShifterMon2 = 0x06,
        .reportMode = MON_REPORT_MODE_AT_FAILURE_ONLY,
        .rxEn = 0x1,
        .txBpmPhaseErrThresh = 0x1555,
        .txBpmAmplErrThresh = 30,
        .phaseShifterThreshMax = 40,
        .phaseShifterThreshMin = 0,
        .reserved = 0x0,
    },
    {
        .profileIndx = 0x0,
        .phaseShifterMonCnfg = 0xC2,
        .phaseShifterMon1 = 0x06,
        .phaseShifterMon2 = 0x06,
        .reportMode = MON_REPORT_MODE_AT_FAILURE_ONLY,
        .rxEn = 0x1,
        .txBpmPhaseErrThresh = 0x1555,
        .txBpmAmplErrThresh = 30,
        .phaseShifterThreshMax = 40,
        .phaseShifterThreshMin = 0,
        .reserved = 0x0,
    }
};

rlExtAnaSignalsMonConf_t gExtAnaSigMonCfg =
{
    //.reportMode = MON_REPORT_MODE_AT_FAILURE_ONLY,
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 0,
    .signalInpEnables = 0x3F,
    .signalBuffEnables = 0x1F,
    .signalSettlingTime[0] = 10,
    .signalSettlingTime[1] = 10,
    .signalSettlingTime[2] = 10,
    .signalSettlingTime[3] = 10,
    .signalSettlingTime[4] = 10,
    .signalSettlingTime[5] = 10,
    .signalThresh[0] = 0,
    .signalThresh[1] = 0,
    .signalThresh[2] = 0,
    .signalThresh[3] = 0,
    .signalThresh[4] = 0,
    .signalThresh[5] = 0,
    .signalThresh[6] = 200,
    .signalThresh[7] = 200,
    .signalThresh[8] = 200,
    .signalThresh[9] = 200,
    .signalThresh[10] = 200,
    .signalThresh[11] = 200,
    .reserved1 = 0,
    .reserved2 = 0,
    .reserved3 = 0,
};


rlTxIntAnaSignalsMonConf_t gTxIntAnaSigMonCfg[3] =
{
    {
        .profileIndx = 0,
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .txPhShiftDacMonThresh = 0,
        .reserved1 = 0,
    },

    {
        .profileIndx = 0,
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .txPhShiftDacMonThresh = 0,
        .reserved1 = 0,
    },

    {
        .profileIndx = 0,
        .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
        .txPhShiftDacMonThresh = 0,
        .reserved1 = 0,
    }

};

rlRxIntAnaSignalsMonConf_t gRxIntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 0,
    .reserved1 = 0,
};


rlPmClkLoIntAnaSignalsMonConf_t gPmClkLoIntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .sync20GSigSel = 0,
    .sync20GMinThresh = 0,
    .sync20GMaxThresh = 0,
    .reserved0 = 0,
    .reserved1 = 0,
};

rlGpadcIntAnaSignalsMonConf_t gGpadcIntAnaSigMonCfg =
{
    .reportMode = MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK,
    .reserved0 = 0,
    .reserved1 = 0,
    .reserved2 = 0,
};


rlRxMixInPwrMonConf_t gRxMixInpwrMonCfg =
{
    .profileIndx = 0,
    .reportMode = MON_REPORT_MODE_AT_FAILURE_ONLY,
    .txEnable = 3,
    .reserved0 = 0,
    .thresholds = 0x2100,  /* setting 226mv*/
    .reserved1 = 0,
    .reserved1 = 0,
};

/* Digital Monitoring Enable Mask Config */
rlMonDigEnables_t gDigMonitorEn = { 0 };

/* Digital Monitoring Latent Fault Config */
rlDigMonPeriodicConf_t gDigMonitorPeriod =
{
     .reportMode = 0, /* report digital monitor report at every monitor period */
     .reserved0  = 0,
     .reserved1  = 0,
     .periodicEnableMask = 0xD, /* Register_read_en, DFE_STC_En, frame_timing_monitor_en */
     .reserved2  = 0
};

rlRxNoiseMonConf_t gRxNoiseMonCfg = {0};
/* it is populated by CLI CMD */
rlRxSatMonConf_t gRxIfSatMonCfg = {0};
/* it is populated by CLI CMD */
rlSigImgMonConf_t gSigImgMonCfg = {0};

/* Monitoring enable Mask config */
rlMonAnaEnables_t  gRfAnaMonitorEn = {0};
