/*
 *   @file  rfmonitor_defaultCfg.c
 *
 *   @brief
 *      Contains all the RF and Monitor default configuration values
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
/* Standard Include Files */
#include <stdio.h>
#include <stdbool.h>

/* mmWave SK Include File */
#include <ti/control/mmwavelink/mmwavelink.h>
#include "common/rfmonitor_internal.h"
#include "common/rfmonitor.h"

#define LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW                          (0U)
#define CENTER_RF_FRQ_IN_PROFILES_SWEEP_BW                          (1U)
#define HIGHEST_RF_FRQ_IN_PROFILES_SWEEP_BW                         (2U)
#define HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW           (7U)
#define ROUND_TO_INT32(X)   ((int32_t) (X))
#define CONV_FREQ_GHZ_TO_CODEWORD(X) ROUND_TO_INT32(X * (1.0e9/53.644))
#define CONV_SLOPE_MHZ_PER_US_TO_CODEWORD(X) (ROUND_TO_INT32(X * (1000.0/48.279)))

/* channel config */
#define MMW_RX_CHANNEL_EN               0xF
#if defined(SOC_XWR16XX)
#define MMW_TX_CHANNEL_EN                0x3
#elif (defined(SOC_XWR68XX) || defined(SOC_XWR18XX))
#define MMW_TX_CHANNEL_EN               0x7
#endif
/* ADC out format config */
#define MMW_NUM_ADC_BIT                 0x2 /* 0:12b, 1:14b, 2:16 bit */
#define MMW_ADC_OUT_FMT                 0x1 /* 0:real, 1:complex1x, 2:complex2x */
/* Low power mode config */
#define MMW_LOWPWR_ADC_MODE             0x0 /* 0: regular ADC, 1: Low power ADC */

/* Profile configuration */
#define MMW_PROFILECFG_ID               0x0
#define MMW_PROFILE_START_FREQ_GHZ      (76.01f)
#define MMW_PROFILECFG_START_FREQ_VAL   (CONV_FREQ_GHZ_TO_CODEWORD(MMW_PROFILE_START_FREQ_GHZ))
#define MMW_PROFILECFG_IDLE_TIME        (1000U)
#define MMW_PROFILECFG_RAMP_END_TIME    (6000U)
#define MMW_PROFILECFG_ADC_START_TIME   (600U)
#define MMW_PROFILECFG_TX_BACKOFF_CODE  (0U)
#define MMW_PROFILECFG_TX_PH_SHIFTER    (0U)
#define MMW_SRR_FREQ_SLOPE_MHZ_PER_US   (5.333f)
#define MMW_PROFILECFG_FREQ_SLOPE       (CONV_SLOPE_MHZ_PER_US_TO_CODEWORD(MMW_SRR_FREQ_SLOPE_MHZ_PER_US))
#define MMW_PROFILECFG_TX_START_TIME    (0U)
#define MMW_PROFILECFG_NUM_ADC_SAMPLE   (256U)
#define MMW_PROFILECFG_SAMPLE_RATE      (8000U)
#define MMW_PROFILECFG_HPF_FREQ1        RL_RX_HPF1_175_KHz
#define MMW_PROFILECFG_HPF_FREQ2        RL_RX_HPF2_350_KHz
#define MMW_PROFILECFG_RX_GAIN          (44U)

/* chirp configuration */
#define MMW_CHIRPCFG_START_IDX             0x0
#define MMW_CHIRPCFG_END_IDX               0x1
#define MMW_CHIRPCFG_PROFILE_ID            0x0
#define MMW_CHIRPCFG_IDLE_TIME_VAR         0x0
#define MMW_CHIRPCFG_ADC_START_TIME_VAR    0x0
#define MMW_CHIRPCFG_FREQ_SLOPE_VAR        0x0
#define MMW_CHIRPCFG_START_FREQ_VAR        0x0
#define MMW_CHIRPCFG_TX_ENABLE             0x3

/* Frame Configuration */
#define MMW_FRAMECFG_CHIRP_START_IDX       0x0
#define MMW_FRAMECFG_CHIRP_END_IDX         0x1
#define MMW_FRAMECFG_NUM_LOOOPS            64
#define MMW_FRAMECFG_NUM_FRAMES            0x0
#define MMW_FRAMECFG_FRAME_PERIOD          20000000
#define MMW_FRAMECFG_TRIGGER_SELECT        0x1
#define MMW_FRAMECFG_TRIGGER_DELAY         0x0

/* calibration monitoring time unit */
#define MMW_CAL_MON_TIME_UNIT           0x1

/* run time calibration config */
#define MMW_ONE_TIME_CALIB_MASK         0x710
#define MMW_PERIODIC_CALIB_MASK         0x710
#define MMW_RUNTIME_CALIB_PERIOD        0x5
#define MMW_CALIBRATION_REPORT_EN       0x1

/* RF Analog monitor enable mask bit as defined in @ref rlMonAnaEnables_t
 * Set 0: disable or 1: enable that monitor */
#define MMW_ANAMON_TEMP_EN                    (1U)
#define MMW_ANAMON_RX_GAIN_PHASE_EN           (1U)
#define MMW_ANAMON_RX_NOISE_EN                (1U)
#define MMW_ANAMON_RX_IFSTAGE_EN              (1U)
#define MMW_ANAMON_TX0_POWER_EN               (1U)
#define MMW_ANAMON_TX1_POWER_EN               (1U)
#define MMW_ANAMON_TX2_POWER_EN               (1U)
#define MMW_ANAMON_TX0_BALLBREAK_EN           (1U)
#define MMW_ANAMON_TX1_BALLBREAK_EN           (1U)
#define MMW_ANAMON_TX2_BALLBREAK_EN           (1U)
#define MMW_ANAMON_TX_GAIN_PHASE_EN           (1U)
#define MMW_ANAMON_TX0_BPM_EN                 (1U)
#define MMW_ANAMON_TX1_BPM_EN                 (1U)
#define MMW_ANAMON_TX2_BPM_EN                 (1U)
#define MMW_ANAMON_SYNTH_FREQ_EN              (1U)
#define MMW_ANAMON_EXT_ANALOG_SIGNALS_EN      (1U)
#define MMW_ANAMON_INT_TX0_SIGNALS_EN         (1U)
#define MMW_ANAMON_INT_TX1_SIGNALS_EN         (1U)
#define MMW_ANAMON_INT_TX2_SIGNALS_EN         (1U)
#define MMW_ANAMON_INT_RX_SIGNALS_EN          (1U)
#define MMW_ANAMON_PMCLKLO_SIGNALS_EN         (1U)
#define MMW_ANAMON_GPADC_SIGNALS_EN           (1U)
#define MMW_ANAMON_PLL_CONTROL_VOLTAGE_EN     (1U)
#define MMW_ANAMON_DCC_CLOCK_FREQ_EN          (1U)
#define MMW_ANAMON_RX_IF_SATURATION_EN        (1U)
#define MMW_ANAMON_RX_SIG_IMG_BAND_EN         (1U)
#define MMW_ANAMON_RX_MIXER_INPUT_POWER_EN    (1U)

#define MMW_ANALOG_MONITOR_EN_MASK      ((MMW_ANAMON_RX_MIXER_INPUT_POWER_EN << 26) | \
                                         (MMW_ANAMON_RX_SIG_IMG_BAND_EN      << 25) | \
                                         (MMW_ANAMON_RX_IF_SATURATION_EN     << 24) | \
                                         (MMW_ANAMON_DCC_CLOCK_FREQ_EN       << 23) | \
                                         (MMW_ANAMON_PLL_CONTROL_VOLTAGE_EN  << 22) | \
                                         (MMW_ANAMON_GPADC_SIGNALS_EN        << 21) | \
                                         (MMW_ANAMON_PMCLKLO_SIGNALS_EN      << 20) | \
                                         (MMW_ANAMON_INT_RX_SIGNALS_EN       << 19) | \
                                         (MMW_ANAMON_INT_TX2_SIGNALS_EN      << 18) | \
                                         (MMW_ANAMON_INT_TX1_SIGNALS_EN      << 17) | \
                                         (MMW_ANAMON_INT_TX0_SIGNALS_EN      << 16) | \
                                         (MMW_ANAMON_EXT_ANALOG_SIGNALS_EN   << 15) | \
                                         (MMW_ANAMON_SYNTH_FREQ_EN           << 14) | \
                                         (MMW_ANAMON_TX2_BPM_EN              << 13) | \
                                         (MMW_ANAMON_TX1_BPM_EN              << 12) | \
                                         (MMW_ANAMON_TX0_BPM_EN              << 11) | \
                                         (MMW_ANAMON_TX_GAIN_PHASE_EN        << 10) | \
                                         (MMW_ANAMON_TX2_BALLBREAK_EN        <<  9) | \
                                         (MMW_ANAMON_TX1_BALLBREAK_EN        <<  8) | \
                                         (MMW_ANAMON_TX0_BALLBREAK_EN        <<  7) | \
                                         (MMW_ANAMON_TX2_POWER_EN            <<  6) | \
                                         (MMW_ANAMON_TX1_POWER_EN            <<  5) | \
                                         (MMW_ANAMON_TX0_POWER_EN            <<  4) | \
                                         (MMW_ANAMON_RX_IFSTAGE_EN           <<  3) | \
                                         (MMW_ANAMON_RX_NOISE_EN             <<  2) | \
                                         (MMW_ANAMON_RX_GAIN_PHASE_EN        <<  1) | \
                                         (MMW_ANAMON_TEMP_EN                 <<  0))

/* Digital monitor enable mask bit as defined in @ref rlMonDigEnables_t
 * Set 0: disable or 1: enable that monitor */
#define MMW_DIG_MON_CR4_VIM_TEST_EN                    (1U)
#define MMW_DIG_MON_CRC_TEST_EN                        (1U)
#define MMW_DIG_MON_RAMPGEN_ECC_EN                     (1U)
#define MMW_DIG_MON_DFE_PARITY_EN                      (1U)
#define MMW_DIG_MON_DFE_MEM_ECC_EN                     (1U)
#define MMW_DIG_MON_RAMPGEN_LOCK_EN                    (1U)
#define MMW_DIG_MON_FRC_LOCKSTEP_EN                    (1U)
#define MMW_DIG_MON_ESM_TEST_EN                        (1U)
#define MMW_DIG_MON_DFE_STC_EN                         (1U)
#define MMW_DIG_MON_ATCM_ECC_EN                        (1U)
#define MMW_DIG_MON_ATCM_PARITY_EN                     (1U)
#define MMW_DIG_MON_FFT_TEST_EN                        (1U)
#define MMW_DIG_MON_RTI_TEST_EN                        (1U)

#define MMW_DIGITAL_MONITOR_EN_MASK     ((MMW_DIG_MON_RTI_TEST_EN << 25) | \
                                         (MMW_DIG_MON_FFT_TEST_EN << 24) | \
                                         (MMW_DIG_MON_ATCM_PARITY_EN << 20) | \
                                         (MMW_DIG_MON_ATCM_ECC_EN << 19) | \
                                         (MMW_DIG_MON_DFE_STC_EN << 17) | \
                                         (MMW_DIG_MON_ESM_TEST_EN << 16) | \
                                         (MMW_DIG_MON_FRC_LOCKSTEP_EN << 11) | \
                                         (MMW_DIG_MON_RAMPGEN_LOCK_EN << 10) | \
                                         (MMW_DIG_MON_DFE_MEM_ECC_EN << 9) | \
                                         (MMW_DIG_MON_DFE_PARITY_EN << 8) | \
                                         (MMW_DIG_MON_RAMPGEN_ECC_EN << 7) | \
                                         (MMW_DIG_MON_RAMPGEN_ECC_EN << 6) | \
                                         (MMW_DIG_MON_CR4_VIM_TEST_EN << 1))
/* rlDigMonPeriodicConf_t -> periodicEnableMask */
#define MMW_DIG_MON_LATENT_EN_MASK  (1)
/* production mode latest fault test */
#define MMW_DIGITAL_MONITOR_TEST_MODE (0)

/* Monitor Reporting Mode to set to BSS */
#define MMW_MONITOR_REPORT_MODE          MON_REPORT_MODE_AT_FAILURE_ONLY

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

#if defined(SOC_XWR18XX)
/**
 **********************************************************
  AS PER RECOMMENDATION FROM THE xWR1843BOOST EVM User guide
 **********************************************************
   set LDO bypass since the board has 1.0V RF supply 1
   and 1.0V RF supply 2. Please update this API if this
   assumption is not valid else it may DAMAGE your board!
 **********************************************************
 */
rlRfLdoBypassCfg_t gRfLdoBypassCfg =
{
    .ldoBypassEnable   = 3, /* 1.0V RF supply 1 and 1.0V RF supply 2 */
    .supplyMonIrDrop   = 1, /* IR drop of 3% */
    .ioSupplyIndicator = 0, /* 3.3 V IO supply */
};
#elif defined(SOC_XWR68XX)
/**
 * @brief
 *  Variable for LDO BYPASS config, PLease consult your
 * board/EVM user guide before changing the values here
 */
rlRfLdoBypassCfg_t gRfLdoBypassCfg =
{
    .ldoBypassEnable   = 0, /* 1.0V RF supply 1 and 1.0V RF supply 2 */
    .supplyMonIrDrop   = 0, /* IR drop of 3% */
    .ioSupplyIndicator = 0, /* 3.3 V IO supply */
};
#endif

/*********** RF Configuration ***************/
/* Channel Config */
rlChanCfg_t gRfChannelCfg =
{
 .txChannelEn = MMW_TX_CHANNEL_EN,
 .rxChannelEn = MMW_RX_CHANNEL_EN,
 .cascading   = 0
};

/* Populate the ADC Output configuration: */
rlAdcOutCfg_t gAdcOutCfg =
{
 .fmt =
 {
      .b2AdcBits    = MMW_NUM_ADC_BIT,
      .b8FullScaleReducFctr = 0,
      .b2AdcOutFmt = MMW_ADC_OUT_FMT,
 },
 .reserved0 = 0,
 .reserved1 = 0
};


rlLowPowerModeCfg_t gLowPowerModeCfg =
{
 .reserved = 0,
 .lpAdcMode = MMW_LOWPWR_ADC_MODE
};

/* Profile Config */
rlProfileCfg_t gProfileCfg =
{
 .profileId            = MMW_PROFILECFG_ID,
 .pfVcoSelect          = 0,
 .pfCalLutUpdate       = 0,
 .startFreqConst       = MMW_PROFILECFG_START_FREQ_VAL,
 .idleTimeConst        = MMW_PROFILECFG_IDLE_TIME,
 .rampEndTime          = MMW_PROFILECFG_RAMP_END_TIME,
 .adcStartTimeConst     = MMW_PROFILECFG_ADC_START_TIME,
 .txOutPowerBackoffCode= MMW_PROFILECFG_TX_BACKOFF_CODE,
 .txPhaseShifter       = MMW_PROFILECFG_TX_PH_SHIFTER,
 .freqSlopeConst       = MMW_PROFILECFG_FREQ_SLOPE,
 .txStartTime          = MMW_PROFILECFG_TX_START_TIME,
 .numAdcSamples        = MMW_PROFILECFG_NUM_ADC_SAMPLE,
 .digOutSampleRate     = MMW_PROFILECFG_SAMPLE_RATE,
 .hpfCornerFreq1       = MMW_PROFILECFG_HPF_FREQ1,
 .hpfCornerFreq2       = MMW_PROFILECFG_HPF_FREQ2,
 .rxGain               = MMW_PROFILECFG_RX_GAIN
};

/* chirp config */
rlChirpCfg_t gChirpCfg =
{
 .chirpStartIdx        = MMW_CHIRPCFG_START_IDX,
 .chirpEndIdx          = MMW_CHIRPCFG_END_IDX,
 .profileId            = MMW_CHIRPCFG_PROFILE_ID,
 .idleTimeVar          = MMW_CHIRPCFG_IDLE_TIME_VAR,
 .adcStartTimeVar      = MMW_CHIRPCFG_ADC_START_TIME_VAR,
 .freqSlopeVar         = MMW_CHIRPCFG_FREQ_SLOPE_VAR,
 .startFreqVar         = MMW_CHIRPCFG_START_FREQ_VAR,
 .txEnable             = MMW_CHIRPCFG_TX_ENABLE
};

/* frame config */
rlFrameCfg_t gFrameCfg =
{
  .chirpStartIdx       = MMW_FRAMECFG_CHIRP_START_IDX,
  .chirpEndIdx         = MMW_FRAMECFG_CHIRP_END_IDX,
  .numLoops            = MMW_FRAMECFG_NUM_LOOOPS,
  .numFrames           = MMW_FRAMECFG_NUM_FRAMES,
  .framePeriodicity    = MMW_FRAMECFG_FRAME_PERIOD,
  .triggerSelect       = MMW_FRAMECFG_TRIGGER_SELECT,
  .frameTriggerDelay   = MMW_FRAMECFG_TRIGGER_DELAY
};

/* Calibration Configuration */
rlRfCalMonTimeUntConf_t gCalMonTimeUnitConf =
{
    .calibMonTimeUnit = MMW_CAL_MON_TIME_UNIT,
    .numOfCascadeDev = 1,
    .devId = 1,
    .reserved = 0
};

rlRunTimeCalibConf_t gRunTimeCalibCfg =
{
    .oneTimeCalibEnMask = MMW_ONE_TIME_CALIB_MASK,  /* Enable All Run time Calibration */
    .periodicCalibEnMask = MMW_PERIODIC_CALIB_MASK, /* Enable All Run time Calibration */
    .calibPeriodicity = MMW_RUNTIME_CALIB_PERIOD,
    .reportEn = MMW_CALIBRATION_REPORT_EN,
    .txPowerCalMode = 0,
    .reserved1 = 0
};

/* Monitoring enable Mask config */
rlMonAnaEnables_t  gRfAnaMonitorEn =
{
   .enMask    = MMW_ANALOG_MONITOR_EN_MASK,
   .ldoScEn   = 0
};

/* Temperature Monitor */
rlTempMonConf_t gTempMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
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
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .txSel = 0,                         /* TX0 */
    .rxGainAbsThresh = 360,             /* 4dB = 53dB - 48dB, 9dB = 48-39 , programmed value = 48*/
    .rxGainMismatchErrThresh = 45,      /* Rel gain : 3.0dB */
    .rxGainFlatnessErrThresh = 65535,   /* set to max value */
    .rxGainPhaseMismatchErrThresh = (30 * (1U << 16)) / 360U, /* 15 degree */
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

rlRxNoiseMonConf_t gRxNoiseMonCfg =
{
 /* This Monitor is de-featured */
};


rlRxIfStageMonConf_t gRxIfStageMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0,
    .reserved1 = 0,
    .hpfCutoffErrThresh = 20,
    .lpfCutoffErrThresh = 100, /* defeatured, set to max */
    .ifaGainErrThresh = 30,
    .reserved2 = 0,
};

/* Tx0 Power Monitoring Config */
rlTxPowMonConf_t gTx0PowMonCfg =
{
    .profileIndx = 0x0,
    .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
    .reserved0 = 0x0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved1 = 0x0,
    .txPowAbsErrThresh = 35,
    .txPowFlatnessErrThresh = 50,
    .reserved2 = 0x0,
    .reserved3 = 0x0
};

/* Tx1 Power Monitoring Config */
rlTxPowMonConf_t gTx1PowMonCfg =
{
    .profileIndx = 0x0,
    .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
    .reserved0 = 0x0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved1 = 0x0,
    .txPowAbsErrThresh = 35,
    .txPowFlatnessErrThresh = 45,
    .reserved2 = 0x0,
    .reserved3 = 0x0
};

/* Tx2 Power Monitoring Config */
rlTxPowMonConf_t gTx2PowMonCfg =
{
    .profileIndx = 0x0,
    .rfFreqBitMask = HIGHEST_CENTER_LOWEST_RF_FRQ_IN_PROFILES_SWEEP_BW,
    .reserved0 = 0x0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved1 = 0x0,
    .txPowAbsErrThresh = 35,
    .txPowFlatnessErrThresh = 60,
    .reserved2 = 0x0,
    .reserved3 = 0x0
};

/* Tx0 Ball Break Monitoring Config */
rlTxBallbreakMonConf_t  gTx0BallBreakMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0x0,
    .txReflCoeffMagThresh = -15,
#ifdef SOC_XWR68XX
        .monStartFreqConst = 0x58E3A1CD, /* 60GHz */
#else
        .monStartFreqConst = 0x0,
#endif
        .reserved1 = 0x0,
};

/* Tx1 Ball Break Monitoring Config */
rlTxBallbreakMonConf_t  gTx1BallBreakMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0x0,
    .txReflCoeffMagThresh = -15,
#ifdef SOC_XWR68XX
        .monStartFreqConst = 0x58E3A1CD, /* 60GHz */
#else
        .monStartFreqConst = 0x0,
#endif
        .reserved1 = 0x0,
};

/* Tx2 Ball Break Monitoring Config */
rlTxBallbreakMonConf_t  gTx2BallBreakMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0x0,
    .txReflCoeffMagThresh = -15,
#ifdef SOC_XWR68XX
        .monStartFreqConst = 0x58E3A1CD, /* 60GHz */
#else
        .monStartFreqConst = 0x0,
#endif
        .reserved1 = 0x0,
};

rlSynthFreqMonConf_t gSynthFreqMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .freqErrThresh = 1000,
    .monStartTime = 35,     /* >=6us into ramp*/
    .monitorMode = 0,
    .reserved1 = 0,
    .reserved2 = 0,
};

rlExtAnaSignalsMonConf_t gExtAnaSigMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
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

/* TX0 internal signal path monitoring config */
rlTxIntAnaSignalsMonConf_t gTx0IntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .txPhShiftDacMonThresh = 0,
    .reserved1 = 0
};

/* TX1 internal signal path monitoring config */
rlTxIntAnaSignalsMonConf_t gTx1IntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .txPhShiftDacMonThresh = 0,
    .reserved1 = 0
};

/* TX2 internal signal path monitoring config */
rlTxIntAnaSignalsMonConf_t gTx2IntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .txPhShiftDacMonThresh = 0,
    .reserved1 = 0
};

rlRxIntAnaSignalsMonConf_t gRxIntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0,
    .reserved1 = 0,
};

rlPmClkLoIntAnaSignalsMonConf_t gPmClkLoIntAnaSigMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .sync20GSigSel = 0,
    .sync20GMinThresh = 0,
    .sync20GMaxThresh = 0,
    .reserved0 = 0,
    .reserved1 = 0,
};

rlGpadcIntAnaSignalsMonConf_t gGpadcIntAnaSigMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0,
    .reserved1 = 0,
    .reserved2 = 0,
};

rlPllContrVoltMonConf_t gPllConVoltMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 0,
    .signalEnables = 7, /* It monitors APLL control voltage and Synth VCO1 control voltage. Not Synth VCO2 control voltage */
    .reserved1 = 0,
};

rlDualClkCompMonConf_t gDualClkCompMonCfg =
{
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .reserved0 = 10,
    .dccPairEnables = 0x3F,
    .reserved1 = 0,
};

/* CQ is not available in this application
rlRxSatMonConf_t gRxIfSatMonCfg =
{
};

rlSigImgMonConf_t gSigImgMonCfg =
{
};
*/
rlRxMixInPwrMonConf_t gRxMixInpwrMonCfg =
{
    .profileIndx = 0,
    .reportMode = MMW_MONITOR_REPORT_MODE,
    .txEnable = 3,
    .reserved0 = 0,
    .thresholds = 0x2100,
    .reserved1 = 0,
    .reserved1 = 0,
};


/* Digital Monitoring Enable Mask Config */
rlMonDigEnables_t gDigMonitorEn =
{
   .enMask   = MMW_DIGITAL_MONITOR_EN_MASK,
   .testMode = MMW_DIGITAL_MONITOR_TEST_MODE,
   .reserved0 = 0,
   .reserved1 = 0,
   .reserved2 = 0
};

/* Digital Monitoring Latent Fault Config */
rlDigMonPeriodicConf_t gDigMonitorPeriod =
{
   .reportMode  = MON_REPORT_MODE_AT_FAILURE_ONLY,
   .reserved0   = 0,
   .reserved1   = 0,
   .periodicEnableMask = MMW_DIG_MON_LATENT_EN_MASK,
   .reserved2   = 0
};

