/*
 *   @file  rfmonitor_internal.h
 *
 *   @brief
 *      Mmwave link RF monitoring internal header file
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

#ifndef RFMONITOR_INTERNAL_H
#define RFMONITOR_INTERNAL_H

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/control/mmwavelink/mmwavelink.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Monitoring Settings */
#define MON_REPORT_MODE_PERIODIC_WITHOUT_THRESHOLD_CHECK            (0U)
#define MON_REPORT_MODE_AT_FAILURE_ONLY                             (1U)
#define MON_REPORT_MODE_PERIODIC_WITH_THRESHOLD_CHECK               (2U)

/* RF Analog monitor enable mask bit as defined in @ref rlMonAnaEnables_t */
#define RFMON_ANAMON_TEMP_EN_BIT                    (0U)
#define RFMON_ANAMON_RX_GAIN_PHASE_EN_BIT           (1U)
#define RFMON_ANAMON_RX_NOISE_EN_BIT                (2U)
#define RFMON_ANAMON_RX_IFSTAGE_EN_BIT              (3U)
#define RFMON_ANAMON_TX0_POWER_EN_BIT               (4U)
#define RFMON_ANAMON_TX1_POWER_EN_BIT               (5U)
#define RFMON_ANAMON_TX2_POWER_EN_BIT               (6U)
#define RFMON_ANAMON_TX0_BALLBREAK_EN_BIT           (7U)
#define RFMON_ANAMON_TX1_BALLBREAK_EN_BIT           (8U)
#define RFMON_ANAMON_TX2_BALLBREAK_EN_BIT           (9U)
#define RFMON_ANAMON_TX_GAIN_PHASE_EN_BIT           (10U)
#define RFMON_ANAMON_TX0_BPM_EN_BIT                 (11U)
#define RFMON_ANAMON_TX1_BPM_EN_BIT                 (12U)
#define RFMON_ANAMON_TX2_BPM_EN_BIT                 (13U)
#define RFMON_ANAMON_SYNTH_FREQ_EN_BIT              (14U)
#define RFMON_ANAMON_EXT_ANALOG_SIGNALS_EN_BIT (15U)
#define RFMON_ANAMON_INT_TX0_SIGNALS_EN_BIT        (16U)
#define RFMON_ANAMON_INT_TX1_SIGNALS_EN_BIT        (17U)
#define RFMON_ANAMON_INT_TX2_SIGNALS_EN_BIT        (18U)
#define RFMON_ANAMON_INT_RX_SIGNALS_EN_BIT         (19U)
#define RFMON_ANAMON_PMCLKLO_SIGNALS_EN_BIT         (20U)
#define RFMON_ANAMON_GPADC_SIGNALS_EN_BIT           (21U)
#define RFMON_ANAMON_PLL_CONTROL_VOLTAGE_EN_BIT (22U)
#define RFMON_ANAMON_DCC_CLOCK_FREQ_EN_BIT          (23U)
#define RFMON_ANAMON_RX_IF_SATURATION_EN_BIT        (24U)
#define RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT         (25U)
#define RFMON_ANAMON_RX_MIXER_INPUT_POWER_EN_BIT        (26U)

/**
 * @brief
 *  RF monitor and calibration report configuration
 *
 * @details
 *  The structure specifies the configuration which is required to configure
 *  monitor and calibration report to CLI UART.
 */
typedef struct RFMonCalibReportCfg_t
{
    /**
     * @brief   Flag to enable calibration report to CLI UART
     */
    uint8_t     enCalibReport;

    /**
     * @brief   Flag to enable monitor stats during inter-frame period to CLI UART
     */
    uint8_t     enMonStats;

    /**
     * @brief   Flag to enable monitor report to CLI UART
     *          0 - Quite mode, turn off failure report
     *          1 - Active mode, report is sent when Async event is received from RF
     *          N - Report is sent after N frame period
     */
    uint8_t     enMonReportMode;

    /**
     * @brief   Flag to enable rxGainPhase raw value report to CLI UART
     */
    uint8_t     enRxGainPhaseReport;
}RFMonCalibReportCfg;

/**
 * @brief
 *  RF monitor configuration
 *
 * @details
 *  The structure specifies the configuration which is required to configure
 *  and setup the RF digital and analog monitors.
 */
typedef struct RFMonitorCfg_t
{
    /*! \brief
    * RX saturation monitoring configuration
    */
    rlRxSatMonConf_t        *rxSatMonCfg;

    /*! \brief
    * Signal and image band energy monitoring configuration
    */
    rlSigImgMonConf_t       *sigImageBandMonCfg;

    /**
     * @brief   RF monitor and calibration report configuration
     */
    RFMonCalibReportCfg     reportCfg;

    /**
     * @brief   RF digital monitors enable configuration
     */
    rlMonDigEnables_t       *rfDigMonitorEn;

    /**
     * @brief   Digital monitoring latent fault reporting configuration
     */
    rlDigMonPeriodicConf_t  *rfDigMonPeriodicCfg;

    /**
     * @brief   RF analog monitors enable configuration
     */
    rlMonAnaEnables_t       *rfAnaMonitorEn;

    /**
     * @brief   RF temperature monitor configuration
     */
    rlTempMonConf_t         *tempMonCfg;

    /**
     * @brief   Programmed rxGain in profile config for the monitering profile
     */
    uint16_t                programmedRxGain;

    /**
     * @brief   RX gain and phase monitoring configuration
     */
    rlRxGainPhaseMonConf_t  *rxGainPhaseMonCfg;

    /**
     * @brief   RX Noise Monitoring Configuration
     */
    rlRxNoiseMonConf_t      *rxNoiseFigMonCfg;
    /**
     * @brief   TX power monitoring configuration
     */
    rlAllTxPowMonConf_t      allTxPowerMonCfg;

    /**
     * @brief    TX ballbreak monitoring configuration
     */
    rlAllTxBallBreakMonCfg_t  allTxBallbreakMonCfg;

    /**
     * @brief    Synthesizer frequency monitoring configuration
     */
    rlSynthFreqMonConf_t    *synthFreqMonCfg;

    /**
     * @brief    Internal signals for PLL control voltage monitoring configuration
     */
    rlPllContrVoltMonConf_t *pllConVoltMonCfg ;

    /**
     * @brief    Internal signals for DCC based clock monitoring configuration
     */
    rlDualClkCompMonConf_t  *dualClkCompMonCfg;

    /**
     * @brief    RX IF stage monitoring configuration
     */
    rlRxIfStageMonConf_t    *rxIfStageMonCfg;

    /**
     * @brief    TX BPM monitoring configuration [De-featured]
     */
    rlAllTxBpmMonConf_t     allTxBpmMonCfg;

    /**
     * @brief    External analog signals monitoring configuration
     */
    rlExtAnaSignalsMonConf_t *extAnaSigMonCfg;

    /**
     * @brief    Internal signals in the TX path monitoring configuration
     */
    rlAllTxIntAnaSignalsMonConf_t allTxIntAnaSigMonCfg;

    /**
     * @brief    Internal signals in the RX path monitoring configuration
     */
    rlRxIntAnaSignalsMonConf_t *rxIntAnaSigMonCfg;

    /**
     * @brief    Internal signals for PM, CLK and LO monitoring configuration
     */
    rlPmClkLoIntAnaSignalsMonConf_t *pmClkLoIntAnaSigMonCfg;

    /**
     * @brief    Internal signals for GPADC monitoring configuration
     */
    rlGpadcIntAnaSignalsMonConf_t *gpadcIntAnaSigMonCfg;

    /**
     * @brief    RX mixer input power monitoring configuration
     */
    rlRxMixInPwrMonConf_t   *rxMixInpwrMonCfg;

    /*! \brief
    * TX gain and phase mismatch monitoring configuration
    */
    rlTxGainPhaseMismatchMonConf_t *txGainPhMisMonCfg;

    /**
     * @brief    Programmed RX gain from the profile set by rxGainPhase monitor
     */
    uint16_t                rxGain;
}RFMonitorCfg;

/* RF gain and phase monitoring raw value */
typedef struct rfMonRxGainRawValue_t
{
    /**
     * @brief    Converted loopback power array in dBm
     */
    uint32_t     loopBackPower[RL_MON_RF_FREQ_CNT];

    /**
     * @brief    Converted RX gain(in 0.1dB) array
     */
    uint32_t    rxGainVal[RL_MON_RF_FREQ_CNT][RL_RX_CNT];

    /**
     * @brief    Converted RX phase(1LSB = 360/2^16) array
     */
    uint32_t    rxPhaseVal[RL_MON_RF_FREQ_CNT][RL_RX_CNT];

    /**
     * @brief    Converted RX noise
     */
    float       rxNoise;
}rfMonRxGainRawValue;

/* Monitoring Report Structures */
typedef struct rfMonReport_t
{
    /**
     * @brief    Mask for monitor failures
     */
    uint32_t                    monFailureMask;

    /**
     * @brief    Global counter for number of FTTIs
     */
    uint32_t                    gNumFtti;

    /**
     * @brief    The report header includes common information 
     *           across all enabled monitors like current FTTI number 
     *           and current temperature.
     *           event: RL_RF_AE_MON_REPORT_HEADER_SB
     */
    rlMonReportHdrData_t       rfMonHdrReport;

    /**
     * @brief    This report contains the measured temperature near various RF analog and digital modules.
     *           The xWR device sends this to host at the programmed periodicity or when failure occurs.
     *           Event:RL_RF_AE_MON_TEMPERATURE_REPORT_SB
     */
    rlMonTempReportData_t      monTempReport;

    /**
     * @brief    This API is a Monitoring report which RadarSS sends to the host,
     *           containing the measured RX gain and phase values. RadarSS sends this
     *           to host at the programmed periodicity or when failure occurs, as programmed
     *           by the configuration API SB. 
     *           Event: RL_RF_AE_MON_RX_GAIN_PHASE_REPORT
     */
    rlMonRxGainPhRep_t         monRxGainPhReport;

    /**
    *@brief This is the Monitoring report which RadarSS sends
    *       to the host, containing the measured RX noise figure values
    *       corresponding to the full IF band of a profile. RadarSS sends
    *       this to host at the programmed periodicity or when failure occurs,
    *       as programmed by the configuration API SB. Event: RL_RF_AE_MON_RX_NOISE_FIG_REPORT
     */
    rlMonRxNoiseFigRep_t        monRxNoiseFigReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing the measured
     *           RX IF filter attenuation values at the given IF frequencies. RadarSS sends this to host
     *           at the programmed periodicity or when failure occurs, as programmed by the configuration API SB.
     *
     *           Event: RL_RF_AE_MON_RX_IF_STAGE_REPORT
     */
    rlMonRxIfStageRep_t        monRxIfStageReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing the
     *           measured TX power values during an explicit monitoring chirp. RadarSS sends this to
     *           host at the programmed periodicity or when failure occurs, as programmed by the
     *           configuration API SB. Same structure is application for Tx0/Tx1/Tx2 power report.
     *           Event: RL_RF_AE_MON_TXn_POWER_REPORT
     */
    rlMonTxPowRep_t            monTx0powReport;
    rlMonTxPowRep_t            monTx1powReport;
    rlMonTxPowRep_t            monTx2powReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing the measured
     *           TX reflection coefficient's magnitude values, meant for detecting TX ball break. RadarSS sends
     *           this to host at the programmed periodicity or when failure occurs, as programmed by the
     *           configuration API SB.
     *           Same strucuture is applicable for Tx0/Tx1/Tx2 ball break report. 
     *           Event: RL_RF_AE_MON_TXn_BALLBREAK_REPORT
     */
    rlMonTxBallBreakRep_t      monTx0BallbreakReport;
    rlMonTxBallBreakRep_t      monTx1BallbreakReport;
    rlMonTxBallBreakRep_t      monTx2BallbreakReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing the measured
     *           TX1 BPM error values. RadarSS sends this to host at the programmed periodicity or when failure
     *           occurs, as programmed by the configuration API SB. Same structure is applicable for
     *           Tx0/Tx1/Tx2 BPM report data.
     *
     *           Event: RL_RF_AE_MON_TXn_BPM_REPORT
     */
    rlMonTxBpmRep_t            monTx0BpmReport;
    rlMonTxBpmRep_t            monTx1BpmReport;
    rlMonTxBpmRep_t            monTx2BpmReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing information
     *           related to measured frequency error during the chirp. RadarSS sends this to host at the
     *           programmed periodicity or when failure occurs, as programmed by the configuration API SB.
     *
     *           Event: RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT
     */
    rlMonSynthFreqRep_t        monSynthFreqReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing the external
     *           signal voltage values measured using the GPADC. RadarSS sends this to host at the programmed
     *           periodicity or when failure occurs, as programmed by the configuration API SB.
     *
     *           Event: RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT
     */
    rlMonExtAnaSigRep_t        monExtAnaSigReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing information
     *           about Internal TX internal analog signals. RadarSS sends this to host at the programmed
     *           periodicity or when failure occurs, as programmed by the configuration API SB. Same structure
     *           is applicable for Tx0/Tx1/Tx2 monitoring report.
     *
     *           Event: RL_RF_AE_MON_TXn_INT_ANA_SIG_REPORT
     */
    rlMonTxIntAnaSigRep_t      monTx0IntAnaSigReport;
    rlMonTxIntAnaSigRep_t      monTx1IntAnaSigReport;
    rlMonTxIntAnaSigRep_t      monTx2IntAnaSigReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing information
     *           about Internal RX internal analog signals. RadarSS sends this to host at the programmed
     *           periodicity or when failure occurs, as programmed by the configuration API SB.
     *           Event: RL_RF_AE_MON_RX_INT_ANALOG_SIG_REPORT
     */
    rlMonRxIntAnaSigRep_t      monRxIntAnaSigReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing information
     *           about Internal PM, CLK and LO subsystems' internal analog signals. RadarSS sends this to host
     *           at the programmed periodicity or when failure occurs, as programmed by the configuration API SB.
     *
     *           Event: RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT
     */
    rlMonPmclkloIntAnaSigRep_t  monPmClkIntAnaSigReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing information
     *           about Internal PM, CLK and LO subsystems' internal analog signals. RadarSS sends this to host
     *           at the programmed periodicity or when failure occurs, as programmed by the configuration API
     *           Event: RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT
     */
    rlMonGpadcIntAnaSigRep_t   monGpadcIntAnaSigReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing the measured PLL
     *           control voltage values during explicit monitoring chirps. RadarSS sends this to host at the
     *           programmed periodicity or when failure occurs, as programmed by the configuration API SB.
     *           Event: RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT
     */
    rlMonPllConVoltRep_t       monPllConvVoltReport;

    /**
     * @brief    This is the Monitoring report which RadarSS sends to the host, containing information about
     *           the relative frequency measurements. RadarSS sends this to host at the programmed periodicity or
     *           when failure occurs, as programmed by the configuration API SB.
     *           Event: RL_RF_AE_MON_DCC_CLK_FREQ_REPORT
     */
    rlMonDccClkFreqRep_t       monDccClkFreqReport;

    /**
     * @brief    This is the Monitoring report which the xWR device sends to the host, containing the
     *           measured RX mixer input voltage swing values. The xWR device sends this to host at the
     *           programmed periodicity or when failure occurs, as programmed by the configuration API SB.
     *           Event: RL_RF_AE_MON_RX_MIXER_IN_PWR_REPORT
     */
    rlMonRxMixrInPwrRep_t      monRxMixrInPwrReport;

    /**
     * @brief    Sensors GPADC measurement data
     *           Event: RL_RF_AE_GPADC_MEAS_DATA_SB
     */
    rlRecvdGpAdcData_t         monGpAdcDataReport;

    /**
     * @brief    Latent fault digital monitoring status data for following event
     *           Event: RL_RF_AE_DIG_LATENTFAULT_REPORT_AE_SB
     */
    rlDigLatentFaultReportData_t monDigLatentFaultReport;

    /**
     * @brief    Periodical Digital test report
     *           Event: RL_RF_AE_MON_DIG_PERIODIC_REPORT_SB
     */
    rlDigPeriodicReportData_t   monDigPeriodReport;

    /**
     * @brief    Runtime calibration report
     */
    rlRfRunTimeCalibReport_t    runtimeCalibReport;

    /**
     * @brief    Saved raw values from Rx Gain and phase report
     *           These values can be used during inter-frame time for report purpose
     */
    rfMonRxGainRawValue         monRxGainPhRawVal;
}rfMonReport;

/* Monitoring failure counter Structures */
typedef struct rfMonFailureReport_t
{
    /**
     * @brief    Failure count based on failure report
     */
    uint32_t            tempMonFailureCnt;
    uint32_t            rxIfStageFailureCnt;
    uint32_t            rxGainPhaseFailureCnt;
    uint32_t            rxNoiseFigFailureCnt;
    uint32_t            tx0PowerFailureCnt;
    uint32_t            tx1PowerFailureCnt;
    uint32_t            tx2PowerFailureCnt;
    uint32_t            tx0BallbreakFailureCnt;
    uint32_t            tx1BallbreakFailureCnt;
    uint32_t            tx2BallbreakFailureCnt;
    uint32_t            tx0BpmFailureCnt;
    uint32_t            tx1BpmFailureCnt;
    uint32_t            tx2BpmFailureCnt;
    uint32_t            synthFreqFailureCnt;
    uint32_t            extAnaSigFailureCnt;
    uint32_t            tx0IntAnaSigFailureCnt;
    uint32_t            tx1IntAnaSigFailureCnt;
    uint32_t            tx2IntAnaSigFailureCnt;
    uint32_t            rxIntAnaSigFailureCnt;
    uint32_t            pmClkLoIntAnaSigFailureCnt;
    uint32_t            gpadcIntAnaSigFailureCnt;
    uint32_t            pllConVoltFailureCnt;
    uint32_t            dualClkCompFailureCnt;
    uint32_t            rxMixInpwrFailureCnt;
    uint32_t            rfDigMonFailureCnt;
    uint32_t            rfDigPeriodFailureCnt;
    uint32_t            runtimeCalibFailureCnt;
}rfMonFailureReport;


/*******************************************************************************************
            Configuration defaults
********************************************************************************************/
extern rlTempMonConf_t gTempMonCfg;
extern rlRxGainPhaseMonConf_t gRxGainPhaseMonCfg;
extern rlRxIfStageMonConf_t gRxIfStageMonCfg;
extern rlSynthFreqMonConf_t gSynthFreqMonCfg;
extern rlExtAnaSignalsMonConf_t gExtAnaSigMonCfg;
extern rlRxIntAnaSignalsMonConf_t gRxIntAnaSigMonCfg;
extern rlPmClkLoIntAnaSignalsMonConf_t gPmClkLoIntAnaSigMonCfg;
extern rlGpadcIntAnaSignalsMonConf_t gGpadcIntAnaSigMonCfg;
extern rlPllContrVoltMonConf_t gPllConVoltMonCfg;
extern rlDualClkCompMonConf_t gDualClkCompMonCfg;
extern rlRxMixInPwrMonConf_t gRxMixInpwrMonCfg;
extern rlTxPowMonConf_t gTxPowMonCfg[];
extern rlTxBallbreakMonConf_t gTxBallbreakMonCfg[];
extern rlTxBpmMonConf_t gTxBpmMonCfg[];
extern rlMonDigEnables_t gDigMonitorEn;
extern rlDigMonPeriodicConf_t gDigMonitorPeriod;
extern rlRxNoiseMonConf_t gRxNoiseMonCfg;
extern rlRxSatMonConf_t gRxIfSatMonCfg;
extern rlSigImgMonConf_t gSigImgMonCfg;
extern rlTxIntAnaSignalsMonConf_t gTxIntAnaSigMonCfg[];
extern rlMonAnaEnables_t  gRfAnaMonitorEn;

/*******************************************************************************************
            External Funcation API
********************************************************************************************/
extern void RFMon_initCfg(void);
extern void RFMon_reportStatsToHost(uint8_t checkFTTI);

#ifdef __cplusplus
}
#endif

#endif /* RFMONITOR_INTERNAL_H */
