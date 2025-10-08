/*
 *   @file  rfmonitor.h
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

#ifndef RFMONITOR_H
#define RFMONITOR_H

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
#include "common/rfmonitor_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! @brief   Temperature monitor Report */
#define RF_TEMPERATURE_MON         (0)
#define RF_RXGAIN_PHASE_MON        (1)
#define RF_RX_NOISE_MON            (2)
#define RF_RXIFA_STAGE_MON         (3)
#define RF_TX0_POWER_MON           (4)
#define RF_TX1_POWER_MON           (5)
#define RF_TX2_POWER_MON           (6)
#define RF_TX0_BALLBREAK_MON       (7)
#define RF_TX1_BALLBREAK_MON       (8)
#define RF_TX2_BALLBREAK_MON       (9)
#define RF_TX_GAIN_PAHSE_MON       (10)
#define RF_TX0_BPM_MON             (11)
#define RF_TX1_BPM_MON             (12)
#define RF_TX2_BPM_MON             (13)
#define RF_SYNTH_FREQ_MON          (14)
#define RF_EXT_ANA_SIG_MON         (15)
#define RF_TX0_INT_ANA_SIG_MON     (16)
#define RF_TX1_INT_ANA_SIG_MON     (17)
#define RF_TX2_INT_ANA_SIG_MON     (18)
#define RF_RX_INT_ANA_SIG_MON      (19)
#define RF_PMCLK_LO_SIG_MON        (20)
#define RF_GPADC_SIG_MON           (21)
#define RF_PLL_CTRL_VOL_MON        (22)
#define RF_DCC_CLK_FREQ_MON        (23)
#define RF_RX_IF_SATUR_MON         (24)
#define RF_RX_SIG_IMG_BAND_MON     (25)
#define RF_RX_MIXER_IN_PWR_MON     (26)
#define RF_MON_RESERVED0           (27)
#define RF_MON_RESERVED1           (28)
#define RF_DIG_PERIODIC_MON        (29) /* for Digital periodic monitor report */
#define RF_DIG_LATENT_FAULT_MON    (30) /* Digital latent fault monitor report */
#define RF_RUN_TIME_CALIB          (31) /* Run time calibration */

extern volatile uint32_t gmonReportData;

/*******************************************************************************************
            External Funcation API
********************************************************************************************/
extern int32_t CLI_RFMonitorExtensionInit(int32_t argc, char* argv[]);
extern int32_t CLI_RFMonitorExtensionHandler(int32_t argc, char* argv[]);
extern void RFMon_resetFailureReports(  void);
extern int32_t RFMon_config(int32_t *errCode);
extern int32_t RFMon_reportHandler(uint16_t msgId, uint16_t asyncSB, uint8_t *payload);
extern void RFMon_updateCQEnMask(uint8_t rxSatMonEn, uint8_t sigImgMonEn);
extern void RFMon_monReportToHost(void);
extern void RFMon_reportStatsToHost(uint8_t checkFTTI);



#ifdef __cplusplus
}
#endif

#endif /* RFMONITOR_H */

