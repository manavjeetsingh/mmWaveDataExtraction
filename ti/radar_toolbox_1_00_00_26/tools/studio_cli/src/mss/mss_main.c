/**
 *   @file  mss_main.c
 *
 *   @brief
 *      This is the main file which implements the mmWave Studio CLI App
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
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


/* BIOS/XDC Include Files. */
#include <xdc/std.h>
#include <xdc/cfg/global.h>
#include <xdc/runtime/IHeap.h>
#include <xdc/runtime/System.h>
#include <xdc/runtime/Error.h>
#include <xdc/runtime/Memory.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Task.h>
#include <ti/sysbios/knl/Event.h>
#include <ti/sysbios/knl/Semaphore.h>
#include <ti/sysbios/knl/Clock.h>
#include <ti/sysbios/heaps/HeapBuf.h>
#include <ti/sysbios/heaps/HeapMem.h>
#include <ti/sysbios/knl/Event.h>
#include <ti/sysbios/family/arm/v7a/Pmu.h>
#include <ti/sysbios/family/arm/v7r/vim/Hwi.h>
#include <ti/sysbios/utils/Load.h>


/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/esm/esm.h>
#include <ti/drivers/crc/crc.h>
#include <ti/drivers/gpio/gpio.h>
#include <ti/drivers/mailbox/mailbox.h>
#include <ti/drivers/pinmux/pinmux.h>
#include <ti/control/dpm/dpm.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/uart/UART.h>
#include <ti/utils/mathutils/mathutils.h>


/* Demo Include Files */
#include "mss/mmw_mss.h"
#include "mss/mmw_cli.h"
#include "common/mmwl_if.h"
#include <common/rfmonitor_internal.h>

/* Profiler Include Files */
#include <ti/utils/cycleprofiler/cycle_profiler.h>
#include <ti/drivers/pinmux/pinmux.h>
#include "common/mmw_output.h"
#include "common/rfmonitor.h"
#include "common/mmw_rfparser.h"
#include "common/mmw_res.h"

/**
 * @brief Task Priority settings:
 * Monitor task is at higher priority because of potential async messages from BSS
 * that need quick action in real-time.
 *
 * CLI task must be at a lower priority than Monitor Task
 */
#define MMWDEMO_CLI_TASK_PRIORITY                  5
#define MMWDEMO_MONITOR_CTRL_TASK_PRIORITY         3


/* These address offsets are in bytes, when configure address offset in hardware,
   these values will be converted to number of 128bits
   Buffer at offset 0x0U is reserved by BSS, hence offset starts from 0x800
 */
#define MMW_DEMO_CQ_SIGIMG_ADDR_OFFSET          0x800U
#define MMW_DEMO_CQ_RXSAT_ADDR_OFFSET           0x1000U

/* CQ data is at 16 bytes alignment for multiple chirps */
#define MMW_DEMO_CQ_DATA_ALIGNMENT            16U


/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
MmwDemo_MSS_MCB    gMmwMssMCB;

/*Counter for number of chirps*/
volatile uint32_t gLinkChirpCnt = 0U;

/*Counter for number of frames*/
volatile uint32_t gLinkFrameCnt = 0U;

/* semaphore for Monitor report streaming Task */
Semaphore_Handle  gMonReportStreamSem;

extern RFMonitorCfg     gRFMonitorCfg;
extern MMW_CLI_AllCfg  gCLICmdCfg;

/* Store the monitoring report bits which are received from BSS */
volatile uint32_t gmonReportData;

/* this variable is used to trigger RF_Init (boot time calibration) in BSS */
volatile char isRfInitCmd = 0;
/* if valid to send monitoring report now.
 * It is being set only when sensorStart done properly */
volatile char gIsMonitorSend = 0;

/**
 * @brief
 *  Global Variable for LDO BYPASS config   
 **********************************************************
  AS PER RECOMMENDATION FROM THE xWR1843BOOST EVM User guide
 **********************************************************
   set LDO bypass since the board has 1.0V RF supply 1 
   and 1.0V RF supply 2. Please update this API if this 
   assumption is not valid else it may DAMAGE your board!
 **********************************************************
 **********************************************************
 */
rlRfLdoBypassCfg_t gRFLdoBypassCfg =
{
    .ldoBypassEnable   = 3, /* 1.0V RF supply 1 and 1.0V RF supply 2 */
    .supplyMonIrDrop   = 1, /* IR drop of 3% */
    .ioSupplyIndicator = 0, /* 3.3 V IO supply */
};

/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/

extern void MmwDemo_CLIInit(uint8_t taskPriority);
extern void MmwaveLink_setLogFunc(RL_PRINT_FUNC func);


/**************************************************************************
 ************************* Millimeter Wave Demo Functions prototype *************
 **************************************************************************/

/* MMW demo functions for datapath operation */
static void MmwDemo_dataPathOpen(void);
static void MmwDemo_dataPathStart (void);
static void MmwDemo_initTask(UArg arg0, UArg arg1);
static void MmwDemo_platformInit(MmwDemo_platformCfg *config);
/* external sleep function when in idle (used in .cfg file) */
void MmwDemo_sleep(void);

/* Edma related functions */
static void MmwDemo_edmaInit(void);
static void MmwDemo_edmaOpen(void);
static void MmwDemo_EDMA_transferControllerErrorCallbackFxn(EDMA_Handle handle,
                EDMA_transferControllerErrorInfo_t *errorInfo);
static void MmwDemo_EDMA_errorCallbackFxn(EDMA_Handle handle, EDMA_errorInfo_t *errorInfo);

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/


/**************************************************************************
 ************************* Millimeter Wave Demo Functions **********************
 **************************************************************************/
/**
 *  @b Description
 *  @n
 *      Send assert information through CLI.
 */
void _MmwDemo_debugAssert(int32_t expression, const char *file, int32_t line)
{
    if (!expression) {
        MmwDemo_CLI_write ("Exception: %s, line %d.\n",file,line);
    }
}

/**
 *  @b Description
 *  @n
 *      Utility function to control GPIO LED State
 *
 *  @param[in] LED On: 1, Off: 0
 *
 *  @retval none
 */
void MmwDemo_gpioStateCtrl(uint32_t ledState)
{
    /* The sensor has been stopped successfully. Switch off the LED */
    GPIO_write (gMmwMssMCB.cfg.platformCfg.SensorStatusGPIO, ledState);
    /* if LED is requested to switched off by the application,
     * then it is assumed that we got sensor-stop Async-event message.
     * So change the sensorState
     */
    if(ledState == 0)
    {
        gMmwMssMCB.sensorState = MmwDemo_SensorState_STOPPED;
    }
}

/**
 *  @b Description
 *  @n
 *      Utility function to apply configuration to specified sub-frame
 *
 *  @param[in] srcPtr Pointer to configuration
 *  @param[in] offset Offset of configuration within the parent structure
 *  @param[in] size   Size of configuration
 *  @param[in] subFrameNum Sub-frame Number (0 based) to apply to, broadcast to
 *                         all sub-frames if special code MMWDEMO_SUBFRAME_NUM_FRAME_LEVEL_CONFIG
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void MmwDemo_CfgUpdate(void *srcPtr, uint32_t offset, uint32_t size, int8_t subFrameNum)
{    
    /* if subFrameNum undefined, broadcast to all sub-frames */
    if(subFrameNum == MMWDEMO_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            memcpy((void *)((uint32_t) &gMmwMssMCB.subFrameCfg[indx] + offset), srcPtr, size);
        }
    }
    else
    {
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        memcpy((void *)((uint32_t) &gMmwMssMCB.subFrameCfg[subFrameNum] + offset), srcPtr, size);
    }
}

/**
 *  @b Description
 *  @n
 *      Utility function to tranmist monitoring report data over UART.
 *
 *  @param[in] uartHandle UART handle
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static void MmwDemo_transmitProcessedOutput(UART_Handle uartHandle)
{
    MmwDemo_output_message_header header;
    uint32_t tlvIdx = 0;
    uint32_t numPaddingBytes;
    uint32_t packetLen;
    uint8_t padding[MMW_OUTPUT_MSG_SEGMENT_LEN];
    MmwDemo_output_message_tl   tl[MMWDEMO_OUTPUT_MSG_MAX];
    extern rfMonReport gRFMonReport;
    
    /* Clear message header */
    memset((void *)&header, 0, sizeof(MmwDemo_output_message_header));

    /********** Packet format *********/
    /*
     * Header         [4B] : 0x1234 5678
     * version        [4B] : [32:24] Major, [23:16] Minor, [15:8] bugFix, [7:0] build version
     * totalPacketLen [4B] : total packet length
     * frameNumber    [4B] : current frame count
     * platform       [2B] : 0xFFFFA1843
     * numTlv         [2B] : total no. of TLV in this packet
     *
     * TLV format:
     * TLV Type       [2B] : TLV type
     * TLV length     [2B] : current TLV length
     * TLV data       [TLV_Len]
     **********************************/
    /* Header: */
    /* added prefix of 'FFF' before each device, AWR and IWR devices are same here */
#ifdef SOC_XWR18XX
    header.platform =  0xFFFA1843;
#endif
#ifdef SOC_XWR16XX
    header.platform =  0xFFFA1642;
#endif
#ifdef SOC_XWR68XX
    header.platform =  0xFFFA6843;
#endif
#ifdef SOC_XWR14XX
    header.platform =  0xFFFA1443;
#endif
    /* add Magic word (2x2B) to header to distinguish b/w different packaets */
    header.magicWord[0] = MESSAGE_MAGIC_WORD1;
    header.magicWord[1] = MESSAGE_MAGIC_WORD2;
    /* embed SDK Version to header field */
    header.version =    MMWAVE_SDK_VERSION_BUILD |
                        (MMWAVE_SDK_VERSION_BUGFIX << 8) |
                        (MMWAVE_SDK_VERSION_MINOR << 16) |
                        (MMWAVE_SDK_VERSION_MAJOR << 24);

    packetLen = sizeof(MmwDemo_output_message_header);

    /*********************************************************************************
     * gmonReportData variable contains the bit status of each type of monitoring
     * report recieved from BSS at every FTTI. So here we check each bit status and
     * based on that status send the monitoring report. Clear that specific monitor
     * type bit in gmonReportData after that report is sent over UART.
     *********************************************************************************/

    /* check if Temperature monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData & (1U << RF_TEMPERATURE_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TEMPERATURE_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTempReportData_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if RX Gain monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_RXGAIN_PHASE_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_RX_GAIN_PHASE_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonRxGainPhRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Rx Noise monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_RX_NOISE_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_RX_NOISE_FIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonRxNoiseFigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Tx0 Power monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX0_POWER_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX0_POWER_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxPowRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Tx1 Power monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX1_POWER_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX1_POWER_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxPowRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Tx2 Power monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX2_POWER_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX2_POWER_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxPowRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Tx0 Ballbreak monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX0_BALLBREAK_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX0_BALLBRK_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxBallBreakRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Tx1 Ballbreak monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX1_BALLBREAK_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX1_BALLBRK_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxBallBreakRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Tx2 Ballbreak monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX2_BALLBREAK_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX2_BALLBRK_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxBallBreakRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Ext signal Voltage monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_EXT_ANA_SIG_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_EXT_ANALOG_SIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonExtAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if RX IF filter monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_RXIFA_STAGE_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_RX_IF_STAGE_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonRxIfStageRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if SYNTH Freq monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_SYNTH_FREQ_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_SYNTHESIZER_FREQ_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonSynthFreqRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if PLL Control monitoring report is received from BSS in last FTTI.
     * And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_PLL_CTRL_VOL_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_PLL_CONTROL_VOLT_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonPllConVoltRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if TX0 Internal analog signals monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX0_INT_ANA_SIG_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX0_INT_ANA_SIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxIntAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if TX1 Internal analog signals monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX1_INT_ANA_SIG_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX1_INT_ANA_SIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxIntAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if TX2 Internal analog signals monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_TX2_INT_ANA_SIG_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_TX2_INT_ANA_SIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonTxIntAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Internal PM, CLK and LO monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_PMCLK_LO_SIG_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_PMCLKLO_INT_ANA_SIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonPmclkloIntAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if RX internal analog signals monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_RX_INT_ANA_SIG_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_RX_INT_ANA_SIG_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonRxIntAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if GPADC input DC signals monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_GPADC_SIG_MON))
    {
        tl[tlvIdx].type = MMW_RF_GPADC_INT_ANA_SIG_REP;
        tl[tlvIdx].length = sizeof(rlMonGpadcIntAnaSigRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if DCC Clock Measurement monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData  & (1U << RF_DCC_CLK_FREQ_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_DCC_CLK_FREQ_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlMonDccClkFreqRep_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Run time calibration report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData & (1U<< RF_RUN_TIME_CALIB))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_RUNTIME_CALIB_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlRfRunTimeCalibReport_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if Latent fault digital monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData & (1U<< RF_DIG_LATENT_FAULT_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_DIG_LATENTFAULT_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlDigLatentFaultReportData_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* check if periodic digital monitoring report is received from BSS in
     * last FTTI. And set the Type & length of this monitoring type to TLV */
    if (gmonReportData & (1U<< RF_DIG_PERIODIC_MON))
    {
        tl[tlvIdx].type = (uint16_t)MMW_RF_DIG_PERIODIC_REP;
        tl[tlvIdx].length = (uint16_t)sizeof(rlDigPeriodicReportData_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }

    header.numTLVs = tlvIdx;
    /* Round up packet length to multiple of MMW_OUTPUT_MSG_SEGMENT_LEN */
    header.totalPacketLen = MMW_OUTPUT_MSG_SEGMENT_LEN *
            ((packetLen + (MMW_OUTPUT_MSG_SEGMENT_LEN-1))/MMW_OUTPUT_MSG_SEGMENT_LEN);
    header.frameNumber = gLinkFrameCnt;

    if(tlvIdx == 0)
    {
        goto EXIT;
    }

    /* write Header */
    UART_writePolling (uartHandle,
                       (uint8_t*)&header,
                       sizeof(MmwDemo_output_message_header));
    
    tlvIdx = 0;
    /* write TLV struct (type+Length) and then monitor report based on if
     * received that monitor type in last FTTI. And after sending that specific
     * Monitor report, clear that report status bit in gmonReportData  */
   if (gmonReportData & (1U << RF_TEMPERATURE_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTempReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TEMPERATURE_MON);
    }
    
    if (gmonReportData & (1U << RF_RXGAIN_PHASE_MON))
    {
       UART_writePolling (uartHandle,
                          (uint8_t*)&tl[tlvIdx],
                          sizeof(MmwDemo_output_message_tl));

       /* Address translation is done when buffer is received*/
       UART_writePolling (uartHandle,
                          (uint8_t*)&gRFMonReport.monRxGainPhReport,
                          tl[tlvIdx].length);
       tlvIdx++;
       gmonReportData &= ~(1U << RF_RXGAIN_PHASE_MON);
    }

    if (gmonReportData & (1U << RF_RX_NOISE_MON))
    {
       UART_writePolling (uartHandle,
                          (uint8_t*)&tl[tlvIdx],
                          sizeof(MmwDemo_output_message_tl));

       /* Address translation is done when buffer is received*/
       UART_writePolling (uartHandle,
                          (uint8_t*)&gRFMonReport.monRxNoiseFigReport,
                          tl[tlvIdx].length);
       tlvIdx++;
       gmonReportData &= ~(1U << RF_RX_NOISE_MON);
    }

    if (gmonReportData & (1U << RF_TX0_POWER_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx0powReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX0_POWER_MON);
    }
    
    if (gmonReportData & (1U << RF_TX1_POWER_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx1powReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX1_POWER_MON);
    }
    
    if (gmonReportData & (1U << RF_TX2_POWER_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx2powReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX2_POWER_MON);
    }

    if (gmonReportData & (1U << RF_TX0_BALLBREAK_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx0BallbreakReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX0_BALLBREAK_MON);
    }
    
    if (gmonReportData & (1U << RF_TX1_BALLBREAK_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx1BallbreakReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX1_BALLBREAK_MON);
    }
    if (gmonReportData & (1U << RF_TX2_BALLBREAK_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx2BallbreakReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX2_BALLBREAK_MON);
    }
    if (gmonReportData & (1U << RF_EXT_ANA_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monExtAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_EXT_ANA_SIG_MON);
    }
    if (gmonReportData & (1U << RF_RXIFA_STAGE_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monRxIfStageReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_RXIFA_STAGE_MON);
    }
    if (gmonReportData & (1U << RF_SYNTH_FREQ_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monSynthFreqReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_SYNTH_FREQ_MON);
    }
    if (gmonReportData & (1U << RF_PLL_CTRL_VOL_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monPllConvVoltReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_PLL_CTRL_VOL_MON);
    }
    if (gmonReportData & (1U << RF_TX0_INT_ANA_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx0IntAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX0_INT_ANA_SIG_MON);
    }
    if (gmonReportData & (1U << RF_TX1_INT_ANA_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx1IntAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX1_INT_ANA_SIG_MON);
    }
    if (gmonReportData & (1U << RF_TX2_INT_ANA_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monTx2IntAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_TX2_INT_ANA_SIG_MON);
    }
    if (gmonReportData & (1U << RF_PMCLK_LO_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monPmClkIntAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_PMCLK_LO_SIG_MON);
    }
    if (gmonReportData & (1U << RF_RX_INT_ANA_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monRxIntAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_RX_INT_ANA_SIG_MON);
    }
    if (gmonReportData & (1U << RF_GPADC_SIG_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monGpadcIntAnaSigReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_GPADC_SIG_MON);
    }
    if (gmonReportData & (1U << RF_DCC_CLK_FREQ_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monDccClkFreqReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_DCC_CLK_FREQ_MON);
    }
    
    if (gmonReportData & (1U<< RF_RUN_TIME_CALIB))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.runtimeCalibReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_RUN_TIME_CALIB);
    }

    if (gmonReportData & (1U<< RF_DIG_LATENT_FAULT_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monDigLatentFaultReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_DIG_LATENT_FAULT_MON);
    }

    if (gmonReportData & (1U<< RF_DIG_PERIODIC_MON))
    {
        UART_writePolling (uartHandle,
                           (uint8_t*)&tl[tlvIdx],
                           sizeof(MmwDemo_output_message_tl));

        /* Address translation is done when buffer is received*/
        UART_writePolling (uartHandle,
                           (uint8_t*)&gRFMonReport.monDigPeriodReport,
                           tl[tlvIdx].length);
        tlvIdx++;
        gmonReportData &= ~(1U << RF_DIG_PERIODIC_MON);
    }

    /* Send padding bytes */
    numPaddingBytes = MMW_OUTPUT_MSG_SEGMENT_LEN - (packetLen & (MMW_OUTPUT_MSG_SEGMENT_LEN-1));
    if (numPaddingBytes<MMW_OUTPUT_MSG_SEGMENT_LEN)
    {
        UART_writePolling (uartHandle,
                            (uint8_t*)padding,
                            numPaddingBytes);
    }

EXIT:
    return;
}

/**
 *  @b Description
 *  @n
 *      Function to stream the monitor report over UART.
 *      This function is triggered as Task and wait on semaphore to
 *      either trigger rf_init or transmit the Monitor report (post
 *      sensor start).
 *
 *  @retval None.
 */
static void MonReportStreamTask (UArg arg0, UArg arg1)
{
    while (1)
    {
        /* Pending on the link semaphore */
        Semaphore_pend (gMonReportStreamSem, BIOS_WAIT_FOREVER);

        if(isRfInitCmd)
        {
            /* do the boot time calibration before profile config*/
            MmwaveLink_rfCalibration();
            MmwaveLink_devCfg();
            isRfInitCmd = 0;
        }
        else
        {
            /* report based on requested mode */
            if(gmonReportData && (gIsMonitorSend))
            {
                /* report at every frame or Nth frame */
                if((gRFMonitorCfg.reportCfg.enMonReportMode == 1) || \
                   ((gRFMonitorCfg.reportCfg.enMonReportMode > 1) && \
                    (gLinkFrameCnt % gRFMonitorCfg.reportCfg.enMonReportMode) == 0))
#ifdef TWO_UART_COMMUNICATION
                MmwDemo_transmitProcessedOutput(gMmwMssMCB.loggingUartHandle);
#else
                MmwDemo_transmitProcessedOutput(gMmwMssMCB.commandUartHandle);
#endif
            }
        }               
    }
}

/**
 *  @b Description
 *  @n
 *      Function to Setup the HSI Clock. Required for LVDS streaming.
 *
 *  @retval
 *      0  - Success.
 *      <0 - Failed with errors
 */
int32_t MmwDemo_mssSetHsiClk(void)
{
    rlDevHsiClk_t                           hsiClkgs;
    int32_t                                 retVal;

    /*************************************************************************************
     * Setup the HSI Clock through the mmWave Link:
     *************************************************************************************/
    memset ((void*)&hsiClkgs, 0, sizeof(rlDevHsiClk_t));

    /* Setup the HSI Clock as per the Radar Interface Document:
     * - This is set to 600Mhz DDR Mode */
    hsiClkgs.hsiClk = 0x9;

    /* Setup the HSI in the radar link: */
    retVal = rlDeviceSetHsiClk(RL_DEVICE_MAP_CASCADED_1, &hsiClkgs);
    if (retVal != RL_RET_CODE_OK)
    {
        /* Error: Unable to set the HSI clock */
        System_printf ("Error: Setting up the HSI Clock Failed [Error %d]\n", retVal);
        return retVal;
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      mmw demo helper Function to do one time sensor initialization. 
 *      User need to fill gMmwMssMCB.cfg.openCfg before calling this function
 *
 *  @param[in]  isFirstTimeOpen     If true then issues MMwave_open
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_openSensor(bool isFirstTimeOpen)
{
    int32_t             retVal;

    /*  Open mmWave module, this is only done once */
    if (isFirstTimeOpen == true)
    {
        retVal = rlRfSetLdoBypassConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlRfLdoBypassCfg_t*)&gRFLdoBypassCfg);        
        if(retVal != 0)
        {            
            System_printf("Error: rlRfSetLdoBypassConfig retVal=%d\n", retVal);
            return -1;
        }
        
        /**********************************************************
         **********************************************************/         

        /* Open mmWave module, this is only done once */
        /* Setup the calibration frequency:*/
        gMmwMssMCB.cfg.openCfg.freqLimitLow  = 760U;
        gMmwMssMCB.cfg.openCfg.freqLimitHigh = 810U;

        /* start/stop async events */
        gMmwMssMCB.cfg.openCfg.disableFrameStartAsyncEvent = false;
        gMmwMssMCB.cfg.openCfg.disableFrameStopAsyncEvent  = false;

        /* No custom calibration: */
        gMmwMssMCB.cfg.openCfg.useCustomCalibration        = false;
        gMmwMssMCB.cfg.openCfg.customCalibrationEnableMask = 0x0;

        /* calibration monitoring base time unit
         * setting it to one frame duration as the demo doesnt support any 
         * monitoring related functionality
         */
        gMmwMssMCB.cfg.openCfg.calibMonTimeUnit            = 1;

        /*Set up HSI clock*/
        if(MmwDemo_mssSetHsiClk() < 0)
        {
            System_printf ("Error: MmwDemo_mssSetHsiClk failed.\n");
            return -1;
        }

        /* Open the datapath modules that runs on MSS */
        MmwDemo_dataPathOpen();
    }
    return 0;
}


/**
 *  @b Description
 *  @n
 *      Perform Data path driver open 
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_dataPathOpen(void)
{
    if(gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_CONTINUOUS)
    {
        gMmwMssMCB.adcBufHandle = Mmw_ADCBufOpen(gMmwMssMCB.socHandle, 1);
    }
    else
    {
        gMmwMssMCB.adcBufHandle = Mmw_ADCBufOpen(gMmwMssMCB.socHandle, 0);
    }
    if(gMmwMssMCB.adcBufHandle == NULL)
    {
        MmwDemo_debugAssert(0);
    }
}

/**
 *  @b Description
 *  @n
 *      Function to configure CQ.
 *
 *  @param[in] subFrameCfg Pointer to sub-frame config
 *  @param[in] numChirpsPerChirpEvent number of chirps per chirp event
 *  @param[in] validProfileIdx valid profile index
 *
 *  @retval
 *      0 if no error, else error (there will be system prints for these).
 */
static int32_t Mmw_configCQ(MmwDemo_SubFrameCfg *subFrameCfg,
                                uint8_t numChirpsPerChirpEvent,
                                uint8_t validProfileIdx)
{
    ADCBuf_CQConf               cqConfig;
    int32_t                     retVal;
    uint16_t                    cqChirpSize;
 
    if( CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
            RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
            RFMON_ANAMON_RX_IF_SATURATION_EN_BIT))
    {
        /* CQ driver config */
        memset((void *)&cqConfig, 0, sizeof(ADCBuf_CQConf));
        cqConfig.cqDataWidth = 0; /* 16bit for mmw demo */
        cqConfig.cq1AddrOffset = MMW_DEMO_CQ_SIGIMG_ADDR_OFFSET; /* CQ1 starts from the beginning of the buffer */
        cqConfig.cq2AddrOffset = MMW_DEMO_CQ_RXSAT_ADDR_OFFSET;  /* Address should be 16 bytes aligned */

        retVal = ADCBuf_control(gMmwMssMCB.adcBufHandle, ADCBufMMWave_CMD_CONF_CQ, (void *)&cqConfig);
        if (retVal < 0)
        {
            System_printf ("Error: MMWDemoDSS Unable to configure the CQ\n");
            MmwDemo_debugAssert(0);
        }
    }

    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
            RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT,
            RFMON_ANAMON_RX_SIG_IMG_BAND_EN_BIT))
    {
        /* This is for 16bit format in mmw demo, signal/image band data has 2 bytes/slice
           For other format, please check DFP interface document
         */
        cqChirpSize = (gRFMonitorCfg.sigImageBandMonCfg->numSlices + 1) * sizeof(uint16_t);
        cqChirpSize = MATHUTILS_ROUND_UP_UNSIGNED(cqChirpSize, MMW_DEMO_CQ_DATA_ALIGNMENT);
        subFrameCfg->sigImgMonTotalSize = cqChirpSize * numChirpsPerChirpEvent;
    }

    if (CSL_FEXTR(gRFMonitorCfg.rfAnaMonitorEn->enMask,
            RFMON_ANAMON_RX_IF_SATURATION_EN_BIT,
            RFMON_ANAMON_RX_IF_SATURATION_EN_BIT))
    {
        /* This is for 16bit format in mmw demo, saturation data has one byte/slice
           For other format, please check DFP interface document
         */
        cqChirpSize = (gRFMonitorCfg.rxSatMonCfg->numSlices + 1) * sizeof(uint8_t);
        cqChirpSize = MATHUTILS_ROUND_UP_UNSIGNED(cqChirpSize, MMW_DEMO_CQ_DATA_ALIGNMENT);
        subFrameCfg->satMonTotalSize = cqChirpSize * numChirpsPerChirpEvent;
    }

    return(retVal);
}


/**
 *  @b Description
 *  @n
 *      The function is used to configure the data path based on the chirp profile.
 *      After this function is executed, the data path processing will ready to go
 *      when the ADC buffer starts receiving samples corresponding to the chirps.
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_dataPathConfig (void)
{
    uint16_t            rxChanOffset[SYS_COMMON_NUM_RX_CHANNEL];
    int32_t                         errCode;
    MmwDemo_SubFrameCfg             *subFrameCfg;
    int8_t                          subFrameIndx;
    MmwDemo_RFParserOutParams       RFparserOutParams;
    uint8_t             numOfSubFrames;


    if(gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_FRAME)
    {
        numOfSubFrames = gCLICmdCfg.cliCtrlCfg.u.advancedFrameCfg.frameCfg.frameSeq.numOfSubFrames;
    }
    else
    {
        numOfSubFrames = 1;
    }

    /* Get RF frequency scale factor */
    gMmwMssMCB.rfFreqScaleFactor = SOC_getDeviceRFFreqScaleFactor(gMmwMssMCB.socHandle, &errCode);
    if (errCode < 0)
    {
        System_printf ("Error: Unable to get RF scale factor [Error:%d]\n", errCode);
        MmwDemo_debugAssert(0);
    }

    /* Reason for reverse loop is that when sensor is started, the first sub-frame
     * will be active and the ADC configuration needs to be done for that sub-frame
     * before starting (ADC buf hardware does not have notion of sub-frame, it will
     * be reconfigured every sub-frame). This cannot be alternatively done by calling
     * the Mmw_ADCBufConfig function only for the first sub-frame because this is
     * a utility API that computes the rxChanOffset that is part of ADC dataProperty
     * which will be used by range DPU and therefore this computation is required for
     * all sub-frames.
     */
    for(subFrameIndx = numOfSubFrames -1; subFrameIndx >= 0;
        subFrameIndx--)
    {
        subFrameCfg  = &gMmwMssMCB.subFrameCfg[subFrameIndx];

        /*****************************************************************************
         * Data path :: Algorithm Configuration
         *****************************************************************************/

        /* Parse the profile and chirp configs and get the valid number of TX Antennas */
        errCode = Mmw_RFParser_parseConfig(&RFparserOutParams, subFrameIndx,
                                         &gCLICmdCfg.openCfg, &gCLICmdCfg.cliCtrlCfg,
                                         &gCLICmdCfg.profChirpCfg,
                                         &subFrameCfg->adcBufCfg,
                                         gMmwMssMCB.rfFreqScaleFactor,
                                         false/* no BPM in 18xx demo */);

        /* if number of doppler chirps is too low, interpolate to be able to detect
         * better with CFAR tuning. E.g. a 2-pt FFT will be problematic in terms
         * of distinguishing direction of motion */
        if (RFparserOutParams.numDopplerChirps <= 4)
        {
            RFparserOutParams.dopplerStep = RFparserOutParams.dopplerStep / (8 / RFparserOutParams.numDopplerBins);
            RFparserOutParams.numDopplerBins = 8;
        }

        if (errCode != 0)
        {
            System_printf ("Error: MmwDemo_RFParser_parseConfig [Error:%d]\n", errCode);
            goto exit;
        }

        subFrameCfg->numRangeBins = RFparserOutParams.numRangeBins;
        /* Workaround for range DPU limitation for FFT size 1024 and 12 virtual antennas case*/
        if ((RFparserOutParams.numVirtualAntennas == 12) && (RFparserOutParams.numRangeBins == 1024))
        {
            subFrameCfg->numRangeBins = 1022;
            RFparserOutParams.numRangeBins = 1022;
        }

        subFrameCfg->numDopplerBins = RFparserOutParams.numDopplerBins;
        subFrameCfg->numChirpsPerChirpEvent = RFparserOutParams.numChirpsPerChirpEvent;
        subFrameCfg->adcBufChanDataSize = RFparserOutParams.adcBufChanDataSize;
        subFrameCfg->numAdcSamples = RFparserOutParams.numAdcSamples;
        subFrameCfg->numChirpsPerSubFrame = RFparserOutParams.numChirpsPerFrame;
        subFrameCfg->numVirtualAntennas = RFparserOutParams.numVirtualAntennas;

        errCode = Mmw_ADCBufConfig(gMmwMssMCB.adcBufHandle,
                                   gCLICmdCfg.openCfg.chCfg.rxChannelEn, 1,
                                 subFrameCfg->adcBufChanDataSize,
                                 &subFrameCfg->adcBufCfg,
                                 &rxChanOffset[0]);
        if (errCode < 0)
        {
            System_printf("Error: ADCBuf config failed with error[%d]\n", errCode);
            MmwDemo_debugAssert (0);
        }
        errCode = Mmw_configCQ(subFrameCfg, RFparserOutParams.numChirpsPerChirpEvent,
                           RFparserOutParams.validProfileIdx);
    }
exit:
    return errCode;
}

/**
 *  @b Description
 *  @n
 *      The function is used to Start data path to handle chirps from front end.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_dataPathStart (void)
{
    CBUFF_OperationalMode operationMode;
    
    DebugP_log0("App: Issuing DPM_start\n");

    /* in case of continuous modem, set ADC buffer to continuous mode with sample size */
    if(gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_CONTINUOUS)
    {
        ADCBuf_control(gMmwMssMCB.adcBufHandle, ADCBufMMWave_CMD_START_CONTINUOUS_MODE, \
                            (void *)&gCLICmdCfg.cliCtrlCfg.u.continuousModeCfg.dataTransSize);
        operationMode = CBUFF_OperationalMode_CONTINUOUS;
    }
    else
    {
       operationMode = CBUFF_OperationalMode_CHIRP; 
    }
    /* Configure HW LVDS stream for the first sub-frame that will start upon
     * start of frame */
    if (gMmwMssMCB.subFrameCfg[0].lvdsStreamCfg.dataFmt != MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED)
    {
        MmwDemo_configLVDSHwData(0, operationMode);
    }
}

/**
 *  @b Description
 *  @n
 *      mmw demo helper Function to start sensor.
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_startSensor(bool isFirstTime)
{
    int32_t     errCode, retVal = -1;

    /*****************************************************************************
     * Data path :: start data path first
     *****************************************************************************/
    MmwDemo_dataPathStart();

    /* if Continuous mode is requested */
    if(gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_CONTINUOUS)
    {
        retVal = MmwaveLink_ContMode(1);
    }
    else
    {
        /*****************************************************************************
         * RF monitor configuration
         *****************************************************************************/
        if(RFMon_config(&errCode) < 0)
        {
            System_printf("Error: RF monitor configuration failed with error[%d] \n", errCode);
            return -1;
        }
        retVal = MmwaveLink_frameTrigger(0);
        /* set monitoring report send only when frame start is done with async-event msg */
        if(retVal == 0)
        {
            gIsMonitorSend = 1;
        }
    }

    /*****************************************************************************
     * The sensor has been started successfully. Switch on the LED 
     *****************************************************************************/
    if(retVal == 0)
    {
        MmwDemo_gpioStateCtrl(1);
    }

    gMmwMssMCB.sensorStartCount++;
    return retVal;
}

/* On Sensor Stop CMD or at frame-End Async event call this function to
 * deactivate the LVDS Session which can be reconfigure and re-triggered.
 */
void deactivateLVDSSession()
{
    int32_t errCode =0;

    /* disable Monitor reporting and clear the monitor report */
    gIsMonitorSend = 0;
    gmonReportData = 0;

    /* Delete any active streaming session */
    if(gMmwMssMCB.lvdsStream.hwSessionHandle != NULL)
    {
        /* Evaluate need to deactivate h/w session:
         * One sub-frame case:
         *   if h/w only enabled, deactivation never happened, hence need to deactivate
         *   if h/w and s/w both enabled, then s/w would leave h/w activated when it is done
         *   so need to deactivate
         *   (only s/w enabled cannot be the case here because we are checking for non-null h/w session)
         * Multi sub-frame case:
         *   Given stop, we must have re-configured the next sub-frame by now which is next of the
         *   last sub-frame i.e we must have re-configured sub-frame 0. So if sub-frame 0 had
         *   h/w enabled, then it is left in active state and need to deactivate. For all
         *   other cases, h/w was already deactivated when done.
         */
        if (/*(gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames == 1) ||
            ((gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames > 1) && */
             (gMmwMssMCB.subFrameCfg[0].lvdsStreamCfg.dataFmt != MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED))
        {
            if (CBUFF_deactivateSession(gMmwMssMCB.lvdsStream.hwSessionHandle, &errCode) < 0)
            {
                System_printf("CBUFF_deactivateSession failed with errorCode = %d\n", errCode);
                MmwDemo_debugAssert(0);
            }
        }
        MmwDemo_LVDSStreamDeleteHwSession();
    }

    /* Delete s/w session if it exists. S/w session never needs to be deactivated in stop because
     * it always (unconditionally) deactivates itself upon completion.
     */
    if(gMmwMssMCB.lvdsStream.swSessionHandle != NULL)
    {
        MmwDemo_LVDSStreamDeleteSwSession();
    }

}

/**
 *  @b Description
 *  @n
 *      Stops the RF and datapath for the sensor. Blocks until both operation are completed.
 *      Prints epilog at the end.
 *
 *  @retval  None
 */
void MmwDemo_stopSensor(void)
{
    int32_t retVal;

    /* if Continuous mode is requested */
    if(gCLICmdCfg.cliCtrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_CONTINUOUS)
    {
        retVal = MmwaveLink_ContMode(0);
        ADCBuf_control(gMmwMssMCB.adcBufHandle, ADCBufMMWave_CMD_STOP_CONTINUOUS_MODE, NULL);
    }
    else
    {
        /* Stop sensor RF , data path will be stopped after RF stop is completed */
        retVal = MmwaveLink_frameStop(0);
    }

    /* Deactivate LVDS Session */
    deactivateLVDSSession();

    if(retVal == 0)
    {
        /* off the GPIO LED */
        MmwDemo_gpioStateCtrl(0);
    }
    gMmwMssMCB.sensorStopCount++;

    /* print for user */
    System_printf("Sensor has been stopped: startCount: %d stopCount %d\n",
                  gMmwMssMCB.sensorStartCount,gMmwMssMCB.sensorStopCount);
}

/**
 *  @b Description
 *  @n
 *      Platform specific hardware initialization.
 *
 *  @param[in]  config     Platform initialization configuraiton
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_platformInit(MmwDemo_platformCfg *config)
{
    /* Setup the PINMUX to bring out the UART-1 */
    Pinmux_Set_OverrideCtrl(MMW_SOC_PIN_UART1_TX, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(MMW_SOC_PIN_UART1_TX, MMW_SOC_PIN_UART1_TX_FUNC);
    Pinmux_Set_OverrideCtrl(MMW_SOC_PIN_UART1_RX, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(MMW_SOC_PIN_UART1_RX, MMW_SOC_PIN_UART1_RX_FUNC);
#ifdef TWO_UART_COMMUNICATION
    /* Setup the PINMUX to bring out the UART-3 */
    Pinmux_Set_OverrideCtrl(MMW_SOC_PIN_UART3_TX, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(MMW_SOC_PIN_UART3_TX, MMW_SOC_PIN_UART3_TX_FUNC);
#endif
    /**********************************************************************
     * Setup the PINMUX:
     * - GPIO Output: Configure pin K13 as GPIO_2 output
     **********************************************************************/
    Pinmux_Set_OverrideCtrl(MMW_SOC_PIN_GPIO2, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(MMW_SOC_PIN_GPIO2, MMW_SOC_PIN_GPIO2_FUNC);

    /**********************************************************************
     * Setup the GPIO:
     * - GPIO Output: Configure pin K13 as GPIO_2 output
     **********************************************************************/
    config->SensorStatusGPIO    = MMW_SOC_PIN_GPIO2_OUT;

    /* Initialize the DEMO configuration: */
    config->sysClockFrequency   = MSS_SYS_VCLK;
    config->loggingBaudRate     = 921600;
    config->commandBaudRate     = 921600;

    /**********************************************************************
     * Setup the DS3 LED on the EVM connected to GPIO_2
     **********************************************************************/
    GPIO_setConfig (config->SensorStatusGPIO, GPIO_CFG_OUTPUT);
}

/**
 *  @b Description
 *  @n
 *      Init EDMA.
 */
static void MmwDemo_edmaInit(void)
{
    int32_t errorCode;

    errorCode = EDMA_init(EDMA_INSTANCE_1);
    if (errorCode != EDMA_NO_ERROR)
    {
        System_printf ("Debug: EDMA instance %d initialization returned error %d\n", errorCode);
        MmwDemo_debugAssert (0);
        return;
    }

    memset(&gMmwMssMCB.EDMA_errorInfo, 0, sizeof(gMmwMssMCB.EDMA_errorInfo));
    memset(&gMmwMssMCB.EDMA_transferControllerErrorInfo, 0, sizeof(gMmwMssMCB.EDMA_transferControllerErrorInfo));
}

/**
 *  @b Description
 *  @n
 *      Call back function for EDMA CC (Channel controller) error as per EDMA API.
 *      Declare fatal error if happens, the output errorInfo can be examined if code
 *      gets trapped here.
 */
static void MmwDemo_EDMA_errorCallbackFxn(EDMA_Handle handle, EDMA_errorInfo_t *errorInfo)
{
    gMmwMssMCB.EDMA_errorInfo = *errorInfo;
    MmwDemo_debugAssert(0);
}

/**
 *  @b Description
 *  @n
 *      Call back function for EDMA transfer controller error as per EDMA API.
 *      Declare fatal error if happens, the output errorInfo can be examined if code
 *      gets trapped here.
 */
static void MmwDemo_EDMA_transferControllerErrorCallbackFxn(EDMA_Handle handle,
                EDMA_transferControllerErrorInfo_t *errorInfo)
{
    gMmwMssMCB.EDMA_transferControllerErrorInfo = *errorInfo;
    MmwDemo_debugAssert(0);
}

/**
 *  @b Description
 *  @n
 *      Open EDMA and set all the essential settings to it.
 */
static void MmwDemo_edmaOpen(void)
{
    int32_t             errCode;
    EDMA_instanceInfo_t edmaInstanceInfo;
    EDMA_errorConfig_t  errorConfig;
    uint8_t             tc;

    /* open EDMA driver */
    gMmwMssMCB.edmaHandle = EDMA_open(EDMA_INSTANCE_1, &errCode, &edmaInstanceInfo);
    gMmwMssMCB.numEdmaEventQueues = edmaInstanceInfo.numEventQueues;

    if (gMmwMssMCB.edmaHandle == NULL)
    {
        System_printf("Error: Unable to open the EDMA Instance err:%d\n",errCode);
        MmwDemo_debugAssert (0);
        return;
    }

    errorConfig.isConfigAllEventQueues = true;
    errorConfig.isConfigAllTransferControllers = true;
    errorConfig.isEventQueueThresholdingEnabled = true;
    errorConfig.eventQueueThreshold = EDMA_EVENT_QUEUE_THRESHOLD_MAX;
    errorConfig.isEnableAllTransferControllerErrors = true;

    gMmwMssMCB.isPollEdmaError = false;
    if (edmaInstanceInfo.isErrorInterruptConnected == true)
    {
        errorConfig.callbackFxn = MmwDemo_EDMA_errorCallbackFxn;
    }
    else
    {
        errorConfig.callbackFxn = NULL;
        gMmwMssMCB.isPollEdmaError = true;
    }

    /* edma transfer controller error callback funtion */
    errorConfig.transferControllerCallbackFxn = MmwDemo_EDMA_transferControllerErrorCallbackFxn;
    gMmwMssMCB.isPollEdmaTransferControllerError = false;

    for(tc = 0; tc < edmaInstanceInfo.numEventQueues; tc++)
    {
        if (edmaInstanceInfo.isTransferControllerErrorInterruptConnected[tc] == false)
        {
            errorConfig.transferControllerCallbackFxn = NULL;
            gMmwMssMCB.isPollEdmaTransferControllerError = true;
            break;
        }
    }

    /* set eDMA configuration error monitoring */
    if ((errCode = EDMA_configErrorMonitoring(gMmwMssMCB.edmaHandle, &errorConfig)) != EDMA_NO_ERROR)
    {
        MmwDemo_debugAssert (0);
        return;
    }
}

/**
 *  @b Description
 *  @n
 *      System Initialization Task which initializes the various
 *      components in the system.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_initTask(UArg arg0, UArg arg1)
{
    int32_t             errCode;
    UART_Params         uartParams;
    Task_Params         taskParams;
    Semaphore_Params    semParams;
    int32_t             i;

    /* Debug Message: */
    System_printf("Debug: Launched the Initialization Task\n");

    /*****************************************************************************
     * Initialize the mmWave SDK components:
     *****************************************************************************/
    /* Initialize the UART */
    UART_init();

    /* Initialize the GPIO */
    GPIO_init();

    /* Initialize the Mailbox */
    Mailbox_init(MAILBOX_TYPE_MSS);

    MmwDemo_edmaInit();

    MmwDemo_edmaOpen();

    /* Initialize LVDS streaming components */
    if ((errCode = MmwDemo_LVDSStreamInit()) < 0 )
    {
        System_printf ("Error: MMWDemoDSS LVDS stream init failed with Error[%d]\n",errCode);
        return;
    }

    /* initialize cq configs to invalid profile index to be able to detect
     * unconfigured state of these when monitors for them are enabled.
     */
    for(i = 0; i < RL_MAX_PROFILES_CNT; i++)
    {
        gMmwMssMCB.cqSatMonCfg[i].profileIndx    = (RL_MAX_PROFILES_CNT + 1);
        gMmwMssMCB.cqSigImgMonCfg[i].profileIndx = (RL_MAX_PROFILES_CNT + 1);
    }

    /* Platform specific configuration */
    MmwDemo_platformInit(&gMmwMssMCB.cfg.platformCfg);

    /*****************************************************************************
     * Open the mmWave SDK components:
     *****************************************************************************/
    /* Setup the default UART Parameters */
    UART_Params_init(&uartParams);
    uartParams.clockFrequency = gMmwMssMCB.cfg.platformCfg.sysClockFrequency;
    uartParams.baudRate       = gMmwMssMCB.cfg.platformCfg.commandBaudRate;
    uartParams.isPinMuxDone   = 1;

    /* Open the UART Instance */
    gMmwMssMCB.commandUartHandle = UART_open(0, &uartParams);
    if (gMmwMssMCB.commandUartHandle == NULL)
    {
        //System_printf("Error: Unable to open the Command UART Instance\n");
        MmwDemo_debugAssert (0);
        return;
    }

#ifdef TWO_UART_COMMUNICATION
    /* Setup the default UART Parameters */
    UART_Params_init(&uartParams);
    uartParams.writeDataMode = UART_DATA_BINARY;
    uartParams.readDataMode = UART_DATA_BINARY;
    uartParams.clockFrequency = gMmwMssMCB.cfg.platformCfg.sysClockFrequency;
    uartParams.baudRate       = gMmwMssMCB.cfg.platformCfg.loggingBaudRate;
    uartParams.isPinMuxDone   = 1U;

    /* Open the Logging UART Instance: */
    gMmwMssMCB.loggingUartHandle = UART_open(1, &uartParams);
    if (gMmwMssMCB.loggingUartHandle == NULL)
    {
        System_printf("Error: Unable to open the Logging UART Instance\n");
        MmwDemo_debugAssert (0);
        return;
    }
#endif
    /* Create binary semaphores which is used for Monitor report over UART. */
    Semaphore_Params_init(&semParams);
    semParams.mode              = Semaphore_Mode_BINARY;
    gMonReportStreamSem = Semaphore_create(0, &semParams, NULL);
    /*****************************************************************************
     * mmWaveLink instead of mmwave 
     *****************************************************************************/
    /* Setup and initialize the mmWave Link: */
    if (MmwaveLink_initLink (MMW_MMWAVELINK_DEVICE_MAP, RL_PLATFORM_MSS ) < 0)
    {
        return;
    }
    /* pass CLI function pointer to MMWL_IF */
    MmwaveLink_setLogFunc(MmwDemo_CLI_write);
    /*****************************************************************************
     * Launch the mmWave control execution task
     * - This should have a higher priroity than any other task which uses the
     *   mmWave control API
     *****************************************************************************/
    Task_Params_init(&taskParams);
    taskParams.priority  = MMWDEMO_MONITOR_CTRL_TASK_PRIORITY;
    taskParams.stackSize = 3*1024;
    gMmwMssMCB.taskHandles.mmwaveCtrl = Task_create(MonReportStreamTask, &taskParams, NULL);

    /*****************************************************************************
     * Initialize the Profiler
     *****************************************************************************/
    Cycleprofiler_init();

    /*****************************************************************************
     * Initialize the CLI Module:
     *****************************************************************************/
    MmwDemo_CLIInit(MMWDEMO_CLI_TASK_PRIORITY);

    return;
}

/**
 *  @b Description
 *  @n
 *     Function to sleep the R4F using WFI (Wait For Interrupt) instruction.
 *     When R4F has no work left to do,
 *     the BIOS will be in Idle thread and will call this function. The R4F will
 *     wake-up on any interrupt (e.g chirp interrupt).
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_sleep(void)
{
    /* issue WFI (Wait For Interrupt) instruction */
    asm(" WFI ");
}

/**
 *  @b Description
 *  @n
 *      Interrupt handler callback for frame start ISR.
 *
 *  @retval
 *      Not Applicable.
 */
void Mmwavelink_frameInterrupCallBackFunc(uintptr_t arg)
{
    gLinkChirpCnt = 0;

    /* increment Frame count */
    gLinkFrameCnt++;  

    /* Report Monitoring report if requested */
    if(gRFMonitorCfg.reportCfg.enMonReportMode > 0)
    {   
        Semaphore_post(gMonReportStreamSem);
    }
}


/**
 *  @b Description
 *  @n
 *      Entry point into the Millimeter Wave Demo
 *
 *  @retval
 *      Not Applicable.
 */
int main (void)
{
    Task_Params     taskParams;
    int32_t         errCode;
    SOC_Handle      socHandle;
    SOC_Cfg         socCfg;
    SOC_SysIntListenerCfg    linkFrameCfg;

    /* Initialize the ESM: Don't clear errors as TI RTOS does it */
    ESM_init(0U);

    /* Initialize the SOC configuration: */
    memset ((void *)&socCfg, 0, sizeof(SOC_Cfg));

    /* Populate the SOC configuration: */
    socCfg.clockCfg = SOC_SysClock_INIT;
    socCfg.mpuCfg = SOC_MPUCfg_CONFIG;
    socCfg.dssCfg = SOC_DSSCfg_UNHALT; //TODO JIT: check if we need to unhalt DSS, for eDMA (LVDS)??

    /* Initialize the SOC Module: This is done as soon as the application is started
     * to ensure that the MPU is correctly configured. */
    socHandle = SOC_init (&socCfg, &errCode);
    if (socHandle == NULL)
    {
        System_printf ("Error: SOC Module Initialization failed [Error code %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return -1;
    }    
    
    /* Register frame interrupt */
    memset((void *)&linkFrameCfg, 0 , sizeof(SOC_SysIntListenerCfg));
    linkFrameCfg.systemInterrupt    = MMW_MSS_FRAME_START_INT;
    linkFrameCfg.listenerFxn        = Mmwavelink_frameInterrupCallBackFunc;
    linkFrameCfg.arg                = (uintptr_t)NULL;
    if ((SOC_registerSysIntListener(socHandle, &linkFrameCfg, &errCode)) == NULL)
    {
        System_printf("Error: Unable to register frame interrupt listener , error = %d\n", errCode);
        return -1;
    }
    
    /* Check if the SOC is a secure device */
    if (SOC_isSecureDevice(socHandle, &errCode))
    {
        /* Disable firewall for JTAG and LOGGER (UART) which is needed by all unit tests */
        SOC_controlSecureFirewall(socHandle, 
                                  (uint32_t)(SOC_SECURE_FIREWALL_JTAG | SOC_SECURE_FIREWALL_LOGGER),
                                  SOC_SECURE_FIREWALL_DISABLE,
                                  &errCode);
    }

    /* Initialize and populate the demo MCB */
    memset ((void*)&gMmwMssMCB, 0, sizeof(MmwDemo_MSS_MCB));

    gMmwMssMCB.socHandle = socHandle;
    
    /* Debug Message: */
    System_printf ("**********************************************\n");
    System_printf ("Debug: Launching the CLI Demo on MSS\n");
    System_printf ("**********************************************\n");

    /* Initialize the Task Parameters. */
    Task_Params_init(&taskParams);
    taskParams.stackSize = 10*1024;
    gMmwMssMCB.taskHandles.initTask = Task_create(MmwDemo_initTask, &taskParams, NULL);

    /* Start BIOS */
    BIOS_start();
    return 0;
}


