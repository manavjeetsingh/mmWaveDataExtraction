/*
 *   @file  mmwl_if.c
 *
 *   @brief
 *      The file contains common functions which test the mmWave Link API
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

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

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
#include <ti/drivers/edma/edma.h>

#ifdef SUBSYS_MSS
#include <ti/sysbios/family/arm/v7r/vim/Hwi.h>
#else
#include <ti/sysbios/family/c64p/Hwi.h>
#endif

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/drivers/mailbox/mailbox.h>
#include <ti/control/mmwavelink/mmwavelink.h>
#include <ti/drivers/crc/crc.h>
#include <ti/control/mmwavelink/include/rl_driver.h>
#include <ti/drivers/adcbuf/ADCBuf.h>
#include "common/mmwl_if.h"
#include "mss/mmw_mss.h"


/**************************************************************************
 *************************** Local Structures *****************************
 **************************************************************************/

/**
 * @brief
 *  Mmwave Link Master Control Block
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Mmwave Link.
 */
typedef struct MmwaveLink_MCB
{
    /**
     * @brief   Handle to the BSS Mailbox
     */
    Mbox_Handle              bssMailbox;

    /**
     * @brief   Semaphore handle for the mmWave Link
     */
    Semaphore_Handle            linkSemaphore;

    /**
     * @brief   mmWave Link Spawning function
     */
    RL_P_OSI_SPAWN_ENTRY        spawnFxn;

    /**
     * @brief   Status of the BSS:
     */
    volatile uint32_t           bssStatus;

    /**
     * @brief   Counter which tracks of the number of times the spawn function was
     * overrun.
     */
    uint32_t                    spawnOverrun;
    /**
     * @brief   Handle to the CRC Channel
     */
    CRC_Handle                  crcHandle;
}MmwaveLink_MCB;

/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

/* Global Variable for tracking information required by the mmWave Link */
MmwaveLink_MCB    gMmwaveLinkMCB;
uint32_t gMonitorHdrCnt = 0U;
uint32_t gInitTimeCalibStatus = 0U;
uint32_t gRunTimeCalibStatus = 0U;
volatile uint32_t gFrameStartStatus = 0U;
volatile uint32_t gMmwlChecksumFailAsyncEvent = 0U;
volatile uint32_t gMmwlCrcFailAsyncEvent = 0U;
rlUInt16_t gMonitoringStatus = 0U;
rlUInt8_t isGetGpAdcMeasData = 0U;

/**
 * @brief
 *  Millimeter Wave Demo statistics
 *
 * @details
 *  The structure is used to hold the statistics information for the
 *  Millimeter Wave demo
 */
typedef struct MMWL_Stats_t
{
    /*! @brief   Counter which tracks the number of frame trigger events from BSS */
    uint64_t     frameTriggerReady;
    
    /*! @brief   Counter which tracks the number of failed calibration reports
     *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     failedTimingReports;

    /*! @brief   Counter which tracks the number of calibration reports received
     *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     calibrationReports;

     /*! @brief   Counter which tracks the number of sensor stop events received
      *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     sensorStopped;
}MMWL_Stats;

MMWL_Stats   mmwl_stats;


RL_P_EVENT_HANDLER g_MailboxInterruptFunc;

/* eDMA handler */
EDMA_Handle EdmaHandle;
/* Tag set based on eDMA data trasnfer done */
volatile uint8_t eDMADataTransferDone = 0U;

/* received GPAdc Data over Async Event */
rlRecvdGpAdcData_t rcvGpAdcData = {0};

/*! CQ MEMORY */
#pragma DATA_SECTION(gMmwCQ, ".cqdata");
uint8_t gMmwCQ [8*1024];

/* Negative test configuration structure */
typedef struct retryNegativeTest
{
    /* Type of corruption
       0 - No corruption
       1 - Checksum corruption
       2 - CRC corruption  */
    uint16_t corruptType;
    /* Event to corrupt
       0 - Command
       1 - Response
       2 - Async Event  */
    uint16_t corruptEvent;
}retryNegativeTest_t;


/* Async Event Enable and Direction configuration */
rlRfDevCfg_t rfDevCfg = {0};

/* Calibration Data storage */
rlCalibrationData_t calibData = { 0 };

/* Phase Shifter Calibration Data storage */
rlPhShiftCalibrationData_t phShiftCalibData = { 0 };
rlRxGainTempLutData_t rxGainTempLutData = { 0 };
rlTxGainTempLutData_t txGainTempLutData = { 0 };
retryNegativeTest_t   retryNegativeTestCfg = { 0 };
uint16_t hit_count = 0;
/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/
extern rlInt32_t Osal_mutexCreate(rlOsiMutexHdl_t* mutexHdl, rlInt8_t* name);
extern rlInt32_t Osal_mutexLock(rlOsiMutexHdl_t* mutexHdl, rlOsiTime_t timeout);
extern rlInt32_t Osal_mutexUnlock(rlOsiMutexHdl_t* mutexHdl);
extern rlInt32_t Osal_mutexDelete(rlOsiMutexHdl_t* mutexHdl);
extern rlInt32_t Osal_semCreate(rlOsiSemHdl_t* semHdl, rlInt8_t* name);
extern rlInt32_t Osal_semWait(rlOsiSemHdl_t* semHdl, rlOsiTime_t timeout);
extern rlInt32_t Osal_semSignal(rlOsiSemHdl_t* semHdl);
extern rlInt32_t Osal_semDelete(rlOsiSemHdl_t* semHdl);
extern void RFMon_reportHandler(uint16_t msgId, uint16_t asyncSB, uint8_t *payload);

/**************************************************************************
 ************************* Link Unit Test Functions ***********************
 **************************************************************************/


RL_PRINT_FUNC g_PrintFunc = NULL;

void MmwaveLink_setLogFunc(RL_PRINT_FUNC func)
{
    g_PrintFunc = func;
}

/**
 *  @b Description
 *  @n
 *      Mailbox registered function which is invoked on the reception of data
 *
 *  @retval
 *      Success - Communicate Interface Channel Handle
 *  @retval
 *      Error   - NULL
 */
static void MmwaveLink_mboxCallbackFxn (Mbox_Handle handle, Mailbox_Type remoteEndpoint)
{
    /* Indicate to the Radar Link that a message has been received. */
    g_MailboxInterruptFunc(0, NULL);
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to open the communication
 *      interface channel
 *
 *  @retval
 *      Success - Communicate Interface Channel Handle
 *  @retval
 *      Error   - NULL
 */
static rlComIfHdl_t MmwaveLink_mboxOpen(rlUInt8_t deviceIndex, uint32_t flags)
{
    Mailbox_Config  cfg;
    int32_t         errCode;

    /* Initialize the mailbox configuration: */
    if(Mailbox_Config_init(&cfg) < 0)
    {
        System_printf("Error: Unable to initialize mailbox configuration\n");
        return NULL;
    }

    cfg.writeMode    = MAILBOX_MODE_POLLING;
    cfg.readMode     = MAILBOX_MODE_CALLBACK;
    cfg.readCallback = MmwaveLink_mboxCallbackFxn;

    /* Open the Mailbox to the BSS */
    gMmwaveLinkMCB.bssMailbox = Mailbox_open(MAILBOX_TYPE_BSS, &cfg, &errCode);
    if (gMmwaveLinkMCB.bssMailbox == NULL)
    {
        System_printf("Error: Unable to open the Mailbox Instance [Error code %d]\n", errCode);
        return NULL;
    }
    System_printf("Debug: BSS Mailbox Handle %p\n", gMmwaveLinkMCB.bssMailbox);
    return gMmwaveLinkMCB.bssMailbox;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to close the communication
 *      interface channel
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static int32_t MmwaveLink_mboxClose(rlComIfHdl_t fd)
{
    int32_t errCode;

    /* Close the Mailbox */
    errCode = Mailbox_close ((Mbox_Handle)fd);
    if (errCode < 0)
        System_printf ("Error: Unable to close the BSS Mailbox [Error code %d]\n", errCode);

    return errCode;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to read data from the communication
 *      interface channel
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static int32_t MmwaveLink_mboxRead(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len)
{
    int32_t    status;
    
    status = Mailbox_read((Mbox_Handle)fd, pBuff, len);
    
    return status;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to write data to the communication
 *      interface channel
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static int32_t MmwaveLink_mboxWrite(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len)
{
    int32_t    status;

    /*
      Currently, the mmwavelink can not detect the error condition where it did not receive a mailbox layer ACK from BSS.

      For instance:
      - The mmwavelink may try to send a message before an ACK was received for the previous message.
      - The mmwavelink may try to resend a message that did not receive a mmwavelink layer ACK back from BSS. It is possible that the
      message did not receive a mailbox layer ACK as well from BSS.

      In either case, Mailbox_writeReset() has to be called before another message is sent to BSS.

      The mmwavelink has no hooks to call the Mailbox_writeReset().
      Therefore, a write reset is done if it is detected that a mailbox layer ACK was not received for the
      previous message (MAILBOX_ETXFULL).
     */
 
    status = Mailbox_write((Mbox_Handle)fd, pBuff, len);
    if(status == MAILBOX_ETXFULL)
    {
        Mailbox_writeReset((Mbox_Handle)fd);
        status = Mailbox_write((Mbox_Handle)fd, pBuff, len);
    }

    return status;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to power on the AR1XX Device
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MmwaveLink_enableDevice(rlUInt8_t deviceIndex)
{
    return 0;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to power off the AR1XX Device
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MmwaveLink_disableDevice(rlUInt8_t deviceIndex)
{
    System_printf("Debug: Disabling the device\n");
    return 0;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to mask the interrupts.
 *      In the case of Mailbox communication interface the driver will
 *      handle the interrupt management. This function is a dummy stub
 *
 *  @retval
 *      Not applicable
 */
static void MmwaveLink_maskHostIRQ(rlComIfHdl_t fd)
{
    return;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to umask the interrupts.
 *      In the case of the mailbox driver we will flush out and close the
 *      read buffer.
 *
 *  @retval
 *      Not applicable
 */
void MmwaveLink_unmaskHostIRQ(rlComIfHdl_t fd)
{
    Mailbox_readFlush((Mbox_Handle)fd);
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to register the Interrupt Handler.
 *      In the case of the Mailbox the driver does the interrupt registeration and
 *      so this function is a dummy stub.
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MmwaveLink_registerInterruptHandler(rlUInt8_t deviceIndex, RL_P_EVENT_HANDLER pHandler, void* pValue)
{
    g_MailboxInterruptFunc = pHandler;
    return 0;
}


/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to handle asynchronous events
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
 uint16_t errorStatusCnt[5] = {0};

static void MmwaveLink_asyncEventHandler(uint8_t devIndex, uint16_t sbId, uint16_t sbLen, uint8_t *payload)
{
    uint16_t asyncSB = RL_GET_SBID_FROM_UNIQ_SBID(sbId);
    uint16_t msgId   = RL_GET_MSGID_FROM_SBID(sbId);

    /* Process the received message: */
    switch (msgId)
    {
        case RL_RF_ASYNC_EVENT_MSG:
        {
            /* Received Asychronous Message: */
            switch (asyncSB)
            {
                case RL_RF_AE_CPUFAULT_SB:
                {
                    rlCpuFault_t* rfCpuFault = (rlCpuFault_t*)payload;
                    System_printf ("Debug: CPU Fault has been detected\n");
                    g_PrintFunc ("ERROR: Fault \n type: %d, lineNum: %d, LR: 0x%x \n"
                                    "PrevLR: 0x%x, spsr: 0x%x, sp: 0x%x, PC: 0x%x \n"
                                    "Status: 0x%x, Source: %d, AxiErrType: %d, AccType: %d, Recovery Type: %d \n",
                                    rfCpuFault->faultType,
                                    rfCpuFault->lineNum,
                                    rfCpuFault->faultLR,
                                    rfCpuFault->faultPrevLR,
                                    rfCpuFault->faultSpsr,
                                    rfCpuFault->faultSp,
                                    rfCpuFault->faultAddr,
                                    rfCpuFault->faultErrStatus,
                                    rfCpuFault->faultErrSrc,
                                    rfCpuFault->faultAxiErrType,
                                    rfCpuFault->faultAccType,
                                    rfCpuFault->faultRecovType);
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_RF_AE_ESMFAULT_SB:
                {
                    g_PrintFunc ("ERROR: ESM Fault. Group1:[0x%x] Group2:[0x%x]\n",
                    ((rlBssEsmFault_t*)payload)->esmGrp1Err, ((rlBssEsmFault_t*)payload)->esmGrp2Err);
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_RF_AE_ANALOG_FAULT_SB:
                {
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_RF_AE_INITCALIBSTATUS_SB:
                {                    
                    gInitTimeCalibStatus = 1;
                    break;
                }
                case RL_RF_AE_FRAME_TRIGGER_RDY_SB:
                {
                    mmwl_stats.frameTriggerReady++;
                    break;
                }
                case RL_RF_AE_MON_TIMING_FAIL_REPORT_SB:
                {
                    System_printf ("Debug: Monitoring FAIL Report received \n");
                    mmwl_stats.failedTimingReports++;
                    break;
                }
                case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
                {
                    mmwl_stats.calibrationReports++;
                    RFMon_reportHandler(msgId, asyncSB, payload);
                    break;
                }
                case RL_RF_AE_MON_DIG_PERIODIC_REPORT_SB:
                case RL_RF_AE_MON_TEMPERATURE_REPORT_SB:
                case RL_RF_AE_MON_RX_GAIN_PHASE_REPORT:
                case RL_RF_AE_MON_RX_NOISE_FIG_REPORT:
                case RL_RF_AE_MON_RX_IF_STAGE_REPORT:
                case RL_RF_AE_MON_TX0_POWER_REPORT:
                case RL_RF_AE_MON_TX1_POWER_REPORT:
                case RL_RF_AE_MON_TX2_POWER_REPORT:
                case RL_RF_AE_MON_TX0_BALLBREAK_REPORT:
                case RL_RF_AE_MON_TX1_BALLBREAK_REPORT:
                case RL_RF_AE_MON_REPORT_HEADER_SB:
                case RL_RF_AE_DIG_LATENTFAULT_REPORT_SB:
                {
                    RFMon_reportHandler(msgId, asyncSB, payload);
                    break;
                }
                case RL_RF_AE_GPADC_MEAS_DATA_SB:
                {
                    isGetGpAdcMeasData = 1U;
                    memcpy(&rcvGpAdcData, payload, sizeof(rlRecvdGpAdcData_t));
                    break;
                }

                case RL_RF_AE_FRAME_END_SB:
                {
                    mmwl_stats.sensorStopped++;
                    MmwDemo_gpioStateCtrl(0);
                    /* Deactivate LVDS Session */
                    deactivateLVDSSession();
                    System_printf ("Debug:  Frame Stop Async Event \n");
                    break;
                }
                default:
                {
                    System_printf ("Error: Asynchronous Event SB Id %d not handled with msg ID [0x%x] \n", asyncSB,msgId);
                    break;
                }
            }
            break;
        }
        case RL_RF_ASYNC_EVENT_1_MSG:
        {
            switch (asyncSB)
            {
                /* Monitoring Report */
                case RL_RF_AE_MON_TX2_BALLBREAK_REPORT:
                case RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT:
                case RL_RF_AE_MON_TX0_BPM_REPORT:
                case RL_RF_AE_MON_TX1_BPM_REPORT:
                case RL_RF_AE_MON_TX2_BPM_REPORT:
                case RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT:
                case RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT:
                case RL_RF_AE_MON_TX0_INT_ANA_SIG_REPORT:
                case RL_RF_AE_MON_TX1_INT_ANA_SIG_REPORT:
                case RL_RF_AE_MON_TX2_INT_ANA_SIG_REPORT:
                case RL_RF_AE_MON_RX_INT_ANALOG_SIG_REPORT:
                case RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT:
                case RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT:
                case RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT:
                case RL_RF_AE_MON_DCC_CLK_FREQ_REPORT:
                case RL_RF_AE_MON_RX_MIXER_IN_PWR_REPORT:
                {
                    RFMon_reportHandler(msgId, asyncSB, payload);
                    break;
                }

                default:
                {
                    System_printf ("Error: Asynchronous Event SB Id %d not handled with msg ID 0x%x\n", asyncSB,msgId);
                    break;
                }
            }
            break;
        }
        /* Async Event from MMWL */
        case RL_MMWL_ASYNC_EVENT_MSG:
        {
            switch (asyncSB)
            {
                case RL_MMWL_AE_MISMATCH_REPORT:
                {
                    /* link reports protocol error in the async report from BSS */
                    MmwDemo_debugAssert(0);
                    break;
                }            
                case RL_MMWL_AE_INTERNALERR_REPORT:
                {
                    /* link reports internal error during BSS communication */
                    MmwDemo_debugAssert(0);
                    break;
                }
            }
            break;
        }
        default:
        {
            System_printf ("Error: Asynchronous message %d is NOT handled\n", msgId);
            break;
        }
    }
    return;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to call the function in a different context
 *      This function is invoked from the Interrupt context.
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MmwaveLink_spawn (RL_P_OSI_SPAWN_ENTRY pEntry, const void* pValue, uint32_t flags)
{
    /* Record the function which is to be spawned. */
    if (gMmwaveLinkMCB.spawnFxn != NULL)
        gMmwaveLinkMCB.spawnOverrun++;

    /* Record the entry to be spawned. */
    gMmwaveLinkMCB.spawnFxn = pEntry;

    /* Post the semaphore and wake up the link management task */
    Semaphore_post (gMmwaveLinkMCB.linkSemaphore);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      Radar Link Registered Callback function to compute the CRC.
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t MmwaveLink_computeCRC(rlUInt8_t* data, rlUInt32_t dataLen, rlUInt8_t crcType, rlUInt8_t* crc)
{
    CRC_SigGenCfg   signGenCfg;
    int32_t         errCode;
    uint64_t        signature;
    uint32_t        index;
    uint8_t*        ptrSignature;
    uint8_t         crcLength;

    /* Initialize the signature generation configuration */
    memset ((void *)&signGenCfg, 0, sizeof(CRC_SigGenCfg));

    /* Allocate a unique transaction id: */
    if (CRC_getTransactionId (gMmwaveLinkMCB.crcHandle, &signGenCfg.transactionId, &errCode) < 0)
    {
        System_printf ("Error: CRC Driver Get transaction id failed [Error code %d]\n", errCode);
        return -1;
    }

    /* Populate the signature generation configuration: */
    signGenCfg.ptrData = (uint8_t*)data;
    signGenCfg.dataLen = dataLen;

    /* Compute the signature for the specific data on Channel-1 */
    if (CRC_computeSignature (gMmwaveLinkMCB.crcHandle, &signGenCfg, &errCode) < 0)
    {
        System_printf ("Error: CRC Driver compute signature failed [Error code %d]\n", errCode);
        return -1;
    }

    /* Get the Signature for Channel */
    if (CRC_getSignature (gMmwaveLinkMCB.crcHandle, signGenCfg.transactionId, &signature, &errCode) < 0)
    {
        System_printf ("Error: CRC Driver get signature failed [Error code %d]\n", errCode);
        return -1;
    }

    /* Get the pointer to the CRC Signature: */
    ptrSignature = (uint8_t*)&signature;

    /* Determine the length of the CRC: */
    switch (crcType)
    {
        case RL_CRC_TYPE_16BIT_CCITT:
        {
            crcLength = 2;
            break;
        }
        case RL_CRC_TYPE_32BIT:
        {
            crcLength = 4;
            break;
        }
        case RL_CRC_TYPE_64BIT_ISO:
        {
            crcLength = 8;
            break;
        }
        default:
        {
            System_printf ("Error: Unknown CRC Type passed from mmWave Link: %d\n", crcType);
            return -1;
        }
    }

    /* Copy the CRC signature into CRC output array */
    for(index = 0U; index < crcLength; index++)
    {
        *(crc + index) = *(ptrSignature + index);
    }

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the task which handles the mmWave Link communication
 *      messages between the BSS and MSS.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwaveLink_mmwaveLinkMgmtTask (UArg arg0, UArg arg1)
{
    RL_P_OSI_SPAWN_ENTRY    spawnFxn;
    uintptr_t               key;
    Semaphore_Params        semParams;

    /* Debug Message: */
    System_printf("Debug: Launched the mmwaveLink Management Task\n");

    /* Initialize the mmWave Link Semaphore: */
    Semaphore_Params_init(&semParams);
    semParams.mode  = Semaphore_Mode_BINARY;
    gMmwaveLinkMCB.linkSemaphore = Semaphore_create(0, &semParams, NULL);

    /* Execute forever: */
    while (1)
    {
        /* Pending on the link semaphore */
        Semaphore_pend (gMmwaveLinkMCB.linkSemaphore, BIOS_WAIT_FOREVER);

        /* Critical Section: We record the spawn function which is to be executed */
        key = Hwi_disable();
        spawnFxn = gMmwaveLinkMCB.spawnFxn;
        gMmwaveLinkMCB.spawnFxn = NULL;
        Hwi_restore (key);

        /* Execute the spawn function: */
        spawnFxn (NULL);
    }
}


/**
 *  @b Description
 *  @n
 *      The function is used to get and display the version information
 *      using the mmWave link API.
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MmwaveLink_getVersion (void)
{
    rlVersion_t verArgs;
    int32_t     retVal;
    /* currently patch binaries are available for AWR16 ES2.0 & IWR16 ES2.0 only*/
    int8_t rfPatchBuildVer, rfPatchDebugVer;

    /* Get the version string: */
    retVal = rlDeviceGetVersion(RL_DEVICE_MAP_INTERNAL_BSS, &verArgs);
    if (retVal != 0)
    {
        System_printf ("Error: Unable to get the device version from mmWave link [Error %d]\n", retVal);
        return -1;
    }

    /* Display the version information */
    System_printf ("RF H/W Version    : %02d.%02d\n",
                    verArgs.rf.hwMajor, verArgs.rf.hwMinor);
    System_printf ("RF F/W Version    : %02d.%02d.%02d.%02d.%02d.%02d.%02d\n",
                    verArgs.rf.fwMajor, verArgs.rf.fwMinor, verArgs.rf.fwBuild, verArgs.rf.fwDebug,
                    verArgs.rf.fwYear, verArgs.rf.fwMonth, verArgs.rf.fwDay);
    rfPatchDebugVer = ((verArgs.rf.patchBuildDebug) & 0x0F);
    rfPatchBuildVer = (((verArgs.rf.patchBuildDebug) & 0xF0) >> 4);

    System_printf ("RF F/W Patch Version : %02d.%02d.%02d.%02d.%02d.%02d.%02d\n",
                    verArgs.rf.patchMajor, verArgs.rf.patchMinor, rfPatchBuildVer, rfPatchDebugVer,
                    verArgs.rf.patchYear, verArgs.rf.patchMonth, verArgs.rf.patchDay);
    System_printf ("mmWaveLink Version: %02d.%02d.%02d.%02d\n",
                    verArgs.mmWaveLink.major, verArgs.mmWaveLink.minor,
                    verArgs.mmWaveLink.build, verArgs.mmWaveLink.debug);
    return 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize and setup the mmWave link
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
int32_t MmwaveLink_initLink (rlUInt8_t deviceType, rlUInt8_t platform)
{
    Task_Params            taskParams;
    CRC_Config          crcCfg;
    rlClientCbs_t       RlApp_ClientCtx;
    int32_t             errCode;

    /* Initialize and populate the Mmwave Link MCB */
    memset ((void*)&gMmwaveLinkMCB, 0, sizeof(MmwaveLink_MCB));

    /*****************************************************************************
     * Start CRC driver:
     *****************************************************************************/

    /* Setup the default configuration: */
    CRC_initConfigParams(&crcCfg);

#if (MMWAVELINK_CRC_TYPE == CRC_TYPE_16BIT)
    /*******************************************************************************
     * This is the configuration for the 16bit CRC Type:
     *******************************************************************************/
    crcCfg.channel  = CRC_Channel_CH1;
    crcCfg.mode     = CRC_Operational_Mode_FULL_CPU;
    crcCfg.type     = CRC_Type_16BIT;
    crcCfg.bitSwap  = CRC_BitSwap_MSB;
    crcCfg.byteSwap = CRC_ByteSwap_ENABLED;
    crcCfg.dataLen  = CRC_DataLen_16_BIT;
#elif (MMWAVELINK_CRC_TYPE == CRC_TYPE_32BIT)
    /*******************************************************************************
     * This is the configuration for the 32bit CRC Type:
     *******************************************************************************/
    crcCfg.channel  = CRC_Channel_CH1;
    crcCfg.mode     = CRC_Operational_Mode_FULL_CPU;
    crcCfg.type     = CRC_Type_32BIT;
    crcCfg.bitSwap  = CRC_BitSwap_LSB;
    crcCfg.byteSwap = CRC_ByteSwap_DISABLED;
    crcCfg.dataLen  = CRC_DataLen_32_BIT;
#elif (MMWAVELINK_CRC_TYPE == CRC_TYPE_64BIT)
    /*******************************************************************************
     * This is the configuration for the 64bit CRC Type:
     *******************************************************************************/
    crcCfg.channel  = CRC_Channel_CH1;
    crcCfg.mode     = CRC_Operational_Mode_FULL_CPU;
    crcCfg.type     = CRC_Type_64BIT;
    crcCfg.bitSwap  = CRC_BitSwap_MSB;
    crcCfg.byteSwap = CRC_ByteSwap_ENABLED;
    crcCfg.dataLen  = CRC_DataLen_32_BIT;
#endif

    /* Open the CRC Driver */
    gMmwaveLinkMCB.crcHandle = CRC_open (&crcCfg, &errCode);
    if (gMmwaveLinkMCB.crcHandle == NULL)
    {
        System_printf ("Error: Unable to open the CRC Channel [Error Code %d]\n", errCode);
        return -1;
    }
    System_printf("Debug: CRC Channel %p has been opened successfully\n", gMmwaveLinkMCB.crcHandle);

    /*****************************************************************************
     * Launch the Mmwave Link Tasks:
     *****************************************************************************/

    /* Initialize and Launch the mmWave Link Management Task: */
    Task_Params_init(&taskParams);
    taskParams.priority = 5;
    Task_create(MmwaveLink_mmwaveLinkMgmtTask, &taskParams, NULL);

    /*****************************************************************************
     * Initialize the mmWave Link: We need to have the link management task
     * operational to be able to process the SPAWN function.
     *****************************************************************************/

    /* Reset the client context: */
    memset ((void *)&RlApp_ClientCtx, 0, sizeof(rlClientCbs_t));

    RlApp_ClientCtx.ackTimeout  = 1000U;

    /* Setup the crc Type in the mmWave link and synchronize this with the
     * created CRC Channel. */
    if (crcCfg.type == CRC_Type_16BIT)
        RlApp_ClientCtx.crcType = RL_CRC_TYPE_16BIT_CCITT;
    else if (crcCfg.type == CRC_Type_32BIT)
        RlApp_ClientCtx.crcType = RL_CRC_TYPE_32BIT;
    else
        RlApp_ClientCtx.crcType = RL_CRC_TYPE_64BIT_ISO;

    /* Setup the platform on which the mmWave Link executes */
    RlApp_ClientCtx.platform  = platform;
    RlApp_ClientCtx.arDevType = deviceType;

    /* Initialize the Communication Interface API: */
    RlApp_ClientCtx.comIfCb.rlComIfOpen     = MmwaveLink_mboxOpen;
    RlApp_ClientCtx.comIfCb.rlComIfClose    = MmwaveLink_mboxClose;
    RlApp_ClientCtx.comIfCb.rlComIfRead     = MmwaveLink_mboxRead;
    RlApp_ClientCtx.comIfCb.rlComIfWrite    = MmwaveLink_mboxWrite;

    /* Initialize OSI Mutex Interface */
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexCreate = Osal_mutexCreate;
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexLock   = Osal_mutexLock;
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexUnLock = Osal_mutexUnlock;
    RlApp_ClientCtx.osiCb.mutex.rlOsiMutexDelete = Osal_mutexDelete;

    /* Initialize OSI Semaphore Interface */
    RlApp_ClientCtx.osiCb.sem.rlOsiSemCreate    = Osal_semCreate;
    RlApp_ClientCtx.osiCb.sem.rlOsiSemWait      = Osal_semWait;
    RlApp_ClientCtx.osiCb.sem.rlOsiSemSignal    = Osal_semSignal;
    RlApp_ClientCtx.osiCb.sem.rlOsiSemDelete    = Osal_semDelete;

    /* Initialize OSI Queue Interface */
    RlApp_ClientCtx.osiCb.queue.rlOsiSpawn      = MmwaveLink_spawn;

    /* Initialize OSI Timer Interface */
    RlApp_ClientCtx.timerCb.rlDelay             = NULL;

    /* Initialize the CRC Interface */
    RlApp_ClientCtx.crcCb.rlComputeCRC          = MmwaveLink_computeCRC;

    /* Initialize Device Control Interface */
    RlApp_ClientCtx.devCtrlCb.rlDeviceDisable            = MmwaveLink_disableDevice;
    RlApp_ClientCtx.devCtrlCb.rlDeviceEnable             = MmwaveLink_enableDevice;
    RlApp_ClientCtx.devCtrlCb.rlDeviceMaskHostIrq        = MmwaveLink_maskHostIRQ;
    RlApp_ClientCtx.devCtrlCb.rlDeviceUnMaskHostIrq      = MmwaveLink_unmaskHostIRQ;
    RlApp_ClientCtx.devCtrlCb.rlRegisterInterruptHandler = MmwaveLink_registerInterruptHandler;

    /* Initialize the Asynchronous Event Handler: */
    RlApp_ClientCtx.eventCb.rlAsyncEvent    = MmwaveLink_asyncEventHandler;

    /* Power on the Device */
    if (rlDevicePowerOn(1U, RlApp_ClientCtx) != 0)
    {
        System_printf("Error: Power on request to the BSS failed\n");
        return -1;
    }

    System_printf("Debug: Power on request successfully passed to the BSS\n");

    return 0;
}

/**
 *  @b Description
 *  @n
 *      The function is to get the BSS Boot up status
 *
 *   @param[out] ptrStatusCfg  RF Boot Status Config
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
int32_t MmwaveLink_getRfBootupStatus (rlRfBootStatusCfg_t *ptrStatusCfg)
{
    int32_t         retVal;

    /* Get RF Boot status */
    retVal = rlGetRfBootupStatus(RL_DEVICE_MAP_INTERNAL_BSS, ptrStatusCfg);

    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: rlGetRfBootupStatus retVal=%d\n", retVal);
        return retVal;
    }

    System_printf("Debug: Finished get radarSS bootup status to BSS\n");

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to Enable/Disable Continous mode
 *      using the mmWave link API.
 *
 *   @param[in] 1: Enable, 0: Disable
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
/* enable the continuous streaming mode */
int32_t MmwaveLink_ContMode(rlUInt16_t enDisable)
{
    rlContModeEn_t          contModeEnable;
    int32_t retVal;

    /* Start the sensor in continuous mode: */
    memset ((void*)&contModeEnable, 0, sizeof(rlContModeEn_t));

    /* Populate the continuous mode configuration: */
    contModeEnable.contModeEn = enDisable;
    retVal = rlEnableContMode (RL_DEVICE_MAP_INTERNAL_BSS, &contModeEnable);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is to Trigger the frame and wait for
 *      async-event message.
 *
 *   @param[in] waitAe : Wait for async-event message
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
int32_t MmwaveLink_frameTrigger(uint8_t waitAe)
{
    int32_t retVal, timeOutCnt = 0xFFFFF;

    /* reset the flag first */
    mmwl_stats.frameTriggerReady = 0;

    retVal = rlSensorStart(RL_DEVICE_MAP_INTERNAL_BSS);

    if(waitAe)
    {
        do{
            Task_sleep(50);
            if(timeOutCnt-- == 0)
            {
                retVal = -1;//MMW_CLI_ERROR_CMD_EXECUTE_TIME_OUT;
                break;
            }
        }while(mmwl_stats.frameTriggerReady == 0);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is to Stop the frame and wait for
 *      async-event message.
 *
 *   @param[in] waitAe : Wait for async-event message
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
int32_t MmwaveLink_frameStop(uint8_t waitAe)
{
    int32_t retVal, timeOutCnt = 0xFFFFF;

    retVal = rlSensorStop(RL_DEVICE_MAP_INTERNAL_BSS);

    if(waitAe)
    {
        do{
            Task_sleep(50);
            if(timeOutCnt-- == 0)
            {
                retVal = -1;//MMW_CLI_ERROR_CMD_EXECUTE_TIME_OUT;
                break;
            }
        }while(mmwl_stats.frameTriggerReady == 0);
    }
    return retVal;
}

