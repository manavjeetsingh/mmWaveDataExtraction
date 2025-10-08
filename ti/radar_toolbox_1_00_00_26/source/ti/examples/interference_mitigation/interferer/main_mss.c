/*
 *   @file  main_mss.c
 *
 *   @brief
 *      This is the CW interferer which executes on the R4 on
 *      the AR16xx and AR18xx
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
#include <ti/sysbios/family/arm/v7r/vim/Hwi.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/drivers/pinmux/pinmux.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/esm/esm.h>
#include <ti/drivers/mailbox/mailbox.h>
#include <ti/drivers/adcbuf/ADCBuf.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/crc/crc.h>
#include <ti/control/mmwavelink/mmwavelink.h>
#include <ti/control/mmwavelink/include/rl_driver.h>

//As per mmwavelink, 1mHz is 2^26/(3.6*10^3) or 18641.351
#define START_FREQ_CONST_VAL_PER_MHZ 18641 

/**
 * @brief
 *  Mmwave Link Master Control Block
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Mmwave Link.
 */
typedef struct interferer_MCB
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
}interferer_MCB;

/**************************************************************************
 *************************** Global Variables *****************************
 **************************************************************************/

/*! @brief   ADCBUF handle */
ADCBuf_Handle             adcbufHandle;

interferer_MCB    gInterfererMCB;

RL_P_EVENT_HANDLER g_MailboxInterruptFunc;

uint32_t gInitTimeCalibStatus = 0U;

/**************************************************************************
 *************************** Configurations *******************************
 **************************************************************************/
const rlChanCfg_t chCfg  =
{
    .rxChannelEn = (1U<<3U)|(1U<<2U)|(1U<<1U)|(1U<<0U),
#ifdef SOC_XWR16XX
    .txChannelEn = (1U<<1U)|(1U<<0U),
#else
#ifdef SOC_XWR18XX
    .txChannelEn = (1U<<2U)|(1U<<1U)|(1U<<0U),
#endif
#endif
    .cascading   = 0x0,
    .cascadingPinoutCfg   = 0x0,
};

const rlAdcOutCfg_t adcOutCfgArgs =
{
    .fmt.b2AdcBits = 2U, //16 bits,
    .fmt.b2AdcOutFmt = 2U, //complex with image band,
    .fmt.b8FullScaleReducFctr = 0U,
    .reserved0  = 0x0,
    .reserved1  = 0x0,
};

const rlLowPowerModeCfg_t lowPowerModeCfg =
{
    .reserved  = 0x0,
    .lpAdcMode = 1U,
};

const rlRfLdoBypassCfg_t rfLdoBypassCfg =
{
    .ldoBypassEnable   = 1,
    .supplyMonIrDrop   = 0,
    .ioSupplyIndicator = 0,
};

const rlContModeCfg_t contModeCfg =
{
    .startFreqConst = 78000*START_FREQ_CONST_VAL_PER_MHZ, //~78.000 gHz
    .txOutPowerBackoffCode = 0,
    .txPhaseShifter = 0,
    .digOutSampleRate = 10000,
    .hpfCornerFreq1 = 0,
    .hpfCornerFreq2 = 0,
    .rxGain = 30,
    .vcoSelect = 0x0,
    .reserved  = 0x0
};

const rlContModeEn_t contModeEn =
{
    .contModeEn = 1,
    .reserved   = 0
};
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
extern rlInt32_t Osal_delay(rlUInt32_t delay);

/**************************************************************************
 ****************************** Functions *********************************
 **************************************************************************/

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
static void interferer_mboxCallbackFxn (Mbox_Handle handle, Mailbox_Type remoteEndpoint)
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
static rlComIfHdl_t interferer_mboxOpen(rlUInt8_t deviceIndex, uint32_t flags)
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
    cfg.readCallback = interferer_mboxCallbackFxn;

    /* Open the Mailbox to the BSS */
    gInterfererMCB.bssMailbox = Mailbox_open(MAILBOX_TYPE_BSS, &cfg, &errCode);
    if (gInterfererMCB.bssMailbox == NULL)
    {
        System_printf("Error: Unable to open the Mailbox Instance [Error code %d]\n", errCode);
        return NULL;
    }
    return gInterfererMCB.bssMailbox;
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
static int32_t interferer_mboxClose(rlComIfHdl_t fd)
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
static int32_t interferer_mboxRead(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len)
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
static int32_t interferer_mboxWrite(rlComIfHdl_t fd, uint8_t* pBuff, uint16_t len)
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
 *      Radar Link Registered Callback function to compute the CRC.
 *
 *  @retval
 *      Success - 0
 *  @retval
 *      Error   - <0
 */
static rlInt32_t interferer_computeCRC(rlUInt8_t* data, rlUInt32_t dataLen, rlUInt8_t crcType, rlUInt8_t* crc)
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
    if (CRC_getTransactionId (gInterfererMCB.crcHandle, &signGenCfg.transactionId, &errCode) < 0)
    {
        System_printf ("Error: CRC Driver Get transaction id failed [Error code %d]\n", errCode);
        return -1;
    }

    /* Populate the signature generation configuration: */
    signGenCfg.ptrData = (uint8_t*)data;
    signGenCfg.dataLen = dataLen;

    /* Compute the signature for the specific data on Channel-1 */
    if (CRC_computeSignature (gInterfererMCB.crcHandle, &signGenCfg, &errCode) < 0)
    {
        System_printf ("Error: CRC Driver compute signature failed [Error code %d]\n", errCode);
        return -1;
    }

    /* Get the Signature for Channel */
    if (CRC_getSignature (gInterfererMCB.crcHandle, signGenCfg.transactionId, &signature, &errCode) < 0)
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
static void interferer_mmwaveLinkMgmtTask (UArg arg0, UArg arg1)
{
    RL_P_OSI_SPAWN_ENTRY    spawnFxn;
    uintptr_t               key;
    Semaphore_Params        semParams;

    /* Initialize the mmWave Link Semaphore: */
    Semaphore_Params_init(&semParams);
    semParams.mode  = Semaphore_Mode_BINARY;
    gInterfererMCB.linkSemaphore = Semaphore_create(0, &semParams, NULL);

    /* Execute forever: */
    while (1)
    {
        /* Pending on the link semaphore */
        Semaphore_pend (gInterfererMCB.linkSemaphore, BIOS_WAIT_FOREVER);

        /* Critical Section: We record the spawn function which is to be executed */
        key = Hwi_disable();
        spawnFxn = gInterfererMCB.spawnFxn;
        gInterfererMCB.spawnFxn = NULL;
        Hwi_restore (key);

        /* Execute the spawn function: */
        spawnFxn (NULL);
    }
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
static rlInt32_t interferer_spawn (RL_P_OSI_SPAWN_ENTRY pEntry, const void* pValue, uint32_t flags)
{
    /* Record the function which is to be spawned. */
    if (gInterfererMCB.spawnFxn != NULL)
        gInterfererMCB.spawnOverrun++;

    /* Record the entry to be spawned. */
    gInterfererMCB.spawnFxn = pEntry;

    /* Post the semaphore and wake up the link management task */
    Semaphore_post (gInterfererMCB.linkSemaphore);
    return 0;
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
static rlInt32_t interferer_enableDevice(rlUInt8_t deviceIndex)
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
static rlInt32_t interferer_disableDevice(rlUInt8_t deviceIndex)
{
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
static void interferer_maskHostIRQ(rlComIfHdl_t fd)
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
void interferer_unmaskHostIRQ(rlComIfHdl_t fd)
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
static rlInt32_t interferer_registerInterruptHandler(rlUInt8_t deviceIndex, RL_P_EVENT_HANDLER pHandler, void* pValue)
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

static void interferer_asyncEventHandler(uint8_t devIndex, uint16_t sbId, uint16_t sbLen, uint8_t *payload)
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
                case RL_RF_AE_INITCALIBSTATUS_SB:
                {
                    gInitTimeCalibStatus = ((rlRfInitComplete_t*)payload)->calibStatus;
                    if(gInitTimeCalibStatus != 0U)
                    {
                        System_printf ("Debug: Init time calibration status [0x%x] \n", gInitTimeCalibStatus);
                    }
                    else
                    {
                        System_printf ("Error: All Init time calibrations Failed:\n");
                    }
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
        default:
        {
            System_printf ("Error: Asynchronous Event SB Id %d not handled with msg ID [0x%x] \n", asyncSB,msgId);
            break;
        }
    }
}

int32_t interferer_initLink (rlUInt8_t deviceType, rlUInt8_t platform)
{
    Task_Params         taskParams;
    CRC_Config          crcCfg;
    rlClientCbs_t       RlApp_ClientCtx;
    int32_t             errCode;

    /* Initialize and populate the Mmwave Link MCB */
    memset ((void*)&gInterfererMCB, 0, sizeof(interferer_MCB));

    /*****************************************************************************
     * Start CRC driver:
     *****************************************************************************/

    /* Setup the default configuration: */
    CRC_initConfigParams(&crcCfg);

    /*******************************************************************************
     * This is the configuration for the 16bit CRC Type:
     *******************************************************************************/
    crcCfg.channel  = CRC_Channel_CH1;
    crcCfg.mode     = CRC_Operational_Mode_FULL_CPU;
    crcCfg.type     = CRC_Type_16BIT;
    crcCfg.bitSwap  = CRC_BitSwap_MSB;
    crcCfg.byteSwap = CRC_ByteSwap_ENABLED;
    crcCfg.dataLen  = CRC_DataLen_16_BIT;


    /* Open the CRC Driver */
    gInterfererMCB.crcHandle = CRC_open (&crcCfg, &errCode);
    if (gInterfererMCB.crcHandle == NULL)
    {
        System_printf ("Error: Unable to open the CRC Channel [Error Code %d]\n", errCode);
        return -1;
    }

    /*****************************************************************************
     * Launch the Mmwave Link Tasks:
     *****************************************************************************/

    /* Initialize and Launch the mmWave Link Management Task: */
    Task_Params_init(&taskParams);
    taskParams.priority = 5;
    Task_create(interferer_mmwaveLinkMgmtTask, &taskParams, NULL);

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
    RlApp_ClientCtx.comIfCb.rlComIfOpen     = interferer_mboxOpen;
    RlApp_ClientCtx.comIfCb.rlComIfClose    = interferer_mboxClose;
    RlApp_ClientCtx.comIfCb.rlComIfRead     = interferer_mboxRead;
    RlApp_ClientCtx.comIfCb.rlComIfWrite    = interferer_mboxWrite;

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
    RlApp_ClientCtx.osiCb.queue.rlOsiSpawn      = interferer_spawn;

    /* Initialize OSI Timer Interface */
    RlApp_ClientCtx.timerCb.rlDelay             = NULL;

    /* Initialize the CRC Interface */
    RlApp_ClientCtx.crcCb.rlComputeCRC          = interferer_computeCRC;

    /* Initialize Device Control Interface */
    RlApp_ClientCtx.devCtrlCb.rlDeviceDisable            = interferer_disableDevice;
    RlApp_ClientCtx.devCtrlCb.rlDeviceEnable             = interferer_enableDevice;
    RlApp_ClientCtx.devCtrlCb.rlDeviceMaskHostIrq        = interferer_maskHostIRQ;
    RlApp_ClientCtx.devCtrlCb.rlDeviceUnMaskHostIrq      = interferer_unmaskHostIRQ;
    RlApp_ClientCtx.devCtrlCb.rlRegisterInterruptHandler = interferer_registerInterruptHandler;

    /* Initialize the Asynchronous Event Handler: */
    RlApp_ClientCtx.eventCb.rlAsyncEvent    = interferer_asyncEventHandler;

    /* Power on the Device */
    if (rlDevicePowerOn(1U, RlApp_ClientCtx) != 0)
    {
        System_printf("Error: Power on request to the BSS failed\n");
        return -1;
    }
    return 0;
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
void interferer_startTest(uint8_t platformType, uint8_t platformCore)
{
    int32_t retVal = 0;

    /* Setup and initialize the mmWave Link: */
    if (interferer_initLink (platformType, platformCore ) < 0)
    {
        System_printf("Error: interferer_initLink failed\n");
        return;
    }


    rlRfBootStatusCfg_t statusCfg = {0};
    /* Get RF Boot status */
    retVal = rlGetRfBootupStatus(RL_DEVICE_MAP_INTERNAL_BSS, &statusCfg);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: rlGetRfBootupStatus retVal=%d\n", retVal);
        return;
    }


    rlRfDevCfg_t rfDevCfg = {0};
    /* Set Device configuration */
    retVal = rlRfSetDeviceCfg(RL_DEVICE_MAP_INTERNAL_BSS, &rfDevCfg);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: rlRfSetDeviceCfg retVal=%d\n", retVal);
        return;
    }


    /* Set channel configuration */
    retVal = rlSetChannelConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlChanCfg_t*)&chCfg);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: setChannelConfig retVal=%d\n", retVal);
        return;
    }


    /* Set ADC out configuration */
    retVal = rlSetAdcOutConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlAdcOutCfg_t*)&adcOutCfgArgs);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
     
        System_printf("Error: setAdcOutConfig retVal=%d\n", retVal);
        return;
    }


    /* Set mmWave Link low power mode Configuration to the BSS */
    if((platformType == RL_AR_DEVICETYPE_16XX) || (platformType == RL_AR_DEVICETYPE_14XX))
    {
        /* Set Low power mode configuration */
        retVal = rlSetLowPowerModeConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlLowPowerModeCfg_t*)&lowPowerModeCfg);

        /* Check for mmWaveLink API call status */
        if(retVal != 0)
        {
            /* Error: Link reported an issue. */
            System_printf("Error: setLowPowerMode retVal=%d\n", retVal);
            return;
        }
    }


    rlRfInitCalConf_t data = { 0 };
    data.calibEnMask = 0x1FF0;
    /* Enable/Disable calibrations */
    retVal = rlRfInitCalibConfig(RL_DEVICE_MAP_INTERNAL_BSS, &data);
    /* Check for mmWaveLink API call status */
    if (retVal != 0)
    {
        System_printf ("Error: Unable to rlRfInitCalibConfig [Error %d]\n", retVal);
        return;
    }


    /* RF Initialization */
    retVal = rlRfInit(RL_DEVICE_MAP_INTERNAL_BSS);
    if (retVal != 0)
    {
        System_printf ("Error: Unable to start RF [Error %d]\n", retVal);
        return;
    }
    while(gInitTimeCalibStatus == 0U)
    {
        /* Sleep and poll again: */
        Task_sleep(1);
    }
    gInitTimeCalibStatus = 0U;
    System_printf("Debug: RF start successfully\n");


    /* Set LDO bypass configuration */
    retVal = rlRfSetLdoBypassConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlRfLdoBypassCfg_t*)&rfLdoBypassCfg);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: rlRfSetLdoBypassConfig retVal=%d\n", retVal);
        return;
    }


    /* Set continue mode configuration */
    retVal = rlSetContModeConfig(RL_DEVICE_MAP_INTERNAL_BSS, (rlContModeCfg_t*)&contModeCfg);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: rlSetContModeConfig retVal=%d\n", retVal);
        return;
    }


    /* Enable Continous mode */
    retVal = rlEnableContMode(RL_DEVICE_MAP_INTERNAL_BSS, (rlContModeEn_t*)&contModeEn);
    /* Check for mmWaveLink API call status */
    if(retVal != 0)
    {
        /* Error: Link reported an issue. */
        System_printf("Error: rlEnableContMode retVal=%d\n", retVal);
        return;
    }

    System_printf("Interferer ready at %d mHz continuous wave\n",
                     contModeCfg.startFreqConst/START_FREQ_CONST_VAL_PER_MHZ);

    //Sleep indefinitely
    Task_setPri(Task_self(), -1);

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
static void interferer_initTask(UArg arg0, UArg arg1)
{

    /* Initialize the Mailbox */
    Mailbox_init(MAILBOX_TYPE_MSS);
#ifdef SOC_XWR16XX
    interferer_startTest(RL_AR_DEVICETYPE_16XX, RL_PLATFORM_MSS);
#else
#ifdef SOC_XWR18XX
    interferer_startTest(RL_AR_DEVICETYPE_18XX, RL_PLATFORM_MSS);   
#endif
#endif

    BIOS_exit(0);
    return;
}


/**
 *  @b Description
 *  @n
 *      Entry point
 *
 *  @retval
 *      Not Applicable.
 */
int32_t main (void)
{
    Task_Params        taskParams;
    int32_t         errCode;
    SOC_Handle      socHandle;
    SOC_Cfg         socCfg;

    /* Initialize the ESM: */
    ESM_init(0U);

    /* Initialize the SOC confiugration: */
    memset ((void *)&socCfg, 0, sizeof(SOC_Cfg));

    /* Populate the SOC configuration: */
    socCfg.clockCfg = SOC_SysClock_INIT;

    /* Initialize the SOC Module: This is done as soon as the application is started
     * to ensure that the MPU is correctly configured. */
    socHandle = SOC_init (&socCfg, &errCode);
    if (socHandle == NULL)
    {
        System_printf ("Error: SOC Module Initialization failed [Error code %d]\n", errCode);
        return -1;
    }

    /* Wait for BSS powerup */
    if (SOC_waitBSSPowerUp(socHandle, &errCode) < 0)
    {
        /* Debug Message: */
        System_printf ("Debug: SOC_waitBSSPowerUp failed with Error [%d]\n", errCode);
        return 0;
    }
    ADCBuf_Params       ADCBufparams;
    /*****************************************************************************
     * Initialize ADCBUF driver
     *****************************************************************************/
    ADCBuf_init();

    /* ADCBUF Params initialize */
    ADCBuf_Params_init(&ADCBufparams);
    ADCBufparams.chirpThresholdPing = 1;
    ADCBufparams.chirpThresholdPong = 1;
    ADCBufparams.continousMode  = 0;
    ADCBufparams.socHandle      = socHandle;

    adcbufHandle = ADCBuf_open(0, &ADCBufparams);
    if (adcbufHandle == NULL)
    {
        System_printf("Error: Unable to open the ADCBUF Instance\n");
        return -1;
    }

    /* Initialize the Task Parameters. */
    Task_Params_init(&taskParams);
    taskParams.priority = 5;
    Task_create(interferer_initTask, &taskParams, NULL);

    /* Start BIOS */
    BIOS_start();
    return 0;
}
